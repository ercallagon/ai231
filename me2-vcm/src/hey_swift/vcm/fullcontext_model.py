"""Scratch BC8 with shared temporal CNN and masked attentive statistics pooling."""
import torch
from torch import nn
from torch.nn import functional as F
from hey_swift.vcm.broadcast import build_broadcast_model


class TemporalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(128,128,5,padding=2*dilation,dilation=dilation,groups=128,bias=False),
            nn.BatchNorm1d(128),nn.SiLU(),nn.Conv1d(128,128,1,bias=False),
            nn.BatchNorm1d(128),nn.Dropout(.10))

    def forward(self,x):
        return F.silu(x+self.net(x))


class ContextBC8(nn.Module):
    def __init__(self,tax):
        super().__init__()
        core=build_broadcast_model(20,{n:tax.slot_classes(n) for n in tax.slotted_intents})
        self.stem,self.blocks,self.tail=core.stem,core.blocks,core.tail
        self.time_pool=nn.AvgPool2d((1,2),stride=(1,2))
        self.project=nn.Sequential(nn.Conv1d(256,128,1,bias=False),nn.BatchNorm1d(128),nn.SiLU())
        self.temporal=nn.Sequential(*(TemporalBlock(d) for d in (1,3,9)))
        self.attention=nn.Sequential(nn.Conv1d(128,64,1),nn.Tanh(),nn.Conv1d(64,1,1))
        self.embedding=nn.Sequential(nn.Linear(256,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(.10))
        self.intent_head=nn.Linear(256,20)
        self.slot_head_map=nn.ModuleDict({n:nn.Linear(256,len(tax.slot_classes(n))) for n in tax.slotted_intents})

    def forward(self,mel,frame_mask):
        x=self.tail(self.blocks(self.time_pool(self.stem(mel)))).mean(dim=2)
        x=self.temporal(self.project(x))
        mask=F.max_pool1d(frame_mask,2,2)
        scores=self.attention(x)+(1-mask)*-10000.
        weights=F.softmax(scores,dim=-1)*mask
        weights=weights/weights.sum(dim=-1,keepdim=True).clamp_min(1e-8)
        mean=(weights*x).sum(dim=-1)
        variance=(weights*(x-mean.unsqueeze(-1)).square()).sum(dim=-1)
        z=self.embedding(torch.cat((mean,variance.clamp_min(1e-6).sqrt()),dim=1))
        return {"intent":self.intent_head(z),"slots":{n:head(z) for n,head in self.slot_head_map.items()}}
