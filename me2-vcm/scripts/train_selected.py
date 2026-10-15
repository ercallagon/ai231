#!/usr/bin/env python3
"""Optional scratch recipe. A fresh output and --execute are required."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--gpu",type=int,default=0)
    ap.add_argument("--data-root",type=Path,default=ROOT/"data/vcm/raw")
    ap.add_argument("--fetch",action="store_true")
    ap.add_argument("--execute",action="store_true")
    args=ap.parse_args()
    command=dict(architecture="bc8_fullcontext_tcn_asp",seed=42,epochs=120,batch=32,
        minimum_epochs=40,patience=20,output=str(args.output),gpu_count=1,
        initialization="scratch",external_pretrained=False,heldout_evaluation=False)
    if not args.execute:
        print(json.dumps(command,indent=2))
        return
    if args.output.exists():
        ap.error("Choose a fresh output directory")
    stats=subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.free","--format=csv,noheader,nounits"],text=True)
    row=next((list(map(int,line.split(","))) for line in stats.splitlines() if int(line.split(",")[0])==args.gpu),None)
    if row is None or row[1]>=40 or row[2]<16384:
        raise RuntimeError("Selected GPU must be below 40% utilization with at least 16 GiB free")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
    import numpy as np
    import torch
    from torch.nn import functional as F
    from hey_swift.vcm.fullcontext import FullContextFeatures,FLOOR
    from hey_swift.vcm.fullcontext_model import ContextBC8
    from hey_swift.vcm import release_data as D
    from hey_swift.vcm.archived_metric_core import metrics
    from hey_swift.vcm.slot_labels import _class_weights,_slot_loss_and_stats
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.30)
    args.output.mkdir(parents=True)
    (args.output/"COMMAND.json").write_text(json.dumps(command,indent=2))
    data={}
    extractor=FullContextFeatures()
    exclusions=json.loads((ROOT/"data/training_exclusions.json").read_text())
    for split in ("train","val"):
        rows=D.manifest(split);D.ensure_files(rows,args.data_root,args.fetch)
        x=np.lib.format.open_memmap(args.output/(split+".npy"),mode="w+",dtype="float32",shape=(len(rows),1,80,600))
        mask=np.zeros((len(rows),1,600),np.float32)
        valid=np.zeros(len(rows),bool)
        for i,wave in D.waves(rows,args.data_root):
            mel,m,receipt=extractor.transform(wave)
            if mel is not None:
                x[i]=mel;mask[i]=m;valid[i]=True
            else:
                x[i]=np.nan
        x.flush()
        eligible=valid.copy()
        if split=="train":
            eligible &= np.array([r["audio_path"] not in exclusions for r in rows])
            if eligible.sum()!=25330:
                raise RuntimeError("Training eligibility differs from the frozen experiment")
        y,ys,_=D.labels(rows,split)
        data[split]=(x,mask,valid,eligible,rows,y,ys)
    class TrainingData:
        def __init__(self):
            self.ix=np.flatnonzero(data["train"][3]);self.epoch=0
        def __len__(self): return len(self.ix)
        def __getitem__(self,j):
            i=int(self.ix[j]);x,mask,valid,eligible,rows,y,ys=data["train"]
            n=int(mask[i].sum());rng=np.random.default_rng([42,self.epoch,i])
            content=np.array(x[i,:,:,:n]);gain=10**(float(rng.uniform(-3,3))/10)
            content=np.log(np.maximum(np.exp(content)-1e-5,0)*gain+1e-5).astype(np.float32)
            for _ in range(2):
                w=int(rng.integers(0,min(12,max(1,n//10))+1));start=int(rng.integers(0,n-w+1))
                content[:,:,start:start+w]=FLOOR
                w=int(rng.integers(0,13));start=int(rng.integers(0,80-w+1));content[:,start:start+w,:]=FLOOR
            start=int(rng.integers(0,600-n+1));out=np.full((1,80,600),FLOOR,np.float32);m=np.zeros((1,600),np.float32)
            out[:,:,start:start+n]=content;m[:,start:start+n]=1
            return out,m,int(y[i]),int(ys[i]),rows[i]["is_synthetic"]=="0"
    device=torch.device("cuda:0");model=ContextBC8(D.tax()).to(device)
    train=TrainingData();real_fraction=np.mean([data["train"][4][i]["is_synthetic"]=="0" for i in train.ix])
    multiplier=.3*(1-real_fraction)/(.7*real_fraction)
    cw=_class_weights(data["train"][5][train.ix].tolist(),20,device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=6,min_lr=1e-6)
    best=-1.;bad=0;history=[]
    with (args.output/"train.log").open("x") as log:
        for epoch in range(120):
            start=time.monotonic();model.train();train.epoch=epoch
            loader=torch.utils.data.DataLoader(train,batch_size=32,shuffle=True,num_workers=0,pin_memory=True,
                generator=torch.Generator().manual_seed(42+epoch))
            loss_total=0.;correct=total=0
            for x,m,y,ys,real in loader:
                x,m,y,ys=x.to(device),m.to(device),y.to(device),ys.to(device)
                weight=torch.where(real.to(device),multiplier,1.)
                opt.zero_grad(set_to_none=True);out=model(x,m)
                ce=F.cross_entropy(out["intent"],y,weight=cw,label_smoothing=.05,reduction="none")
                loss=(ce*weight).sum()/(cw[y]*weight).sum()
                slot_loss,_,_=_slot_loss_and_stats(out,y,ys,D.tax(),torch.nn.CrossEntropyLoss())
                if slot_loss is not None: loss=loss+.75*slot_loss
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
                total+=len(y);correct+=int((out["intent"].argmax(1)==y).sum());loss_total+=loss.item()*len(y)
            vx,vm,valid,_,rows,_,_=data["val"]
            logits=np.zeros((len(rows),20),np.float32)
            slots={n:np.zeros((len(rows),len(D.tax().slot_classes(n))),np.float32) for n in D.tax().slotted_intents}
            ix=np.flatnonzero(valid);model.eval()
            with torch.no_grad():
                for k in range(0,len(ix),32):
                    batch=ix[k:k+32]
                    out=model(torch.from_numpy(np.array(vx[batch])).to(device),torch.from_numpy(vm[batch]).to(device))
                    logits[batch]=out["intent"].cpu().numpy()
                    for n,v in out["slots"].items(): slots[n][batch]=v.cpu().numpy()
            met=metrics(logits,slots,valid,rows,"val")
            score=.7*met["overall_intent"]["rate"]+.3*met["real_intent"]["rate"]
            improved=score>best
            if improved: best=score;bad=0
            else: bad+=1
            sched.step(score)
            ck=dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                epoch=epoch+1,selection_score=score,arch="bc8_fullcontext_tcn_asp",seed=42)
            torch.save(ck,args.output/"last.pt")
            if improved: torch.save(ck,args.output/"best.pt")
            record=dict(epoch=epoch+1,loss=loss_total/total,train_augmented_accuracy=correct/total,
                val_intent=met["overall_intent"]["rate"],val_real_intent=met["real_intent"]["rate"],
                selection_score=score,patience_used=bad,seconds=time.monotonic()-start)
            history.append(record);log.write(json.dumps(record)+"\n");log.flush();print(json.dumps(record),flush=True)
            (args.output/"history.json").write_text(json.dumps(history,indent=2))
            if epoch+1>=40 and bad>=20: break
    print("Scratch run complete. New weights require validation calibration; do not assume the released threshold transfers.")


if __name__=="__main__":
    main()
