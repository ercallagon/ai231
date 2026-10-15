"""Exact metric function used by the frozen six-second experiment."""
from .release_data import labels, tax


def metrics(logits,slots,valid,rs,split,T=1.,tau=0.):
    import numpy as np
    from scipy.special import softmax
    y,ys,unsupported=labels(rs,split);t=tax();real=np.array([r["is_synthetic"]=="0" for r in rs])
    probabilities=softmax(logits/T,axis=1);pred=np.where(valid,probabilities.argmax(1),20)
    correct=valid&(pred==y);joint=correct.copy();known=np.ones(len(rs),bool)
    for name in t.slotted_intents:
        ix=y==t.intent_to_idx[name];known[ix&(ys<0)&~unsupported]=False
        joint[ix&(ys>=0)] &= slots[name].argmax(1)[ix&(ys>=0)]==ys[ix&(ys>=0)]
    hard_reject=(pred==15)&(slots["TEMPERATURE"].argmax(1)==3)
    accepted=valid&(pred!=19)&~hard_reject&(probabilities.max(1)>=tau)
    invalid=(y==19)|unsupported;supported=known&~invalid
    success=accepted&joint&supported;semantic=(invalid&~accepted)|success
    def stat(mask,flag):
        n=int(mask.sum());k=int((mask&flag).sum())
        return dict(n=n,correct=k,rate=k/n if n else None)
    cm=np.zeros((20,21),int);np.add.at(cm,(y,pred),1)
    families={}
    f1=[]
    for i,name in enumerate(list(t.intents)+[t.oos_class]):
        ix=y==i;tp=int(cm[i,i]);p=tp/max(1,int(cm[:,i].sum()));r=tp/max(1,int(ix.sum()));f=2*p*r/max(1e-12,p+r);f1.append(f)
        families[name]=dict(intent=stat(ix,correct),joint=stat(ix&supported,joint),accepted_joint=stat(ix&supported,success),
            precision=p,recall=r,f1=f,failures=int((ix&~valid).sum()),false_reject=int((ix&supported&~accepted).sum()))
    return dict(overall_intent=stat(np.ones(len(rs),bool),correct),real_intent=stat(real,correct),
        synthetic_intent=stat(~real,correct),supported_joint=stat(supported,joint),
        supported_accepted_joint=stat(supported,success),real_supported_accepted=stat(supported&real,success),
        synthetic_supported_accepted=stat(supported&~real,success),semantic=stat(known,semantic),
        wrong_supported_accepts=int((accepted&supported&~joint).sum()),false_rejects=int((~accepted&supported).sum()),
        oos_false_accept=stat(y==19,accepted),unsupported_false_accept=stat(unsupported,accepted),
        unsupported_temperature_false_accept=stat(unsupported&(y==15),accepted),
        temperature_joint=stat((y==15)&supported,joint),temperature_accepted=stat((y==15)&supported,success),
        failures=int((~valid).sum()),unscored=int((~known).sum()),macro_f1=float(np.mean(f1)),
        confusion=cm.tolist(),failure_column=20,families=families,T=float(T),threshold=float(tau))
