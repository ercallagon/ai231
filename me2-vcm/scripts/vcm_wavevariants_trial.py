#!/usr/bin/env python3
"""Isolated matched BC8 continuation with optional training waveform views."""
import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import vcm_bc8_source_finetune as source
import vcm_bc8_trial as reference
import vcm_grouped_bc8_trial as grouped
from hey_swift.vcm import train_repro
from hey_swift.vcm.augmentation import augment_mel
from hey_swift.vcm.protocol import sha256

STAGE = Path("work/vcm_wavevariants_v1")
PARENT = Path("experiments/vcm/run_043_hf_grouped_v2b_bc8_sourceft_s42")
TAGS = {"control":"run_046_hf_grouped_v2b_bc8_aug_control_s42",
        "wave":"run_047_hf_grouped_v2b_bc8_wavevariants_s42"}


def verify():
    ready=json.loads((STAGE/"readiness.json").read_text())
    if not ready["passed"]: raise RuntimeError("Preparation not validated")
    for path,digest in ready["sha256"].items():
        if sha256(path)!=digest: raise ValueError(f"Trial artifact changed: {path}")
    return ready


class VariantDataset(source.SourceDataset):
    def __init__(self,config,*args,**kwargs):
        super().__init__(config,*args,**kwargs)
        self.use_views=self.augment and config["experiment"]["waveform_views"]
        self.views=np.load(STAGE/"mels.npy",mmap_mode="r") if self.use_views else None
        self.view_index=json.loads((STAGE/"index.json").read_text()) if self.use_views else {}

    def __getitem__(self,i):
        row=self.rows[i]; key=row["audio_path"]
        view_rng=np.random.default_rng(np.random.SeedSequence([self.seed,int(self.epoch),i,101]))
        use=self.use_views and key in self.view_index and view_rng.random()<.5
        mel=np.array(self.views[self.view_index[key]] if use else self.cache[self.index[key]],copy=True)
        if self.augment:
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,int(self.epoch),i]))
            mel=augment_mel(mel,self.augmentation,rng)
        sample=(torch.from_numpy(mel),int(self.labels[i]),int(self.slots[i]))
        if self.augment:
            return (*sample,self.real_multiplier if row["is_synthetic"]=="0" else 1.)
        return sample


@contextlib.contextmanager
def engine(config):
    parent=source.PARENT; source.PARENT=PARENT
    try:
        with source.refinement_engine(config,{}):
            train_repro.CachedDataset=VariantDataset
            yield
    finally:
        source.PARENT=parent


def train(arm):
    ready=verify(); config=str(STAGE/f"{arm}.yaml")
    reference.check_candidate(config)
    with engine(config): output=train_repro.run(TAGS[arm],config)
    proof={"parent":str(PARENT),"parent_checkpoint_sha256":sha256(PARENT/"best.pt"),
        "arm":arm,"preparation_sha256":sha256(STAGE/"readiness.json"),
        "waveform_views":arm=="wave","new_training_utterances":False,
        "maximum_variants":ready["variant_count"],"test_holdout_or_challenge_used_for_choices":False}
    (output/"finetune_provenance.json").write_text(json.dumps(proof,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["train","eval","calibrate","export"])
    parser.add_argument("--arm",choices=list(TAGS),required=True)
    parser.add_argument("--split",default="val",choices=["val","challenge","test_core","test_negatives","test"])
    parser.add_argument("--exp-dir"); parser.add_argument("--config")
    args=parser.parse_args(); torch.set_num_threads(1)
    config=args.config or str(STAGE/f"{args.arm}.yaml")
    if args.mode=="train": train(args.arm); return
    grouped.TAG,grouped.CONFIG=TAGS[args.arm],config
    import sys
    sys.argv=[sys.argv[0],args.mode,"--config",config]
    if args.mode=="eval":
        sys.argv.extend(["--split",args.split])
        if args.exp_dir: sys.argv.extend(["--exp-dir",args.exp_dir])
    grouped.main()


if __name__=="__main__": main()
