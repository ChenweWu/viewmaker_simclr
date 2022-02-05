#!/usr/bin/env python
__author__ = "Bryan Gopal"

from analysis import bootstrap, test_metric
from dataprocessing import PhysionetDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from systems import SimCLR, DownstreamTask
from utils import configs as cfg, MODE
import torch
import wandb
def main():
  seed_everything(6)
  
  if MODE == "bd": 
    bootstrap(cfg)
  elif MODE == "tm":
    test_metric(cfg)
  elif MODE == "pt": 
    print(torch.cuda.device_count())
    pre_train()
  elif MODE == "vm":
    print(torch.cuda.device_count())
    pre_train_vm()
  else: 
    for split in range(cfg.misc.num_splits):
      _shared_downstream(split)

def pre_train():
  model = SimCLR(**cfg.args)
  dm = PhysionetDataModule(**cfg.dm)

  trainer = Trainer(callbacks=_get_callbacks(), **cfg.trainer)
  trainer.tune(model, datamodule=dm)
  trainer.fit(model, datamodule=dm)
def pre_train_vm():
  model = SimCLR_VM(**cfg.args)
  dm = PhysionetDataModule(**cfg.dm)
  config = {
  "budget": 0.02,
  "model": "basic_cnn",
  "learning_rate": "auto-lr-find",
  "batch_size": 64,
  }  

  wandb.init(project='3kg_vm_tests', entity='viewmaker-ecg',sync_tensorboard=True, config=config)
  wandb.run.name="budget 0.02 basic cnn"
  trainer = Trainer(callbacks=_get_callbacks(), **cfg.trainer)
  trainer.tune(model, datamodule=dm)
  trainer.fit(model, datamodule=dm)
def _shared_downstream(split: int):
  model = DownstreamTask(_get_ds_encoder(), **cfg.args)
  dm = PhysionetDataModule(**cfg.dm)
  dm.fractionate(cfg.misc.frac, split)

  trainer = Trainer(callbacks=_get_callbacks(), **cfg.trainer)
  trainer.tune(model, datamodule=dm)
  trainer.fit(model, datamodule=dm)
  trainer.test(ckpt_path="best", datamodule=dm, verbose=False)

def _get_ds_encoder() -> SimCLR:
  args = cfg.misc.dummy_args
  path = cfg.misc.path
  model = None
  if path:
    model = SimCLR.load_from_checkpoint(path, encoder=args.encoder, num_channels=args.num_channels, strict=False)
  else:
    model = SimCLR(**args)
  return model.enable_downstream()

def _get_callbacks():
  m = cfg.metric
  return [
    EarlyStopping(  **m, patience=30),
    ModelCheckpoint(**m, every_n_epochs=1, save_top_k=1, filename=f"{{epoch}}-{{{m.monitor}:.4f}}")
  ]

if __name__ == "__main__":
  main()
