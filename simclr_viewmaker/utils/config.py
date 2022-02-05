from attrdict import AttrDict
from .data_info import *
from .parsing import args
from .paths import *
from dataprocessing.transforms import *

MODE = args.mode

def get_num_encoder_channels(system: str):
  return NUM_LEADS if system == "pclr" else 1

def get_configs():
  if MODE not in ["pt", "le", "ft", "vm"]: return AttrDict(vars(args))

  _CLOCS_SYSTEMS = ["cmsc", "cmlc", "cmsmlc"]

  encoder = args.encoder
  num_encoder_channels = get_num_encoder_channels(args.system)

  _transform = None
  _args = None
  _metric = None
  _dataset_type = "PhysionetDataset"
  _misc = None
  _resume_path = args.ckpt_path
  if MODE == "pt":
    if args.system in _CLOCS_SYSTEMS:
      _transform_type = SingleRandomTransform
    else:
      _transform_type = DoubleRandomTransform
    
    _transform = _transform_type(
      RandomRotation(args.max_theta),
      RandomScale(args.max_scale),
      RandomGaussian(args.gaussian),
      RandomTimeMask(args.tmask_p)
    )

    _args = {
      "encoder": encoder,
      "num_channels": num_encoder_channels,
      "hparams": {
        "lr": args.lr,
        "temp": args.temp,
        "embedding_dim": args.embedding_dim
      }
    }
    
    _metric = {
      "monitor": f"{MODE}_val_loss",
      "mode": "min"
    }

    if args.system == "simclr":
      _dataset_type = "PhysionetDataset"
    elif args.system == "simcg-l" or args.system == "cmlc" or args.system == "simcg-l-vm":
      _dataset_type = "CMLCDataset"
    elif args.system == "cmsc":
      _dataset_type = "CMSCDataset"
    elif args.system == "simcg-sl" or args.system == "cmsmlc":
      _dataset_type = "CMSMLCDataset"
    
  elif MODE == "vm":
    if args.system in _CLOCS_SYSTEMS:
      _transform_type = VMSingleRandomTransform
    else:
      _transform_type = VMDoubleRandomTransform
    
    _transform = _transform_type()

    _args = {
      "encoder": encoder,
      "num_channels": num_encoder_channels,
      "hparams": {
        "lr": args.lr,
        "temp": args.temp,
        "embedding_dim": args.embedding_dim
      }
    }
    
    _metric = {
      "monitor": f"{MODE}_val_loss",
      "mode": "min"
    }

    if args.system == "simclr":
      _dataset_type = "PhysionetDataset"
    elif args.system == "simcg-l" or args.system == "cmlc" or args.system == "simcg-l-vm":
      _dataset_type = "CMLCDataset"
    elif args.system == "cmsc":
      _dataset_type = "CMSCDataset"
    elif args.system == "simcg-sl" or args.system == "cmsmlc":
      _dataset_type = "CMSMLCDataset"

  else:
    _args = {
      "prefix": MODE,
      "num_classes": NUM_CLASSES,
      "hparams": {
        "lr": args.lr
      }
    }

    _metric = {
      "monitor": f"{MODE}_val_auroc",
      "mode": "max"
    }

    _ds_pt_path = None
    if args.ckpt_path and ("pt" in args.ckpt_path.split("/") or "vm" in args.ckpt_path.split("/")):
      _ds_pt_path = args.ckpt_path
      _resume_path = None
    
    _misc = {
      "num_splits": args.num_splits,
      "frac": args.ds_frac,
      "path": _ds_pt_path,
      "dummy_args": {
        "encoder": encoder,
        "num_channels": num_encoder_channels,
        "hparams": {
          "lr": args.lr,
          "temp": 0,
          "embedding_dim": 1
        }
      }
    }

  _trainer_settings = {
    "gpus": 1,
    "sync_batchnorm": True, 
    "default_root_dir": CHECKPOINT_DIR,
    "deterministic": True,
    "fast_dev_run": args.fast_dev_run,
    "terminate_on_nan": True,
    "max_epochs": 250,
    "precision": 16,
    "auto_lr_find": args.auto_lr_find,
    "auto_scale_batch_size": args.auto_scale_batch_size,
    "resume_from_checkpoint": _resume_path
  }

  return AttrDict({
    "metric": _metric,
    "args": _args,
    "trainer": _trainer_settings,
    "dm": {
      "dataset_type": _dataset_type,
      "transform": _transform,
      "batch_size": args.batch_size,
      "num_workers": args.num_workers
    },
    "misc": _misc
  })

configs = get_configs()
