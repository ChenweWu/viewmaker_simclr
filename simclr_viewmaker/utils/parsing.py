from argparse import ArgumentParser
from attrdict import AttrDict
import os
from psutil import cpu_count
from systems import __encoders__
from typing import List

def parse_args():
  def valid_fraction(frac):
    frac = float(frac)
    if frac <= 0 or frac > 1:
      raise ValueError(f"Specified fraction not inside (0,1]: {frac}")
    return frac
  
  def get_encoder_idx(path_list: List[str]):
    intersect = list(set(path_list) & set(__encoders__))
    if len(intersect) != 1:
      raise ValueError("Could not determine encoder type from checkpoint path.")

    encoder_idx = path_list.index(intersect[0])
    if encoder_idx == len(path_list) - 1:
      raise ValueError("Improperly formatted checkpoint path: expected at least one "
                      f"directory after encoder directory \"{path_list[encoder_idx]}\"")

    return encoder_idx
  
  def parse_ckpt_path(path: str):
    path_list = path.split("/")
    encoder_idx = get_encoder_idx(path_list)

    return AttrDict({
      "system": path_list[encoder_idx - 1],
      "encoder": path_list[encoder_idx],
      "experiment": "/".join(path_list[encoder_idx - 1:encoder_idx + 4]),
      "dir": "/".join(path_list)
    })

  model = ArgumentParser(add_help=False)
  startup = model.add_mutually_exclusive_group(required=True)
  startup.add_argument("--ckpt_path", type=os.path.expanduser, default=None)
  startup.add_argument("--encoder", type=str, choices=__encoders__)

  train = ArgumentParser(add_help=False)
  hparams = train.add_argument_group("hyperparameters")
  hparams.add_argument("--lr", type=float, default=1e-3)

  settings = train.add_argument_group("training settings")
  settings.add_argument("--fast_dev_run", action="store_true")
  settings.add_argument("--auto_lr_find", action="store_true")
  settings.add_argument("--auto_scale_batch_size", action="store_true")

  data = ArgumentParser(add_help=False)

  dataset = data.add_argument_group("dataset settings")
  dataset.add_argument("--clear_data", action="store_true")
  dataset.add_argument("--crop_time", type=int, default=5)

  dataloading = data.add_argument_group("dataloading settings")
  dataloading.add_argument("--batch_size", type=int, default=512)
  dataloading.add_argument("--num_workers", type=int, default=cpu_count())
  
  parser = ArgumentParser()
  subparsers = parser.add_subparsers(dest="mode", required=True)

  pt = subparsers.add_parser("pt", parents=[model, train, data], help="pretraining")
  pt.add_argument("--system", type=str, default="simclr", 
      choices=["simclr", "simcg-l", "simcg-sl", "cmlc", "cmsc", "cmsmlc", "pclr"])
    
  vm = subparsers.add_parser("vm", parents=[model, train, data], help="viewmaker pretraining")
  vm.add_argument("--system", type=str, default="simcg-l-vm", 
      choices=["simclr", "simcg-l", "simcg-l-vm", "simcg-sl", "cmlc", "cmsc", "cmsmlc", "pclr"])

  augs = pt.add_argument_group("augmentations", description="will ignore if given ckpt_path")
  augs.add_argument("--tmask_p", type=float, default=0)
  augs.add_argument("--max_theta", type=float, default=0)
  augs.add_argument("--max_scale", type=float, default=1)
  augs.add_argument("--gaussian", action="store_true")

  augs_vm = vm.add_argument_group("viewmaker augmentations", description="will ignore if given ckpt_path")
#   augs_vm.add_argument("--distortion_budget", type=float, default=0.02)

  pt_hparams = pt.add_argument_group("pretraining hyperparameters")
  pt_hparams.add_argument("--temp", type=float, default=0.1)
  pt_hparams.add_argument("--embedding_dim", type=int, default=128)
    
  vm_hparams = vm.add_argument_group("viewmaker pretraining hyperparameters")
  vm_hparams.add_argument("--temp", type=float, default=0.1)
  vm_hparams.add_argument("--embedding_dim", type=int, default=128)


  ds = ArgumentParser(add_help=False)
  ds.add_argument("ds_frac", type=valid_fraction)
  ds.add_argument("num_splits", type=int)

  subparsers.add_parser("le", parents=[model, train, data, ds], help="linear evaluation")
  subparsers.add_parser("ft", parents=[model, train, data, ds], help="full fine-tuning")

  metric = ArgumentParser(add_help=False)
  metric.add_argument("--metric", choices=["auroc", "f1"], default="auroc")
  metric.add_argument("--per_class", action="store_true")

  bd = subparsers.add_parser("bd", parents=[metric, data], help="bootstrapped differences")
  bd.add_argument("baseline_dir", type=os.path.expanduser)
  bd.add_argument("comparison_dirs", type=os.path.expanduser, nargs="+")

  tm = subparsers.add_parser("tm", parents=[metric, data], help="test metric")
  tm.add_argument("checkpoint_dirs", type=os.path.expanduser, nargs="+")
  tm.add_argument("--bootstrap", action="store_true")

  args = parser.parse_args()

  if args.mode == "bd": 
    args.baseline = parse_ckpt_path(args.baseline_dir)
    args.comparisons = [parse_ckpt_path(comp_dir) for comp_dir in args.comparison_dirs]
    
    del args.baseline_dir
    del args.comparison_dirs
    
    return args
  elif args.mode == "tm":
    args.checkpoints = [parse_ckpt_path(ckpt_dir) for ckpt_dir in args.checkpoint_dirs]
    
    del args.checkpoint_dirs

    return args

  if args.ckpt_path:
    path_list = args.ckpt_path.split("/")
    encoder_idx = get_encoder_idx(path_list)

    args.system = path_list[encoder_idx - 1]
    args.encoder = path_list[encoder_idx]
    args.dir = path_list[encoder_idx + 1]

  elif args.mode != "pt" and args.mode != "vm":
    args.system = "simclr"
    args.dir = "random_init"

  if args.mode == "pt":
    if args.ckpt_path:
      _aug_types = {_aug.dest: _aug.type for _aug in augs._group_actions}
      for _aug_pair in args.dir.split(":"):
        if "=" not in _aug_pair: continue
        _name, _val = _aug_pair.split("=")
        if hasattr(args, _name): 
          setattr(args, _name, _aug_types[_name](_val))
    else:
      _aug_list = []
      for arg in augs._group_actions:
        _val = getattr(args, arg.dest)
        if _val != arg.default: _aug_list.append(f"{arg.dest}={_val}")
      
      if len(_aug_list) == 0:
        args.dir = "no_augs"
      else:
        args.dir = ":".join(_aug_list)
    
    if args.system == "cmsc": args.batch_size *= 8
    elif args.system == "cmlc": args.batch_size *= 2
        
        
  if args.mode == "vm":
    if args.ckpt_path:
      _aug_types = {_aug.dest: _aug.type for _aug in augs._group_actions}
      for _aug_pair in args.dir.split(":"):
        if "=" not in _aug_pair: continue
        _name, _val = _aug_pair.split("=")
        if hasattr(args, _name): 
          setattr(args, _name, _aug_types[_name](_val))
    else:
      _aug_vm_list = []
      for arg in augs_vm._group_actions:
        _val = getattr(args, arg.dest)
        if _val != arg.default: _aug_vm_list.append(f"{arg.dest}={_val}")
      
      if len(_aug_vm_list) == 0:
        args.dir = "no_vm_aug_hyperparams"
      else:
        args.dir = ":".join(_aug_vm_list)
    
    if args.system == "cmsc": args.batch_size *= 8
    elif args.system == "cmlc": args.batch_size *= 2

  if args.fast_dev_run: 
    args.dir = "fast_dev_run"
  
  return args

args = parse_args()