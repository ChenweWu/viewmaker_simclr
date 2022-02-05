import os
from .parsing import args
from shutil import rmtree
from typing import List

GENERAL_PATH = os.path.expanduser(os.getenv("SSL_ECG_PATH"))
RAW_DATA_PATH = f"{GENERAL_PATH}/raw"
SAVED_DATA_PATH = f"{GENERAL_PATH}/data"
SPLITS_PATH = f"{GENERAL_PATH}/splits.csv"

if args.clear_data and os.path.exists(SAVED_DATA_PATH): rmtree(SAVED_DATA_PATH)
os.makedirs(SAVED_DATA_PATH, exist_ok=True)

def all_paths_exist(*paths: List[str]):
  for path in paths:
    if not os.path.exists(path): return False
  return True

if args.mode in ["pt", "le", "ft","vm"]:
  _suffix_dir = f"{args.mode}/{args.ds_frac}" if (args.mode not in ["pt", "vm"]) else args.mode
  CHECKPOINT_DIR = f"{GENERAL_PATH}/checkpoints/{args.system}/{args.encoder}/{args.dir}/{_suffix_dir}"
  os.makedirs(CHECKPOINT_DIR, exist_ok=True)
