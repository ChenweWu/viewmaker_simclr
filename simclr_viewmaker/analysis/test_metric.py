from attrdict import AttrDict
from dataprocessing import PhysionetDataModule
from .helpers import *
import torch
from tqdm import tqdm

def test_metric(cfg: AttrDict):
  dm = PhysionetDataModule("PhysionetDataset", None, cfg.batch_size, cfg.num_workers)
  dm.setup("test")

  output_messages = []
  for checkpoint in tqdm(cfg.checkpoints):
    metric = initialize_metric(cfg.metric, cfg.per_class)
    if cfg.bootstrap: metric = bootstrap_metric(metric)
    
    models = load_all_checkpoints(checkpoint.system, checkpoint.encoder, checkpoint.dir)

    with torch.no_grad():
      for x, y in tqdm(dm.test_dataloader(), leave=False):
        for model in tqdm(models, leave=False):
          metric(model(x).softmax(dim=-1), y.long())

    result = metric.compute()
    if cfg.bootstrap:
      result = calculate_bootstrapped_stats(result["raw"])

    output_messages.append(f"RESULTS for {checkpoint.experiment}:\n{get_result_message(result)}")

    metric.reset()
  
  print_all_messages(output_messages)