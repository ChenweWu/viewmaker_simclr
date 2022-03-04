

## Installation 

### Virtual Environment - Conda
Before your first use, create the conda environment:
```
conda create -n selfecg python=3.8.6
```

Before every use, activate the virtual environment:
```
conda activate selfecg
```

Install the dependencies from requirements.txt:
```
pip install -r requirements.txt
```

## Directory Formatting
Set the environment variable `3KG_PATH`, which is a path that data is read from and model results are saved to. The directory should be structured as follows:
```
<3KG_PATH>
  /data        : Where the formatted tensor datasets will be saved to (automatically created/overwritten)
  /raw         : Where all the .mat and .hea files should reside for all ECGs in the desired dataset. No
                 subdirectories are allowed for this folder. It is expected that all files follow the 
                 formatting set by Physionet 2020 (user must provide)
  /checkpoints : Where all model checkpoints will be saved (automatically created)
  splits.csv   : A CSV file containing all diagnosis labels and fold numbers (user must provide).
```

## Running 

Models are trained and evaluated by executing `python run.py` with the following flags:
```
dir: What enclosing folder all generated files related to a run (checkpoints, logs, etc.) should be saved under.
ds_frac: The percentage of labels that the model is allowed to see in the downstream (either linear evaluation or full fine-tune) setting. Must be a floating point number within (0,1].
system: Whether to use the default SimCLR pretraining setup, or any of the CLOCS implementation systems (CMLC, CMSC, CMSMLC).
encoder: What model architecture to train with.
``` 

**Evaluating**

## Pre-trained Models
Model checkpoints from our experiments are available for download here (need to add).
