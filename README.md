# viewmaker_simclr

For pretraining, python run.py -vm --encoder [basic_cnn/resnet] --batch_size --auto_lr_find
For linear evaluation, python run.py -le --checkpoint [path]

This project is developed by Chenwei Wu & Kathryn Wantlin from the 3kg base code of Ryan Han and Bryan Gopal.

A viewmaker network is trained adversarially with encoders to auto generate augmentation views for ECG data and pretrained models could be used for downstream tasks.

Data used are from physionet challenge 2020.
