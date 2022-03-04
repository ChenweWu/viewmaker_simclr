![image](https://user-images.githubusercontent.com/36363910/156784737-1cc4bb54-4673-4b6c-a935-eacf723ef867.png)

# viewmaker_simclr

For pretraining, python run.py -vm --encoder [basic_cnn/resnet] --batch_size --auto_lr_find
For linear evaluation, python run.py -le --checkpoint [path]

This project is developed by Chenwei Wu & Kathryn Wantlin from the 3kg base code of Ryan Han and Bryan Gopal.

A viewmaker network is trained adversarially with encoders to auto generate augmentation views for ECG data and pretrained models could be used for downstream tasks.

Data used are from physionet challenge 2020.

# Problem Formulation

Motivation: 
• Generative networks are noteworthy on natural images, let's make it work in the field of ECG? (where SSL recently just start to beat SL) Do they medically make sense?

e.g. Alex's ICLR 2021 paper generates view on a CIFAR cat

![image](https://user-images.githubusercontent.com/36363910/156784794-d9f41104-ffd3-4ed2-be32-8040932686b1.png)

Target: 
• Develop viewmaker networks, which are generative models with stochastic boundaries for data augmentations, via Pytorch Lightning, to adversarially auto-learn and generate augmentations on 12-lead electrocardiogram (ECG) sensor data for contrastive learning tasks, so as to reduce the rigorous trial and error by human experts. 

• Investigate and compare the performance of viewmaker networks to those of other previous contrastive methods, in particular whether viewmaker networks learned views that are medically sensible, and whether they are more robust to corruptions commonly observed in ECG data collection settings. 

# Approach
![image](https://user-images.githubusercontent.com/36363910/156784897-75fae05f-2190-4a76-bdfa-9e28c7540ee3.png)

1.Training viewmaker adversarially with encoder by gradient reversal layer

![image](https://user-images.githubusercontent.com/36363910/156784922-a4194a7d-7fb0-4267-ae14-b37271557a9b.png)

2.Viewmaker architecture

![image](https://user-images.githubusercontent.com/36363910/156784953-d509e14d-16a2-4e02-a70f-de7d73842221.png)

3.Patient supervised contrastive loss

![image](https://user-images.githubusercontent.com/36363910/156785023-e27f2f76-7ae3-4047-a2dd-c76b56bc1c91.png)

![image](https://user-images.githubusercontent.com/36363910/156785032-ef073d45-9618-405b-ba7c-c2d2a91d6556.png)

4.per lead vs not per lead


# Future steps

Combine with Hand Crafted SOTA?
ECG->VCG
(Inverse Dowers transformation to 3d space)
VCG:                                  ECG:
Rotation                            Time Masking
Scale

![image](https://user-images.githubusercontent.com/36363910/156785134-b82efeee-fa58-4e6b-b698-e460c1e4df45.png)

# Current Results & Next Steps

![image](https://user-images.githubusercontent.com/36363910/156785203-c9280c2c-03df-4fdc-8a3e-0eaa18fc96b7.png)

Interpreting results With Clinicians






