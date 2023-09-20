 # Source Code for Contrastive Generative Learning in Neuro-Imaging

**Abstract:**

Diabetes and major depressive disorder (MDD) have shown associations with brain structure alterations. This project intends to disentangle the diabetes and/or MDD specific patterns based on the 3D T1 structural brain magnetic resonance imaging (MRI) scans. At first, the lightweight Simple Fully Convolutional Network (SFCN) and the linear bias correction (LBC) are applied to estimate the brain age gap (BAG) for each scan. Then BAG is used as a bio-marker to select the scans which have the potential to intensify the diseases-related patterns. Afterwards, a target dataset, which consists of the brain MRI scans from subjects diagnosed with diabetes and/or MDD, and a background dataset, which consists of scans from a healthy control (HC) population, are used as the inputs of contrastive variational auto-encoder (cVAE)-liked models to isolate the desired features. Experiments on different values of the total correction (TC) loss weight $\gamma$, cyclical annealing schedules on the Kullback–Leibler (KL) divergence loss weight $\beta$ and ablation studies on the discriminator and the KL loss were conducted. It is discovered that a malfunctioning discriminator can lead to an ineffective learning in the latent space, and converting a VAE into a deterministic regularized auto-encoder (RAE) might help with the improvement of model performance. The desired patterns tend to cluster in the patent space, but no obvious groupings consistent with the scan types are discovered.



**Please also down load the [SFCN github repo](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/) folder and put it in the same direction of this folder. Running of the files in this fold need help functions from the SFCN code. `SFCN_path` in the files might need to change accordingly.**

**The data are collected from UKBB and saved in the fubini server. If one needs to re-extract the data from UKBB, please change the file saving path to your own device.**



# Virtual Environments

````
os
sys
argparse
logging
numpy
pandas
torch
sklearn
matplotlib
scipy
nibabel
nilearn
seedir
zipfile
````



# Data collection and Brain Age Estimation

* `nii_and_h5_file_format.ipynb`
* `UKBB_20252.ipynb`
* `R_mdd.ipynb`
* `Inference_brain_age.ipynb`



**Scan type information is saved in folder `./data_info`. `data_info\T1_MNI_20252_2_0\all3_HC` stores the information about the dataset used to train the cVAE models.**



# Source files to build and train the models:

* `cVAE_trainer.py`
* `cVAE_utils.py`
* `utils.py`
* `SFCN_trainer.py`



# Bash Files

(treat baseline as Model 0 here)

## training

`bash train_cVAE-{i}.sh` -- replace {i} with model index



**If one don't want to train the models by themselves, they can download the pre-trained model by the project [here](https://drive.google.com/file/d/1AOqojkdLyn2AWT7zTtmVR9zFUYjgo59F/view?usp=sharing). Please unzip the downloaded folder and put into this folder. Then run the following codes to see the loss curves and evaluation results.**



## plotting loss curves

run `plot-{i}.ipynb` in jupyter labs -- replace {i} with model index

## evaluating models

run `Evaluate-M{i}.ipynb` in jupyter labs -- replace {i} with model index