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