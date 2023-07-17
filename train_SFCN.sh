#!/usr/bin/env bash
###########
# USAGE NOTE:
###########

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/miniconda3/bin/activate ConGele

# Define a location for all your experiments to save
save_root_dir="./trained_SFCN"
# NAME YOUR EXPERIMENT HERE
#trial_name="test1"
trial_name="UKBB_HC"

# Data fetching
dataset="./data_info/T1_MNI_20252_2_0/HC_info.csv"
val_rate=0.2

batch_size=20 # can set to 20
logging_interval=25

# early stop
patience=7
# maximum training epoch
max_epoch=100

# echo blablabla

# Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.
python SFCN_trainer.py --save_root_dir ${save_root_dir}  \
                --trial_name ${trial_name}  \
                --dataset ${dataset}  \
                --val_rate ${val_rate}  \
                --batch_size ${batch_size}  \
                --logging_interval ${logging_interval}  \
                --patience ${patience}  \
                --max_epoch ${max_epoch}  \

                ### ADDITIONAL ARGUMENTS HERE ###


