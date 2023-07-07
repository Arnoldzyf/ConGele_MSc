#!/usr/bin/env bash
###########
# USAGE NOTE:
###########

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/miniconda3/bin/activate ConGele

# Define a location for all your experiments to save
save_root_dir="./trained_cVAE_3DCNN"
# NAME YOUR EXPERIMENT HERE
trial_name="UKBB567"

# Data fetching
target_df="./data_info/T1_MNI_20252_2_0/MDD_dataset_567.csv"
background_df="./data_info/T1_MNI_20252_2_0/HC_dataset_567.csv"
validation_ratio=0.2

batch_size=10
logging_interval=4
plot_background=true # ??

# early stop
patience=7

# echo blablabla

# Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.
python cVAE_trainer.py --save_root_dir ${save_root_dir}  \
                --trial_name ${trial_name}  \
                --target_df ${target_df}  \
                --background_df ${background_df}  \
                --validation_ratio ${validation_ratio}  \
                --batch_size ${batch_size}  \
                --logging_interval ${logging_interval}  \
                --plot_background ${plot_background}  \
                --patience ${patience}  \

                ### ADDITIONAL ARGUMENTS HERE ###


