#!/usr/bin/env bash
###########
# USAGE NOTE:

# cVAE asd model, UKBB567 train first 40, val first 10 (random state 37)
###########

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/miniconda3/bin/activate ConGele

# Define a location for all your experiments to save
save_root_dir="./trained_cVAE_3DCNN"
# NAME YOUR EXPERIMENT HERE
#trial_name="UKBB567_1"
#trial_name="test_default"

# model arch
cVAE_type="asd" # model type: "default", "asd",

# Data fetching
target_df="./data_info/T1_MNI_20252_2_0/MDD_dataset_567.csv"
background_df="./data_info/T1_MNI_20252_2_0/HC_dataset_567.csv"
validation_ratio=0.2
random_state=37
normalization_type="min-max-2" # normalization type: "min-max", "None", "min-max-2"
enlarge=0.1  # normalization enlarging scale

batch_size=6
logging_interval=1
plot_background=true # ??

# loss computation
alpha=$((160 * 192 * 160 / 20)).0  # the denominator to scale down the validation loss
beta=1
gamma=1

# early stop
patience=7
# maximum training epoch
max_epoch=30

# echo blablabla

# Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.
python cVAE_trainer.py --save_root_dir ${save_root_dir}  \
                --trial_name ${trial_name}  \
                --cVAE_type ${cVAE_type}  \
                --target_df ${target_df}  \
                --background_df ${background_df}  \
                --validation_ratio ${validation_ratio}  \
                --random_state ${random_state}  \
                --normalization_type ${normalization_type}  \
                --enlarge ${enlarge}  \
                --batch_size ${batch_size}  \
                --logging_interval ${logging_interval}  \
                --plot_background ${plot_background}  \
                --alpha ${alpha}  \
                --beta ${beta}  \
                --gamma ${gamma}  \
                --patience ${patience}  \
                --max_epoch ${max_epoch}  \

                ### ADDITIONAL ARGUMENTS HERE ###


