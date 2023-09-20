#!/usr/bin/env bash
###########
# USAGE NOTE:
###########

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/miniconda3/bin/activate ConGele

# learning rate
# lr=0.001 # default 0.0003

# batch and log
batch_size=10
logging_interval=6

# loss computation
alpha=250000 # 250000, 20
beta=1
gamma=0.01 # 100, 2

# Define a location for all your experiments to save
save_root_dir="./ConGele" # "./trained_cVAE_3DCNN"
# NAME YOUR EXPERIMENT HERE
#trial_name="UKBB567_1"
trial_name="Model-1"
#trial_name="test_run-batch_size_${batch_size}"

# model arch
cVAE_type="asd" # model type: "default", "asd",
s_dim=32
z_dim=32

# Data fetching
target_df="./data_info/T1_MNI_20252_2_0/all3_HC/tg_dataset_604.csv"  # "./data_info/T1_MNI_20252_2_0/MDD_HC/MDD_dataset_567.csv"
background_df="./data_info/T1_MNI_20252_2_0/all3_HC/bg_dataset_604.csv"  # "./data_info/T1_MNI_20252_2_0/MDD_HC/HC_dataset_567.csv"
validation_ratio=0.2
random_state=42
normalization_type="min-max-4" # normalization type: "min-max", "None", "min-max-2", "min-max-3", "min-max-4"
enlarge=0.1  # normalization enlarging scale, no use if min-max-4

plot_background=1 # ??

# early stop
patience=7
min_step=0
# maximum training epoch
max_epoch=800

# echo blablabla

# Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.



python cVAE_trainer.py --save_root_dir ${save_root_dir}  \
                --trial_name ${trial_name}  \
                --cVAE_type ${cVAE_type}  \
                --s_dim ${s_dim}  \
                --z_dim ${z_dim}  \
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
                --min_step ${min_step}  \
                --max_epoch ${max_epoch}  \
#                --lr ${lr}  \

                ### ADDITIONAL ARGUMENTS HERE ###




