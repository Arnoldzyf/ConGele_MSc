#!/usr/bin/env bash
###########
# USAGE NOTE:
###########

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/miniconda3/bin/activate ConGele

# Define a location for all your experiments to save
save_root_dir="./hp_tuning-3DcVAE_asd-UKBB567"
# NAME YOUR EXPERIMENT HERE
#trial_name="UKBB567_1"
#trial_name="beta-gamma"

# model arch
cVAE_type="asd" # model type: "default", "asd",

# Data fetching
target_df="./data_info/T1_MNI_20252_2_0/MDD_dataset_567.csv"
background_df="./data_info/T1_MNI_20252_2_0/HC_dataset_567.csv"
validation_ratio=0.2
random_state=33
normalization_type="min-max-2" # normalization type: "min-max", "None", "min-max-2"
enlarge=0.1  # normalization enlarging scale

batch_size=6
logging_interval=1
plot_background=true # ??

# loss computation
alpha=$((160 * 192 * 160 / 20)).0  # the denominator to scale down the validation loss
#beta=1
#gamma=1

# early stop
patience=7
# maximum training epoch
max_epoch=30

# echo blablabla


for beta in 0.5 1 2
do
  gamma=1
  trial_name="beta_${beta}-gamma_${gamma}"
  echo
  echo "***************************************"
  echo ${trial_name}

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

done


for gamma in 0.5 1 2
do
  beta=1
  trial_name="beta_${beta}-gamma_${gamma}"
  echo
  echo "***************************************"
  echo ${trial_name}

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

done




