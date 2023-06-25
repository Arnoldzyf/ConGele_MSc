# `Data_info.csv`

For info about subjects in the `\data\ukb\imaging\raw\t1_structural_nifti_20252` folder

* Columns:
  * `f.eid`: subjects ID,
  * `f.21003.2.0`: [Age when attended assessment centre (3rd instance)](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21003),
  * `f.20126.0.0`: [Bipolar and major depression status](https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20126),
  * `f.2976.2.0`: [Age diabetes diagnosed](https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=2976)

Please see detailed intro in the last section "Brief Analysis" in the `UKBB_20252.ipynb` file.

# `UKBB_20252.ipynb`

extract information about "age at assessment",  "depression status", "age diabetes diagnosed" of the subjects in the `\data\ukb\imaging\raw\t1_structural_nifti_20252` folder

plot several scans of one sample in `t1_structural_nifti_20252` -- need to use `T1/T1_brain_to_MNI.nii.gz`.

# `SFCN_trainer.py` 

functions to train and validate SFCN

initialized by the pretrained model

# `cVAE_trianer.py`

functions to train and validate the cVAE

use test class for smaller input `(6,9,6)` for now.

apply early stop and can resume training

the file can be run directly, for running options please check `get_args` function in the file

(will add test class later, will add logging files later)

# `cVAE_utils.py`

cVAE architecture

# `Training.ipynb`

Code to build cVAE via Keras

Most copied from the original cVAE repository

(160, 192, 160) is too large for my PC to experiment

# `nii_and_h5_file_format.ipynb` : 

Info about how to handle `.nii` and `.h5` files.

#  `Inference_brain_age.ipynb`:  

Use pre-trained SFCN model to predict brain age of the 20 scans in the small dataset

**Inference**：

* data flow: 

  .h5 file -> numpy array -> reshape to (batch_size, 1, 160, 192, 160) -> tensor as model input

* bins: 

  14-94, with 2 years interval

**Result:**

* 20 scans in total, all around 53 years old

  ​	<img src="readme.assets/image-20230528022702265.png" alt="image-20230528022702265" style="zoom:67%;" />

* Bias Correction:

  **SOMETHING IS WRONG HERE**

  ​	<img src="readme.assets/image-20230528022405369.png" alt="image-20230528022405369" style="zoom:67%;" />