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