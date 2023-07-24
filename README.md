# Baseline Semantic Segmentation Model for CV4Africa Challenge on the Legacy of Spatial Apartheid in South Africa
## Setup
* Pytorch 2.0.1 
* Torchvision 0.13.1 
* OpenCV 
* Pillow 
* Numpy

## Training
* Setup the data under DATA_ROOT and modify the train, val, test files with paths accordingly. You can use the default path data_renamed/
* Run the following command
```
python train.py --model_type unet --experiment unet_testing --data_root DATA_ROOT
```

## Inference
* Download the trained weights, that can be used to reproduce the sample submission in the starter kit, [here](https://www.dropbox.com/scl/fi/m8a4jnkhao6hy0fl9ee8n/unet_fixed.zip?rlkey=je4j3jshgcthrr8ohgif3jpsw&dl=0)
* Run the following command
```

```
* The output results will be saved in 
