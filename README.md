# Baseline Semantic Segmentation for CV4Africa Challenge

[CV4Africa Workshop](https://ro-ya-cv4africa.github.io/homepage/event_workshop.html)

[Challenge on the Legacy of Spatial Apartheid in South Africa](https://codalab.lisn.upsaclay.fr/competitions/14259)

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
python inference.py --model_dir unet_fixed/ --model_type unet --out_dir OUTDIR
```
* The output results will be saved under OUTDIR/

## Licence
The code is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## Acknowledgements
Parts of the code have been re-used from the following repositories:
* [MEDVT](https://github.com/rkyuca/medvt) 
* [UNET](https://github.com/milesial/Pytorch-UNet) 
* [RePRI](https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation) 
