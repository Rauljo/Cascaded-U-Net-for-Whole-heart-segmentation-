# Cascaded-U-Net-for-Whole-heart-segmentation-

The following code can be used for training and making predictions with a two-staged cascaded U-Net segmentation model. It has been used specifically for whole heart segmentation on 7 different labels. 

## Training:

### First stage:

Just need to execute `python Train.py` with some concrete arguments:
- **num_labels = 1**: in this way you indicate that you are training a segmentation model for just one label (the heart as a whole). Data will be preprocessed so that all labels are converted into just one. 

And some optional arguments:
- depth: depth of the network
- start_filts: number of filters in the first layer
- cross_val: wether you want to use 6 folds crossvalidation and receive results on `resultados.csv` or you just want to training a model without validation set. In that case, the final checkpoint will be saved in the root directory. 
- max_epochs
- img_path: path to read images from
- logs_folder: where to log training data
- tam: size of the images you want to work with
- augment: wether to use augmentation or not

## Second stage:

In this case, just do not use `num_labels=1` argument and the network will be trained in all labels separately. 

The second stage should be trained on the cropped images. For this purpose you have to execute `Croped_dataset.py` code with the desired arguments:
- source_folder
- destination_folder
- margin_value: how much margin does the cuboid over the heart have
- crop_type: "fixed" or "variable". The fixed type uses a fixed experimental cuboid. The variable depends on the concrete subject you are segmenting. 

## Predicting:

Just need to execute `python Prediccion.py` with the desired arguments:
- model_pred: path to second stage model
- model_crop: path to first stage model
- images_folder: path to source images
- destination_folder: path where inferences will be stored
- tam_pred: size in which the second stage model works
- tam_crop: size in which the first stage model works
- crop_margin: the margin on each dimension to use for the crop (<1)

## Postprocessing:

Just need to execute `python postprocesado.py` with the desired arguments:
- source_folder: path to source images
- largest_pulmonar: 0 maintain everything on pulmonary artery. 1 maintain only largest component also on pulmonary artery

Source images will be rewriten maintaining only largest component. 