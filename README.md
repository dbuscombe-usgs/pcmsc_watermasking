# pcmsc westcoast watermasker
Implementation of a residual U-Net model for detecting water in west coast usgs planecam imagery

Written by Daniel Buscombe for the USGS Coastal Change Hazards Program

Design and Concept by Andy Ritchie, Jon Warrick, and Daniel Buscombe, @ the USGS Coastal Change Hazards Program

## What it does
You provide 
1) a directory of jpg images 
2) [optional] a command line flag (`-m`) indicating whether to print a black and white mask to jpg file, and what type
3) [optional] a command line flag (`-l`) indicating whether to keep a log file 
4) [optional] a command line flag (`-p`) indicating whether to use parallel processing

and it creates a data archive in compressed numpy npz format that contains the following fields:

1) model metadata
	* number of test-time-augmented outputs weighted-averaged 
	* area in pixels^2 to be considered either a pixel island or hole 
	* mean variance in prediction 
	* median variance in prediction 
	* min variance in prediction 
	* max variance in prediction 
	* mean probability of land
	* median probability of land
	* min probability of land 
	* max probability of land 
	* Otsu threshold for land 
	* Otsu threshold for land 
	* Mean model certainty
	* Otsu threshold confidence
	* Otsu threshold variance
	* certainty output code (0=bad, 1=ok, 2=good) 
	* Proportion of land pixels in conservative mask 
	* Proportion of land pixels in liberal mask

2) original image exif data (in a human readable dictionary format)
3) probability of land (8 bit, divide by 100 to get a probability of land )
4) confidence in estimate (8 bit, divide by 100 to get a confidence)
5) variance around that estimate (8 bit, divide by 100 to get a variance)
6) conservative land mask (1 bit): this is the most conservative (i.e. calling pixels land only if it is very certain)
7) liberal land mask (1 bit): this is the most liberal (i.e. calling pixels land only if it is only moderately certain)

## Input flags 
* `-m` : 0=no mask [default], 1= liberal mask, 2= conservative mask
* `-l` : 0=no log, 1=use log [default]
* `-p` : 0=no parallel proc, 1=use paralel proc [default]

The program will:
1) read the names of the latest and greatest model parameters in `best_current_model.py`
2) read configuration files and create a setting dictionary to pass to the main function
3) prompt the user to select a directory of images 
4) depending on user inputs, run the `main` function in serial or parallel to create the watermask outputs

then on each image, the following procedures are carried out (in `main()`):

1) read the image into memory at both scales (there are two models, trained using different scales)
2) standardize both images
3) read the models from json_string and add weights from h5 file
4) use each model on the image to estimate a stack of test-time augmented outputs
5) resize to original image size, then weighted average to create an average map of probabilities of land 
6) compute variance, certainty and confidence rasters
7) compute masks using an Otsu threshold and remove islands and holes
8) save masks if user inputs say so 
9) get exif data and compile model metadata 
10) write out npz file containing all outputs (listed above)

## Current ensemble model

### 8/10/21
Merges outputs from:
1) watermask_planecam_1024_20210802
2) watermask_planecam_2048_20210802

trained on datasets: 

1. watermask-oblique-planecam-data-july2021_2048.zip
6308 augmented images and binary masks resizd to 1536 x 2048 x 3 pix

2. watermask-oblique-planecam-data-july2021_768_1024.zip
5259 augmented images and binary masks resizd to 768 x 1024 x 3 pix


%# EOF
