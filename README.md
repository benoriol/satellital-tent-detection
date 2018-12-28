# Satellital tent detection
The steps to follow for our approach are:
1. Generating a dataset for image segmentation given background images and foreground images. Applied to satellite image segmentation.
2. Training ML model for a masking task
3. Evaluation of the models

The explanation will be divided in these three parts, starting with generation.

## Generation of a dataset
For this part (included in /generation) there are 2 important files in addition to the **data** folder, which contains the folder **examples**, that includes different backgrounds and tents. The files are the following.

**generate.py**

Here, the main code for generation exists. This is the file used to generate. To get familiar with it you can run the help options (`-help`). Recall that the date is a mandatory parameter.

Let's try an example:

`python generate.py -hf data/examples/tent1.png -bf data/examples/background8.png -date 19970305 -l 5 -s 1.5 -d 0x0 -h 60 -margin 0 -m`

This would generate 5 images with around 60 tents occupying all the background used. These would output in data/[date]/output, in addition with their masks in data/[date]/mask and a metadata file containing the output path, the mask path and the real number of tents for every image generated. If the `-m` parameter is called or if there is no mask available for a tent, the number of tents will not be truthful.

Comments about the use:
- The images are expected to be .jpg, .png or .gif.
- The tent images are expected to have the same name as their masks without including "-mask" (i.e. "tent1.png and tent1-mask.png").

This file also relies on the **auxiliar.py** file, which contains additional functionalities.
Another thing we can do is to input a folder of tents not only one type of tent. To use an example, we have included the folder **data/examples/tents-example** containing a group of tents with their masks respectively.



**script.sh**

Here there is a collection of calls to the last file so that a lot of variation of parameters (like type of tents or backgrounds or rotation, etcetera.) can be made to actually generate a big dataset.



## Training ML model for a masking task

## Evaluation of the models
