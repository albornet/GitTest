# Description:
# Reads a BMP file, transforms it into a numpy array, and crops the white background that may lie behind the stimulus
# Intended to be used by the script LaminartWithSegmentation.py

# Parameters:
# imName --> string containing the name of the file to read, e.g. : "Stimuli/HalfLineFlanks5.bmp"
# onlyZerosAndOnes --> 0 if you want aliasing, 1 if you want no aliasing, 2 if you want (aliased+notaliased)/2
# addCropx,y --> integer defined to crop the stimulus array even more to make it smaller, for computational purposes
    # e.g. : addCropx = 50 will remove the 50 far-left columns and 50 far-right columns

# Imports:
from PIL import Image
import numpy as np

# Function:
def readAndCropBMP(imName, onlyZerosAndOnes=0, addCropx=0, addCropy=0):

    # Read the image and transforms it into a [0 to 254] array
    im = Image.open(imName)
    im = np.array(im)
    if len(np.shape(im)) > 2:                # if the pixels are [r,g,b] and not simple integers
        im = np.mean(im,2)                   # mean each pixel to integer (r+g+b)/3
    if onlyZerosAndOnes == 1:
        im = np.round(im/im.max())*im.max()  # either 0 or im.max()
    if onlyZerosAndOnes == 2:
        im = 0.5*(im + np.round(im/im.max())*im.max())  # less aliasing ...
    im *= 254.0/im.max()                     # array of floats between 0.0 and 254.0 (more useful for the Laminart script)

    white = im.max()                         # to detect the parts to crop

# Cropping:
    # Remove the upper white background (delete white rows above)
    usefulImg = np.where(im[:,int(len(im[0,:])/2)] != white)[0]
    indexesToRemove = range(usefulImg[0])
    im = np.delete(im,indexesToRemove,axis=0)

    # Remove the left and right white background (delete white columns left and right)
    usefulImg = np.where(im[0,:] != white)[0]
    indexesToRemove = np.append(np.array(range(usefulImg[0])),np.array(range(usefulImg[-1]+1,len(im[0,:]))))
    im = np.delete(im,indexesToRemove,axis=1)

    # Remove the bottom white background (delete white rows below)
    usefulImg = np.where(im[:,0] != white)[0]
    indexesToRemove = range(usefulImg[-1]+1,len(im[:,0]))
    im = np.delete(im,indexesToRemove,axis=0)

    # Crops the image even more, if wanted (for computation time)
    if addCropx != 0:
        indexesToRemove = np.append(np.array(range(addCropx)),np.array(range(len(im[0,:])-addCropx,len(im[0,:]))))
        im = np.delete(im,indexesToRemove,axis=1) # horizontal crop (remove columns on both sides)
    if addCropy != 0:
        indexesToRemove = np.append(np.array(range(addCropy)),np.array(range(len(im[:,0])-addCropy,len(im[:,0]))))
        im = np.delete(im,indexesToRemove,axis=0) # vertical crop (remove lines on both sides)

    return im, np.shape(im)[0], np.shape(im)[1]