'''
This file contains all the functions related to the loading of the images,  
the processing (normalization, augmentation, etc.), and building the batches.
In this project, we do not store the images in memory. Instead, we get the 
list of names and we load one image at a time. 
This might not be very efficient, though. 
'''

import os
import numpy as np  
import imageio as imageio  
from skimage.transform import rescale               
import scipy.ndimage as ndi 
import skimage.exposure as exp   
from tensorflow.keras.utils import to_categorical


# -----------------------------------------------------------------------------
#               BUILD BATCHES
# -----------------------------------------------------------------------------

def build_training_batch_myeloid(imgName, args=None):
    """ This function builds a training batch for the Myeloid dataset.
    # Arguments
        imgName:    list of lists, the name (path included) of the images.
                    1st dimension is class. 2nd dim is images within that class.
      Within args, it is required:
        input_shape:   array of ints, shape of the CNN input (row, col, chn)
        batch_classes: array of ints,
        batch_unit:    int, size of a unit of a batch
        batch_sets:    int, the number of batch_units in a batch
        batch_size:    int, the final size of the batch (sets * unit)
     # Returns
        batch_img:  4D numpy array, the batch of the images.
        batch_lbl:  2D numpy array, the labels of the batch. 
    """
    batch_img = np.zeros(shape=(args.batch_size,
                                args.image_shape[0],
                                args.image_shape[1],
                                args.image_shape[2]), dtype=np.float32)
    batch_lbl = np.zeros(shape=(args.batch_size,))

    id = 0
    for ii in range(args.batch_sets):
        for jj in range(args.batch_unit):
            # For all the classes to consider (aa), select one randomly (iC), 
            # and then select one image randomly (iM).
            aa = np.argwhere(args.batch_classes == jj)      
            iC = int(aa[int(np.random.uniform(0,len(aa)))]) 
            iM = int(np.random.uniform(0,len(imgName[iC]))) 
            img = load_image_4Augmentation(imgName[iC][iM], 
                                           is_training=True, args=args)
            batch_img[id] = img[0]
            batch_lbl[id] = iC
            id = id + 1

    # Shuffle the patches within the batch
    ix = np.random.permutation(args.batch_size)
    batch_img = batch_img[ix] 
    batch_lbl = batch_lbl[ix] 

    # CROPPING. At the CNN input: This affects the data
    if args.crop_input:       
        batch_img = crop_batch_4CNN(batch_img, args.input_shape)

    # SCALING. At the CNN input: This affects the data
    if args.scale_input is not None and args.scale_input != 1:       
        batch_img = rescale_batch_4CNN(batch_img, args.scale_input)

    # WEIGHTING 
    if args.weighting_flag:
        batch_wgh = np.zeros(shape=(batch_lbl.shape[0],1), dtype=np.float32)
        for ii in args.weighting_class:
            batch_wgh[batch_lbl == ii] = args.weighting_class[ii]
    else:
        batch_wgh = None
    
    # Labels to categorical
    batch_lbl = to_categorical(batch_lbl, num_classes=len(args.classes), 
                               dtype='float32')
    return batch_img, batch_lbl, batch_wgh    



def build_training_batch_leukemia(posImgName, negImgName, args=None):
    """ This function builds a training batch for the Leukemia dataset.
    It selects randomly 50/50 from positive and negative images.
    # Arguments
        posImgName:    str, the name (path included) of the positive image.
        negImgName:    str, the name (path included) of the negative image.
      Within args, it is required:
        input_shape:   array of ints, shape of the CNN input (row, col, chn)
        batch_classes: array of ints,
        batch_unit:    int, size of a unit of a batch
        batch_sets:    int, the number of batch_units in a batch
        batch_size:    int, the final size of the batch (sets * unit)
     # Returns
        batch_img:  4D numpy array, the batch of the images.
        batch_lbl:  2D numpy array, the labels of the batch. 
    """
    nPos = int(args.batch_size//2)
    nNeg = args.batch_size - nPos
    batch_img = np.zeros(shape=(args.batch_size,
                                args.image_shape[0],
                                args.image_shape[1],
                                args.image_shape[2]), dtype=np.float32)
    batch_lbl = np.append(np.zeros(shape=(nNeg,1), dtype=np.float32), 
	                      np.ones(shape=(nPos,1),  dtype=np.float32), axis=0)

    id = 0
    for ii in range(nNeg):
        ix = int(np.random.uniform(0, len(negImgName)))
        img = load_image_4Augmentation(negImgName[ix], is_training=True, args=args)
        batch_img[id] = img[0]
        id = id + 1
  
    for ii in range(nPos):
        ix = int(np.random.uniform(0, len(posImgName)))
        img = load_image_4Augmentation(posImgName[ix], is_training=True, args=args)
        batch_img[id] = img[0]
        id = id + 1

    # Shuffle the patches within the batch
    ix = np.random.permutation(args.batch_size)
    batch_img = batch_img[ix] 
    batch_lbl = batch_lbl[ix] 

    # CROPPING. At the CNN input: This affects the data
    if args.crop_input:       
        batch_img = crop_batch_4CNN(batch_img, args.input_shape)

    # SCALING. At the CNN input: This affects the data
    if args.scale_input is not None and args.scale_input != 1:       
        batch_img = rescale_batch_4CNN(batch_img, args.scale_input)

    # WEIGHTING 
    if args.weighting_flag:
        batch_wgh = np.zeros(shape=(batch_lbl.shape[0],1), dtype=np.float32)
        for ii in args.weighting_class:
            batch_wgh[batch_lbl == ii] = args.weighting_class[ii]
    else:
        batch_wgh = None
    
    # Labels to categorical
    batch_lbl = to_categorical(batch_lbl, num_classes=len(args.classes), 
                               dtype='float32')
    return batch_img, batch_lbl, batch_wgh    



def build_validation_batch(imgName, imgLabel, args=None):
    """ This function builds a batch for validation or testing.
    # Arguments
        imgName:    string, the name (path included) of the images.
        imgLabel:   int, the label of the images
      Within args, it is required:
        input_shape: array of ints, shape of the CNN input (row, col, chn)
     # Returns
        batch_img:  4D numpy array, the batch of the images.
        batch_lbl:  2D numpy array, the labels of the batch, for CNN. 
    """
    batch_img = np.zeros(shape=(len(imgName),
                                args.image_shape[0],
                                args.image_shape[1],
                                args.image_shape[2]), dtype=np.float32)
    batch_lbl = imgLabel * np.ones(shape=(len(imgName),1), dtype=np.float32)
    batch_lbl = to_categorical(batch_lbl, num_classes=len(args.classes), 
                               dtype='float32')
    for ii in range(len(imgName)):
        img = load_image_4Augmentation(imgName[ii], is_training=False, 
                                       args=args)
        batch_img[ii] = img[0]

    # CROPPING. At the CNN input: This affects the data
    if args.crop_input:       
        batch_img = crop_batch_4CNN(batch_img, args.input_shape)

    # SCALING. At the CNN input: This affects the data
    if args.scale_input is not None and args.scale_input != 1:       
        batch_img = rescale_batch_4CNN(batch_img, args.scale_input)
    return batch_img, batch_lbl    



# -----------------------------------------------------------------------------
#               LOAD AN IMAGE & APPLY AUGMENTATION
# -----------------------------------------------------------------------------

def load_image_4Augmentation(img_name, is_training=True, args=None):
    """ This function load an image and apply augmentation and preprocessing. 
    Depending on args, it will be turned into float32: 
    - That would be necessary to apply rotation or deformations.
    - If so, labels need to be rounded afterwards.
    The function loadimage_4CNN already transpose the dimensions if required, 
    thus it is not done here (all augmentations deal with it).
    # Arguments
        img_name:    string, name of the image (including path).
        is_training: boolean, to indicate whether it is training or testing.               
      Within args:
        flag_augmentation:     boolean, to apply augmentation  
        scale_value:           float, to rescale the image
        batch_normalization:   boolean, whether to apply normalization.
        batch_standardization: boolean, whether to apply standardization.        
    # Returns
        img:        4D numpy array, float32, image (NHWC).
    """
    # Load one image
    img = load_image_4CNN(img_name)
    if args.flag_augmentation and is_training:
        # Apply augmentation/distorsions
        #img = adjust_gamma(img, gamma=np.random.uniform(0.7, 1.3))
        #img = add_gaussian_noise(img, scale=np.random.uniform(0.0, .05))
        #img = blur_images(img, sigma=np.random.uniform(0.0, 1.0))
        nFlip = np.random.choice(4, 1)
        flip_type = (None, 'UD', 'LR', '2A')
        if nFlip > 0:
            img = flip_images(img, flip_type=flip_type[int(nFlip)])
        # Rotate (only in angles of 90 degrees)
        angle = np.random.uniform(-90, 90)
        img = ndi.rotate(img, angle, axes=(2,1), reshape=False, 
                         mode='constant', cval=0.0)
    # Normalize or standardize on the batch, if indicated
    if args.batch_normalization:
        img = batch_normalize(img) 
    if args.batch_standardization:
        img = batch_standardize(img)  
    return img


def batch_normalize(img): 
    return np.divide((img - np.nanmin(img)), (np.nanmax(img) - np.nanmin(img)))  
  

def batch_standardize(img): 
    return (img - img.mean())/img.std  


# -----------------------------------------------------------------------------       
#               FUNCTIONS TO LOAD, SCALE, AND CROP IMAGES
# -----------------------------------------------------------------------------

def im2double(img, max_value=None):
    """ Convert the input image 'img' (uint8, probably) into a image with 
        double precision (float32), whose values range between 0 and 1. This   
        is the equivalent to im2double in Matlab. Optionally, the range can   
        be set by indicating the maximum value.
    """    
    if max_value is not None:
        return img.astype(np.float32) / max_value
    else:
        info = np.iinfo(img.dtype) # Get the data type of the input image
        return img.astype(np.float32) / info.max 


def load_image_4CNN(img_name, folder=None, to_double=True):
    """ It loads the indicated image from a given directory. The directory can 
    be introduced separately or all together with the name of the file.
    It is loaded as uint8 values, as image (jpeg, png, etc.) are stored.
    The returned image is given in a 4D array, for preparation for the CNN.
    # Arguments:
        img_name:    string, the name of the image, which can include the path. 
        folder:      string, the path of the image.
        to_double:   boolean, to make the image double-range (0-1)
    # Returns:
        numpy array, 4 dimensions (NHWC)           
    """
    if folder is not None:
        img_fullname = os.path.join(folder, img_name) 
    else:
        img_fullname = img_name 
    img = imageio.imread(img_fullname).astype(np.uint8)
    if len(img.shape) == 2:
        img = img[np.newaxis,:,:,np.newaxis]   # Add axis N and C.
    elif len(img.shape) == 3:                  # This problem has RGBA images!
        img = img[np.newaxis,:,:,0:3]          # Add axis N
    if to_double:
        img = im2double(img)
    return img  


def rescale_batch_4CNN(batch_image, scale_value, flag_round=False): 
    """It rescales the batch based on a scale factor. Note how the final shape 
    is rounded.
    # Arguments:
        batch_image:   numpy array, 4 dimensions (NHWC).
        scale_value:   float, scaling factor
        flag_round:    boolean, to round the values once has been scaled.
    # Returns:
        numpy array.
    """
    Ni, hi, wi, ci = batch_image.shape
    new_batch = np.zeros(shape=(Ni,int(hi*scale_value),int(wi*scale_value),ci))    
    for ii in range(Ni):
        if ci == 1:
            img = batch_image[ii,:,:,0] 
            img = rescale(img, scale_value, preserve_range=True, 
                          multichannel=False).astype(batch_image.dtype)
            new_batch[ii,:,:,0] = img
        else:
            img = batch_image[ii]
            img = rescale(img, scale_value, preserve_range=True, 
                          multichannel=True).astype(batch_image.dtype)
            new_batch[ii] = img
    if flag_round:
        new_batch = np.round(new_batch)
    return new_batch 


def crop_batch_4CNN(batch_image, output_shape): 
    """It crops the image into the output_shape, taking the center part. 
    It assumes that the output_shape is smaller than the input image size. 
    Only Height and Weight are cropped.
    # Arguments:
        batch_image:  numpy array, 4 dimensions (NHWC). Original shape.
        output_shape: list int (H,W,C), desired output image shape.
    # Returns:
        numpy array.
    """
    _, h, w, _ = batch_image.shape
    ho, wo, _ = output_shape 
    ih = (h - ho)//2
    iw = (w - wo)//2
    new_batch = batch_image[:, ih:ih+ho, iw:iw+wo, :]
    return new_batch 

  
def load_image_names(folder, ending=None):
    """ It loads the names of all images from directory. If given an ending,
    only those files will be returned.	
    """
    img_files = sorted(os.listdir(folder)) 
    if ending is not None:
        img_files = [ii for ii in img_files if ii.endswith(ending)]
    img_number = len(img_files)     
    print("  Loading names of", str(img_number), "images...")   
    return np.asarray(img_files)    


def check_file_exist(file_name, folder=None):
    """ It check whether a SINGLE file exists or not. The path (folder) can be 
    given all together (fileName=path+name) or separated.
    """
    if folder is not None:
        file_fullname = os.path.join(folder, file_name) 
    else:
        file_fullname = file_name
    return os.path.isfile(file_fullname)


# -----------------------------------------------------------------------------        
#                 AUGMENTATION
# -----------------------------------------------------------------------------

def flip_images(img, flip_type='UD'):
    """ Flips all images vertically (Up-Down) or horizontally (Left-Right). 
    By default, flip is Up-Down (UD). 
        - For Left-Right, introduce 'LR'.
        - For both flips, introduce '2A'.
    """ 
    assert(len(img.shape) == 4)     
    if flip_type == 'LR':
        axis_flip = 2                    # Left-Right 
    elif flip_type == '2A':
        axis_flip = (1, 2)               # Both
    else:
        axis_flip = 1                    # Up-Down           
    img = np.flip(img, axis=axis_flip)
    return img


def blur_images(img, sigma=1.0):
    """ Blurs the image using a Gaussian filter. Sigma=0 does not blur. """
    assert(len(img.shape) == 4) 
    newIm = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        newIm[i] = ndi.gaussian_filter(img[i], sigma=(sigma,sigma,0))
    return newIm

  
def add_gaussian_noise(img, scale=0.05):
    """ Adds Gaussian noise to the image. Rescale in the end. """ 
    assert(len(img.shape) == 4) 
    newIm = np.random.normal(img, scale=scale)
    np.putmask(newIm, newIm > 1, 1)
    np.putmask(newIm, newIm < 0, 0)
    return newIm

  
def adjust_gamma(img, gamma=1.0):
    """ Adjust the gamma. Default gamma=1.0 does not change anything. """ 
    assert (len(img.shape)==4)  
    newIm = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        newIm[i] = exp.adjust_gamma(img[i], gamma)
    return newIm

  
def adjust_contrast(img):
    """ Adjust the contrast based on the sigmoid adjustment.     
        The values indicated are the default ones in the documentation. """ 
    assert (len(img.shape)==4)  
    newIm = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        newIm[i] = exp.adjust_sigmoid(img[i], cutoff=0.5, gain=10)
    return newIm