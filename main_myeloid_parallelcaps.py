'''
Parallel Dynamic-Routing Convolutional Capsules (Parallel DR-ConvCaps)

Original Paper by Juan P. Vigueras-Guillen (https://arxiv.org/abs/...)
Code written by: Juan P. Vigueras-Guillen
    Part of the code for DR-ConvCaps is a conversion (from PyTorch to 
    Tensorflow 2.x) of the code made by Lei Yang (yl-1993) and Moein Hasani 
    (Moeinh77): https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch
    Yang and Hasain's code (EM routing) was transformed to DR routing and 
    some changes were made.
If you use significant portions of this code or the ideas from this paper, 
please cite it. If you have any questions, please email me at:
    J.P.ViguerasGuillen@gmail.com


This is the main file regarding the so-called "parallel capsules", which uses  
the Myeloid dataset. From here you can train and test the network. 


# DESCRIPTION OF THE DATA:
(1) We used the dataset from the paper "Human-level recognition of blast cells  
in acute myeloid leukaemia with convolutional neural networks", Nature Machine 
Intelligence 1, 538-544 (2019). 
Link: https://www.nature.com/articles/s42256-019-0101-9
Dataset available at The Cancer Imaging Archive (TCIA):
https://doi.org/10.7937/tcia.2019.36f5o9ld
(2) This dataset contains 18365 images of leukocytes, size 400x400 pixels.
There are 15 unbalanced classes: largest class has 8484 images, whereas the 
smallest one only has 11 images. We selected this dataset because:
    (a) we can prove that large cell images can be solved with DR-ConvCaps;
    (b) we can evaluate how a high unbalanced set is handled by DR-ConvCaps.


# DESCRIPTION OF THE TRAINING:
We tested several ways of balancing the batches. For this, we created the 
concept of 'groups', where classes are set into one group and the batches are 
build by randomly selecting one image from each group. For example:
    (a) Each class has its own group, therefore a batch always contains an 
    example of each class (small classes will be overpresented during training).
    (b) The classes with less examples are grouped together.
To perform the training, we build our own functions to build the batches and we 
used the keras functions 'train_on_batch' and 'test_on_batch'. Functions to keep 
track of the loss and metrics were created. Thus, an epoch here is simply a  
specific number of iterations.


# THE CONCEPT OF PARALLEL CAPSULES
Given a network with a CNN section and a capsule section (with 3 layers in this 
case: primary, mid, and class capsules), the idea is simply to branch the CNN 
section such that we create as many branches as number of mid-capsules. Thus, 
each branch will end up with a single mid-capsule, which are then concatenated 
with the mid-capsules from the other branches and then routed to the next 
layer of capsules (in this case, the final class-capsules).

There are many ways to branch the CNN. This is how we coded it:
Let's assume we have a 5 CNNs (the initial block plus 4 ResNeXt blocks).
- In CNN_filters, we indicate the filters for the branched ResNeXt blocks.
- In CNN_strides, we indicate the strides for the branched ResNeXt blocks.
- In CNN_branches, we describes how the branching will be distributed.
- The common blocks (non-branched) are not indicated in those arguments.

Example:
Let' assume we want to make 10 branches by:
- No subdivision in the first basic block.
- No subdivision in the 1st ResNeXt block.
- Subdividing in 2 branches in the 2nd ResNeXt block. Thus, initIdxStage = 3
- Subdividing in 2 branches in the 3rd block for every previous branch.
- Subdividing in 2 or 3 branches in the 4th block (so we end up in 10 branches).
Thus, -->  args.CNN_branches = [[2],
                                [2, 2],
                                [2, 2, 3, 3]]
This branching-method allows for uneven branching.
If there is no branching in the last ResNeXt block, then place those CNN-blocks 
within the Capsule-section (see code for more details).

We have not asserted (in the code) that the branching annotation is consistent, 
but basically:
   sum(CNN_branches[i]) == len(CNN_branches[i+1])

'''

import os
import argparse
import numpy as np 
from split_data import split_dataset_myeloid
from networks_myeloid_parallel import CapsNet_DR_parallel
from train import train_model_myeloid
from test import test_model_myeloid


def launch_experiment(args):
    """ Given the input arguments of the experiment (args), it ...
    (1) loads the data info and split it intro train/vaid/test; 
    (2) builds the network, and load weights if indicated;
    (3) trains the model (if indicated);
    (4) tests the model (if indicated).
    """                   
    trn_img, val_img, tst_img = split_dataset_myeloid(args) 
    model = CapsNet_DR_parallel(args.input_shape, args=args)
    model.summary()
    if args.weights:
        model.load_weights(args.save_folder + args.weights) # , by_name=True
    if args.train == 1:
        train_model_myeloid(model, trn_img, val_img, args)
    if args.test == 1:
        test_model_myeloid(model, tst_img, args)
    print('Experiment finished!')



if __name__ == '__main__':
    # Only the arguments that need to be changed regularly are added in the 
    # parser. Otherwise, they are defined below.
    parser = argparse.ArgumentParser(description='Tensorflow-Keras DR-ConvCaps')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='The directory of the data.')
    parser.add_argument('--save_folder', type=str, required=True,
                        help='The directory where to save the data.')
    parser.add_argument('--weights', type=str, default='',
                        help='The mame of trained_model.h5; Set to "" for none.' 
                             'Uses path from args.save_folder')
    parser.add_argument('--epochs', type=int, default=700, 
                        help='The number of epochs for training.')
    parser.add_argument('--current_epoch', type=int, default=0,
                        help='If weights are loaded, set the current epoch.')
    parser.add_argument('--iterations', type=int, default=500,
                        help='The number of iterations for one epoch.')      
    parser.add_argument('--iter2eval', type=int, default=500,
                        help='The iterations to launch validation set.')                
    parser.add_argument('--num_folds', type=int, default=5,
                        help='The number of folds to separate the dataset.')
    parser.add_argument('--test_fold', type=int, default=1,
                        help='The index of the fold to use as test set.')
    parser.add_argument('--vald_fold', type=int, default=1,
                        help='The index of the fold to use as validation set,'
                             'but once the images from the test set have been '
                             'removed; i.e. a second K-split is performed.')
    parser.add_argument('--train', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable training.')
    parser.add_argument('--test', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable testing.')
    
    args = parser.parse_args()
    
    # General model
    args.image_shape = (400, 400, 3)   # This is the original image size
    args.input_shape = (400, 400, 3)   # This is the input to the model
    args.scale_input = None  # float, to scale image to input; 'None' or 1. 
    args.crop_input = False  # To crop the image into input_shape (centered) 

    # Learning rate
    args.lr_init = 0.001
    args.lr_decay = 0.99    

    # CNN section (only related to the branching)
    args.initIdxStage = 4  # The number of the block where branching starts
    args.CNN_filters = [[256, 256, 512],
                        [256, 256, 512]]
    args.CNN_strides = [2, 2]
    args.CNN_branches = [[2],
                         [4, 4]]

    # Capsule section
    args.cap_numbCap = [32,  8, 15]  # Number of capsule per layer
    args.cap_sizeCap = [16, 16, 16]  # Size of the capsule
    args.cap_Kvalues = [ 3,  3,  1]  # Kernel of the ConvCaps
    args.cap_strides = [ 1,  1,  1]  # Strides in the ConvCaps
    args.cap_routing = 3             # Number of iterations in DR

    # CNN section
    args.drop_rate = None 
    
    # Batches and classes
    # * In our first design, each class has its own group (batch_classes). 
    # * Alternatively, we grouped several classes into one group.
    # * The final batch size can be increased by adding several units of batches
    args.batch_classes = np.arange(15) # Representation of the groups-classes 
    #args.batch_classes = np.array((0, 1, 2, 1, 3, 4, 1, 5, 1, 1, 1, 1, 1, 1, 1))
    args.batch_unit = np.max(args.batch_classes) + 1      
    args.batch_sets = 1    
    args.batch_size = int(args.batch_unit * args.batch_sets)
    args.classes = np.arange(15)   # The labels of the classes
    args.weighting_flag = False    # To whether apply sample weighting based ... 
    args.weighting_class = {0: 1,  # ... on the classes, using this dictionary.
                            1: 1,        
                            2: 1,
                            3: 1, 
                            4: 1, 
                            5: 1, 
                            6: 1, 
                            7: 1, 
                            8: 1, 
                            9: 1, 
                           10: 1, 
                           11: 1, 
                           12: 1, 
                           13: 1, 
                           14: 1} 
    args.class_folders = ['NGS/', 'NGB/', 'LYT/', 'LYA/', 'MON/', 'EOS/', \
                          'BAS/', 'MYO/', 'PMO/', 'PMB/', 'MYB/', 'MMZ/', \
                          'MOB/', 'EBO/', 'KSC/']
    args.class_descrip = ['Neutrophil (Segmented)', 'Neutrophil (band)     ', \
                          'Lymphocite (typical)  ', 'Lymphocite (atypical) ', \
                          'Monocyte              ', 'Eosinophil            ', \
                          'Basophil              ', 'Myeloblast            ', \
                          'Promyelocyte          ', 'Promyelocyte (bilobed)', \
                          'Myelocyte             ', 'Metamyelocyte         ', \
                          'Monoblast             ', 'Erythroblast          ', \
                          'Smudge cell           ']
    args.class_folders = [args.data_folder + ii for ii in args.class_folders]

    # Augmentation
    args.flag_augmentation = True
    args.batch_normalization = False    
    args.batch_standardization = False     

    # Saving the model and variables (we saved the model at each epoch; this 
    # can be changed in train_model.py)
    args.save_model = True
    args.save_variables = True
    args.save_folder = args.save_folder + 'Fold_' + \
        str(args.test_fold).zfill(2) + '/'
    if not os.path.exists(args.save_folder):
        print('Creating the folder... ')
        os.makedirs(args.save_folder)
    else:
        print('Folder already exists! ')
    
    # -------------------------------------------------------------------------
    # Run the experiment
    launch_experiment(args)






    

