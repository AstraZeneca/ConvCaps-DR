'''
Deep Dynamic-Routing Convolutional Capsules (DR-ConvCaps), which includes
    - Dense DR-ConvCaps: it uses dense connections within the capsule section.
    - Residual DR-ConvCaps: it uses residual connections within the capsule 
      section.
    - Reduced DR-ConvCaps: it reduces the number of CNNs prior to capsules.

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


This is the main file for the project, regarding the Leukemia dataset (another 
main script related to the Myeloid dataset is also available). 
From here you can train and test the network. 
Certain manipulations to the DR-ConvCaps network must be done manually here and 
in the network-python file (instructions in the code). Those manipulations are: 
    - The depth of the network (how many capsule layers to include). For this, 
      change the "capsule section" below and add more layers in the network 
      function (file networks.py)
    - Whether to add dense or residual connections (we already created several 
      functions in networks.py, each one dealing with a specific type).
Please see the README for detailed instructions for this project.


# DESCRIPTION OF THE DATA:
(1) We used the dataset from the ISBI 2019 challenge "Classification of Normal 
versus Malignant Cells in B-ALL White Blood Cancer Microscopic Images". 
Link: https://www.kaggle.com/general/71031
Link: https://biomedicalimaging.org/2019/challenges/
(2) This dataset contains images of leukocytes, size 450x450 pixels. The dataset 
is already divided into training/test (here we use the preliminary test set, 
which was provided with the labels; a final test set was later available but the 
labels were not disclosed, thus it is not used here).
    - The training set has 7272 positive images (belonging with patients with 
      B-lineage Acute Lmphblastic Leukemia, B-ALL, under the folder "all"), and 
      3389 negative images (under the folder "hem").
      Originally, the training images were subdivided in 3 folders, saved in bmp. 
      We transfored into png (to reduce size) and place them in the same folder.
    - The test set has 1867 images. The labels were provided in a CSV file.
The dataset (transformed as indicated above) can be downlaoded here: 
https://drive.google.com/file/d/1RfNt9Zw9x8rX25Mi8QGGzi3C5IMBZPrW/



# DESCRIPTION OF THE TRAINING:
For many experiments, it was not possible to train from scratch with this 
dataset (network could not converge). However, if we used transfer learning 
using the weights from the Myeloid experiment, all networks could converge.
 - Thus, we include a line for transfer learning.
To perform the training, we build our own functions to build the batches and we 
used the keras functions 'train_on_batch' and 'test_on_batch'. Functions to keep 
track of the loss and metrics were created. Thus, an epoch here is simply a  
specific number of iterations.
'''

import os
import argparse
import numpy as np 
from split_data import split_dataset_leukemia
from networks_leukemia import CapsNet_DR, Residual_CapsNet_DR, Dense_CapsNet_DR
from train import train_model_leukemia
from test import test_model_leukemia


def launch_experiment(args):
    """ Given the input arguments of the experiment (args), it ...
    (1) loads the data info and split it intro train/vaid/test; 
    (2) builds the network; 
    (3) loads the weights from two possible places: another model by transfer 
        learning, or from a specific previous epoch in the current model;
    (3) trains the model (if indicated);
    (4) tests the model (if indicated).
    """                   
    trnPos_img, trnNeg_img, valPos_img, valNeg_img, tst_img, tst_lbl, tst_nam \
        = split_dataset_leukemia(args) 
    model = CapsNet_DR(args.input_shape, args=args) # Change here the network
    model.summary()
    if args.transfer_learning:
        model.load_weights(args.save_folder + args.transfer_learning, 
                           by_name=True) 
    if args.weights:
        model.load_weights(args.save_folder + args.weights) 
    if args.train == 1:
        train_model_leukemia(model, trnPos_img, trnNeg_img, valPos_img, 
                             valNeg_img, args)
    if args.test == 1:
        test_model_leukemia(model, tst_img, tst_lbl, tst_nam, args)
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
                             'It uses the path of save_folder')
    parser.add_argument('--transfer_learning', type=str, default='',
                        help='The model.h5 from which to do transfer learning.' 
                             'Set to "" for none. Uses path args.save_folder')
    parser.add_argument('--epochs', type=int, default=500, 
                        help='The number of epochs for training.')
    parser.add_argument('--current_epoch', type=int, default=0,
                        help='If weights are loaded, set the current epoch.')
    parser.add_argument('--iterations', type=int, default=750,
                        help='The number of iterations for one epoch.')      
    parser.add_argument('--iter2eval', type=int, default=750,
                        help='The iterations to launch validation set.')                
    parser.add_argument('--num_folds', type=int, default=10,
                        help='The number of folds to divide the training set.')
    parser.add_argument('--vald_fold', type=int, default=1,
                        help='The index of the fold to use as validation set')
    parser.add_argument('--test_fold', type=int, default=1,
                        help='The index of the test fold. NOT USED HERE!'
                             'The test set is already independent.')
    parser.add_argument('--train', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable training.')
    parser.add_argument('--test', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable testing.')
    
    args = parser.parse_args()
    
    # General model
    args.image_shape = (450, 450, 3)   # This is the original image size
    args.input_shape = (400, 400, 3)   # This is the input to the model
    args.scale_input = 0.8889  # float, to scale image to input; 'None' or 1 for no scaling
    args.crop_input = False  # To crop the image into input_shape (centered) 

    # Learning rate
    args.lr_init = 0.001
    args.lr_decay = 0.99    

    # Capsule section
    args.cap_numbCap = [32,  8,  8,  8,  8,  2]  # Number of capsule per layer
    args.cap_sizeCap = [16, 16, 16, 16, 16, 16]  # Size of the capsule
    args.cap_Kvalues = [ 3,  3,  3,  3,  3,  1]  # Kernel of the ConvCaps
    args.cap_strides = [ 1,  1,  1,  1,  1,  1]  # Strides in the ConvCaps
    args.cap_routing = 3                         # Number of iterations in DR

    # CNN section
    args.drop_rate = None 
    
    # Batches and classes
    # * Even though there is some unbalance in the dataset, we do not see 
    #   necessary to balance the batches.  
    args.batch_size = 14
    args.classes = (0, 1)          # The labels of the classes
    args.weighting_flag = False    # To whether apply sample weighting based ... 
    args.weighting_class = {0: 1, 1: 1}  # ... on the classes, using this dict.                         

    # Augmentation
    args.flag_augmentation = True
    args.batch_normalization = False    
    args.batch_standardization = False     

    # Saving the model and variables (we saved the model at each epoch; this 
    # can be changed in train_model.py)
    args.save_model = True
    args.save_variables = True

    # Directories (training set -positive and negatives-, and test set)
    args.neg_folder = args.data_folder + 'Training/hem/'   
    args.pos_folder = args.data_folder + 'Training/all/'
    args.tst_folder = args.data_folder + 'C-NMC_test_prelim_phase_data/'
    args.tst_labels = args.data_folder + 'C-NMC_test_prelim_phase_data_labels.csv'
    
    if not os.path.exists(args.save_folder):
        print('Creating the folder... ')
        os.makedirs(args.save_folder)
    else:
        print('Folder already exists! ')
    
    # -------------------------------------------------------------------------
    # Run the experiment
    launch_experiment(args)



