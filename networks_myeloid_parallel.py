"""
This file contains several network definitions: the default network, and with 
dense connections or residual connections within the capsule section.
Check the notes in each network to understand the limitations of each network.
NOTE: In all the examples below, we used the 6 capsule-layers scheme. In our 
experiments, we tested from 2 capsule layers up to 9 capsule layers. It is 
straightforward to add or remove capsule layers: simply add/remove it here and 
update the arguments in the "capsule section" from the main python file. 
"""

import numpy as np
import tensorflow as tf  
import tensorflow.keras.layers as layers 
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Nadam

from cnn_blocks import conv_block, attention_NLA
from convcaps_dr import primary_caps_DR, conv_caps_DR, activation_caps_DR, squash
from losses import margin_loss

K.set_image_data_format('channels_last')


# ----------------------------------------------------------------------------
# Parallel CapsNet
# ----------------------------------------------------------------------------

def branching_caps(x, idxStage, idxElemt, args, printShape=False):
    """ This function builds the branching (in the CNN section) up to the 
    merging of the capsules (in this case, in the mid-caps).
    See that we branch based on the info from args.CNN_branches.
    The if-else deals with the branching part: 
    - Any previous non-branched CNN block should be place outside in the main 
      function.
    - Any posterior non-branched CNN block should be place in the beginning of 
      the non-branched code.
    # Arguments
        x:            input_tensor 
        idxStage:     int, the number of the stage where the ResNeXt start 
                      building up. The first stage is zero. This is used as 
                      index to the python lists in args.
        idxElemt:     int, the node/element in a stage. This is used as 
                      index to the python lists in args (CNN_branches).
        args:         namespace, with all the arguments required to build the 
                      network.
        printShape:   boolean, to print the shapes at the different steps
    """
    # Branched section (CNN section) ------------------------------------------
    if idxStage < len(args.CNN_branches):   
        nbranches = args.CNN_branches[idxStage][idxElemt]
        baseElemt = int(np.sum(args.CNN_branches[idxStage][0:idxElemt]))
        if printShape: print('Stage:   ', idxStage+1)
        if printShape: print('CNN - Number of branches:   ', nbranches)
        
        for ii in range(nbranches):
            x1 = conv_block(x, 7, args.CNN_filters[idxStage], 
                            cardinality=args.CNN_filters[idxStage][0]//4, 
                            strides=args.CNN_strides[idxStage], 
                            stage=idxStage + args.initIdxStage, 
                            block='b' + str(baseElemt+ii),
                            drop_rate=args.drop_rate)
            if printShape: print('Output of CNN layer: ', x1.shape)
            
            # Branching ---
            x1 = branching_caps(x1, idxStage+1, baseElemt+ii, args, 
                                printShape=printShape)
                                
            # Concatenate the branches (in the Capsule axis)
            x2 = x1 if ii == 0 else layers.Concatenate(axis=3)([x2, x1])
        return x2
        
        
    # Non-branched section (Capsule section) ----------------------------------
    else:     
        # Attention
        #x1 = attention_NLA(x, 64, type_att='sum', name='attentionNLA' + str(idxElemt))
    
        if printShape: print('Input to Primary-Caps:   ', x.shape)    
        x1 = primary_caps_DR(oCaps=args.cap_numbCap[0], oPose=args.cap_sizeCap[0], 
                             k=args.cap_Kvalues[0], strides=args.cap_strides[0], 
                             padding='SAME', name='Prim_caps' + str(idxElemt))(x)
        if printShape: print('Output Primary-Caps:   ', x1.shape)
        

        # Middle-cap 1
        x1 = conv_caps_DR(iCaps=args.cap_numbCap[0], iPose=args.cap_sizeCap[0], 
                          oCaps=1, oPose=args.cap_sizeCap[1], 
                          k=args.cap_Kvalues[1], strides=args.cap_strides[1], 
                          padding='SAME', iters=args.cap_routing,
                          conv_cap=True, name='Mid1_caps' + str(idxElemt))(x1)
        if printShape: print('Output Mid-Caps 1:   ', x1.shape)
        return x1



def CapsNet_DR_parallel(input_shape, args):
    """ A Capsule Network on with DR routing.
    # Arguments
        input_shape:    data shape, 3d, [width, height, channels]
        args:           A namespace with the parameters of the network
    """
    inputs = layers.Input(shape=input_shape)
    print('Input:', inputs.shape)


    # CNN section -------------------------------------------------------------
    # First layer with a single CNN to extract basic features
    x = layers.Conv2D(64, 7, strides=2, padding='same',
                      kernel_initializer='he_normal', name='1a_conv')(inputs)
    x = layers.BatchNormalization(name='1a_conv_bn')(x)
    x = layers.Activation('relu', name='1a_conv_act')(x)
    print('Output 1st conv layer: ', x.shape)

    # Second layer of CNNs
    x = conv_block(x, 7, [64, 64, 128], cardinality=16, strides=2, 
                   stage=2, block='a', drop_rate=args.drop_rate)
    print('Output 2nd conv layer: ', x.shape)

    # Third layer of CNNs
    x = conv_block(x, 7, [128, 128, 256], cardinality=32, strides=2, 
                   stage=3, block='a', drop_rate=args.drop_rate)
    print('Output 3rd conv layer: ', x.shape)   
    
    # Branching up to Mid-Caps ------------------------------------------------
    x = branching_caps(x, idxStage=0, idxElemt=0, args=args, printShape=True)
    print('Output after branching:', x.shape)
    
 
    # Class capsules ----------------------------------------------------------
    x = conv_caps_DR(iCaps=args.cap_numbCap[1], iPose=args.cap_sizeCap[1], 
                     oCaps=args.cap_numbCap[2], oPose=args.cap_sizeCap[2], 
                     k=args.cap_Kvalues[2], iters=args.cap_routing,
                     conv_cap=False, last_cap=True, w_shared=True, 
                     name='Class_caps')(x)
    print('Class capsules, activations:', x.shape)

    # Layer Out
    out_caps = activation_caps_DR(act_type='default', name='capsnet_out')(x)
    print('Output out_caps (lenght layer):', out_caps.shape) 

    # Compile the model 
    model = Model(inputs=inputs, outputs=out_caps, name='CapsNets_DR_Par')
    nadam = Nadam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=nadam,
                  loss=margin_loss,
                  metrics='categorical_accuracy')
    return model

