"""
This is an exact copy of 'networks_myeloid' but where the only layer that is 
different (the output layer of the network) is renamed. Thus, when doing 
transfer learning from Myeloid models, only this is necessary:
    model.load_weights(args.save_folder + args.transfer_learning, by_name=True) 

This file contains several network definitions: the default network, and with 
dense connections or residual connections within the capsule section.
Check the notes in each network to understand the limitations of each network.
NOTE: In all the examples below, we used the 6 capsule-layers scheme. In our 
experiments, we tested from 2 capsule layers up to 9 capsule layers. It is 
straightforward to add or remove capsule layers: simply add/remove it here and 
update the arguments in the "capsule section" from the main python file. 
"""

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
# Defaultl CapsNet
# ----------------------------------------------------------------------------

def CapsNet_DR(input_shape, args):
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

    # Forth layer of CNNs
    x = conv_block(x, 7, [256, 256, 512], cardinality=64, strides=2, 
                   stage=4, block='a', drop_rate=args.drop_rate)
    print('Output 4th conv layer: ', x.shape)

    # Fifth layer of CNNs
    x = conv_block(x, 7, [512, 512, 1024], cardinality=128, strides=2, 
                   stage=5, block='a', drop_rate=args.drop_rate)
    print('Output 5th conv layer: ', x.shape)
    
    # Attention
    #x = attention_NLA(x, 128, type_att='sum', name='attentionNLA')
    
    # Capsule section ---------------------------------------------------------
    x = primary_caps_DR(oCaps=args.cap_numbCap[0], oPose=args.cap_sizeCap[0], 
                        k=args.cap_Kvalues[0], strides=args.cap_strides[0], 
                        padding='SAME', name='Prim_caps')(x)
    print('Output Primary-Caps:   ', x.shape)

    # Middle-cap 1
    x = conv_caps_DR(iCaps=args.cap_numbCap[0], iPose=args.cap_sizeCap[0], 
                     oCaps=args.cap_numbCap[1], oPose=args.cap_sizeCap[1], 
                     k=args.cap_Kvalues[1], strides=args.cap_strides[1], 
                     batch=args.batch_size, iters=args.cap_routing,
                     padding='SAME', conv_cap=True, name='Mid1_caps')(x)
    print('Output Mid-Caps 1:   ', x.shape)
    
    # Middle-cap 2 
    x = conv_caps_DR(iCaps=args.cap_numbCap[1], iPose=args.cap_sizeCap[1], 
                     oCaps=args.cap_numbCap[2], oPose=args.cap_sizeCap[2], 
                     k=args.cap_Kvalues[2], strides=args.cap_strides[2], 
                     batch=args.batch_size,  iters=args.cap_routing,
                     padding='SAME', conv_cap=True, name='Mid2_caps')(x)
    print('Output Mid-Caps 2:   ', x.shape)
    
    # Middle-cap 3 
    x = conv_caps_DR(iCaps=args.cap_numbCap[2], iPose=args.cap_sizeCap[2], 
                     oCaps=args.cap_numbCap[3], oPose=args.cap_sizeCap[3], 
                     k=args.cap_Kvalues[3], strides=args.cap_strides[3], 
                     batch=args.batch_size, iters=args.cap_routing,
                     padding='SAME', conv_cap=True, name='Mid3_caps')(x)
    print('Output Mid-Caps 3:   ', x.shape)
    
    # Middle-cap 4 
    x = conv_caps_DR(iCaps=args.cap_numbCap[3], iPose=args.cap_sizeCap[3], 
                     oCaps=args.cap_numbCap[4], oPose=args.cap_sizeCap[4], 
                     k=args.cap_Kvalues[4], strides=args.cap_strides[4], 
                     batch=args.batch_size, iters=args.cap_routing,
                     padding='SAME', conv_cap=True, name='Mid4_caps')(x)
    print('Output Mid-Caps 4:   ', x.shape)
 
    # Class capsules 
    x = conv_caps_DR(iCaps=args.cap_numbCap[4], iPose=args.cap_sizeCap[4], 
                     oCaps=args.cap_numbCap[5], oPose=args.cap_sizeCap[5], 
                     k=args.cap_Kvalues[5], iters=args.cap_routing,
                     conv_cap=False, last_cap=True, w_shared=True, 
                     batch=args.batch_size, name='Class_caps_4Leukemia')(x)
    print('Class capsules, activations:', x.shape)

    # Layer Out
    out_caps = activation_caps_DR(act_type='default', name='capsnet_out')(x)
    print('Output out_caps (lenght layer):', out_caps.shape) 

    # Compile the model 
    model = Model(inputs=inputs, outputs=out_caps, name='CapsNets_DR')
    nadam = Nadam(lr=args.lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=nadam,
                  loss=margin_loss,
                  metrics='categorical_accuracy')
    return model



# ----------------------------------------------------------------------------
# Residual CapsNet
# ----------------------------------------------------------------------------
"""
Adding residual connections within the capsule section is straightforward: we 
simply include an "add" layer to sum the output of the previous capsule layer 
with the current one. This only requires that the output of the capsule layers 
have the same shape.  
Things to consider:
- Since Primary_Caps usually have more capsules, this layer is not included in 
  the residual scheme. Thus, we start the residual connections in the 2nd Mid-Cap.
- A "squash" activation can be added after the residual connection to normalize 
  the new capsules. In our experiments, we tested both cases: with and without 
  squashing at those points.
- Residual connections do neither add complexity to the network nor requires 
  more resources.
"""

def Residual_CapsNet_DR(input_shape, args):
    """ A Capsule Network on with DR routing and residual connections.
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

    # Forth layer of CNNs
    x = conv_block(x, 7, [256, 256, 512], cardinality=64, strides=2, 
                   stage=4, block='a', drop_rate=args.drop_rate)
    print('Output 4th conv layer: ', x.shape)

    # Fifth layer of CNNs
    x = conv_block(x, 7, [512, 512, 1024], cardinality=128, strides=2, 
                   stage=5, block='a', drop_rate=args.drop_rate)
    print('Output 5th conv layer: ', x.shape)

    
    # Capsule section -------------------------------------------------
    x = primary_caps_DR(oCaps=args.cap_numbCap[0], oPose=args.cap_sizeCap[0], 
                        k=args.cap_Kvalues[0], strides=args.cap_strides[0], 
                        padding='SAME', name='Prim_caps')(x)
    print('Output Primary-Caps:   ', x.shape)

    # Middle-cap 1 
    x = conv_caps_DR(iCaps=args.cap_numbCap[0], iPose=args.cap_sizeCap[0], 
                     oCaps=args.cap_numbCap[1], oPose=args.cap_sizeCap[1], 
                     k=args.cap_Kvalues[1], strides=args.cap_strides[1], 
                     batch=args.batch_size, iters=args.cap_routing,
                     padding='SAME', conv_cap=True, name='Mid1_caps')(x)
    print('Output Mid-Caps 1:   ', x.shape)
    
    # Middle-cap 2 with residual connection
    x1 = conv_caps_DR(iCaps=args.cap_numbCap[1], iPose=args.cap_sizeCap[1], 
                     oCaps=args.cap_numbCap[2], oPose=args.cap_sizeCap[2], 
                     k=args.cap_Kvalues[2], strides=args.cap_strides[2], 
                     batch=args.batch_size, iters=args.cap_routing,
                     padding='SAME', conv_cap=True, name='Mid2_caps')(x)
    print('Output Mid-Caps 2:   ', x1.shape)
    x = layers.Add(name='Mid2_res')([x, x1])
    x = layers.Lambda(squash, name='Squash_Mid2')(x)
    
    # Middle-cap 3 with residual connection
    x1 = conv_caps_DR(iCaps=args.cap_numbCap[2], iPose=args.cap_sizeCap[2], 
                     oCaps=args.cap_numbCap[3], oPose=args.cap_sizeCap[3], 
                     k=args.cap_Kvalues[3], strides=args.cap_strides[3], 
                     batch=args.batch_size, iters=args.cap_routing,
                     padding='SAME', conv_cap=True, name='Mid3_caps')(x)
    print('Output Mid-Caps 3:   ', x1.shape)
    x = layers.Add(name='Mid3_res')([x, x1])
    x = layers.Lambda(squash, name='Squash_Mid3')(x)
    
    # Middle-cap 4 with residual connection
    x1 = conv_caps_DR(iCaps=args.cap_numbCap[3], iPose=args.cap_sizeCap[3], 
                     oCaps=args.cap_numbCap[4], oPose=args.cap_sizeCap[4], 
                     k=args.cap_Kvalues[4], strides=args.cap_strides[4], 
                     batch=args.batch_size, iters=args.cap_routing,
                     padding='SAME', conv_cap=True, name='Mid4_caps')(x)
    print('Output Mid-Caps 4:   ', x1.shape)
    x = layers.Add(name='Mid4_res')([x, x1])
    x = layers.Lambda(squash, name='Squash_Mid4')(x)
 
    # Class capsules 
    x = conv_caps_DR(iCaps=args.cap_numbCap[4], iPose=args.cap_sizeCap[4], 
                     oCaps=args.cap_numbCap[5], oPose=args.cap_sizeCap[5], 
                     k=args.cap_Kvalues[5], iters=args.cap_routing,
                     conv_cap=False, last_cap=True, w_shared=True, 
                     batch=args.batch_size, name='Class_caps_4Leukemia')(x)
    print('Class capsules, activations:', x.shape)

    # Layer Out
    out_caps = activation_caps_DR(act_type='default', name='capsnet_out')(x)
    print('Output out_caps (lenght layer):', out_caps.shape) 

    # Compile the model 
    model = Model(inputs=inputs, outputs=out_caps, name='Residual_CapsNets_DR')
    nadam = Nadam(lr=args.lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=nadam,
                  loss=margin_loss,
                  metrics='categorical_accuracy')
    return model



# ----------------------------------------------------------------------------
# Fully Residual CapsNet
# ----------------------------------------------------------------------------
"""
A fully residual capsnets is a small variation of the residual network where all 
layers have residual connections. Note that ResNeXt in the CNN part is already 
residual. So, we perform the following:
- The Primary_Caps are formed by using a ResNeXt block + reshape + squash
- The number of capsules: args.cap_numbCap = [10, 10, 10, 10, 10, 15]
"""

def Fully_Residual_CapsNet_DR(input_shape, args):
    """ A Capsule Network on with DR routing and residual connections.
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

    # Forth layer of CNNs
    x = conv_block(x, 7, [256, 256, 512], cardinality=64, strides=2, 
                   stage=4, block='a', drop_rate=args.drop_rate)
    print('Output 4th conv layer: ', x.shape)

    # Fifth layer of CNNs
    x = conv_block(x, 7, [512, 512, 1024], cardinality=128, strides=2, 
                   stage=5, block='a', drop_rate=args.drop_rate)
    print('Output 5th conv layer: ', x.shape)

    
    # Capsule section -------------------------------------------------
    # Primary Caps by residual block + reshape + squash
    x = conv_block(x, 3, [128, 128, 160], cardinality=32, strides=1, 
                   stage=6, block='a', drop_rate=None)
    x = primary_caps_DR(oCaps=args.cap_numbCap[0], oPose=args.cap_sizeCap[0], 
                        k=args.cap_Kvalues[0], strides=args.cap_strides[0], 
                        do_poses=False, squashed=True, name='Prim_caps')(x)
    print('Output Primary-Caps:   ', x.shape)

    # Middle-cap 1 with residual connection
    x1 = conv_caps_DR(iCaps=args.cap_numbCap[0], iPose=args.cap_sizeCap[0], 
                      oCaps=args.cap_numbCap[1], oPose=args.cap_sizeCap[1], 
                      k=args.cap_Kvalues[1], strides=args.cap_strides[1], 
                      batch=args.batch_size, iters=args.cap_routing,
                      padding='SAME', conv_cap=True, name='Mid1_caps')(x)
    print('Output Mid-Caps 1:   ', x1.shape)
    x = layers.Add(name='Mid1_res')([x, x1])
    x = layers.Lambda(squash, name='Squash_Mid1')(x)
    
    # Middle-cap 2 with residual connection
    x1 = conv_caps_DR(iCaps=args.cap_numbCap[1], iPose=args.cap_sizeCap[1], 
                      oCaps=args.cap_numbCap[2], oPose=args.cap_sizeCap[2], 
                      k=args.cap_Kvalues[2], strides=args.cap_strides[2], 
                      batch=args.batch_size, iters=args.cap_routing,
                      padding='SAME', conv_cap=True, name='Mid2_caps')(x)
    print('Output Mid-Caps 2:   ', x1.shape)
    x = layers.Add(name='Mid2_res')([x, x1])
    x = layers.Lambda(squash, name='Squash_Mid2')(x)
    
    # Middle-cap 3 with residual connection
    x1 = conv_caps_DR(iCaps=args.cap_numbCap[2], iPose=args.cap_sizeCap[2], 
                      oCaps=args.cap_numbCap[3], oPose=args.cap_sizeCap[3], 
                      k=args.cap_Kvalues[3], strides=args.cap_strides[3], 
                      batch=args.batch_size, iters=args.cap_routing,
                      padding='SAME', conv_cap=True, name='Mid3_caps')(x)
    print('Output Mid-Caps 3:   ', x1.shape)
    x = layers.Add(name='Mid3_res')([x, x1])
    x = layers.Lambda(squash, name='Squash_Mid3')(x)
    
    # Middle-cap 4 with residual connection
    x1 = conv_caps_DR(iCaps=args.cap_numbCap[3], iPose=args.cap_sizeCap[3], 
                      oCaps=args.cap_numbCap[4], oPose=args.cap_sizeCap[4], 
                      k=args.cap_Kvalues[4], strides=args.cap_strides[4], 
                      batch=args.batch_size, iters=args.cap_routing,
                      padding='SAME', conv_cap=True, name='Mid4_caps')(x)
    print('Output Mid-Caps 4:   ', x1.shape)
    x = layers.Add(name='Mid4_res')([x, x1])
    x = layers.Lambda(squash, name='Squash_Mid4')(x)
    
    # Class capsules 
    x = conv_caps_DR(iCaps=args.cap_numbCap[4], iPose=args.cap_sizeCap[4], 
                     oCaps=args.cap_numbCap[5], oPose=args.cap_sizeCap[5], 
                     k=args.cap_Kvalues[5], iters=args.cap_routing,
                     conv_cap=False, last_cap=True, w_shared=True, 
                     batch=args.batch_size, name='Class_caps_4Leukemia')(x)
    print('Class capsules, activations:', x.shape)

    # Layer Out
    out_caps = activation_caps_DR(act_type='default', name='capsnet_out')(x)
    print('Output out_caps (lenght layer):', out_caps.shape) 

    # Compile the model 
    model = Model(inputs=inputs, outputs=out_caps, 
                  name='Fully_Residual_CapsNets_DR')
    nadam = Nadam(lr=args.lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=nadam,
                  loss=margin_loss,
                  metrics='categorical_accuracy')
    return model





# ----------------------------------------------------------------------------
# Dense CapsNet
# ----------------------------------------------------------------------------
"""
Adding dense connections within the capsule section is very straightforward: we 
only need to concatenate capsules from different layers along the capsule-axis.
This only requires that the feature maps at the output of the capsule layers 
should have the same size for H, W, and capsule_size.
Things to consider:
- Dense connections are very resource-demanding. Usually, the number of capsules 
  per layer should be reduced to allocate for the dense connections.
- In our experiments, we reduced the network in the following way: 
  * The filters in the 5th CNN were reduce to half.
  * The number of capsules:  args.cap_numbCap = [24, 4, 4, 4, 4, 15]
- We also experimented with the "depth" of the dense connections. In the example 
  below, depth=3. This means that the input of a capsule layer only receives 
  the capsules from the previous 3 capsule layers. 
  At most, depth = num_caps_layers - 1
"""

def Dense_CapsNet_DR(input_shape, args):
    """ A Capsule Network on with DR routing and dense connections.
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

    # Forth layer of CNNs
    x = conv_block(x, 7, [256, 256, 512], cardinality=64, strides=2, 
                   stage=4, block='a', drop_rate=args.drop_rate)
    print('Output 4th conv layer: ', x.shape)

    # Fifth layer of CNNs
    x = conv_block(x, 7, [256, 256, 512], cardinality=64, strides=2, 
                   stage=5, block='a', drop_rate=args.drop_rate)
    print('Output 5th conv layer: ', x.shape)

    
    # Capsule section -------------------------------------------------
    x0 = primary_caps_DR(oCaps=args.cap_numbCap[0], oPose=args.cap_sizeCap[0], 
                         k=args.cap_Kvalues[0], strides=args.cap_strides[0], 
                         padding='SAME', name='Prim_caps')(x)
    print('Output Primary-Caps:   ', x0.shape)

    # Middle-cap 1 with dense connection
    # Concatenate in the Capsule-axis -> pose.shape = [b, h, w, oCap, oPose]
    x1 = conv_caps_DR(iCaps=args.cap_numbCap[0], iPose=args.cap_sizeCap[0], 
                      oCaps=args.cap_numbCap[1], oPose=args.cap_sizeCap[1], 
                      k=args.cap_Kvalues[1], strides=args.cap_strides[1], 
                      batch=args.batch_size, iters=args.cap_routing,
                      padding='SAME', conv_cap=True, name='Mid1_caps')(x0)
    print('Output Mid-Caps 1:   ', x1.shape)
    x = layers.Concatenate(axis=3, name='Mid1_conc')([x0, x1])
    print('Output ConcCaps 1:   ', x.shape)
    
    # Middle-cap 2 with dense connection
    x2 = conv_caps_DR(iCaps=sum(args.cap_numbCap[0:2]), iPose=args.cap_sizeCap[1], 
                      oCaps=args.cap_numbCap[2], oPose=args.cap_sizeCap[2], 
                      k=args.cap_Kvalues[2], strides=args.cap_strides[2], 
                      batch=args.batch_size, iters=args.cap_routing,
                      padding='SAME', conv_cap=True, name='Mid2_caps')(x)
    print('Output Mid-Caps 2:   ', x2.shape)
    x = layers.Concatenate(axis=3, name='Mid2_conc')([x0, x1, x2])
    print('Output ConcCaps 2:   ', x.shape)
    
    # Middle-cap 3 with dense connection
    x3 = conv_caps_DR(iCaps=sum(args.cap_numbCap[0:3]), iPose=args.cap_sizeCap[2], 
                      oCaps=args.cap_numbCap[3], oPose=args.cap_sizeCap[3], 
                      k=args.cap_Kvalues[3], strides=args.cap_strides[3], 
                      batch=args.batch_size, iters=args.cap_routing,
                      padding='SAME', conv_cap=True, name='Mid3_caps')(x)
    print('Output Mid-Caps 3:   ', x3.shape)
    x = layers.Concatenate(axis=3, name='Mid3_conc')([x1, x2, x3])
    print('Output ConcCaps 3:   ', x.shape)
    
    # Middle-cap 4 with dense connection
    x4 = conv_caps_DR(iCaps=sum(args.cap_numbCap[1:4]), iPose=args.cap_sizeCap[3], 
                      oCaps=args.cap_numbCap[4], oPose=args.cap_sizeCap[4], 
                      k=args.cap_Kvalues[4], strides=args.cap_strides[4], 
                      batch=args.batch_size, iters=args.cap_routing,
                      padding='SAME', conv_cap=True, name='Mid4_caps')(x)
    print('Output Mid-Caps 4:   ', x4.shape)
    x = layers.Concatenate(axis=3, name='Mid4_conc')([x2, x3, x4])
    print('Output ConcCaps 4:   ', x.shape)
 
    # Class capsules 
    x = conv_caps_DR(iCaps=sum(args.cap_numbCap[2:5]), iPose=args.cap_sizeCap[4], 
                     oCaps=args.cap_numbCap[5], oPose=args.cap_sizeCap[5], 
                     k=args.cap_Kvalues[5], iters=args.cap_routing,
                     conv_cap=False, last_cap=True, w_shared=True, 
                     batch=args.batch_size, name='Class_caps_4Leukemia')(x)
    print('Class capsules, activations:', x.shape)

    # Layer Out
    out_caps = activation_caps_DR(act_type='default', name='capsnet_out')(x)
    print('Output out_caps (lenght layer):', out_caps.shape) 

    # Compile the model 
    model = Model(inputs=inputs, outputs=out_caps, name='Dense_CapsNets_DR')
    nadam = Nadam(lr=args.r_init, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=nadam,
                  loss=margin_loss,
                  metrics='categorical_accuracy')
    return model



# ----------------------------------------------------------------------------
# Default CapsNet with CNN reduction
# ----------------------------------------------------------------------------
"""
We also experimented with the idea of removing CNNs in the convolutional section 
and, instead, adding strides in the capsule section.
However, a few things needs to be considered in this approach:
- The Primary Caps should be reduced in number, otherwise there is a bottleneck 
  due to the large size of the features at the input of the capsule section.
  However, this shouldn't be a problem if enough memory resources are available. 
- Increasing the Kvalue (the window size of the convolutional capsule) is also
  computationally demanding. In our experiments, we only increased it in the 
  Primary caps. 

For instance, in the below network with 6 capsule-layers and 2 CNNs (we removed 
3 CNNS), we did the following:
- Set the Primary Caps to 8.
- Increase the strides in the first 3 capsule-layers.
- increased the Kvalue of the PrimaryCaps to 5.
    args.cap_numbCap = [ 8,  8,  8,  8,  8, 15]
    args.cap_sizeCap = [16, 16, 16, 16, 16, 16]
    args.cap_Kvalues = [ 5,  3,  3,  3,  3,  1]
    args.cap_strides = [ 2,  2,  2,  1,  1,  1]

For the network with 3 CNNS:
    args.cap_numbCap = [16,  8,  8,  8,  8, 15]
    args.cap_sizeCap = [16, 16, 16, 16, 16, 16]
    args.cap_Kvalues = [ 5,  3,  3,  3,  3,  1]
    args.cap_strides = [ 2,  2,  1,  1,  1,  1]

And for the network with 4 CNNs:
    args.cap_numbCap = [24,  8,  8,  8,  8, 15]
    args.cap_sizeCap = [16, 16, 16, 16, 16, 16]
    args.cap_Kvalues = [ 5,  3,  3,  3,  3,  1]
    args.cap_strides = [ 2,  1,  1,  1,  1,  1]

"""

def Reduced_CapsNetDR(input_shape, args):
    """ A Capsule Network on with EM routing.
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
    # x = conv_block(x, 7, [128, 128, 256], cardinality=32, strides=2, 
                   # stage=3, block='a', drop_rate=args.drop_rate)
    # print('Output 3rd conv layer: ', x.shape)

    # Forth layer of CNNs
    # x = conv_block(x, 7, [256, 256, 512], cardinality=64, strides=2, 
                   # stage=4, block='a', drop_rate=args.drop_rate)
    # print('Output 4th conv layer: ', x.shape)

    # Fifth layer of CNNs
    # x = conv_block(x, 7, [256, 256, 512], cardinality=64, strides=2, 
                   # stage=5, block='a', drop_rate=args.drop_rate)
    # print('Output 5th conv layer: ', x.shape)

    
    # Capsule section -------------------------------------------------
    x = primary_caps_DR(oCaps=args.cap_numbCap[0], oPose=args.cap_sizeCap[0], 
                        k=args.cap_Kvalues[0], strides=args.cap_strides[0], 
                        padding='SAME', name='Prim_caps')(x)
    print('Output Primary-Caps:   ', x.shape)

    # Middle-cap 1 
    x = conv_caps_DR(iCaps=args.cap_numbCap[0], iPose=args.cap_sizeCap[0], 
                     oCaps=args.cap_numbCap[1], oPose=args.cap_sizeCap[1], 
                     k=args.cap_Kvalues[1], strides=args.cap_strides[1], 
                     batch=args.batch_size, iters=args.cap_routing,
                     padding='SAME', conv_cap=True, name='Mid1_caps')(x)
    print('Output Mid-Caps 1:   ', x.shape)
    
    # Middle-cap 2 
    x = conv_caps_DR(iCaps=args.cap_numbCap[1], iPose=args.cap_sizeCap[1], 
                     oCaps=args.cap_numbCap[2], oPose=args.cap_sizeCap[2], 
                     k=args.cap_Kvalues[2], strides=args.cap_strides[2], 
                     batch=args.batch_size, iters=args.cap_routing,
                     padding='SAME', conv_cap=True, name='Mid2_caps')(x)
    print('Output Mid-Caps 2:   ', x.shape)
    
    # Middle-cap 3 
    x = conv_caps_DR(iCaps=args.cap_numbCap[2], iPose=args.cap_sizeCap[2], 
                     oCaps=args.cap_numbCap[3], oPose=args.cap_sizeCap[3], 
                     k=args.cap_Kvalues[3], strides=args.cap_strides[3], 
                     batch=args.batch_size, iters=args.cap_routing,
                     padding='SAME', conv_cap=True, name='Mid3_caps')(x)
    print('Output Mid-Caps 3:   ', x.shape)
    
    # Middle-cap 4 
    x = conv_caps_DR(iCaps=args.cap_numbCap[3], iPose=args.cap_sizeCap[3], 
                     oCaps=args.cap_numbCap[4], oPose=args.cap_sizeCap[4], 
                     k=args.cap_Kvalues[4], strides=args.cap_strides[4], 
                     batch=args.batch_size, iters=args.cap_routing,
                     padding='SAME', conv_cap=True, name='Mid4_caps')(x)
    print('Output Mid-Caps 4:   ', x.shape)
 
    # Class capsules 
    x = conv_caps_DR(iCaps=args.cap_numbCap[4], iPose=args.cap_sizeCap[4], 
                     oCaps=args.cap_numbCap[5], oPose=args.cap_sizeCap[5], 
                     k=args.cap_Kvalues[5], iters=args.cap_routing,
                     conv_cap=False, last_cap=True, w_shared=True, 
                     name='Class_caps_4Leukemia')(x)
    print('Class capsules, activations:', x.shape)

    # Layer Out
    out_caps = activation_caps_DR(act_type='default', name='capsnet_out')(x)
    print('Output out_caps (lenght layer):', out_caps.shape) 

    # Compile the model 
    model = Model(inputs=inputs, outputs=out_caps, name='CapsNets_DR')
    nadam = Nadam(lr=args.lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=nadam,
                  loss=margin_loss,
                  metrics='categorical_accuracy')
    return model