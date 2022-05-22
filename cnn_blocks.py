"""
The functions to build a block of ResNeXt with group convolutions and a block 
of non-local attention
"""

import tensorflow as tf
import tensorflow.keras.layers as layers 
from tensorflow.keras import initializers, regularizers, constraints, activations


class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 name,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 activation=None,
                 groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be " \
                              + " divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be " \
                              + " divisible by the value of groups.")

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(tf.keras.layers.Conv2D(
                filters=self.group_out_num,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activations.get(activation),
                use_bias=use_bias,
                kernel_initializer=initializers.get(kernel_initializer),
                bias_initializer=initializers.get(bias_initializer),
                kernel_regularizer=regularizers.get(kernel_regularizer),
                bias_regularizer=regularizers.get(bias_regularizer),
                activity_regularizer=regularizers.get(activity_regularizer),
                kernel_constraint=constraints.get(kernel_constraint),
                bias_constraint=constraints.get(bias_constraint),
                name=name+str(i),
                **kwargs))

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, 
                    i*self.group_in_num: (i + 1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out




def conv_block(input_tensor, kernel_size, filters, cardinality, stage, block, 
               strides=1, drop_rate=None):
    """A convolutional block. The shortcut might have a convolution layer. 
    # Arguments
        input_tensor: input tensor
        kernel_size:  int, the kernel size of middle conv layer at main path
        filters:      list of integers, the filters at main path
        stage:        int, current stage label, used for generating layer names
        block:        str, current block label, used for generating layer names
        strides:      int, strides for the middle conv layer in the block.
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = str(stage) + block + '_conv' + '_branch'
    bnrm_name_base = str(stage) + block + '_bnor' + '_branch'
    actv_name_base = str(stage) + block + '_actv' + '_branch'
    xtra_name_base = str(stage) + block 
  
    x = layers.Conv2D(filters1, (1, 1), strides=1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2_a')(input_tensor)
    x = layers.BatchNormalization(name=bnrm_name_base + '2_a')(x)
    x = layers.Activation('relu', name=actv_name_base + '2_a')(x)
    
    x = GroupConv2D(input_channels=filters1, output_channels=filters2,
                    kernel_size=kernel_size, strides=strides, padding='same', 
                    groups=cardinality, kernel_initializer='he_normal',
                    name=conv_name_base + '2_b')(x)       
    x = layers.BatchNormalization(name=bnrm_name_base + '2_b')(x)
    x = layers.Activation('relu', name=actv_name_base + '2_b')(x)

    x = layers.Conv2D(filters3, (1, 1), strides=1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2_c')(x)
    x = layers.BatchNormalization(name=bnrm_name_base + '2_c')(x)

    if strides > 1 or input_tensor.shape[-1] != filters3:
        shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '1')(input_tensor)
        shortcut = layers.BatchNormalization(name=bnrm_name_base + '1')(shortcut)
    else:
        shortcut = input_tensor

    x = layers.Add(name=xtra_name_base + '_add_0')([x, shortcut])
    x = layers.Activation('relu', name=xtra_name_base + '_actv_0')(x)
    if drop_rate is not None:
        x = layers.Dropout(rate=drop_rate, name=xtra_name_base + '_drop_0')(x)
    return x



def attention_NLA(x, ichannels, g=None, type_att='sum', name=None, 
                  return_attention=False):
    """ An attention block inspired from the paper 'Wang 2018 - Non-local neural 
    networks'. Several options are available:
    - If a tensor from a lower-level is given (already upsampled), this is used   
      to build phi and gee, otherwise the default Wang version is used.
    - The attention maps can be either multiplied (only one map) or summed (same  
      number of maps).
    # Arguments
        x:            input tensor, where attention is applied.
        ichannels:    float, number of channels in the first reduction.
        g:            lower-level input tensor, used to determine where to pay 
                      attention. This should already be upsampled.
        type_att:     how the attention is applied: 'sum' or 'mul'.
        name:         string, block label.
    # Returns
        Output tensor for the block.
    """  
    if g is None: g = x 
    
    # Apply conv1x1 to create 3 different tensors
    the_x = layers.Conv2D(ichannels, 1, strides=1, name=name + '_conv_the_x')(x)
    phi_x = layers.Conv2D(ichannels, 1, strides=1, name=name + '_conv_phi_x')(g)
    gee_x = layers.Conv2D(ichannels, 1, strides=1, name=name + '_conv_gee_x')(g)
    _,h,w,c = the_x.shape  

    # Reshape the 3 tensors: 
    # the_x and gee_x become (Batch, C, HW). phi_x becomes (Batch, HW, C)
    the_x = layers.Reshape((int(h*w),c), name=name + '_reshape_the_x')(the_x)
    phi_x = layers.Reshape((int(h*w),c), name=name + '_reshape_phi_x')(phi_x)
    gee_x = layers.Reshape((int(h*w),c), name=name + '_reshape_gee_x')(gee_x)
    the_x = layers.Permute((2,1), name=name + '_permute_the_x')(the_x)
    gee_x = layers.Permute((2,1), name=name + '_permute_gee_x')(gee_x)

    # Apply matrix multiplication - zet_x becomes (Batch, HW, C)
    eta_x = tf.matmul(the_x, phi_x)
    eta_x = layers.Activation('softmax', name=name + '_actv_eta_x')(eta_x)
    zet_x = tf.matmul(eta_x, gee_x)

    # Reshape to return to original dimensions -> (Batch, H, W, C)
    zet_x = layers.Reshape((h,w,c), name=name + '_reshape_zet_x')(zet_x)

    # Convolve again and add/multiply to the original input
    if type_att == 'sum':
        att_x = layers.Conv2D(x.shape[3], 1, strides=1, name=name + '_conv_att_x')(zet_x)
        att_x = layers.Activation('relu', name=name + '_actv_relu')(att_x)
        psi_x = layers.Add(name=name + '_add_psi_x')([x, att_x])
    elif type_att == 'mul':
        att_x = layers.Conv2D(1, 1, strides=1, name=name + '_conv_att_x')(zet_x)
        att_x = layers.Activation('sigmoid', name=name + '_actv_sigmoid')(att_x)
        psi_x = layers.Multiply(name=name + '_mul_psi_x')([x, att_x])
    if return_attention:
        return psi_x, att_x
    return psi_x
