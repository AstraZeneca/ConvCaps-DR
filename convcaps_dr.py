
import tensorflow as tf  
import tensorflow.keras.layers as layers 
from tensorflow.keras import initializers, regularizers
import numpy as np


class activation_caps_DR(layers.Layer):
    """ Compute the activation of the capsules (vectors). This is used as the 
    last layer of the model to predict the labels. There are several ways to 
    compute the activation:
    * Default: It uses the length (norm) of the vectors. 
    * Linear:  It learns a linear combination of the elements of the capsule 
               (with a conv1D) and then applies sigmoid
    * Linear + Norm: It uses a conv1D to reduce the dimensionality of the 
               capsule and then applies the default norm. This allows the 
               network to compute the norm from a linear combination of the 
               original capsule, thereby some elements of the original capsule 
               might not be used to compute the activation.
    Args:
        x:          tensor, shape=[b, oCaps, oPose]
        act_type:   string, type of activation ('default' or 'linear')
    Returns:        tensor, shape=[b, oCaps]
    """
    def __init__(self, act_type='default', pose=8, **kwargs):
        super(activation_caps_DR, self).__init__()
        self.act_type = act_type
        if act_type == 'linear':
            self.actv = layers.Conv1D(filters=1, kernel_size=1, 
                                      strides=1, padding='same',
                                      use_bias=True, 
                                      activation='sigmoid')    
        elif act_type == 'linear_norm':
            self.actv = layers.Conv1D(filters=pose, kernel_size=1, 
                                      strides=1, padding='same',
                                      use_bias=True)  
        
    def call(self, x, epsilon=1e-12, **kwargs):
        if self.act_type == 'linear':
            _, vec, dim = x.shape 
            actv = self.actv(x)
            actv = tf.reshape(actv, [-1, vec]) 
        elif self.act_type == 'linear_norm':
            _, vec, dim = x.shape 
            actv = self.actv(x)
            actv = tf.sqrt(tf.reduce_sum(tf.square(actv), -1) + epsilon)
        else:
            actv = tf.sqrt(tf.reduce_sum(tf.square(x), -1) + epsilon)
        return actv


def squash(x, axis=-1, epsilon=1e-12):
    """ The non-linear activation used in DR-Capsules. It drives the length 
    of a large vector to near 1 and small vector to 0. """
    squared_norm = tf.reduce_sum(tf.square(x), axis, keepdims=True)
    safe_norm = tf.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = x / safe_norm
    squashed_vector = squash_factor * unit_vector
    return squashed_vector


class primary_caps_DR(layers.Layer):
    """This constructs a primary capsule layer using a convolution layer.
    The convolutions generate feature maps that are considered the poses. 
    The definition of Primary Capsule is from the DR-Capsule paper: 
    * The pose is a vector. 
      NOTE: The poses can be later reshaped into matrices to perform the 
      matrix-multiplication necessary to compute the next-layer capsule if that 
      is the desired type of capsule.
    * The poses are squashed. This makes the norm of the capsule to be within 
      the range 0-1.       
    * Contrary to EM-Caps, here the activation will be considered to be the 
      norm of the vector (not relevant in this layer but only in the last 
      capsule layers). Different from EM-Caps, activations are not computed and  
      added to the tensors (they can be directly computed when necessary).
    * (Optionally) We do not build the poses within the layer but, instead, 
      we assume that the input are already the poses. This is useful to allow 
      for more complex convolutional blocks to generate the first poses 
      instead of the default (and simple) conv2D.
      NOTE: In this case, this layer simply checks that the input features can 
      form equally-sized capsules, and it squashes the capsules (if indicated).
    Args:
        oCaps: 	 	int, number of output capsules
        oPose: 		int, size of pose matrix, p*p
        k: 		 	int, kernel size of convolution
        strides: 	int, strides of convolution
        do_poses:   boolean, flag to indicate that the poses will be done within 
                    the layer with a conv2D (using the previous parameters). 
        squashed:   boolean, flag to indicate the squashing of the poses.
    Shape:
        input:   	(*, h,  w,  iFeat)
        output: 	(*, h', w', oCaps, oPose)
        iFeat is the input number of features 
        h', w' are computed the same way as convolution layer
    """
    def __init__(self, oCaps=32, oPose=16, k=1, strides=1, padding='SAME', 
                 do_poses=True, squashed=True, **kwargs):
        super(primary_caps_DR, self).__init__(**kwargs)
        self.oCaps = oCaps
        self.oPose = oPose   
        self.squashed = squashed 
        self.do_poses = do_poses
        if do_poses:
            self.pose = layers.Conv2D(filters=oCaps*oPose, kernel_size=k, 
                                      strides=strides, padding=padding,
                                      use_bias=True, name=self.name + '_conv')    
    
    def build(self, input_shape):  
        # If the input are the poses, check they are divisible into capsules.
        if not self.do_poses:
            assert input_shape[3] == self.oCaps * self.oPose    

        
    def squash(self, x, axis=-1, epsilon=1e-12):
        """ The non-linear activation used in DR-Capsules. It drives the length 
        of a large vector to near 1 and small vector to 0. """
        squared_norm = tf.reduce_sum(tf.square(x), axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = x / safe_norm
        squashed_vector = squash_factor * unit_vector
        return squashed_vector


    def call(self, x, training=False):         
        pose = self.pose(x) if self.do_poses else x    
        _, h, w, _ = pose.shape 
        pose = tf.reshape(pose, [-1, h, w, self.oCaps, self.oPose])
        if self.squashed: pose = self.squash(pose)    
        return pose



class conv_caps_DR(layers.Layer):
    """ This constructs a convolution capsule layer, whose input is a primary 
    capsule or a convolution capsule, transfering capsule layer L to capsule 
    layer L+1 by DR routing.
    Overall, there are three possibilities for the capsule application and 
    three possibilities for the capsule poses:

    - Capsule application: Convolutional and non-convolutional. 
      * In a convolutional approach, a capsule instantiation (in L+1) uses a 
        small number of capsules (in L) that are close in proximity (k*k, as a 
        conventional convolutional operation does).
      * In a non-convolutional approach (used, for instance, in the last layer 
        of capsules), the spatial information is lost. In this case, two options 
        are possible: sharing and non-sharing weights.
        ** Sharing weights: For each type of capsule, we have an instantiation  
           in each spatial point of the grid (h*w). If we "share weights", those 
           same type of capsules use the same weight_matrix (in practice, we 
           just tile the W matrix over the h*w).  This is assumed to be the 
           original idea of CapsNets.
        ** Not wharing weights: We simply make a larger W matrix considering the 
           spatial resolution (h*w). 
           Con: This increases the number of weights by h*w. 
           Pro: This allows to have different transformations depending on where 
                the capsule lies in the grid.
    
    - Capsule pose: Matrix and vector. 
      * This only changes the function 'transform_view', i.e. how the matrix 
        multiplication is done (DR was originally defined for vectors).
      * If a 'matrix type' is used, there is the option of using two weight 
        matrices, which is necessary if different pose sizes exist for the input 
        and output capsules.
        NOTE: Preliminary tests show that using 2 weight matrices are slightly 
              detrimental. For now, it seems better to use the same pose-size 
              for all layers of capsules (or to use vector-type in the specific 
              interface where the capsules change size).

    The steps in a conv_caps are (see 'call'):
    1. If using a convolutional approach (conv_cap==True), the input capsules 
       are prepared to allow convolutional scheme (function: kernel_tile).
    2. Apply the transformation matrix (W) to the lower-level capsules.  
       Optionally, two matrices are used to allow different cap sizes 
       (function: transform_view).
    3. Optionally: We add the coordinates to the first two elements of the 
       capsules (function: coordinate_addition).
    4. Apply DR routing.

    Args:
        iCaps: 		int, input number of types of capsules
        iPose: 		int, input size of capsules as total number of elements, 4x4=16
        oCaps: 		int, output number on types of capsules
        oPose: 		int, output size of capsules (as total number of elements)
        k: 			int, kernel size of convolution (usually, 1 or 3)
        strides: 	int, strides of convolution (usually, 1 or 2)
        padding:    str, the padding in the conv-caps ('valid' or 'same')
        iters: 		int, number of DR iterations (usually, 3)
        batch: 		int, the size of the batch during training 
        conv_cap:   boolean, to apply convolutional capsules.
        vect_cap:   boolean, to indicate the use of a vector capsule (not matrix)
        last_cap:   boolean, to indicate the last capsule layer (output of model)
        w_shared: 	boolean, to share transformation matrix across h*w; only 
                    used for non-convolutional capsules.
        w_double:   boolean, to use two weight matrices (it allows to change 
                    capsule-sizes for -only- matrix capsules)
        coor_add: 	boolean, to use scaled coordinate addition
        squashed:   boolean, to squash the capsules (it computes a pseudo-norm 
                    assuming the capsule is a vector; see DR-capsule paper).
        use_bias:   boolean, to add a bias term within the DR routing
    Shape:
        input:  	(*, h,  w,  iCaps, iPose)
        output: 	(*, h', w', oCaps, oPose)
        h', w' are computed the same way as a convolutional layer
        parameter size is: k*k*iCaps*oCaps*oPose ???
    """
    def __init__(self, iCaps=32, iPose=16, oCaps=32, oPose=16,   
                 k=3, strides=1, padding='SAME', iters=3, batch=16,
                 conv_cap=True, vect_cap=True,  last_cap=False, 
                 w_shared=True, w_double=False, coor_add=False,
                 squashed=True, use_bias=True, 
                 kernel_initializer='glorot_normal', 
                 regularizer_biases=None,
                 regularizer_weights=None, **kwargs):
        super(conv_caps_DR, self).__init__(**kwargs)
        
        # Main arguments
        self.iCaps = iCaps          
        self.iPose = iPose         
        self.oCaps = oCaps            
        self.oPose = oPose   
        self.k = k                    
        self.strides = strides
        self.padding = padding
        self.iters = iters      
        self.batch = batch        
        self.iP = int(np.sqrt(iPose))     # The cap size as the matrix-side
        self.oP = int(np.sqrt(oPose))    
        self.kernel_initializer = initializers.get(kernel_initializer)    
        self.regularizer_biases = regularizer_biases   
        self.regularizer_weights = regularizer_weights           

        # Check some incompatible flags:
        if conv_cap: w_shared = False
        if vect_cap: w_double = False

        # Flags
        self.conv_cap = conv_cap     # To select a convolutional capsule.
        self.vect_cap = vect_cap     # To set capsules as vectors.
        self.last_cap = last_cap     # To indicate it is the last capsule layer.
        self.w_shared = w_shared     # For non-conv caps, to share the weights.
        self.w_double = w_double     # For matrix caps, to use two W matrices.    
        self.coor_add = coor_add     # To add coordinate addition.
        self.squashed = squashed     # To apply 'squash' to the caps.
        self.use_bias = use_bias     # A bias term within the DR-Routing
        

    def build(self, input_shape):
        # Params
        # A bias term within the DR-routing.
        if self.use_bias:
            self.biases = self.add_weight(shape=[1, 1, self.oCaps, self.oPose, 1], 
                                          initializer=self.kernel_initializer, 
                                          regularizer=self.regularizer_biases, 
                                          trainable=True, 
                                          name=self.name + 'biases')    

        # Output of the W transformation based on the type of capsule:.     
        # Vector: 
        #     W = [iPose, oPose] --> v = XW = [1, iPose][iPose, oPose] = 
        #                                     [1, oPose]
        # Matrix, equal size (iP==oP): 
        #     W = [iP, oP]       --> v = XW = [iP, iP][iP, oP] = [iP, oP]
        # Matrix, unequal sizes (double W): 
        #     Wa = Wb = [iP, oP] --> v = (XWa)'Wb 
        #                        --> XWa = [iP, iP][iP, oP] = [iP, oP]  
        #                        --> Tranpose XWa: (XWa)' = [oP, iP] 		
        #                        --> v = (XWa)'Wb = [oP, iP][iP, oP] = [oP, oP]	
        if self.conv_cap:
            kk = self.k*self.k 
        else:
            kk = 1 if self.w_shared else input_shape[1]*input_shape[2]
        
        iPx = self.iPose if self.vect_cap else self.iP
        oPx = self.oPose if self.vect_cap else self.oP
        
        self.Wa = self.add_weight(shape=[1, kk*self.iCaps, self.oCaps, iPx, oPx], 
                                  initializer=self.kernel_initializer, 
                                  regularizer=self.regularizer_weights,
                                  trainable=True, name=self.name + 'Wa')   
        if self.w_double:
            self.Wb = self.add_weight(shape=[1, kk*self.iCaps, self.oCaps, iPx, oPx], 
                                  initializer=self.kernel_initializer,
                                  regularizer=self.regularizer_weights,                                 
                                  trainable=True, name=self.name + 'Wb') 
      

    def caps_dr_routing(self, u, bx):
        """ The routing-by-agreement.
        Input:
            u:     (bx, iC, oCaps, oPose)
        Output:
            v:     (bx,  1, oCaps, oPose)
        Note that some dimensions are merged, that is:
        - For conv_caps:
            bx == batch_size*oh*ow, iC == k*k*iCaps
        - For non-conv_caps (class-caps):
            bx == batch_size, iC == oh*ow*k*k*iCaps
        STEPS:
        - Initialize the prior 'b' for the coupling coefficients. Note: do not 
          confuse the batch size ('bx' here, but 'b' in other subfunctions)
          with the prior 'b' (named 'b' in the original paper); probably I 
          should change the name!
        - The input u is expanded to deal later with the update of b.  
        - Enter the iterations:
          * Apply softmax to b along the output capsules (oCaps) -> Now called c.
          * Compute the weighted sum of all the predicted output vectors. Thus,
            multiply c with the input. The function `multiply` will broadcast: 
                c.shape = [bx, iC, oCaps,     1, 1]
                u.shape = [bx, iC, oCaps, oPose, 1]
                v.shape = [bx, iC, oCaps, oPose, 1]
            Then sum along the axis of the input capsules (iC).  
                v.shape = [bx,  1, oCaps, oPose, 1]
            Then sum the biases (if indicated) and apply squash along the oPose.
          * Finally, update the prior b by simply multiplying the current  
            output v with the original inputs u and sum it to the current b. 
            See that, in matmul, the 1st term is transposed so it results in:  
                agreement.shape = [bx, iC, oCaps, 1, 1]            
        """
        _, iC, c, _ = u.shape
        assert c == self.oCaps   
        b = tf.zeros(shape=[bx, iC, self.oCaps, 1, 1])
        u = tf.expand_dims(u, 4)    
        assert self.iters > 0, 'The routing-iterations should be > 0.'
        for ii in range(self.iters):
            c = layers.Softmax(axis=2)(b)
            v = tf.multiply(c, u)
            v = tf.reduce_sum(v, axis=1, keepdims=True)
            if self.use_bias: v = v + self.biases
            if self.squashed: v = self.squash(v, axis=3)  
            # Update the prior b.
            if ii < self.iters - 1:  
                v_tiled = tf.tile(v, [1, iC, 1, 1, 1])
                agreement = tf.matmul(u, v_tiled, transpose_a=True)
                b = tf.add(b, agreement)
        # Squeeze the outputs to remove the last axis:
        v = tf.squeeze(v, 4)
        return v


    def kernel_tile(self, x):
        """It deals with the preparation to the 'convolution capsules'. 
        It uses depthwise convolutions to expand the input vector, adding 
        two axis related to the kernel of the convolutions (k).
        Depthwise convolution applies a different filter to each input channel 
        (expanding from 1 channel to 'channel_multiplier' channels for each), 
        then concatenates the results together. The filter is simply 1 in the 
        correct places, so the input values are not changed.
        Shape:
            Input:     (b, h, w, iCaps*(iPose+1))
            Output:    (b, h', w', k, k, iCaps*(iPose+1))
  
        The filter kernels -> (k, k, iCaps*(iPose+1), k*k)
            Refered as: [filter_height, filter_width, in_channels, 
                         channel_multiplier]
        Examples: 
          - Inputs (?, 14, 14, 512) depthwise-convolved with (1, 1, 512, 1),
            and strides=1, does not change anything, giving --> (?, 14, 14, 512)
          - Inputs (?, 14, 14, 512) depthwise-convolved with (3, 3, 512, 9),
            and strides=1 (and padding=valid), it only reduces the output 
            because of the padding and it replicates in the last axis 
            (channels_mutliplier) the elements (from the input vector) that  
            would be related to the "convolution capsules" of each spatial 
            point. Output would be --> (?, 12, 12, 4608)
          - Inputs (?, 14, 14, 512) depthwise-convolved with (3, 3, 512, 9),
            and stride=2 (and padding=valid), gives --> (?, 6, 6, 4608)
        In the last example, the output of the depth-convolution has 
            [in_channels * channel_multiplier] = (512x9) channels.
        The tensor is then reshaped and transformed 
            (?, 6, 6, 4608) into --> (?, 6, 6, 3, 3, 512)
        """
        _, h, w, c = x.shape
        if self.k == 1 and self.strides == 1:
            x = tf.reshape(x, shape=[-1, h, w, 1, 1, c])
            return x, h, w
        else:
            tile_filter = np.zeros(shape=[self.k, self.k, c, self.k*self.k], 
                                   dtype=np.float32)
            for i in range(self.k):
                for j in range(self.k):
                    tile_filter[i, j, :, i * self.k + j] = 1.0 
            tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)       
            x = tf.nn.depthwise_conv2d(x, tile_filter_op, 
                                       strides=[1, self.strides, self.strides, 1], 
                                       padding=self.padding)         
            ob, oh, ow, oc = x.shape
            x = tf.reshape(x, shape=[-1, oh, ow, c, self.k, self.k])
            x = tf.transpose(x, perm=[0, 1, 2, 4, 5, 3])
            return x, oh, ow


    def transform_view(self, x, bx):
        """ This is a matrix multiplication between the input capsules (x) and 
        the weights (W). This is a brief description for the matrix-cap case:
        - First, it adapts both elements for the matrix multiplication:
            x was as [bx, iC, iPose]  --> Reshaped to [bx, iC,     1, iP, iP] 
                                      -->  & tiled to [bx, iC, oCaps, iP, iP]
            W was [1, iC, oCaps, oP, oP] --> Tiled to [bx, iC, oCaps, oP, oP]
        - Then, it applies matmul, resulting in:  v = [bx, iC, oCaps, iP, oP]
            and since iP=oP, this is the same as: v = [bx, iC, oCaps, oP, oP]
		Note that some dimensions are merged, that is:
            Conv_caps:     bx == batch_size*h*w, and iC == k*k*iCaps
            Non-conv_caps: bx == batch_size,     and iC == h*w*iCaps
        For conv_caps:
            Input:     (b*h*w, k*k*iCaps, iPose)
            Output:    (b*h*w, k*k*iCaps, oCaps, oPose)
        For non-conv_caps (w_shared==True):
            Input:     (b, h*w*iCaps, iPose)
            Output:    (b, h*w*iCaps, oCaps, oPose)
        Options:           
        - If we use vector type (self.vect_cap == True):
		       x = [bx, iC, oCaps,     1, iPose]
               W = [bx, iC, oCaps, iPose, oPose]
          thus v = [bx, iC, oCaps,     1, oPose]
        - w_shared ('weights_shared') means sharing the weights in the case of 
          losing the spatial information (only for non-convolutional capsules).
        - If we use two W matrices (w_double), we need to transpose the outcome 
          of the first matmul.
        """
        _, iC, psize = x.shape 
        assert psize == self.iPose

        # Tile the weights
        Wa = self.Wa       
        if self.w_shared:       # Tile h*w times in axis=1 
            Wa = tf.tile(Wa, [1, int(iC / Wa.shape[1]), 1, 1, 1]) 
        Wa = tf.tile(Wa, [bx, 1, 1, 1, 1])    

        # If there are two weight matrices, also tile it.
        if self.w_double: 
            Wb = self.Wb       
            if self.w_shared:   # Tile h*w times in axis=1 
                Wb = tf.tile(Wb, [1, int(iC / Wb.shape[1]), 1, 1, 1]) 
            Wb = tf.tile(Wb, [bx, 1, 1, 1, 1]) 

        # Tile the input
        if self.vect_cap:       # Vector type
            x = tf.expand_dims(x, 2)  
            x = tf.expand_dims(x, 3)         
        else:                   # Matrix type
            x = tf.reshape(x, shape=(-1, iC, 1, self.iP, self.iP))                        
        x = tf.tile(x, [1, 1, self.oCaps, 1, 1]) 

        # Apply matrix multiplication and reshape
        v = tf.matmul(x, Wa)
        if self.w_double:
            v = tf.transpose(v, perm=[0, 1, 2, 4, 3])
            v = tf.matmul(v, Wb)
        v = tf.reshape(v, shape=(bx, iC, self.oCaps, self.oPose)) 
        return v



    def add_coord(self, v, b, h, w):
        """ This adds the coordinate positions (x,y) to the first and
        second elements of the pose matrix. I am not sure whether this is
        how the original authors intended. 
            Input:     (b, h*w*iCaps, oCaps, oPose)
            Output:    (b, h*w*iCaps, oCaps, oPose)
        """
        assert h == w
        v = tf.reshape(v, shape=(b, h, w, self.iCaps, self.oCaps, self.oPose))   
        coord_hh = tf.reshape((tf.range(h, dtype=tf.float32) + 0.50) /h, 
                              [1, h, 1, 1, 1])
        coord_h0 = tf.constant(0.0, shape=[1, h, 1, 1, 1], dtype=tf.float32)
        coord_h = tf.stack([coord_hh, coord_h0] + \
                           [coord_h0 for _ in range(self.oPose-2)], 
                           axis=-1)  # (1, h, 1, 1, 1, oPose)
        coord_ww = tf.reshape((tf.range(w, dtype=tf.float32) + 0.50) /w, 
                              [1, 1, w, 1, 1])   
        coord_w0 = tf.constant(0.0, shape=[1, 1, w, 1, 1], dtype=tf.float32)   
        coord_w = tf.stack([coord_w0, coord_ww] + \
                           [coord_w0 for _ in range(self.oPose-2)], 
                           axis=-1) # (1, 1, w, 1, 1, oPose)
        v = v + coord_h + coord_w 
        v = tf.reshape(v, shape=(b, h*w*self.iCaps, self.oCaps, self.oPose))
        return v

    
    def squash(self, x, axis=-1, epsilon=1e-12):
        """ The non-linear activation used in DR-Capsules. It drives the length 
        of a large vector to near 1 and small vector to 0. """
        squared_norm = tf.reduce_sum(tf.square(x), axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = x / safe_norm
        squashed_vector = squash_factor * unit_vector
        return squashed_vector
    
    
    def call_conv_cap(self, x, b):
        """ The actions for the call in a convolutional-cap layer.  
        Depending on the k-size and strides, the output h/w (oh, ow) might be  
        smaller than the input h/w.
        Input:
            x:         (b, h, w, iCaps, iPose), pose 
            b:          batch size
        Output:
            p_out:     (b, oh, ow, oCaps, oPose), the pose
        """ 
        _, h, w, iC, iP = x.shape
        assert iC == self.iCaps
        assert iP == self.iPose
        # Prepare for the convolutional capsule
        x = tf.reshape(x, shape=(b, h, w, self.iCaps*self.iPose))
        p_in, oh, ow = self.kernel_tile(x)
        p_in = tf.reshape(p_in, shape=(b*oh*ow, self.k*self.k*self.iCaps, 
                                       self.iPose))  # h & w to axis 0 with b.   
        v = self.transform_view(p_in, b*oh*ow)       # Matrix multiplication 
        p_out = self.caps_dr_routing(v, b*oh*ow)     # DR routing
        p_out = tf.reshape(p_out, shape=(b, oh, ow, self.oCaps, self.oPose))
        return p_out
    
    
    def call_nonconv_cap(self, x, b):
        """ The actions for the call in a non-convolutional-cap layer.  
        The spatial information (h/w) is lost in the process, i.e. oh=ow=1. 
        Note that the input might be from a previous conv-cap layer or a 
        previous non-conv-cap layer (in that case, the input h=w=1).
        Input:
            x:         (b, h, w, iCaps, iPose), pose 
            b:          batch size
        Output:
            p_out:     (b, 1, 1, oCaps, oPose), OR
                       (b, oCaps, oPose) for the last capsule layer.
        """ 
        _, h, w, iC, iP = x.shape
        assert iC == self.iCaps
        assert iP == self.iPose
        assert 1 == self.k
        assert 1 == self.strides
        # Reshape, but h*w is placed in the second element
        p_in = tf.reshape(x, shape=(b, h*w*self.iCaps, self.iPose))
        v = self.transform_view(p_in, b)     # Matrix multiplication 
        if self.coor_add: v = self.add_coord(v, b, h, w) # Coordinate addition 
        p_out = self.caps_dr_routing(v, b)   # DR routing
        # At this point: p_out.shape = [b, 1, oCaps, oPose]. Reshape (h=w=1):
        p_out = tf.reshape(p_out, shape=(b, 1, 1, self.oCaps, self.oPose))      
        return p_out
    

    def call(self, x, training=False):
        """ Things to take into account:
        * I need to know the batch size. I use the batch size for training, but 
          knowing that for testing/validation I only use one image. TO FIX!
        """
        b = self.batch if training else 1
        # Convolutional and non-convolutional capsules
        if self.conv_cap: 
            p_out = self.call_conv_cap(x, b)
        else:
            p_out = self.call_nonconv_cap(x, b)
        # If it is the last capsule-layer, just removes the h-w axis, so that: 
        #   pose.shape=[b, oCaps, oPose].
        if self.last_cap:
            return tf.squeeze(p_out, [1, 2])      
        else:
            return p_out

