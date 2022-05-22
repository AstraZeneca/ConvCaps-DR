""" This code is mainly a conversion (from PyTorch to Tensorflow 2.x) of the 
code made by Lei Yang (yl-1993) and Moein Hasani ():
https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch

Some addons have been implemented to Primary-Caps and Conv-Caps
""" 

import tensorflow as tf  
import tensorflow.keras.layers as layers 
from tensorflow.keras import initializers, activations, regularizers
import numpy as np



class primary_caps_EM(layers.Layer):
    """This constructs a primary capsule layer using regular convolution layers.
    The convolutions generate feature maps that are considered the poses. 
    A general capsule has a pose (vector or matrix) and an activation.
    - Poses:
      * In Primary Capsules, the pose is just a vector. Only in the next capsule 
        layers, the primary-cap-vector will be transformed into a matrix if that  
        is the selected type of capsule.
      * (Optionally) We do not build the poses within the layer but, instead, 
        we assume that the input are already the poses. This is useful to allow 
        for more complex convolutional blocks to generate the first poses 
        instead of the default (and simple) conv2D.
      * (Optionally) The poses can be squashed by using the squashed function 
        defined in the DR-Capsule paper. Note that this considers the poses as 
        vectors. Preliminary tests indicate that squash is not a good choice!       
    - Activations ('default', 'norm', 'linear'): 
      * In the default option (as in the EM-paper), a convolutional layer 
        generates them. Note that a 'sigmoid' activation is added in the 
        conv-layer since we want the activations to be in the range 0-1.
      * (Optionally) The norm of the capsule-vector is used as activation.
      * (Optionally) A learnt linear combination of the elements of the capsule 
        is used to compute the activation. This is achieved by using a Conv3D 
        with k=1 and single output feature, given as input the poses in the 
        shape of [None, h, w, oCaps, oPose]. 
      * A subtle point: If we use 'default' activations but without doing the 
        poses, the activations are then computed over the current poses instead 
        of over the previous tensors.
    Args:
        oCaps: 	 	number of output capsules
        oPose: 		size of pose matrix, as the number of elements (p*p)
        k: 		 	kernel size of the convolution
        strides: 	strides of the convolution
        padding:    padding of the convolution
        do_poses:   boolean, flag to indicate that the poses will be done within 
                    the layer with a conv2D (using the previous parameters). 
        squashed:   boolean, flag to indicate the squashing of the poses.
        act_type:   string, the type of activation (see options above).
    Shape:
        input:   	(*, h,  w,  iFeat)
        output: 	pose: (*, h', w', oCaps, oPose)
                    actv: (*, h', w', oCaps, 1)
        iFeat is the input number of features 
        h', w' are computed the same way as convolution layer
    """
    def __init__(self, oCaps=32, oPose=16, k=1, strides=1, padding='SAME', 
                 do_poses=True, squashed=False, act_type='default'):
        super(primary_caps_EM, self).__init__()
        self.oCaps = oCaps
        self.oPose = oPose   
        self.do_poses = do_poses
        self.squashed = squashed
        self.act_type = act_type   
        
        # Poses: whether we make the poses or the input is already the poses.
        if do_poses:
            self.pose = layers.Conv2D(filters=oCaps*oPose, kernel_size=k, 
                                      strides=strides, padding=padding,
                                      use_bias=True)    
        # Types of activations:
        if act_type == 'linear':
            print('Primary-Caps, activation: Linear')
            self.actv = layers.Conv3D(filters=1, kernel_size=1, 
                                      strides=1, padding=padding, 
                                      use_bias=True, activation='sigmoid')   
        elif act_type == 'linear_norm':
            self.actv = layers.Conv3D(filters=int(oPose/2), kernel_size=1, 
                                      strides=1, padding='same', 
                                      use_bias=True)  
        elif act_type == 'norm': 
            print('Primary-Caps, activation: Norm') # Nothing to do  
        
        else: # act_type == 'default':
            print('Primary-Caps, activation: Default')
            self.actv = layers.Conv2D(filters=oCaps, kernel_size=k, 
                                      strides=strides, padding=padding, 
                                      use_bias=True, activation='sigmoid')   
        
    
    def build(self, input_shape):  
        # If the input are poses, check that they are divisible into capsules.
        if not self.do_poses:
            assert input_shape[3] == self.oCaps * self.oPose
    

    def norm_vector(self, x, axis=-1, epsilon=1e-12):
        """ It computes the length of the input vectors along the given axis."""
        squared_norm = tf.reduce_sum(tf.square(x), axis, keepdims=False)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        return safe_norm
        
        
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
        # The pose 
        pose = self.pose(x) if self.do_poses else x
        _, h, w, _ = pose.shape  
        pose = tf.reshape(pose, [-1, h, w, self.oCaps, self.oPose])        
        
        # The activation 
        if self.act_type == 'linear':
            actv = self.actv(pose)
        elif self.act_type == 'norm':
            actv = self.norm_vector(pose)
            actv = activations.sigmoid(actv)
        elif self.act_type == 'linear_norm':
            actv = self.actv(pose)
            actv = self.norm_vector(actv)
            actv = activations.sigmoid(actv)
        else: # 'default'
            actv = self.actv(x) 
        actv = tf.reshape(actv, [-1, h, w, self.oCaps, 1])
        if self.squashed: pose = self.squash(pose)
        return pose, actv



class conv_caps_EM(layers.Layer):
    """ This constructs a convolution capsule layer, whose input is a primary 
    capsule or a convolution capsule, transfering capsule layer L to capsule 
    layer L+1 by EM routing.
    
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
        multiplication is done (EM is actually computed by using vectors).
      * If the matrix type is used, there is the option of using two weight 
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
    4. Apply EM routing.

    Args:
        iCaps: 		input number of types of capsules
        iPose: 		input size of capsules as total number of elements, 4x4=16
        oCaps: 		output number on types of capsules
        oPose: 		output size of capsules (as total number of elements)
        k: 			kernel size of convolution (usually, 1 or 3)
        strides: 	strides of convolution (usually, 1 or 2)
        padding:    the padding in the conv-caps ('valid' or 'same')
        iters: 		number of EM iterations (usually, 3)
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

    Shape:
        input:  	poses: (*, h,  w,  iCaps, iPose)
                    actvs: (*, h,  w,  iCaps, 1)
        output: 	poses: (*, h,  w,  oCaps, oPose) 
                    actvs: (*, h,  w,  oCaps, 1)
        h', w' are computed the same way as a convolutional layer
        parameter size is: k*k*iCaps*oCaps*oPose + iCaps*iPose ???
    """
    def __init__(self, iCaps=32, iPose=16, oCaps=32, oPose=16,   
                 k=3, strides=1, padding='SAME', iters=3, batch=16,
                 conv_cap=True, vect_cap=False, last_cap=False, 
                 w_shared=True, w_double=False, coor_add=False,
                 squashed=False, 
                 kernel_initializer='glorot_normal', 
                 regularizer_biases=None,
                 regularizer_weights=None,
                 **kwargs):
        super(conv_caps_EM, self).__init__(**kwargs)
        
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

        # Constant
        self.eps = 1e-12
        self._lambda = 1e-03         # Lambda scheduler is in the 'call function'
        self.ln_2pi = tf.math.log(2*tf.constant(m.pi))
        

    def build(self, input_shape):
        # Params
        # \beta_u and \beta_a are per capsule type.
        # The total number of trainable parameters between two convolutional 
        # capsule layer types is 4*4*k*k and for the whole layer 
        # is 4*4*k*k*iCaps*oCaps.
        # https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        
        self.beta_u = self.add_weight(shape=[self.oCaps], 
                                      initializer=self.kernel_initializer, 
                                      regularizer=self.regularizer_biases, 
                                      trainable=True, 
                                      name=self.name + 'beta_u')    
        self.beta_a = self.add_weight(shape=[self.oCaps], 
                                      initializer=self.kernel_initializer, 
                                      regularizer=self.regularizer_biases, 
                                      trainable=True,
                                      name=self.name + 'beta_a')   

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
            
    
    def get_config(self):
        config = {
            'num_capsule': self.oCaps,
            'dim_capsule': self.oPose,
            'conv_k': self.k,
            'conv_strides': self.strides,
            'iter_routing': self.iters
        }
        base_config = super(conv_caps_EM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
      
      

    def caps_em_routing(self, v, a_in, bx):
        """ The routing, which involves two steps (E-M).
        Input:
            v:   	   (bx, iC, oCaps, oPose)
            a_in:      (bx, iC, 1)
        Output:
            mu:        (bx, 1, oCaps, oPose)
            a_out:     (bx, oCaps, 1)

        Note that some dimensions are merged, that is:
        - For conv_caps:
            bx == batch_size*oh*ow, iC == k*k*iCaps
        - For non-conv_caps (class-caps):
            bx == batch_size, iC == oh*ow*k*k*iCaps
        """
        _, iC, c, _ = v.shape
        assert c == self.oCaps
        #assert (bx, iC, 1) == a_in.shape     # bx cannot be asserted

        r = (1/self.oCaps) * tf.ones(shape=(bx, iC, self.oCaps), 
                                     dtype=tf.float32)
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, iC)
            if iter_ < self.iters - 1:
                r = self.e_step(a_out, mu, sigma_sq, v)
        return mu, a_out



    def m_step(self, a_in, r, v, iC):
        """ The M-step. The equations are (latex syntasis):
        
        \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}, 
        (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}, 
        cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}, 
        a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

        Input:
            a_in:      (bx, iC, 1)
            r:         (bx, iC, oCaps, 1)
            v:         (bx, iC, oCaps, oPose)
        Local:
            cost_h:    (bx, oCaps, oPose)
            r_sum:     (bx, oCaps, 1)
        Output:
            a_out:     (bx, oCaps, 1)
            mu:        (bx, 1, oCaps, oPose)
            sigma_sq:  (bx, 1, oCaps, oPose)
        """
        r = r * a_in
        # Is this some kind of normalization?
        r = r / (tf.reduce_sum(r, axis=2, keepdims=True) + self.eps) 
        r_sum = tf.reduce_sum(r, axis=1, keepdims=True)
        coeff = r / (r_sum + self.eps)
        coeff = tf.reshape(coeff, shape=[-1, iC, self.oCaps, 1])
        mu = tf.reduce_sum(coeff * v, axis=1, keepdims=True)
        sigma_sq = tf.reduce_sum(coeff * (v - mu)**2, axis=1, 
                                 keepdims=True) + self.eps
        r_sum = tf.reshape(r_sum, shape=[-1, self.oCaps, 1])
        sigma_sq  = tf.reshape(sigma_sq , shape=[-1, self.oCaps, self.oPose])
        cost_h = (self.beta_u[..., tf.newaxis] \
                  + tf.math.log(tf.math.sqrt(sigma_sq)) + self.eps) * r_sum
        a_out = tf.math.sigmoid(
                    self._lambda*(self.beta_a - tf.reduce_sum(cost_h, axis=2)))
        sigma_sq = tf.reshape(sigma_sq, shape=[-1, 1, self.oCaps, self.oPose])
        return a_out, mu, sigma_sq



    def e_step(self, a_out, mu, sigma_sq, v):
        """ The E-step. The equations are (latex syntasis):

        ln(p_j) = - \sum_h \dfrac{(V^h_{ij} - \mu^h_j)^2}{2 (\sigma^h_j)} - 
                    \sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
        r = softmax(ln(a_j*p_j)) = softmax(ln(a_j) + ln(p_j))

        Input:
            mu:        (bx, 1, oCaps, oPose)
            sigma:     (bx, 1, oCaps, oPose)
            a_out:     (bx, oCaps, 1)
            v:         (bx, iC, oCaps, oPose)
        Local:
            ln_p_j_h:  (bx, iC, oCaps, oPose)
            ln_ap:     (bx, iC, oCaps, 1)
        Output:
            r:         (bx, iC, oCaps, 1)
        
        NOTE on EPS: 
        There seems to be a problem after a few dozen epochs where the loss  
        goes to 0 and accuracy drops from 0.95 to 0.10. This happens MAINLY in 
        categorical-crossentropy (and not always; it tends to be an exception). 
        I rarely found it with spread loss. I believe it might be the logs when 
        these tend to zero (very low). From m_step, 'sigma_sq' was already 
        given an epsilon, so that should prevent ln_p_j_h from giving errors.
        Here, I have added eps to ln_ap
        """
        ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq) \
                   - tf.math.log(tf.math.sqrt(sigma_sq) + self.eps) \
                   - 0.5*self.ln_2pi 
        ln_ap = tf.reduce_sum(ln_p_j_h, axis=3) \
                + tf.math.log(tf.reshape(a_out, shape=[-1, 1, self.oCaps]) \
                + self.eps)   
        r = activations.softmax(ln_ap)
        return r



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
		
        Shape: psize
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


    
    def squash(self, x, axis=-1):
        """ The non-linear activation used in DR-Capsules. It drives the length 
        of a large vector to near 1 and small vector to 0. """
        squared_norm = tf.reduce_sum(tf.square(x), axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + self.eps)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = x / safe_norm
        squashed_vector = squash_factor * unit_vector
        return squashed_vector
    
    
    
    def call_conv_cap(self, pose, actv, b):
        """ The actions for the call in a convolutional-cap layer.  
        Depending on the k-size and strides, the output h/w (oh, ow) might be  
        smaller than the input h/w.
        Input:
            pose:      (b, h, w, iCaps, iPose), the input poses
            actv:      (b, h, w, iCaps, 1), the input activations
            b:          batch size
        Output:
            p_out:     (b, oh, ow, oCaps, oPose), the pose
            a_out:     (b, oh, ow, oCaps, 1), the activation
        """ 
        _, h, w, iC, iP = pose.shape
        assert iC == self.iCaps
        assert iP == self.iPose
        
        # Prepare for the convolutional capsule
        pose = tf.reshape(pose, shape=(b, h, w, self.iCaps*self.iPose))
        actv = tf.reshape(actv, shape=(b, h, w, self.iCaps))
        x = tf.concat([pose, actv], axis=3)
        x, oh, ow = self.kernel_tile(x)

        # Separate pose and activations, and reshape (h & w to axis 0 with b).
        p_in = x[:, :, :, :, :, :self.iCaps*self.iPose]            
        a_in = x[:, :, :, :, :, self.iCaps*self.iPose:]            
        p_in = tf.reshape(p_in, shape=(b*oh*ow, self.k*self.k*self.iCaps, 
                                       self.iPose)) 
        a_in = tf.reshape(a_in, shape=(b*oh*ow, self.k*self.k*self.iCaps, 
                                       1))          

        v = self.transform_view(p_in, b*oh*ow) # Matrix multiplication 
        p_out, a_out = self.caps_em_routing(v, a_in, b*oh*ow) # EM routing
        if self.squashed: p_out = self.squash(p_out)   # Squash (if indicated)

        # Reshape  
        p_out = tf.reshape(p_out, shape=(b, oh, ow, self.oCaps, self.oPose))
        a_out = tf.reshape(a_out, shape=(b, oh, ow, self.oCaps, 1))
        return p_out, a_out
    
    
    
    def call_nonconv_cap(self, pose, actv, b):
        """ The actions for the call in a non-convolutional-cap layer.  
        The spatial information (h/w) is lost in the process, i.e. oh=ow=1. 
        Note that the input might be from a previous conv-cap layer or a 
        previous non-conv-cap layer (in that case, the input h=w=1).
        Input:
            pose:      (b, h, w, iCaps, iPose), the input poses
            actv:      (b, h, w, iCaps, 1), the input activations
            b:          batch size
        Output:
            p_out:     (b, 1, 1, oCaps, oPose), the pose
            a_out:     (b, 1, 1, oCaps, 1), the activation
        """ 
        _, h, w, iC, iP = pose.shape
        assert iC == self.iCaps
        assert iP == self.iPose
        assert 1 == self.k
        assert 1 == self.strides

        # Reshape, but h*w is placed in the second element
        p_in = tf.reshape(pose, shape=(b, h*w*self.iCaps, self.iPose))
        a_in = tf.reshape(actv, shape=(b, h*w*self.iCaps, 1))
        v = self.transform_view(p_in, b)              # Matrix multiplication 
        if self.coor_add: v = self.add_coord(v, b, h, w) # Coordinate addition 
        p_out, a_out = self.caps_em_routing(v, a_in, b)  # EM routing
        if self.squashed: p_out = self.squash(p_out)     # Squash (if indicated)

        # Reshape them (h=w=1)
        p_out = tf.reshape(p_out, shape=(b, 1, 1, self.oCaps, self.oPose))
        a_out = tf.reshape(a_out, shape=(b, 1, 1, self.oCaps, 1))
        
        return p_out, a_out
        
    

    def call(self, pose, actv, training=False):
        """ Things to take into account:
        - I need to know the batch size. I can use the 'training' argument, 
          knowing that for testing/validation I only use one image. 
        - The returned arguments will be the two parts of the capsule (pose and 
          activation) concatenated (except if it is the last capsule layer, 
          in which case the activations are only necessary). 
          HOWEVER: 
        - In parallel capsules (or in DenseCaps), I need to concatenate the 
          poses (of all branches) together and later to concatenate the 
          activations at the end. Thus, the concatenation needs to be done 
          OUTSIDE, in the main network function.
        - Regarding the lambda scheduler, my experiments did not show an 
          improvement with that scheduler (so it is commented out). This needs 
          further testing.
        """
        b = self.batch if training else 1
        #self._lambda = self._lambda + 1e-04   # Rough lambda-scheduler
        
        # Convolutional and non-convolutional capsules
        if self.conv_cap: 
            p_out, a_out = self.call_conv_cap(pose, actv, b)
        else:
            p_out, a_out = self.call_nonconv_cap(pose, actv, b)
        
        # If it is the last capsule-layer, just return the activations. 
        # This entails to squeeze the h-w axis, so actv.shape=[b, oCaps, 1].
        if self.last_cap:
            return tf.squeeze(a_out, [1, 2])      
        else:
            return p_out, a_out

