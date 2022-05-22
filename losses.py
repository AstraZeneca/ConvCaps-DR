
import tensorflow as tf

def margin_loss(y_true, y_pred):
    """ Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, 
    this loss should work too. Not tested it.
    # Arguments
        y_true: [None, n_classes]
        y_pred: [None, num_capsule]
    # Returns
        a scalar loss value.
    """
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) \
        + 0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, 1))


margin4loss = tf.Variable(0.2) # Initial value
def spread_loss(y_true, y_pred):
    """ Spread loss. Originally, margin is updated during training. We use a 
    global variable that is updated every epoch (see training-loop).
    # Arguments
        y_true: [None, n_classes]
        y_pred: [None, num_capsule]
    # Returns
        a scalar loss value.
    """
    pos = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
    pos = tf.tile(pos, (1, 15)) # len(args.classes)
    neg = (1.0 - y_true) * y_pred
    pn_dif = margin4loss - (pos - neg)
    spread = (1.0 - y_true) * pn_dif
    return tf.reduce_sum(tf.square(tf.maximum(0.0, spread)), axis=-1)


"""
NOTE: If using spread loss, we update the margin2loss after evey epoch. 
Paste that code at the beginning of the for-loop of the training.

    # Change the variable in the spread_loss related to the epoch
    margin_new = tf.minimum(0.9, (0.2 + iEpoc*0.005))
    K.set_value(margin4loss, margin_new)
    print('New margin in loss: %.2f' % margin4loss)
"""