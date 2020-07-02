import tensorflow as tf
def inference(x, xsize, ysize, W_vals=0, b_vals=0):
    '''
    全局控制网络传递函数
    This is a general-purpose softmax inference layer implementation.
    '''
    W_init = tf.compat.v1.constant_initializer(value=W_vals)
    b_init = tf.compat.v1.constant_initializer(value=b_vals)
    W = tf.compat.v1.get_variable('W', [xsize, ysize], initializer=W_init)
    b = tf.compat.v1.get_variable('b', [ysize],        initializer=b_init)
    output = tf.nn.softmax(tf.matmul(x, W) + b)

    return output