# !/usr/bin/env python3
'''
使用保存的图片训练一个网咯识别碰撞

使用tensorflow 2.0版本改写1.0

'''

import tensorflow as tf
import numpy as np
import pickle
from image_helper import loadgray

# 最后一张图片是发生了碰撞,之前的没有
# SAFESIZE意味着安全的图片数量
SAFESIZE = 5

# 保存视频图像的文件夹
IMAGEDIR = './carpix'

# 通过pkl文件保存参数网络等
PARAMFILE = 'params.pkl'

# 学习的参数
learning_rate = 0.01
training_epochs = 500
batch_size = 100
display_step = 10

def loss(output, y):
    dot_product = y * tf.math.log(output)

    # 将每一个维度折叠为一个值，这是为啥啊，存疑

    xentropy = -tf.reduce_sum(input_tensor=dot_product, axis=1)
     
    loss = tf.reduce_mean(input_tensor=xentropy)

    return loss

def training(cost, global_step):

    tf.compat.v1.summary.scalar('cost', cost)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op

def main():

    # This will get number of pixels in each image (they must all be the same!)
    imgsize = 0

    # Read in images from car, convert to grayscale, scale down, and flatten for use as input
    images = []
    for k in range(SAFESIZE):

        image = loadgray(IMAGEDIR + '/image%03d.png' % k)
                
        imgsize = np.prod(image.shape)
        images.append(image)

    # All but last image is safe (01 = no-crash; 10 = crash)
    targets = []
    for k in range(SAFESIZE-1):
        targets.append([0,1])
    targets.append([1,0])
    
    # with tf.Graph().as_default():

    x = tf.compat.v1.placeholder('float', [None, imgsize]) # car FPV images
    y = tf.compat.v1.placeholder('float', [None, 2])       # 01 = no-crash; 10 = crash

    output = inference(x, imgsize, 2)

    cost = loss(output, y)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = training(cost, global_step)

    sess = tf.compat.v1.Session()

    init_op = tf.compat.v1.global_variables_initializer()

    sess.run(init_op)

    # Training cycle
    for epoch in range(training_epochs):
            
        # Fit training using batch data
        sess.run(train_op, feed_dict={x: images, y: targets})
            
        # Compute average loss
        avg_cost = sess.run(cost, feed_dict={x: images, y: targets})
            
        # Display logs per epoch step
        if epoch%display_step == 0:
            print('Epoch:', '%04d' % epoch, 'cost =', '{:.9f}'.format(avg_cost))

        print('Optimization Finished; saving weights to ' + PARAMFILE)
        params = [sess.run(param) for param in tf.compat.v1.trainable_variables()]
        
        pickle.dump( params, open(PARAMFILE, 'wb'))

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


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    main()
