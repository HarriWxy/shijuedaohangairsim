# !/usr/bin/env python3
'''
collision_testing.py : tests pickled network on ability to predict a collision

'''
import airsim
from AirSimClient import CarClient, CarControls, ImageRequest, AirSimImageType, AirSimClientBase
import os
import time
import tensorflow as tf
import pickle
import sys
import numpy as np 

from image_helper import loadgray, IMAGEDIR
from tf_softmax_layer import inference

TMPFILE = IMAGEDIR + '/active.png'
PARAMFILE = 'params.pkl'
IMGSIZE = 1032
INITIAL_THROTTLE= 0.65
BRAKING_DURATION = 15

tf.compat.v1.disable_eager_execution() #tf 2.0在1.0中不兼容的特性

# 连接到airsim
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset() 
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
# Load saved training params as ordinary NumPy
# 数据序列
W,b = pickle.load(open('params.pkl', 'rb'))



# Placeholder for an image
x = tf.compat.v1.placeholder('float', [None, IMGSIZE])

# Our inference engine, intialized with weights we just loaded
output = inference(x, IMGSIZE, 2, W, b)

# TensorFlow initialization boilerplate
sess = tf.compat.v1.Session()
init_op = tf.compat.v1.global_variables_initializer()
sess.run(init_op)

# Once the brakes come on, we need to keep them on for a while before exiting; otherwise,
# the vehicle will resume moving.
brakingCount = 0

# Loop until we detect a collision
vx=1
vy=0
while True:
    client.moveByVelocityAsync(vx, vy,0, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()
    # Get RGBA camera images from the car
    responses = client.simGetImages([ImageRequest(1, AirSimImageType.Scene)])

    # Save it to a temporary file
    image = responses[0].image_data_uint8
    AirSimClientBase.write_file(os.path.normpath(TMPFILE), image)

    # Read-load the image as a grayscale array
    image = loadgray(TMPFILE)

    # Run the image through our inference engine.
    # Engine returns a softmax output inside a list, so we grab the first
    # element of the list (the actual softmax vector), whose second element
    # is the absence of an obstacle.
    safety = sess.run(output, feed_dict={x:[image]})[0][1]

    # Slam on the brakes if it ain't safe!
    if safety < 0.5:
        # client.moveByAngleRatesThrottleAsync(np.pi/6,np.pi/6,np.pi/6,1,1).join()
        temp=vx
        vx=vy
        vy=temp
       
    # Wait a bit on each iteration
    time.sleep(0.1)

