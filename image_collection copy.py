# !/usr/bin/env python3
'''
image_collection.py : uses AirSim to collect vehicle first-person-view images

Copyright (C) 2017 Jack Baird, Alex Cantrell, Keith Denning, Rajwol Joshi, 
Simon D. Levy, Will McMurtry, Jacob Rosen

This file is part of AirSimTensorFlow

MIT License
'''
import airsim
from AirSimClient import ImageRequest, AirSimImageType, AirSimClientBase
from image_helper import IMAGEDIR
import pprint
import os
import time

def linkToAirsim(client):
    # 连接到airsim
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    # connect to the AirSim simulator 
    print('Connected')

if __name__ == "__main__":
    client = airsim.MultirotorClient()
    linkToAirsim(client)
    # maintain a queue of images of this size
    QUEUESIZE = 10

    # Create image directory if it doesn't already exist
    try:
        os.stat(IMAGEDIR)
    except:
        os.mkdir(IMAGEDIR)
    
    # take off
    client.takeoffAsync().join()
    # 第一波先按照直线飞行采集数据
    vx=2
    vy=0
    # vx,vy,vz,duration(持续时间)
    client.moveByVelocityAsync(vx, vy,0, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()
    imagequeue = []
    # 这里看一下官方文档是一些什么类比较好
    while True:
        client.moveByVelocityAsync(vx, vy,0, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()
        # get RGBA camera images from the car
        responses = client.simGetImages([ImageRequest(1, AirSimImageType.Scene)])  
        
        # add image to queue        
        imagequeue.append(responses[0].image_data_uint8)

        # dump queue when it gets full
        if len(imagequeue) == QUEUESIZE:
            for i in range(QUEUESIZE):
                AirSimClientBase.write_file(os.path.normpath(IMAGEDIR + '/image%03d.png'  % i ), imagequeue[i])
            imagequeue.pop(0)    

        collision_info = client.simGetCollisionInfo()

        if collision_info.has_collided:
            print("Collision at pos %s, normal %s, impact pt %s, penetration %f, name %s, obj id %d" % (
                pprint.pformat(collision_info.position), 
                pprint.pformat(collision_info.normal), 
                pprint.pformat(collision_info.impact_point), 
                collision_info.penetration_depth, collision_info.object_name, collision_info.object_id))
            break

        time.sleep(0.1)

    client.enableApiControl(False)
