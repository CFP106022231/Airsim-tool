import airsim #pip install airsim
import numpy as np
import os
import cv2
import math
import flowpy
import time
import matplotlib.pyplot as plt


client = airsim.MultirotorClient()

save_dir = '../result/230422/'

IMG = []
DEPTH = []
OF = []
POS = []
ORI = []
newpos = np.zeros(3)
newori = np.zeros(4)

#hsv = np.zeros((240, 320, 3)).astype(np.float32)
#hsv[...,1] = 255
#shift = 0.999
#alpha = 1/math.log(0.1 + shift)
#frame = 0
#fps = []

while True:
    try:
#        start = time.time()
        responses = client.simGetImages([
                airsim.ImageRequest("0", 0, False, False), #BGR影像
                
                airsim.ImageRequest("0", 6, True), #光流角度
                airsim.ImageRequest("0", 8, True)]) #光流大小
                # airsim.ImageRequest("0", 2, True)]) # 深度
#        end = time.time()
#        fps.append((1/(end-start)))

        response = responses[0]
        # depth = responses[3]
        theta = responses[1]
        r = responses[2]
        width = response.width
        height = response.height
        # print('0', (ofny.time_stamp - ofnx.time_stamp)/(10**9))
        posi = response.camera_position 
        orie = response.camera_orientation
        

        newpos[0] = posi.x_val
        newpos[1] = posi.y_val
        newpos[2] = posi.z_val
        # print(newpos)

        newori[0] = orie.w_val
        newori[1] = orie.x_val
        newori[2] = orie.y_val
        newori[3] = orie.z_val

        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        # depth1d = np.array(depth.image_data_float).astype(np.float32)
        theta1d = np.array(theta.image_data_float).astype(np.float32)
        r1d = np.array(r.image_data_float).astype(np.float32)
        

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(height, width, 3)
        # depth_show = depth1d.reshape(height, width, 1)

        oftheta = theta1d.reshape(height, width)
        ofr = r1d.reshape(height, width)

        ofx = ofr * np.cos(oftheta)
        ofy = ofr * np.sin(oftheta)
        OF_final = np.dstack((ofr, oftheta)).astype(np.float32)
        
        
#        OF_final = np.dstack((ofx, ofy)).astype(np.float32)
        
#        OF_Final = cv2.resize(OF_final, (hsv.shape[1], hsv.shape[0]))
#        mag, ang = cv2.cartToPolar(OF_final[...,0], OF_final[...,1])
#        hsv[...,0] = ang*180/np.pi/2
#        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
#        hsv[...,0] = oftheta*180/np.pi/2
#        hsv[...,2] = 6 * ofr

#        rad2ang = 180.0/np.pi
##        angle = 180.0 + np.arctan2(OF_final[...,1], OF_final[...,0])*rad2ang
##        angle[angle<0] = 360 + angle[angle<0]
#        angle = np.arctan2(OF_final[...,1], OF_final[...,0])*rad2ang
#        angle = 360 + angle
#        angle = np.fmod(angle, 360.0)
#        norm = np.sqrt(OF_final[...,0]**2 + OF_final[...,1]**2)
#        intensity = np.clip(alpha*np.log(norm + shift), 0.0, 1.0)
#        hsv[...,0] = angle
#        hsv[...,1] = 1.0
#        hsv[...,2] = intensity

#        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#        cv2.imwrite('../result/230216/airsim{}.png'.format(frame), bgr)

        frame += 1
        print('frame',frame,'fps=',1/(end-start))
#        cv2.imshow('Flow', bgr)
#        cv2.imshow('flow', flowpy.flow_to_rgb(OF_final))
#        cv2.waitKey(1)


        
        # print(np.max(ofpy_rgb))

        # print(np.mean(of_rgb[:, :, 0]))
        IMG.append(img_rgb)
        # DEPTH.append(depth_show)
        OF.append(OF_final)
        POS.append(newpos.copy())
        ORI.append(newori.copy())

    except KeyboardInterrupt:
        print('All done')
        break 

IMG = np.array(IMG)
# DEPTH = np.array(DEPTH)
OF = np.array(OF)
POS = np.array(POS)
ORI = np.array(ORI)

np.savez(save_dir + '0522test14.npz', IMG=IMG, DEPTH=DEPTH, OF=OF, POS=POS, ORI=ORI)
#     cv2.imshow('depth', depth_show)
#     cv2.imshow('IMG', img_rgb)
#     cv2.imshow('OF', of_rgb)
#     cv2.waitKey(1)
plt.plot(fps)
