import moviepy.editor as mp
import cv2
import os #To create folders and delete files
from videoprops import get_video_properties  #To get bitrate
import threading
import time
import shutil
from multiprocessing import *
from PIL import Image
import sys
import numpy as np


def fun1(inputVideo,outputVideo,outputAudio,loopCounter,finaOutputVideo,props,tempFolder):
    try:
        #################Reading a video to get the audio###############################
        video2 = mp.VideoFileClip(r""+inputVideo+"")

        #############Extracting the audio and saving it as a .mp3 file###############################
        video2.audio.write_audiofile(r""+outputAudio+"")

        #############Combining Video Clips###############################
        clipList=[]
        for i in range(loopCounter):
            clipList.append(mp.VideoFileClip(r""+tempFolder+"/"+outputVideo+""+str(i+1)+""+".mp4"))
        
        video3 = mp.concatenate_videoclips([clipList[i] for i in range(loopCounter)])

        ####################Combining the audio and the images to create a .mp4 video file###############################
        video3.write_videofile(r""+finaOutputVideo+"",audio=""+outputAudio+"",bitrate=(str(int(props['bit_rate'])-(0.05*int(props['bit_rate'])))))
    except:
        print("Something went wrong while writing the final video")

####################Interpolating a frame given a frame before and after it####################
def Interpolate(img1rgb,img2rgb):
    try:
        img1 = cv2.cvtColor(img1rgb, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2rgb, cv2.COLOR_BGR2GRAY)
        #######################Forward interpolation#######################
        #remap will take the pixels going from img2 to img1 and interpolate a frame after img1
        flow_F = cv2.calcOpticalFlowFarneback(img2,img1,None,0.5,3,15,3,5,1.2,0)

        height_F = flow_F.shape[0]
        width_F = flow_F.shape[1]
        
        y_coords_F, x_coords_F = np.mgrid[0:height_F, 0:width_F]
        coords_F = np.float32(np.dstack([x_coords_F, y_coords_F]))
        
        new_coords_F1 = coords_F + (flow_F*0.4)
        new_coords_F2 = coords_F + (flow_F*0.5)
        new_coords_F3 = coords_F + (flow_F*0.6)

        inter_frame_F1 = cv2.remap(img1rgb, new_coords_F1, None, cv2.INTER_CUBIC)#INTER_CUBIC  INTER_LINEAR
        inter_frame_F2 = cv2.remap(img1rgb, new_coords_F2, None, cv2.INTER_CUBIC)
        inter_frame_F3 = cv2.remap(img1rgb, new_coords_F3, None, cv2.INTER_CUBIC)
        
        #######################Backward interpolation#######################
        #remap will take the pixels going from img1 to img2 and interpolate a before img2
        flow_B = cv2.calcOpticalFlowFarneback(img1,img2,None,0.5,3,15,3,5,1.2,0)

        height_B = flow_B.shape[0]
        width_B = flow_B.shape[1]

        y_coords_B, x_coords_B = np.mgrid[0:height_B, 0:width_B]
        coords_B = np.float32(np.dstack([x_coords_B, y_coords_B]))
        
        new_coords_B1 = coords_B + (flow_B*0.4)
        new_coords_B2 = coords_B + (flow_B*0.5)
        new_coords_B3 = coords_B + (flow_B*0.6)

        inter_frame_B1 = cv2.remap(img2rgb, new_coords_B1, None, cv2.INTER_CUBIC)#INTER_CUBIC  INTER_LINEAR
        inter_frame_B2 = cv2.remap(img2rgb, new_coords_B2, None, cv2.INTER_CUBIC)
        inter_frame_B3 = cv2.remap(img2rgb, new_coords_B3, None, cv2.INTER_CUBIC)

        final_interpolated = (inter_frame_F1*(1/6)) + (inter_frame_F2*(1/6)) + (inter_frame_F3*(1/6)) + (inter_frame_B1*(1/6)) + (inter_frame_B2*(1/6)) + (inter_frame_B3*(1/6))
        
        return final_interpolated

    except:
        print("Something went wrong while Interpolating")


if __name__ == "__main__":
    try:
        inputVideo=str(sys.argv[1])
        outputVideo="Output Video"
        outputAudio="Audio.mp3"
        finalOutputVideo="Final30.mp4"
        tempFolder="Folder"
        frameIndicesFile="Skipped.txt"
        os.mkdir(tempFolder)
        #############Reading a video and extracting video info (I care about the bitrate and fps) and getting the while loop threshold#################
        start_time=time.time()
        props = get_video_properties(""+inputVideo+"")
        fps = round(int(props['avg_frame_rate'].split('/')[0])/int(props['avg_frame_rate'].split('/')[1]))
        video1 = cv2.VideoCapture(""+inputVideo+"")
        limit = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
        totalFrames = 0
        #################Reading the text files that contains the indices of the skipped frames#################
        f = open(frameIndicesFile, 'r+')
        skipped_frames_indices = f.read().split()
        f.close()

        for i in range(len(skipped_frames_indices)):
            skipped_frames_indices[i] = int(skipped_frames_indices[i])

        skipped_frames_indices = np.sort(skipped_frames_indices)

        total_frames = len(skipped_frames_indices) + limit
        print(total_frames)
        #####################Adjusting the frames splitting threshold for memory###############################
        width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        splitter = 0
        if(width*height<=2073600): #1920 x 1080
            splitter = 3000
        elif(width*height<=3686400): #2560 x 1440
            splitter = 2000
        elif(width*height<=8294400): #3840 x 2160
            splitter = 1000
        else:
            splitter = 500

        #################Interpolating the frames#################
        loop_counter = 0
        previous_frame = 0
        frame_list = []
        do_not_append_flag = False
        counter = 0
        index = 0
        for i in range(total_frames):

            if(counter>=splitter):
                loop_counter+=1
                #############################Producing an output video without sound and emptying the frame_list to free up memory################################
                out = cv2.VideoWriter(""+tempFolder+"/"+outputVideo+""+str(loop_counter)+""+".mp4",cv2.VideoWriter_fourcc(*'mp4v'), fps, (len(frame_list[0][0]), len(frame_list[0])))
                for i in range(len(frame_list)):
                    out.write(frame_list[i])
                out.release()
                print(len(frame_list))
                frame_list=[]
                counter = 0
            if(index<len(skipped_frames_indices)):
                if(i==skipped_frames_indices[index]):
                    index+=1
                    success,image = video1.read()
                    if(success):
                        frame1 = previous_frame
                        frame2 = image
                        interpolated_frame = Interpolate(frame1,frame2)
                        interpolated_frame = interpolated_frame.astype(np.uint8) #To write the frame correctly using video writer
                        frame_list.append(interpolated_frame)
                        frame_list.append(image)
                        counter+=2
                        do_not_append_flag = True
                        previous_frame = image
                else:
                    if(do_not_append_flag == False):
                        success,image = video1.read()
                        if(success):
                            frame_list.append(image)
                            counter+=1
                            previous_frame = image

                    if(do_not_append_flag == True):
                        do_not_append_flag = False
            else:
                success,image = video1.read()
                if(success):
                    frame_list.append(image)
                    counter+=1


        #################Writing any remaining frames#################
        if(len(frame_list)>0):
            loop_counter+=1
            print(len(frame_list))
            out = cv2.VideoWriter(""+tempFolder+"/"+outputVideo+""+str(loop_counter)+""+".mp4",cv2.VideoWriter_fourcc(*'mp4v'), fps, (len(frame_list[0][0]), len(frame_list[0])))
            for i in range(len(frame_list)):
                out.write(frame_list[i])
            out.release()

        #################Combining the videos to from one video with sound#################   
        p1 = Process(target=fun1, args=(inputVideo,outputVideo,outputAudio,loop_counter,finalOutputVideo,props,tempFolder)) 
        p1.start()
        p1.join()

        ################################Release all space and windows once done###############################
        video1.release() 
        cv2.destroyAllWindows() 

        ###############################Deleting the audio file and additional video file###############################
        os.remove(""+outputAudio+"")
        shutil.rmtree(""+tempFolder+"")
        print("Finished")
        print(time.time()-start_time)
    except:
        print("Something went wrong. Please do the following then run again:")
        print("Make sure to enter the name of the video correctly. Last time, you entered ",str(sys.argv[1]))
        print("Make sure there is no folder called ",tempFolder,". If there exists one, then please move it or delete it.")