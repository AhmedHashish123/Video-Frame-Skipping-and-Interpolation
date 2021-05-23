import moviepy.editor as mp
import cv2
import os 
from videoprops import get_video_properties 
import time
import shutil
from multiprocessing import *
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

####################Calculating optical flow####################
def Optical_flow(img1, img2, alpha, no_of_iterations):
    Kernel_for_calculating_avg = np.array([[1/12, 1/6, 1/12],
                                        [1/6,    0, 1/6],
                                        [1/12, 1/6, 1/12]], float)

    fx_mask = np.array([[-0.25, 0.25],
                        [-0.25, 0.25]])

    fy_mask = np.array([[-0.25, -0.25],
                        [0.25, 0.25]])

    ft_mask = np.ones((2, 2))*.25

    u = np.zeros([img1.shape[0], img1.shape[1]])
    v = np.zeros([img1.shape[0], img1.shape[1]])

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    fx = cv2.filter2D(img1,-1,fx_mask) + cv2.filter2D(img2,-1,fx_mask)
    fy = cv2.filter2D(img1,-1,fy_mask) + cv2.filter2D(img2,-1,fy_mask)
    ft = cv2.filter2D(img1,-1,-ft_mask) + cv2.filter2D(img2,-1,ft_mask)

    for _ in range(no_of_iterations):
        u_avg = cv2.filter2D(u,-1, Kernel_for_calculating_avg)
        v_avg = cv2.filter2D(v,-1, Kernel_for_calculating_avg)
        num_div_denom = (fx*u_avg + fy*v_avg + ft) / (fx**2 + fy**2 + alpha)
        u = u_avg - (fx * num_div_denom)
        v = v_avg - (fy * num_div_denom)

    return u, v

####################Interpolating a single frame####################
def Getting_frames(imgrgb,new_coords):

    interpolated_image=np.copy(imgrgb)

    temp_first_1 = np.round(new_coords[:,:,1])

    temp_first_1 = temp_first_1.astype(int)

    first_coord = temp_first_1>0
    first_coord = first_coord * temp_first_1

    temp_first_2 = np.copy(first_coord)
    first_coord = temp_first_2<=(imgrgb.shape[0]-1)
    first_coord = first_coord * temp_first_2

    temp_second_1 = np.round(new_coords[:,:,0])

    temp_second_1 = temp_second_1.astype(int)

    second_coord = temp_second_1>0
    second_coord = second_coord * temp_second_1

    temp_second_2 = np.copy(second_coord)
    second_coord = temp_second_2<=(imgrgb.shape[1]-1)
    second_coord = second_coord * temp_second_2


    interpolated_image[:,:,:] = imgrgb[first_coord,second_coord,:]

    return interpolated_image

def Interpolate(img1rgb,img2rgb):
    try:
        img1 = cv2.cvtColor(img1rgb, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2rgb, cv2.COLOR_BGR2GRAY)

        #######################Backward interpolation#######################
        alpha = 0.000001
        #alpha = 1
        no_of_iterations = 8
        u,v = Optical_flow(img2,img1,alpha,no_of_iterations)
        handmade_flow = np.dstack([u, v])

        height_B = u.shape[0]
        width_B = u.shape[1]
        
        y_coords_B, x_coords_B = np.mgrid[0:height_B, 0:width_B]
        coords_B = np.float32(np.dstack([x_coords_B, y_coords_B]))

        new_coords_B1 = coords_B + (handmade_flow * 0.5)
        
        first_interpolated = Getting_frames(img1rgb,new_coords_B1)

        
        final_interpolated = first_interpolated
        
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
        #################Reading the text files that contains the indices of the skipped frames and sorting it#################
        f = open(frameIndicesFile, 'r+')
        skipped_frames_indices = f.read().split()
        f.close()

        for i in range(len(skipped_frames_indices)):
            skipped_frames_indices[i] = int(skipped_frames_indices[i])

        skipped_frames_indices = np.sort(skipped_frames_indices)

        total_frames = len(skipped_frames_indices) + limit
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
                        interpolated_frame = cv2.GaussianBlur(interpolated_frame,(3,3),0)
                        interpolated_frame = interpolated_frame.astype(np.uint8) #To write the frame correctly using video writer
                        frame1 = frame1.astype(np.uint8)
                        frame2 = frame1.astype(np.uint8)
                        final_int = (interpolated_frame*(1/3))+(frame1*(1/3)) + (frame2*(1/3)) 
                        final_int = final_int.astype(np.uint8)
                        frame_list.append(final_int)
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
