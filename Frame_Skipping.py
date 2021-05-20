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

####################Detedcting and removing redundant frames####################
def fun2(frame_list,skipped_frames_list,resizer,similarityThreshold,numberOfFifth,previousFrames,loopCounter,splitter,fileName):
    try:
        i=0
        removed_frames_list=[]
        while i < int(len(frame_list)):
            if(i>0 and i<len(frame_list)-1):
                gray1 = cv2.cvtColor(frame_list[i-1], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame_list[i], cv2.COLOR_BGR2GRAY)
                gray3 = cv2.cvtColor(frame_list[i+1], cv2.COLOR_BGR2GRAY)
                img1 = Image.fromarray(gray1)
                img1 = img1.resize((resizer, resizer), Image.LANCZOS)#resizing to make computations easier
                pixelValues1 = list(img1.getdata())
                avgPixelValue1 = sum(pixelValues1)/len(pixelValues1)
                binaryRepresentation1 = "".join(['1' if (pixel1 >= avgPixelValue1) else '0' for pixel1 in pixelValues1])
                hexadecimalRepresentation1 = str(hex(int(binaryRepresentation1, 2)))[2:]
                img2 = Image.fromarray(gray2)
                img2 = img2.resize((resizer, resizer), Image.LANCZOS)
                pixelValues2 = list(img2.getdata())
                avgPixelValue2 = sum(pixelValues2)/len(pixelValues2)
                binaryRepresentation2 = "".join(['1' if (pixel2 >= avgPixelValue2) else '0' for pixel2 in pixelValues2])
                hexadecimalRepresentation2 = str(hex(int(binaryRepresentation2, 2)))[2:]
                img3 = Image.fromarray(gray3)
                img3 = img3.resize((resizer, resizer), Image.LANCZOS)
                pixelValues3 = list(img3.getdata())
                avgPixelValue3 = sum(pixelValues3)/len(pixelValues3)
                binaryRepresentation3 = "".join(['1' if (pixel3 >= avgPixelValue3) else '0' for pixel3 in pixelValues3])
                hexadecimalRepresentation3 = str(hex(int(binaryRepresentation3, 2)))[2:]

                tempIndex1 = 0
                counter1 = 0
                tempLimit1 = 0
                if(len(hexadecimalRepresentation1)<=len(hexadecimalRepresentation2)):
                    tempLimit1 = len(hexadecimalRepresentation1)
                else:
                    tempLimit1 = len(hexadecimalRepresentation2)
                while(tempIndex1 < tempLimit1):
                    if(hexadecimalRepresentation1[tempIndex1] != hexadecimalRepresentation2[tempIndex1]):
                        counter1 += 1
                    tempIndex1 += 1

                tempIndex2 = 0
                counter2 = 0
                tempLimit2 = 0
                if(len(hexadecimalRepresentation2)<=len(hexadecimalRepresentation3)):
                    tempLimit2 = len(hexadecimalRepresentation2)
                else:
                    tempLimit2 = len(hexadecimalRepresentation3)
                while(tempIndex2 < tempLimit2):
                    if(hexadecimalRepresentation2[tempIndex2] != hexadecimalRepresentation3[tempIndex2]):
                        counter2 += 1
                    tempIndex2 += 1

                if(abs(counter1)<similarityThreshold and abs(counter2)<similarityThreshold):
                    removed_frames_list.append(i)
                    i+=2
                    time.sleep(0.0)
                else:
                    i+=1
                    time.sleep(0.0)
            else:
                i+=1
                time.sleep(0.0)
        with open(fileName, 'a') as file:
            for j in range(len(frame_list)):
                if j in removed_frames_list:
                    file.write("%i\n" % ((previousFrames)+((loopCounter-1)*splitter)+j))
                else:
                    skipped_frames_list.append(frame_list[j])
    except:
        print("Something went wrong with checking the similarity between frames")


if __name__ == "__main__":
    try:
        inputVideo=str(sys.argv[1])
        outputVideo="Output Video"
        outputAudio="Audio.mp3"
        finalOutputVideo="Final.mp4"
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

        #####################Extracting its frames###############################
        counter = 0
        loopCounter = 0
        while totalFrames < limit:
            setter=0
            limiter = 0
            frame_list=[]
            success = True   
            while success:
                if(limiter >= splitter or counter >= limit):
                    break
                else:
                    if(setter == 0):
                        video1.set(1,counter)
                        setter+=1
                    success,image = video1.read()
                    #print(counter)
                    if(success):
                        frame_list.append(image)
                        limiter+=1
                        totalFrames+=1
                        counter+=1
            loopCounter+=1


            #################Detecting redundant frames and filling a list with only the non similar frames#################
            skipped_frames_list=[]
            skipped_frames_list1=[]
            skipped_frames_list2=[]
            skipped_frames_list3=[]
            skipped_frames_list4=[]
            skipped_frames_list5=[]
            resizer = 10
            similarityThreshold = 7
            fileName = frameIndicesFile

            previousFrames1 = 0
            previousFrames2 = previousFrames1 + len(frame_list[0:int(len(frame_list)/5)])
            previousFrames3 = previousFrames2 + len(frame_list[int(len(frame_list)/5):int((len(frame_list)/5)*2)])
            previousFrames4 = previousFrames3 + len(frame_list[int((len(frame_list)/5)*2):int((len(frame_list)/5)*3)])
            previousFrames5 = previousFrames4 + len(frame_list[int((len(frame_list)/5)*3):int((len(frame_list)/5)*4)])

            t1 = threading.Thread(target=fun2,args=(frame_list[0:int(len(frame_list)/5)],skipped_frames_list1,resizer,similarityThreshold,0,previousFrames1,loopCounter,splitter,fileName,))
            t2 = threading.Thread(target=fun2,args=(frame_list[int(len(frame_list)/5):int((len(frame_list)/5)*2)],skipped_frames_list2,resizer,similarityThreshold,1,previousFrames2,loopCounter,splitter,fileName,))
            t3 = threading.Thread(target=fun2,args=(frame_list[int((len(frame_list)/5)*2):int((len(frame_list)/5)*3)],skipped_frames_list3,resizer,similarityThreshold,2,previousFrames3,loopCounter,splitter,fileName,))
            t4 = threading.Thread(target=fun2,args=(frame_list[int((len(frame_list)/5)*3):int((len(frame_list)/5)*4)],skipped_frames_list4,resizer,similarityThreshold,3,previousFrames4,loopCounter,splitter,fileName,))
            t5 = threading.Thread(target=fun2,args=(frame_list[int((len(frame_list)/5)*4):int((len(frame_list)/5)*5)],skipped_frames_list5,resizer,similarityThreshold,4,previousFrames5,loopCounter,splitter,fileName,))
            t1.start()
            t2.start()
            t3.start()
            t4.start()
            t5.start()
            t1.join()
            t2.join()
            t3.join()
            t4.join()
            t5.join()
            skipped_frames_list.extend(skipped_frames_list1)
            skipped_frames_list.extend(skipped_frames_list2)
            skipped_frames_list.extend(skipped_frames_list3)
            skipped_frames_list.extend(skipped_frames_list4)
            skipped_frames_list.extend(skipped_frames_list5)
            
            

            #############################Producing an output video without sound################################
            out = cv2.VideoWriter(""+tempFolder+"/"+outputVideo+""+str(loopCounter)+""+".mp4",cv2.VideoWriter_fourcc(*'mp4v'), fps, (len(skipped_frames_list[0][0]), len(skipped_frames_list[0])))
            for i in range(len(skipped_frames_list)):
                out.write(skipped_frames_list[i])
            out.release()

        p1 = Process(target=fun1, args=(inputVideo,outputVideo,outputAudio,loopCounter,finalOutputVideo,props,tempFolder)) 
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
        print("Make sure you entered the name of the video correctly. you entered ",str(sys.argv[1]))
        print("Make sure there is no folder called ",tempFolder,". If there exists one, then please move it or delete it.")
