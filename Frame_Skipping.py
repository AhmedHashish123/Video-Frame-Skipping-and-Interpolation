import moviepy.editor as mp
import cv2
import os #To create folders and delete files
from videoprops import get_video_properties  #To get bitrate
import threading
import time
import shutil
from multiprocessing import *
from PIL import Image
 
def fun(inputVideo,outputVideo,outputAudio,loopCounter,finaOutputVideo,props,tempFolder):
    #################Reading a video to get the audio###############################
    video2 = mp.VideoFileClip(r""+inputVideo+"")

    #############Extracting the audio and saving it as a .mp3 file###############################
    video2.audio.write_audiofile(r""+outputAudio+"")

    #############Combining Video Clips###############################
    clipList=[]
    for i in range(loopCounter):
        clipList.append(mp.VideoFileClip(r""+tempFolder+"/"+outputVideo+""+str(i+1)+""+".mp4"))
    
    video3 = mp.concatenate_videoclips([clipList[i] for i in range(loopCounter)])#keeps these files open until process ends. Therefore, a new process is created just for this; so that when it ends, the files are closed

    ####################Combining the audio and the images to create a .mp4 video file###############################
    video3.write_videofile(r""+finaOutputVideo+"",audio=""+outputAudio+"",bitrate=(str(int(props['bit_rate'])-(0.05*int(props['bit_rate'])))))
   
def fun1(frame_list,removed_frames_list):
    i1=0
    while i1 < int(len(frame_list)/5):
        #print(i1)
        if(i1>0 and i1<len(frame_list)-1):
            gray11 = cv2.cvtColor(frame_list[i1-1], cv2.COLOR_BGR2GRAY)
            gray21 = cv2.cvtColor(frame_list[i1], cv2.COLOR_BGR2GRAY)
            gray31 = cv2.cvtColor(frame_list[i1+1], cv2.COLOR_BGR2GRAY)
            #print(gray11)
            #print("aaaaaaaaaaaaaaaaaaa")
            img11 = Image.fromarray(gray11)
            #print(img11)
            #print("bbbbbbbbbbbbbbbbbbb")
            img11 = img11.resize((10, 10), Image.LANCZOS)#resizing to make computations easier. LANCZOS is a resampling filter that should produce cleaner results and is equivalent to AntiAliasing
            pixelValues11 = list(img11.getdata())
            #print(pixelValues11)
            #print("ccccccccccccccccccc")
            avgPixelValue11 = sum(pixelValues11)/len(pixelValues11)
            binaryRepresentation11 = "".join(['1' if (pixel11 >= avgPixelValue11) else '0' for pixel11 in pixelValues11])
            hexadecimalRepresentation11 = str(hex(int(binaryRepresentation11, 2)))[2:]#[2:] to remove 0x at the beginning
            img21 = Image.fromarray(gray21)
            img21 = img21.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues21 = list(img21.getdata())
            avgPixelValue21 = sum(pixelValues21)/len(pixelValues21)
            binaryRepresentation21 = "".join(['1' if (pixel21 >= avgPixelValue21) else '0' for pixel21 in pixelValues21])
            hexadecimalRepresentation21 = str(hex(int(binaryRepresentation21, 2)))[2:]
            img31 = Image.fromarray(gray31)
            img31 = img31.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues31 = list(img31.getdata())
            avgPixelValue31 = sum(pixelValues31)/len(pixelValues31)
            binaryRepresentation31 = "".join(['1' if (pixel31 >= avgPixelValue31) else '0' for pixel31 in pixelValues31])
            hexadecimalRepresentation31 = str(hex(int(binaryRepresentation31, 2)))[2:]

            tempIndex11 = 0
            counter11 = 0
            tempLimit11 = 0
            if(len(hexadecimalRepresentation11)<=len(hexadecimalRepresentation21)):
                tempLimit11 = len(hexadecimalRepresentation11)
            else:
                tempLimit11 = len(hexadecimalRepresentation21)
            while(tempIndex11 < tempLimit11):#hamming distance
                if(hexadecimalRepresentation11[tempIndex11] != hexadecimalRepresentation21[tempIndex11]):
                    counter11 += 1
                tempIndex11 += 1

            tempIndex21 = 0
            counter21 = 0
            tempLimit21 = 0
            if(len(hexadecimalRepresentation21)<=len(hexadecimalRepresentation31)):
                tempLimit21 = len(hexadecimalRepresentation21)
            else:
                tempLimit21 = len(hexadecimalRepresentation31)
            while(tempIndex21 < tempLimit21):
                if(hexadecimalRepresentation21[tempIndex21] != hexadecimalRepresentation31[tempIndex21]):
                    counter21 += 1
                tempIndex21 += 1

            if(abs(counter11)<7 and abs(counter21)<7):
                removed_frames_list.append(i1)
                i1+=2
                time.sleep(0.0)
            else:
                i1+=2
                time.sleep(0.0)
        else:
            i1+=2
            time.sleep(0.0)
    return removed_frames_list

def fun2(frame_list,removed_frames_list):
    i2=int(len(frame_list)/5)
    while i2 < int((len(frame_list)/5)*2):
        #print(i2)
        if(i2>int(len(frame_list)/5) and i2<len(frame_list)-1):
            gray12 = cv2.cvtColor(frame_list[i2-1], cv2.COLOR_BGR2GRAY)
            gray22 = cv2.cvtColor(frame_list[i2], cv2.COLOR_BGR2GRAY)
            gray32 = cv2.cvtColor(frame_list[i2+1], cv2.COLOR_BGR2GRAY)
            img12 = Image.fromarray(gray12)
            img12 = img12.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues12 = list(img12.getdata())
            avgPixelValue12 = sum(pixelValues12)/len(pixelValues12)
            binaryRepresentation12 = "".join(['1' if (pixel12 >= avgPixelValue12) else '0' for pixel12 in pixelValues12])
            hexadecimalRepresentation12 = str(hex(int(binaryRepresentation12, 2)))[2:]
            img22 = Image.fromarray(gray22)
            img22 = img22.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues22 = list(img22.getdata())
            avgPixelValue22 = sum(pixelValues22)/len(pixelValues22)
            binaryRepresentation22 = "".join(['1' if (pixel22 >= avgPixelValue22) else '0' for pixel22 in pixelValues22])
            hexadecimalRepresentation22 = str(hex(int(binaryRepresentation22, 2)))[2:]
            img32 = Image.fromarray(gray32)
            img32 = img32.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues32 = list(img32.getdata())
            avgPixelValue32 = sum(pixelValues32)/len(pixelValues32)
            binaryRepresentation32 = "".join(['1' if (pixel32 >= avgPixelValue32) else '0' for pixel32 in pixelValues32])
            hexadecimalRepresentation32 = str(hex(int(binaryRepresentation32, 2)))[2:]

            tempIndex12 = 0
            counter12 = 0
            tempLimit12 = 0
            if(len(hexadecimalRepresentation12)<=len(hexadecimalRepresentation22)):
                tempLimit12 = len(hexadecimalRepresentation12)
            else:
                tempLimit12 = len(hexadecimalRepresentation22)
            while(tempIndex12 < tempLimit12):
                if(hexadecimalRepresentation12[tempIndex12] != hexadecimalRepresentation22[tempIndex12]):
                    counter12 += 1
                tempIndex12 += 1

            tempIndex22 = 0
            counter22 = 0
            tempLimit22 = 0
            if(len(hexadecimalRepresentation22)<=len(hexadecimalRepresentation32)):
                tempLimit22 = len(hexadecimalRepresentation22)
            else:
                tempLimit22 = len(hexadecimalRepresentation32)
            while(tempIndex22 < tempLimit22):
                if(hexadecimalRepresentation22[tempIndex22] != hexadecimalRepresentation32[tempIndex22]):
                    counter22 += 1
                tempIndex22 += 1

            if(abs(counter12)<7 and abs(counter22)<7):
                removed_frames_list.append(i2)
                i2+=2
                time.sleep(0.0)
            else:
                i2+=2
                time.sleep(0.0)
        else:
            i2+=2
            time.sleep(0.0)
    return removed_frames_list

def fun3(frame_list,removed_frames_list):
    i3=int((len(frame_list)/5)*2)
    while i3 < int((len(frame_list)/5)*3):
        #print(i3)
        if(i3>int((len(frame_list)/5)*2) and i3<len(frame_list)-1):
            gray13 = cv2.cvtColor(frame_list[i3-1], cv2.COLOR_BGR2GRAY)
            gray23 = cv2.cvtColor(frame_list[i3], cv2.COLOR_BGR2GRAY)
            gray33 = cv2.cvtColor(frame_list[i3+1], cv2.COLOR_BGR2GRAY)
            img13 = Image.fromarray(gray13)
            img13 = img13.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues13 = list(img13.getdata())
            avgPixelValue13 = sum(pixelValues13)/len(pixelValues13)
            binaryRepresentation13 = "".join(['1' if (pixel13 >= avgPixelValue13) else '0' for pixel13 in pixelValues13])
            hexadecimalRepresentation13 = str(hex(int(binaryRepresentation13, 2)))[2:]
            img23 = Image.fromarray(gray23)
            img23 = img23.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues23 = list(img23.getdata())
            avgPixelValue23 = sum(pixelValues23)/len(pixelValues23)
            binaryRepresentation23 = "".join(['1' if (pixel23 >= avgPixelValue23) else '0' for pixel23 in pixelValues23])
            hexadecimalRepresentation23 = str(hex(int(binaryRepresentation23, 2)))[2:]
            img33 = Image.fromarray(gray33)
            img33 = img33.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues33 = list(img33.getdata())
            avgPixelValue33 = sum(pixelValues33)/len(pixelValues33)
            binaryRepresentation33 = "".join(['1' if (pixel33 >= avgPixelValue33) else '0' for pixel33 in pixelValues33])
            hexadecimalRepresentation33 = str(hex(int(binaryRepresentation33, 2)))[2:]

            tempIndex13 = 0
            counter13 = 0
            tempLimit13 = 0
            if(len(hexadecimalRepresentation13)<=len(hexadecimalRepresentation23)):
                tempLimit13 = len(hexadecimalRepresentation13)
            else:
                tempLimit13 = len(hexadecimalRepresentation23)
            while(tempIndex13 < tempLimit13):
                if(hexadecimalRepresentation13[tempIndex13] != hexadecimalRepresentation23[tempIndex13]):
                    counter13 += 1
                tempIndex13 += 1

            tempIndex23 = 0
            counter23 = 0
            tempLimit23 = 0
            if(len(hexadecimalRepresentation23)<=len(hexadecimalRepresentation33)):
                tempLimit23 = len(hexadecimalRepresentation23)
            else:
                tempLimit23 = len(hexadecimalRepresentation33)
            while(tempIndex23 < tempLimit23):
                if(hexadecimalRepresentation23[tempIndex23] != hexadecimalRepresentation33[tempIndex23]):
                    counter23 += 1
                tempIndex23 += 1

            if(abs(counter13)<7 and abs(counter23)<7):
                removed_frames_list.append(i3)
                i3+=2
                time.sleep(0.0)
            else:
                i3+=2
                time.sleep(0.0)
        else:
            i3+=2
            time.sleep(0.0)
    return removed_frames_list

def fun4(frame_list,removed_frames_list):
    i4=int((len(frame_list)/5)*3)
    while i4 < int((len(frame_list)/5)*4):
        #print(i4)
        if(i4>int((len(frame_list)/5)*3) and i4<len(frame_list)-1):
            gray14 = cv2.cvtColor(frame_list[i4-1], cv2.COLOR_BGR2GRAY)
            gray24 = cv2.cvtColor(frame_list[i4], cv2.COLOR_BGR2GRAY)
            gray34 = cv2.cvtColor(frame_list[i4+1], cv2.COLOR_BGR2GRAY)
            img14 = Image.fromarray(gray14)
            img14 = img14.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues14 = list(img14.getdata())
            avgPixelValue14 = sum(pixelValues14)/len(pixelValues14)
            binaryRepresentation14 = "".join(['1' if (pixel14 >= avgPixelValue14) else '0' for pixel14 in pixelValues14])
            hexadecimalRepresentation14 = str(hex(int(binaryRepresentation14, 2)))[2:]
            img24 = Image.fromarray(gray24)
            img24 = img24.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues24 = list(img24.getdata())
            avgPixelValue24 = sum(pixelValues24)/len(pixelValues24)
            binaryRepresentation24 = "".join(['1' if (pixel24 >= avgPixelValue24) else '0' for pixel24 in pixelValues24])
            hexadecimalRepresentation24 = str(hex(int(binaryRepresentation24, 2)))[2:]
            img34 = Image.fromarray(gray34)
            img34 = img34.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues34 = list(img34.getdata())
            avgPixelValue34 = sum(pixelValues34)/len(pixelValues34)
            binaryRepresentation34 = "".join(['1' if (pixel34 >= avgPixelValue34) else '0' for pixel34 in pixelValues34])
            hexadecimalRepresentation34 = str(hex(int(binaryRepresentation34, 2)))[2:]

            tempIndex14 = 0
            counter14 = 0
            tempLimit14 = 0
            if(len(hexadecimalRepresentation14)<=len(hexadecimalRepresentation24)):
                tempLimit14 = len(hexadecimalRepresentation14)
            else:
                tempLimit14 = len(hexadecimalRepresentation24)
            while(tempIndex14 < tempLimit14):
                if(hexadecimalRepresentation14[tempIndex14] != hexadecimalRepresentation24[tempIndex14]):
                    counter14 += 1
                tempIndex14 += 1

            tempIndex24 = 0
            counter24 = 0
            tempLimit24 = 0
            if(len(hexadecimalRepresentation24)<=len(hexadecimalRepresentation34)):
                tempLimit24 = len(hexadecimalRepresentation24)
            else:
                tempLimit24 = len(hexadecimalRepresentation34)
            while(tempIndex24 < tempLimit24):
                if(hexadecimalRepresentation24[tempIndex24] != hexadecimalRepresentation34[tempIndex24]):
                    counter24 += 1
                tempIndex24 += 1

            if(abs(counter14)<7 and abs(counter24)<7):
                removed_frames_list.append(i4)
                i4+=2
                time.sleep(0.0)
            else:
                i4+=2
                time.sleep(0.0)
        else:
            i4+=2
            time.sleep(0.0)
    return removed_frames_list

def fun5(frame_list,removed_frames_list):
    i5=int((len(frame_list)/5)*4)
    while i5 < int((len(frame_list)/5)*5):
        #print(i5)
        if(i5>int((len(frame_list)/5)*4) and i5<len(frame_list)-1):
            gray15 = cv2.cvtColor(frame_list[i5-1], cv2.COLOR_BGR2GRAY)
            gray25 = cv2.cvtColor(frame_list[i5], cv2.COLOR_BGR2GRAY)
            gray35 = cv2.cvtColor(frame_list[i5+1], cv2.COLOR_BGR2GRAY)
            img15 = Image.fromarray(gray15)
            img15 = img15.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues15 = list(img15.getdata())
            avgPixelValue15 = sum(pixelValues15)/len(pixelValues15)
            binaryRepresentation15 = "".join(['1' if (pixel15 >= avgPixelValue15) else '0' for pixel15 in pixelValues15])
            hexadecimalRepresentation15 = str(hex(int(binaryRepresentation15, 2)))[2:]
            img25 = Image.fromarray(gray25)
            img25 = img25.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues25 = list(img25.getdata())
            avgPixelValue25 = sum(pixelValues25)/len(pixelValues25)
            binaryRepresentation25 = "".join(['1' if (pixel25 >= avgPixelValue25) else '0' for pixel25 in pixelValues25])
            hexadecimalRepresentation25 = str(hex(int(binaryRepresentation25, 2)))[2:]
            img35 = Image.fromarray(gray35)
            img35 = img35.resize((10, 10), Image.LANCZOS)#resizing to make computations easier
            pixelValues35 = list(img35.getdata())
            avgPixelValue35 = sum(pixelValues35)/len(pixelValues35)
            binaryRepresentation35 = "".join(['1' if (pixel35 >= avgPixelValue35) else '0' for pixel35 in pixelValues35])
            hexadecimalRepresentation35 = str(hex(int(binaryRepresentation35, 2)))[2:]

            tempIndex15 = 0
            counter15 = 0
            tempLimit15 = 0
            if(len(hexadecimalRepresentation15)<=len(hexadecimalRepresentation25)):
                tempLimit15 = len(hexadecimalRepresentation15)
            else:
                tempLimit15 = len(hexadecimalRepresentation25)
            while(tempIndex15 < tempLimit15):
                if(hexadecimalRepresentation15[tempIndex15] != hexadecimalRepresentation25[tempIndex15]):
                    counter15 += 1
                tempIndex15 += 1

            tempIndex25 = 0
            counter25 = 0
            tempLimit25 = 0
            if(len(hexadecimalRepresentation25)<=len(hexadecimalRepresentation35)):
                tempLimit25 = len(hexadecimalRepresentation25)
            else:
                tempLimit25 = len(hexadecimalRepresentation35)
            while(tempIndex25 < tempLimit25):
                if(hexadecimalRepresentation25[tempIndex25] != hexadecimalRepresentation35[tempIndex25]):
                    counter25 += 1
                tempIndex25 += 1

            if(abs(counter15)<7 and abs(counter25)<7):
                removed_frames_list.append(i5)
                i5+=2
                time.sleep(0.0)
            else:
                i5+=2
                time.sleep(0.0)
        else:
            i5+=2
            time.sleep(0.0)
    return removed_frames_list

if __name__ == "__main__":

    inputVideo="test.mp4"
    outputVideo="Output Video"
    outputAudio="Audio.mp3"
    finalOutputVideo="Final2.mp4"
    tempFolder="Folder"
    os.mkdir(tempFolder)

    #############Reading a video and extracting video info (I care about the bitrate and fps) and getting the while loop threshold#################
    start_time=time.time()
    props = get_video_properties(""+inputVideo+"")
    fps = round(int(props['avg_frame_rate'].split('/')[0])/int(props['avg_frame_rate'].split('/')[1]))
    video1 = cv2.VideoCapture(""+inputVideo+"")
    limit = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    totalFrames = 0
    #####################Extracting its frames###############################
    counter = 0
    loopCounter = 0
    
    while totalFrames < limit:
        setter=0
        limiter = 0 #limit of the number of frames in the frame_list
        frame_list=[]
        success = True   
        while success:
            if(limiter >= 3000 or counter >= limit):
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


        #################Detecting redundant frames#################
        #print(len(frame_list))

        removed_frames_list=[]
        
        
        
        
        
        t1 = threading.Thread(target=fun1,args=(frame_list,removed_frames_list,))
        t2 = threading.Thread(target=fun2,args=(frame_list,removed_frames_list,))
        t3 = threading.Thread(target=fun3,args=(frame_list,removed_frames_list,))
        t4 = threading.Thread(target=fun4,args=(frame_list,removed_frames_list,))
        t5 = threading.Thread(target=fun5,args=(frame_list,removed_frames_list,))
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

        ##################Removing redundant frames#################
        skipped_frame_list=[]
        print(len(removed_frames_list))
        with open('Skipped_Frames_correct.txt', 'a') as file:
            for j in range(len(frame_list)):
                if j in removed_frames_list:
                    file.write("%i\n" % (((loopCounter-1)*3000)+j))
                else:
                    skipped_frame_list.append(frame_list[j])
        #############################Producing an output video without sound################################
        out = cv2.VideoWriter(""+tempFolder+"/"+outputVideo+""+str(loopCounter)+""+".mp4",cv2.VideoWriter_fourcc(*'mp4v'), fps, (len(skipped_frame_list[0][0]), len(skipped_frame_list[0])))
        for i in range(len(skipped_frame_list)):
            out.write(skipped_frame_list[i])
        out.release()

    p1 = Process(target=fun, args=(inputVideo,outputVideo,outputAudio,loopCounter,finalOutputVideo,props,tempFolder)) 
    p1.start()
    p1.join()

    ################################Release all space and windows once done###############################
    video1.release() 
    cv2.destroyAllWindows() 

    ###############################Deleting the audio file and additional video file###############################
    os.remove(""+outputAudio+"")
    shutil.rmtree(""+tempFolder+"")
    #for i in range(loopCounter):
    #    fd = os.open(""+outputVideo+""+str(i+1)+""+".mp4", os.O_WRONLY)
    #    os.close(fd)
    #    os.remove(""+outputVideo+""+str(i+1)+""+".mp4")
    print("Finished")
    print(time.time()-start_time)