# Video-Frame-Skipping-and-Interpolation
This project will consist of two main modules. One module takes a video as an input and outputs a smaller sized video with less frames and a text file indicating the indices of the removed frames, and the other module takes this smaller video and the text file and interpolates the missing frames. This can be really handy for uploading videos quickly or transferring a small sized video to another pc then interpolating the missing frames to get a video that is close to the original.

# To run Frame_Skipping.py
Simply open a terminal in the same location as the python file and write "python Frame_Skipping.py video.mp4", where "video.mp4" is the name of the video file that you wish to apply the algorithm to. You can change this name as you like.

# To run Interpolate_1.py
Simply open a terminal in the same location as the python file and write "python Interpolate_1.py video.mp4", where "video.mp4" is the name of the video file that was produced from the skipping module. You can change this name as you like.
