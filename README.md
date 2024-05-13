# HighwayCar_Detection_Using_DETR Project Background
This project using DETR Model from <a href="https://huggingface.co/facebook/detr-resnet-50">facebook/detr-resnet-50</a> to detect cars travel on highway from the input video then export the processed output video.
After detecting cars on highway, the program try to calculate the speed for each car 

# Execuation
--First I will create a class called CarDetections to detect cars running on the highway and set up functions to calculate and process frames in the video.
  Then I determine the size of the video, I decide to use the Hough Transform algorithm to identify straight lines 
  After that, I calculated the coordinates and found the most suitable coordinates of the highway in the video 
  Finally, the program will export the processed video and include the speed of the vehicles running on the highway.

--Output: 
  Hough Transform algorithm 
  ![image](https://github.com/nhutnguyen1907/highwayCar_Detection_Using_DETR/assets/93028680/e7e557bd-213e-4c56-b606-9011c49f4853)

  Output video
  ![image](https://github.com/nhutnguyen1907/highwayCar_Detection_Using_DETR/assets/93028680/58e3333a-6609-4863-b024-a75257afa756)

