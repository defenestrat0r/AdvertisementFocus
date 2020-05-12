import cv2 as cv
import sys

#Just creating a face cascade
cascade = cv.CascadeClassifier('/home/junaid/miniconda3/envs/face-detection/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
#cascPath = sys.argv[1]
#faceCasc = cv.CascadeClassifier(cascPath)


#Next we gotta use the laptops camera to capture video
vid = cv.VideoCapture(0) #We can also just put in the filepath for a saved video here, so remember that for future loop purposes?
#There is some issue with opencv and compressed videos tho, so there's some thing called ffmpeg that acts as a front end to decode saved compressed videos
#TODO: Integrate ffmpeg into opencv

#Next we run a loop to read the video frame by frame 
while True:
    returnCode, frame = vid.read() #So the return code will tell us when we run out of frames to read
    if not returnCode:
        break
    
    #Now we just basically do the same thing the first program did
    
    #Convert frame to grayscale
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #Detecting faces
    detected = cascade.detectMultiScale(grayscale)
    
    #Drawing rectangles
    for (col, row, width, height) in detected:
        cv.rectangle(
                    frame, #the original img
                    (col, row), #coordinates of the top left pixel
                    (col + width, row + height), #coordinates of the bottom right pixel
                    (255, 0, 0), #color of rectangle
                    2 #thiccness of rectangle
                    )
    
    cv.imshow('Blammo!', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'): #Basically, we wait for the q key to be pressed. If it is pressed, we exit
        break

#Cleaning up and closing shop
vid.release()
cv.destroyAllWindows()
