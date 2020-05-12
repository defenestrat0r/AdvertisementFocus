import numpy as np
import cv2 as cv
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Makin a Model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

#Just creating a face cascade
cascade = cv.CascadeClassifier('/home/junaid/miniconda3/envs/face-detection/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
#cascPath = sys.argv[1]
#faceCasc = cv.CascadeClassifier(cascPath)

#loading the CNN
model.load_weights('model.h5')

# prevents openCL usage and unnecessary logging messages
cv.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


#Next we gotta use the laptops camera to capture video
vid = cv.VideoCapture(0) #We can also just put in the filepath for a saved video here, so remember that for future loop purposes?
#There is some issue with opencv and compressed videos tho, so there's some thing called ffmpeg that acts as a front end to decode saved compressed videos
#TODO: Integrate ffmpeg into opencv

#Next we run a loop to read the video frame by frame 
while True:
    
    #Now we just basically do the same thing the first program did
    returnCode, frame = vid.read()
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
        
        #This is for predicting the emotion and displaying it on top of the rectangle
        roi_gray = grayscale[row:row + height, col:col + width]
        cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        #emotion_dict[maxindex]) will give you the emotion in the frame
        cv.putText(frame, emotion_dict[maxindex], (col + 20, row - 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    
    cv.imshow('Blammo!', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'): #Basically, we wait for the q key to be pressed. If it is pressed, we exit
        break

#Cleaning up and closing shop
vid.release()
cv.destroyAllWindows()
