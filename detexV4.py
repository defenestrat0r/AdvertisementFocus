import numpy as np
import cv2 as cv
import sys
import ffmpeg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#--------------------------------------------------------------------------------------
#Makin a model
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

#--------------------------------------------------------------------------------------
# creating a face cascade
cascade = cv.CascadeClassifier('/home/junaid/miniconda3/envs/face-detection/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')

#--------------------------------------------------------------------------------------
# loading the CNN
model.load_weights('model.h5')

# prevents openCL usage and unnecessary logging messages
cv.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# to count the frames
count = 1

# array to store happy frames
hap = []

# livestream 
vid = cv.VideoCapture(0)

#-------------------------------------------------------------------------------------- 
# next we run a loop to read the video frame by frame 
while True:
    returnCode, frame = vid.read() 
    if not returnCode:
        break
    
    # convert frame to grayscale
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # detecting faces
    detected = cascade.detectMultiScale(grayscale)
    
    #--------------------------------------------------------------------------------------
    for (col, row, width, height) in detected:        
        # this is for predicting the emotion 
        roi_gray = grayscale[row:row + height, col:col + width]
        cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        
        # this will give you the emotion in the frame
        emotion = emotion_dict[maxindex]
        #print(emotion)
        
        if emotion == "Happy":
            #cv.imwrite("frame%d.jpg" % count, frame)
            hap.append(count)
            
        count += 1
    #--------------------------------------------------------------------------------------
    
    #Show frame by frame
    cv.imshow('Blammo!', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'): #Basically, we wait for the q key to be pressed. If it is pressed, we exit
        break

#Cleaning up and closing shop
vid.release()
cv.destroyAllWindows()

print("Total frames : " + str(count))

print("Happy frames : ")
x = len(hap)
for i in range(0,x):
    print(hap[i])
