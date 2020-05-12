import cv2 as cv

#Importing the image to use 
img = cv.imread('/home/junaid/Documents/Sneks/family')

print(img.shape)

#Converting it to grayscale for Viola-Jones 
grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Loading the haar cascade classifier
cascade = cv.CascadeClassifier('/home/junaid/miniconda3/envs/face-detection/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')

#Detecting faces
detected = cascade.detectMultiScale(grayscale_img)
#These detections are saved as their top-left coordinate + width and height of the rectangle that encompasses their face.

#So we draw rectangles using the information from the detected variable
#And put that in a loop to display every face that has been detected
for (col, row, width, height) in detected:
    cv.rectangle(
                img, #the original img
                (col, row), #coordinates of the top left pixel
                (col + width, row + height), #coordinates of the bottom right pixel
                (255, 0, 0), #color of rectangle
                2 #thiccness of rectangle
                ) 

#Now we display the image
cv.imshow('Ta-Da!', img)
cv.waitKey(0) #thisll wait till a key is pressed to delete the window 
cv.destroyAllWindows() #seems self explanatory
 
