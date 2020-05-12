# Advertisement Focus

## Abstract 
Advertisements play a key role in the business process. This is the age of the internet, and advertisements are present everywhere, but most tend to be ignored and swept to the peripherals of the user’s minds, doomed to be reduced to little more than just background noise. So, what really engages a person when they’re watching something? What disengages them? Those are the questions we hope to answer by analysing user reactions to advertisements. Using facial recognition and sentiment analysis, we hope to discern what the reaction to the advertisements, or parts of the advertisements are so that we may figure out what strategies work when trying to hook a consumer, and what strategies fail.

## Requirements
We used ``Python v3.6.2``

A couple of packages you might want to get this to run
- scipy
- scikit-learning
- numpy
- cv2 (opencv)
- tensorflow

## Usage
Really, the final product is ``detexV4.py``, so go ahead and run that directly if you don't care about what's going on under the hood. Simply run ``python detexV4.py`` on your command line and you're good to go.

``detex.py`` starts slow and only detects faces in the still image called family. You can swap out the static image for one of your own and modify the code to test out the classifier.

``detexV2.py`` takes it a step further and detects faces from a live feed, your webcam.

``detexV3.py`` uses the pretrained model, ``model.h5`` to analyze and classify emotion from a live feed.

``detexV4.py`` combines everything to demonstrate detecting specific emotions and storing that frame number in an array for later reference.

## Additional Information
The model was trained using the ``emotions.py`` program in the data **Tensorflow** folder, with the images in the **Tensorflow > data** folder. The training and test images were taken from Kaggle to classify 7 basic emotions - happy, sad, fear, disgust, angry, neutral and surprise. A weighted CNN was used for our program. 

A different model can be trained with the available test and training data, and can easily be slotted into the ``detexV4.py`` file by swapping out the ``model.h5``.
