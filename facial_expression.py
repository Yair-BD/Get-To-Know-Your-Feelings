import cv2, numpy as np, sys, tensorflow as tf
from keras.models import load_model
from keras.backend import set_session
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
''' Keras took all GPU memory so to limit GPU usage, I have add those lines'''
## End section

class EmotionDetections:
    def __init__(self, image, model):
        self.image_to_detect = image
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        self.model = model
        self.target = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    
    def get_gray_image(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return gray
    
    def get_faces_from_image(self):
        gray_image = self.get_gray_image(self.image_to_detect)
        faces = self.faceCascade.detectMultiScale(gray_image,scaleFactor=1.1)
        return faces
    
    def detect_face_emotions(self, face_crop):
        result = self.target[np.argmax(self.model.predict(face_crop))] # predict the result using our model
        return result
    
    def crop_faces(self, faces):  
        font = cv2.FONT_HERSHEY_SIMPLEX
        list_of_feelings = []
        for (x, y, w, h) in faces:
                cv2.rectangle(self.image_to_detect, (x, y), (x+w, y+h), (0, 255, 0), 2,5) # draw rectangle to main image
                face_crop = self.image_to_detect[y:y+h,x:x+w] # crop face from image
                face_crop = cv2.resize(face_crop,(48,48)) # resize to 48x48
                face_crop = self.get_gray_image(face_crop) # convert to grayscale
                face_crop = face_crop.astype('float32')/255 # normalize
                face_crop = np.asarray(face_crop) # convert to array 
                face_crop = face_crop.reshape(1, 1,face_crop.shape[0],face_crop.shape[1]) # reshape to 1,1,48,48
                result = self.detect_face_emotions(face_crop) # predict the result using our model 
                list_of_feelings.append(result)
                cv2.putText(self.image_to_detect, result, (x,y), font, 1, (200,0,0), 3, cv2.LINE_AA) # write result on image 

        return list_of_feelings, self.image_to_detect

def main(image, model):
    image = EmotionDetections(image, model)
    faces = image.get_faces_from_image()
    list, image_cv_format = image.crop_faces(faces)
    return list, image_cv_format
        
        
    
if __name__=='__main__':
    image = sys.argv[1]
    model = sys.argv[2]
    
    main(image, model)