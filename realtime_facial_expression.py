import numpy as np, cv2
from keras.models import load_model


def gen_frames():  
    camera = cv2.VideoCapture(0)
    model = load_model(f'keras_model/model_5-49-0.62.hdf5')
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    target = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(frame, scaleFactor=1.1)
            # # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
                frame_to_model = frame[y:y + h, x:x + w]
                frame_to_model = cv2.resize(frame, (48, 48))
                frame_to_model = cv2.cvtColor(frame_to_model, cv2.COLOR_BGR2GRAY)
                frame_to_model = frame_to_model.astype('float32') / 255
                frame_to_model = np.asarray(frame_to_model)
                frame_to_model = frame_to_model.reshape(1, 1, frame_to_model.shape[0], frame_to_model.shape[1])
                result = target[np.argmax(model.predict(frame_to_model))]
                cv2.putText(frame, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            