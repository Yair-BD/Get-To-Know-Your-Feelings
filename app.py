from flask import Flask, render_template, Response
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import facial_expression
from realtime_facial_expression import gen_frames
import numpy as np, cv2, base64
from keras.models import load_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'iwertywerty'
app.config['UPLOADED_PHOTOS_DEST'] = 'static/uploads'
PATH_CUTE_VIDEO = 'static\\video\cute.mp4'

photos = UploadSet('photos', IMAGES) # class, extensions
configure_uploads(app, photos)

class UploadForm(FlaskForm): # class for upload photo form 
    photo = FileField(
        validators=[
            FileAllowed(photos, "Only images are allowed!"), # validator for file
            FileRequired('File field should not be empty!') # validator for file
            ]
        )
    submit = SubmitField('Upload') 
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(model), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime')
def realtime():
    return render_template('real_time.html')


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    video = ''
    if form.validate_on_submit():
        image_file = form.photo.data # get image from form
        
        image_array = np.frombuffer(image_file.read(), np.uint8) # convert image to np array 
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED) # decode image

        list_of_feelings, image_cv_format = facial_expression.main(decoded_image, model) # get list of feelings and image with emotions
        
        _, jpeg_data = cv2.imencode('.jpg', image_cv_format) # encode image to jpeg
        b64_data = base64.b64encode(jpeg_data).decode() # convert image to base64
        
        if there_is_bad_feelings(list_of_feelings):
            video = PATH_CUTE_VIDEO 
          
    else:
        list_of_feelings = []
        video = ''
        b64_data = None
    return render_template('index.html', form=form, b64_data=b64_data, list_of_feelings=list_of_feelings, video = video)

def there_is_bad_feelings(list_of_feelings):
    for feelings in list_of_feelings:
        if feelings in ['Angry','Disgust','Fear','Sad']:
            return True
    return False
 

if __name__ == '__main__':
    model = load_model(f'keras_model/model_5-49-0.62.hdf5')
    app.run(debug=False, host= '0.0.0.0')