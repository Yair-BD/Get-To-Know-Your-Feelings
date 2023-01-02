from flask import Flask, render_template, url_for, send_from_directory, Response
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import emotions_detections_class
from realtime_facial_expression import gen_frames
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'iwertywerty'
app.config['UPLOADED_PHOTOS_DEST'] = 'static/uploads'

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

@app.route('/static/uploads/<filename>')
def get_file(filename):
    return send_from_directory('static/uploads', filename)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime')
def realtime():
    return render_template('real_time.html')


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    video = ''
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        list_of_feelings, path_to_image = emotions_detections_class.main(file_url[1:])
        file_url = path_to_image
        if there_is_bad_feelings(list_of_feelings):
            video = 'static\\video\cute.mp4'
          
    else:
        list_of_feelings = []
        video = ''
        file_url = None
    return render_template('index.html', form=form, file_url=file_url, list_of_feelings=list_of_feelings, video = video)

def there_is_bad_feelings(list_of_feelings):
    for feelings in list_of_feelings:
        if feelings in ['Angry','Disgust','Fear','Sad']:
            return True
    return False
 

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='10.0.0.5')