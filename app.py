from flask import Flask, render_template, url_for, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import emotions_detections_class




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


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        print(f"\n\n\n\n{filename}\n\n\n\n")
        file_url = url_for('get_file', filename=filename)
        print(f"\n\n\n\n{file_url}\n\n\n\n")     
        list_of_feelings, path_to_image = emotions_detections_class.main(file_url[1:])
        file_url = path_to_image
    else:
        list_of_feelings = []
        file_url = None
    return render_template('index.html', form=form, file_url=file_url, list_of_feelings=list_of_feelings)

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)