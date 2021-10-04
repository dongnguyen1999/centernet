import logging
import os

# from flask import render_template, Blueprint, request, make_response
from werkzeug.utils import secure_filename

# @blueprint.route('/')
# @blueprint.route('/index')
# def index():
#     # Route to serve the upload form
#     return render_template('index.html',
#                            page_name='Home',
#                            project_name="ndongApi")


# @blueprint.route('/upload', methods=['POST'])
# def upload():
#     # Route to deal with the uploaded chunks
#     log.info(request.form)
#     log.info(request.files)
#     return make_response(('ok', 200))

from flask import Flask, render_template, request, make_response, Response
import time
import cv2

from flask_uploads import UploadSet, configure_uploads
#Initialize the Flask app

app = Flask(__name__)

current_dir = os.getcwd()
app.config['UPLOADED_MEDIA_DEST'] = os.path.join(current_dir, 'tmp')
media = UploadSet('media', ('mp4')) 
configure_uploads(app, (media,))

camera = cv2.VideoCapture(r'C:\Users\nvdkg\Videos\Captures\test_video.mp4', cv2.CAP_DSHOW)
'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        time.sleep(1)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    chunk = request.files['chunk']
    filename = request.form['filename']
    file_uuid = request.form['uuid']
    chunk_index = request.form['index']
    total_chunk = request.form['total_chunk']
    total_size = request.form['total_size']
    chunk_size = request.form['chunk_size']

    # print(filename)
    
    filename = secure_filename(f'chunk{chunk_index}.mp4') # Secure the filename to prevent some kinds of attack
    media.save(chunk, folder=file_uuid, name=filename)

    return make_response(('Uploaded Chunk', 200))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

