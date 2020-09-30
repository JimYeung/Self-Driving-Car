from flask import Flask
import socketio
import eventlet
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
#Flask is used to create bi-direction communication between client & server

sio = socketio.Server() #real-time communication between server & client
app = Flask(__name__)
"""
Event Handler
@app.route('/home')#Telling the flask what route the app should use to trigger the function
def greeting():
    return 'welcome!'

@sio.event() / @sio.on()
    - Once the event handler is triggered, the code will be executed accordingly

sio.emit()
    -emitting events
"""
speed_limit = 30
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66)) #Resize to fit NVidia data architecture
    return img/255


@sio.on('connect') #message, disconnect
def connect(sid, environ):
    print('connected')
    send_control(0,0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
if __name__ =='__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)