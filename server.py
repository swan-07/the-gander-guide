from flask import Flask, render_template, request, redirect, url_for, flash, Response
import json
import cv2
import os
import numpy as np
import midas_processing as mp
import base64
import random
from ultralytics import YOLO
import torch

def get_base_url(port:int) -> str:
    '''
    Returns the base URL to the webserver if available.
    
    i.e. if the webserver is running on coding.ai-camp.org port 12345, then the base url is '/<your project id>/port/12345/'
    
    Inputs: port (int) - the port number of the webserver
    Outputs: base_url (str) - the base url to the webserver
    '''
    
    try:
        info = json.load(open(os.path.join(os.environ['HOME'], '.smc', 'info.json'), 'r'))
        project_id = info['project_id']
        base_url = f'/{project_id}/port/{port}/'
    except Exception as e:
        print(f'Server is probably running in production, so a base url does not apply: \n{e}')
        base_url = '/'
    return base_url

'''
    to run
flask --app server run
    or
python server.py

    to exit 
ctrl + c
'''
global switch, cap, tracker, vision_mode
switch=1
vision_mode=1 # 1 is normal, 2 is computer vision, 3 is rainbows and unicorns
port = 5000
base_url = get_base_url(port)

# Flask App
app = Flask(__name__)
# OpenCV Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tracker = mp.MiDaS()
model = YOLO('weights.pt')
  
def euclid_dist(x1, y1, x2, y2):
    return np.linalg.norm([x2 - x1, y2 - y1])

# Home Page
@app.route(f"{base_url}", methods=['GET', 'POST'])
def index():
    print("Loading Home Page...")
    global switch, cap, vision_mode
    if request.method == 'POST':
        print("Post Request Received")
        print(request.form)
        if request.form.get('option') == 'cv':
            print("Computer Vision Mode")
            vision_mode = 2
        elif request.form.get('option') == 'tv':
            print("Technical Vision Mode")
            vision_mode = 3
        else:
            print("Normal Vision Mode")
            vision_mode = 1
            
    
    elif request.method == 'GET':
        return render_template("index.html")
    return render_template("index.html")

@app.route(f"{base_url}/video_feed/")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global computer_vision, cap, tracker


    while True:
        try:
            success, image = cap.read()
            
        except:
            print("Camera not found")
            break
        
        if vision_mode == 2: # if computer vision mode
            results = model.predict(image)
            for result in results:
                boxes = result.boxes.xyxy
                labels = result.boxes.cls
                for box, label in zip(boxes, labels):
                    x1, y1, x2, y2 = box[:4].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, result.names[int(label)], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            try:
                ret, buffer = cv2.imencode('.jpg', image)
                img_64 = base64.b64encode(buffer)
                #print(img_64)
                #pred = tracker.identifier.predict(img_64)
                #print(pred)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print("exception thrown when trying to encode image")
                pass
        elif vision_mode == 3: # if midas view mode (formerly technical)
            
            tracker.filter(tracker.normalize(tracker.predict(image)), vibrate="Website") # show both vibration info and verbal warning
            image = tracker.website_image

            try:
                ret, buffer = cv2.imencode('.jpg', image)
                img_64 = base64.b64encode(buffer)
                #print(img_64)
                #pred = tracker.identifier.predict(img_64)
                #print(pred)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print("exception thrown when trying to encode image")
                pass
        else:
            try:
                ret, buffer = cv2.imencode('.jpg', image)
                #print(img_64)
                #pred = tracker.identifier.predict(img_64)
                #print(pred)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print("exception thrown when trying to encode image")
                pass


if __name__ == "__main__":
    app.run(debug=True)