from flask import Flask, render_template, request, redirect, url_for, flash, Response
import json
import cv2
import os
import numpy as np
import midas_processing as mp
import base64

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
global switch, cap, tracker, computer_vision
switch=1
computer_vision=True
port = 5000
base_url = get_base_url(port)

# Flask App
app = Flask(__name__)
# OpenCV Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tracker = mp.MiDaS()

# Home Page
@app.route(f"{base_url}")
def index():
    print("Loading Home Page...")
    global switch, cap, computer_vision
    if request.method == 'POST':
        if request.form.get('stop') == 'Start or Stop Video':
            print("Flipping switch")
            if(switch==1):
                switch=0
                cap.release()
                cv2.destroyAllWindows()
            else:
                cap = cv2.VideoCapture(0)
                switch=1
        if request.form.get('cv') == 'Toggle Computer Vision':
            print("flipping camera mode")
            if(computer_vision==True):
                computer_vision=False
            else:
                computer_vision=True
    
    elif request.method == 'GET':
        return render_template("index.html")
    return render_template("index.html")

@app.route(f"{base_url}/video_feed/")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    '''
    Generates frames from camera
    '''
    global computer_vision, cap, tracker
    while True:
        try:
            success, image = cap.read()
            
        except:
            print("Camera not found")
            break
        
        if computer_vision:
            image = tracker.normalize(tracker.predict(image), 255)
            cv2.putText(image, tracker.filter(image, 255), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
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
                pass


if __name__ == "__main__":
    app.run(debug=True)