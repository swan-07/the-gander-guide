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
    '''
    Generates frames from camera
    '''
    # Technical view variables:


    equations = [
        r'$\lim_{{n\to\infty}}\frac{1}{2\pi}\int_{-\pi}^{\pi}f\delta_n(x)e^{-inx}\,dx = \lim_{{n\to\infty}}\frac{1}{2\pi}\int_{-\infty}^{\infty}g\delta_n(t)\int_{-\pi}^{\pi}e^{-inx}\left|x - t\right|^{1-\alpha}\,dx\,dt$',
        r'$a_n = \frac{1}{2\pi}\int_{-\pi}^{\pi}f(x)e^{-inx}\,dx = \frac{1}{2\pi}\int_{-\pi}^{\pi}f_0(x)e^{-inx}\,dx$',
        r'$\int_G \Theta(f_\varepsilon(t))\,d\mu(t) = -\int_G f^4_\varepsilon(t)\,d\mu(t) + (b^2 + 2a^2)\int_G f^2_\varepsilon(t)\,d\mu(t) + 2ab^2\int_G f_\varepsilon(t)\,d\mu(t) + a^2b^2 - a^4$',
        r'$P(GN) \leq \sum_{\mu^2<j\leq\mu} P\left(\frac{{kB_0j}}{{kT}} > q_j\right)\leq \sum_{\mu^2<j\leq\mu} \frac{{4T}}{Nq_j}C_0Tq_j(\log p_{j+1})^{3/2} \leq \frac{{4T}}{N}C_0T\mu(\log p_{\mu+1})^{3/2} \leq \frac{{4T C_0T}}{N\mu}\exp\left(3(\mu + 1)l\gamma\right)$',
        r'$P_k(s) = \frac{1}{h_k} + \frac{e^{-s h_k}}{h_k - 1} + \cdots + \frac{e^{-(h_k-1)s}}{1 - e^{-h_k s}}$'
    ]

    color = np.random.randint(0, 255, (100, 3))

    # Parameters for corner detection
    feature_params = dict(
        maxCorners=30,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7
    )

    # Parameters for optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    old_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_gray)
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
        elif vision_mode == 3: # if technical view mode
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Update the corner points using goodFeaturesToTrack
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

            # Create a new mask image for each frame
            mask = np.zeros_like(image)
            # Draw the tracks and collect intersection points
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                # Only take action if the movement is greater than 8px
                if euclid_dist(a, b, c, d) > 1:
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 8)

                    # Calculate the direction vector of the ray
                    direction = (c - a, d - b)

                    # Extend the ray to cover the entire frame width and height
                    extended_point = (int(a + direction[0] * image.shape[1]), int(b + direction[1] * image.shape[0]))

                    # Draw the extended ray
                    mask = cv2.line(mask, (int(a), int(b)), extended_point, color[i].tolist(), 5)
                    mask = cv2.circle(mask, (int(a), int(b)), 5, color[i].tolist(), -1)

            image = cv2.add(image, mask)  # Combine frame and mask
            old_gray = frame_gray.copy()
        
            # Equation stuff:
            image_size = image.shape
            
            for i in range(8):
                selected_equation = random.choice(equations)

                # Randomly select a 10-character subset
                subset_start = random.randint(0, len(selected_equation) - 18)
                subset_end = subset_start + 18
                equation_subset = selected_equation[subset_start:subset_end]

                # Define the font properties
                font_scale = random.uniform(0.6, 1.2)
                font_thickness = random.randint(1, 3)

                # Randomize text color
                text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # Calculate the text size
                (text_width, text_height), _ = cv2.getTextSize(equation_subset, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

                # Randomize text position
                text_position = (
                    random.randint(0, image_size[1] - text_width),
                    random.randint(text_height, image_size[0])
                )

                # Add the equation subset text to the image
                image = cv2.putText(image, equation_subset, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
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