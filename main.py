import cv2
import midas_processing as mp
import base64

# Path: main.py
camera = cv2.VideoCapture(0)
depth_model = mp.MiDaS()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



while True:
    try:
        success, frame = camera.read()
    except:
        print("There is no camera")
        break
    
    #insert any model calls here
    frame = depth_model.normalize(depth_model.predict(frame))
    
    depth_model.filter(frame)
    #
    #
    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()