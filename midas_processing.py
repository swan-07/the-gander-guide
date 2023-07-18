import torch
import urllib.request
import cv2
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import furniture_detection as fd

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#urllib.request.urlretrieve(url, filename)
def say(something = None, pos = None):
    # x and y to check for furniture there
    print(f"Saying {something}!") if pos is None else print(f"Searching ({pos[0]}, {pos[1]}) for furniture; {something}")# placeholder for text to speech

class MiDaS:
    def __init__(self, height=480, width=640):
        self.model_type = ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]
        self.model_index = 0
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type[self.model_index])
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.FOV = 70.42 # deg
        self.min_angle_for_prompt = 10 # deg
        self.min_danger_for_problem = 230 # arbitrary

        self.bestXs = [0, 0] # init a queue

        self.height, self.width = height, width
        self.depth_filter = np.zeros((self.height, self.width)) # need to create some [720x1280] array of 0-100 values
        for i in range(self.height):
            for j in range(self.width):
                self.depth_filter[i, j] = np.exp( -0.5 * (((j - (self.width//2))) / (self.width / 6)) ** 2)
        
        if self.model_type[self.model_index] == "DPT_Large" or self.model_type[self.model_index] == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform 
        else:
            self.transform = self.midas_transforms.small_transform
        
        self.identifier = fd.FurnitureIdentifier()

        # placeholders for vibrating warnings only
        self.amplitude = 64
        self.period = 0.5 # 0 can mean steady vibrate; max is 1500ms signifying sharp right, near 0ms signifies sharp left

        # state queue for verbal warnings only
        self.states = [4, 4, 4, 4, 4, 4]

        # State definitions:
        # 1: Completely obstructed
        # 2: Obstruction on left
        # 3: Obstruction on right
        # 4: Good
        # 5: Path to the left
        # 6: Path to the right

    def predict(self, img):
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        #print("Time elapsed: ", time.time() - start)
        return output
    
    # seperate methods to normalize and denormalize depth maps
    def normalize(self, img, scale_factor=1):
        # travis webcam is 1280x720
        maximum = np.amax(img)

        img /= maximum
        return img * scale_factor
        
    # local depth map evaluation (test center third of image for depth values closer than XXXXX)
    def filter(self, img, scale_factor=1, vibrate = True):
        output = img / scale_factor
        
        # check for complete obstructedness
        if np.mean(output) > 0.6:
            if vibrate:
                self.amplitude = 128
                self.period = 0
            else:
                if self.states[-3:] == [1, 1, 1] and self.states[:3].count(1) < 2: # noise-forgiving check for the start of a sequence
                    say("Back up; path is blocked")
                self.states.append(1)
        else:

            # Calculate the column-wise sums
            column_sums = np.sum(output * self.depth_filter, axis=0)
            
            if max(column_sums) < self.min_danger_for_problem:
                # blur horizontally to mitigate noise
                blurred = cv2.blur(output, (10, 1))
                
                # Find the most free path with the minimum weighted average value 
                weights = np.linspace(1, 7, self.height).reshape((self.height, 1))
                candidate = np.argmin(np.mean(blurred * weights, axis=0))
                
                # average from a queue of length 3 with lower rows weighted higher
                self.bestXs.append(candidate)
                bestX = round(sum(self.bestXs)/3)
                self.bestXs.pop(0)
                
                # find the angle to correct path and notify 
                angle = int(self.FOV * bestX / self.width - self.FOV / 2)
            
                if vibrate: 
                    self.amplitude = 64
                    self.period = (1499 * (angle + self.FOV / 2) / self.FOV) + 1 # tell the person where they should turn
                else:
                    if angle**2 < 100:
                        if self.states[-3:] == [4, 4, 4] and self.states[:3].count(4) < 2:
                            say("Good")
                        self.states.append(4)
                    elif angle < -10:
                        if self.states[-3:] == [5, 5, 5] and self.states[:3].count(5) < 2:
                            say("Turn left")
                        self.states.append(5)
                    else:
                        if self.states[-3:] == [6, 6, 6] and self.states[:3].count(6) < 2:
                            say("Turn right")
                        self.states.append(6)
            else:
                # an obstruction exists
                right = np.argmax(column_sums) > self.width / 2

                if vibrate:
                    self.amplitude = 64
                    self.period = 1 + int(right) * 1499 # we can only tell the person to turn away from it
                else:
                    if right:
                        if self.states[-3:] == [3, 3, 3] and self.states[:3].count(3) < 2:
                            say(" to the right; turn left", pos=np.unravel_index(np.argmax(output * self.depth_filter), output.shape))
                    elif self.states[-3:] == [2, 2, 2] and self.states[:3].count(2) < 2:
                        say(" to the left; turn right", pos=np.unravel_index(np.argmax(output * self.depth_filter), output.shape))
                    self.states.append(int(right) + 2)

        if not vibrate:
            self.states.pop(0)
    
if __name__ == "__main__":
    midas = MiDaS()
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))
    depth = midas.predict(img)
    depth = midas.filter(depth)
    plt.imshow(depth)
    plt.show()