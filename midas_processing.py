import torch
import urllib.request
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import furniture_detection as fd

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#urllib.request.urlretrieve(url, filename)

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
    def filter(self, img, scale_factor=1):
        output = img / scale_factor
        
        ## START OF NEW STUFF
        str = 'everything is fine'
        # Calculate the column-wise sums
        column_sums = np.sum(output * self.depth_filter, axis=0)
        str = " "
        # Minimum 'danger level' to call it a problem
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
            
            # annotate with a line
            output = cv2.line(output, (bestX, 0), (bestX, self.height), (255, 20, 100), 3)
            
            # find the angle to correct path and notify 
            angle = int(self.FOV * bestX / self.width - self.FOV / 2)
        
            if angle < -self.min_angle_for_prompt:
                str = f"Turn left by {-angle} degrees"
            elif angle > self.min_angle_for_prompt:
                str = f"Turn right by {angle} degrees"
        else:
            angle = int(self.FOV * np.argmax(column_sums) / self.width - self.FOV / 2)

            if angle < 0:
                str = f"Problem({round(max(column_sums))}) on left by {-angle} degrees"
            else:
                str = f"Problem({round(max(column_sums))}) on right by {angle} degrees"
        return str
        ## END OF NEW STUFF
    
    
if __name__ == "__main__":
    midas = MiDaS()
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))
    depth = midas.predict(img)
    depth = midas.filter(depth)
    plt.imshow(depth)
    plt.show()