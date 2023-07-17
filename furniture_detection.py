from roboflow import Roboflow


class FurnitureIdentifier:
    def __init__(self):
        self.rf = Roboflow(api_key="x85B1rA8t0ISXPx6cF4Z")
        self.project = self.rf.workspace().project("furniture-identifier-u2tyo")
        self.model = self.project.version(5).model
    
    def predict(self, img, confidence=40, overlap=30):
        print("predicting objects in image")
        return self.model.predict(img, confidence, overlap).json()

# infer on a local image

if __name__ == "__main__":
    print(FurnitureIdentifier.predict("data/normal living rooms114.jpg", confidence=40, overlap=30).json())