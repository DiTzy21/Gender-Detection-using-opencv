from typing import Any
import torch
import numpy as np
import cv2
from time import time
from PIL import Image  # Import the Image module from the Pillow library

class FaceDetection:

    def __init__(self, capture_index, model_name):
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("using Device : ", self.device)

    # def get_video_capture(self):
    #     return cv2.VideoCapture(self.capture_index)

    def get_image(self, image_path):
        # Load image using Pillow library
        image = Image.open(image_path)
        return np.array(image)  # Convert the PIL Image to a NumPy array

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord, scores = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1], results.xyxy[0][:, 4]
        return labels, cord, scores

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord, scores = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if scores[i] >= 0.5:  # You can adjust the confidence threshold as needed
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                accuracy_text = f"Accuracy: {scores[i]:.2f}"  # Display accuracy on the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 240, 255), 2)
                cv2.putText(frame, accuracy_text, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 2)
        return frame

    def __call__(self, image_path):
        frame = self.get_image(image_path)  # Load the image using the provided path
        frame = cv2.resize(frame, (416, 416))

        start_time = time()
        result = self.score_frame(frame)
        frame = self.plot_boxes(result, frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('yolov5 detection', frame_rgb)
        cv2.waitKey(0)  # Wait for a key press before closing the window

# Usage example:
image_path = 'C:/Users/Aditya Hadi/Documents/ImageProcesss/Yolov5/yolov5/WhatsApp Image55 2023-12-17 at 23.28.02_47ecc835.jpg'
detector = FaceDetection(capture_index=0, model_name='C:/Users/Aditya Hadi/Documents/ImageProcesss/Yolov5/yolov5/best10.pt')
detector(image_path)
