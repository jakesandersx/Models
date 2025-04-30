import cv2
import os
import random
from ultralytics import YOLO

model = YOLO('C:/Users/MSJAIUser/PycharmProjects/Senior Research Project/cicada_yolo/runs/detect/train11/weights/best.pt')

image_folder = './labeled_images/images'
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

selected_images = random.sample(image_paths, min(10, len(image_paths)))

for img_path in selected_images:
    img = cv2.imread(img_path)

    results = model(img)

    for result in results:
        result.show()

        predictions = result.boxes

        print(f"Predictions for {img_path}:")
        for box in predictions.xyxy:
            if len(box) == 6:
                x1, y1, x2, y2, confidence, class_idx = box
                print(f"Detected {model.names[int(class_idx)]} with confidence {confidence:.2f}")
                print(f"Bounding box: ({x1:.2f}, {y1:.2f}), ({x2:.2f}, {y2:.2f})")
            else:
                print("Box does not contain the expected number of values.")
