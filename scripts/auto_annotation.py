import os
from ultralytics import YOLO
from tqdm import tqdm
import shutil
import logging

# Setup logging
logging.basicConfig(
    filename="error_log.txt",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Directories
image_dir = "G:/photos"
output_dir = "./new_labeled_images/images"
annotations_dir = "./new_labeled_images/labels"

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Load YOLO model
try:
    model = YOLO("C:/Users/MSJAIUser/PycharmProjects/Senior Research Project/cicada_yolo/runs/detect/train11/weights/best.pt")
except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    raise

try:
    all_images = [image_file for image_file in os.listdir(image_dir)]
    images_to_do = all_images
except Exception as e:
    logging.error(f"Error listing images in directory {image_dir}: {e}")
    images_to_do = []

# Define classes
classes = ["cicada", "eye", "wing"]
confidence_threshold = 0.6  # Minimum confidence for each class

for image_file in tqdm(images_to_do, desc="Annotating images"):
    try:
        image_path = os.path.join(image_dir, image_file)
        annotations = []

        try:
            results = model.predict(source=image_path, save=False, conf=confidence_threshold, device='cpu', verbose=False)
        except Exception as e:
            logging.error(f"Prediction error for image {image_file}: {e}")
            continue

        # Track detections for required classes
        detected_classes = {class_name: False for class_name in classes}
        class_confidences = {class_name: [] for class_name in classes}

        try:
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)  # Class ID
                    xywhn = box.xywhn[0].tolist()  # Normalized bounding box coordinates
                    conf = box.conf  # Confidence score

                    # Map class ID to class name
                    if 0 <= cls < len(classes):
                        class_name = classes[cls]
                        class_confidences[class_name].append(conf)

                        if conf >= confidence_threshold:
                            detected_classes[class_name] = True
                            # Save annotation if confidence is high enough
                            x_center, y_center, width, height = xywhn
                            annotations.append(f"{cls} {x_center} {y_center} {width} {height}")
        except Exception as e:
            logging.error(f"Error processing predictions for {image_file}: {e}")
            continue

        # Check if all required classes are detected with confidence > 0.6
        if all(detected_classes.values()):
            try:
                # Copy image to output_dir
                shutil.copy(image_path, os.path.join(output_dir, image_file))

                # Write annotations to a text file in annotations_dir
                output_file = os.path.join(annotations_dir, os.path.splitext(image_file)[0] + '.txt')
                with open(output_file, 'w') as f:
                    f.write('\n'.join(annotations))
            except Exception as e:
                logging.error(f"Error saving image or annotations for {image_file}: {e}")
        else:
            logging.info(f"Not all required classes detected with sufficient confidence in {image_file}. Skipping.")

    except Exception as e:
        logging.error(f"Unexpected error with image {image_file}: {e}")

print("Annotation process complete. Check error_log.txt for any issues encountered.")
