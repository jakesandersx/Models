import os
import random
import shutil
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

labeled_images_dir = "./new_labeled_images"
images_dir = os.path.join(labeled_images_dir, "images")
labels_dir = os.path.join(labeled_images_dir, "labels")

output_base_dir = "USABLE_ANNOTATIONS_3"
train_images_dir = os.path.join(output_base_dir, "train/images")
train_labels_dir = os.path.join(output_base_dir, "train/labels")
valid_images_dir = os.path.join(output_base_dir, "valid/images")
valid_labels_dir = os.path.join(output_base_dir, "valid/labels")
test_images_dir = os.path.join(output_base_dir, "test/images")
test_labels_dir = os.path.join(output_base_dir, "test/labels")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(valid_images_dir, exist_ok=True)
os.makedirs(valid_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

random.seed(42)
random.shuffle(image_files)

n_total = len(image_files)
n_train = int(n_total * 0.85)
n_valid = int(n_total * 0.10)

train_files = image_files[:n_train]
valid_files = image_files[n_train:n_train + n_valid]
test_files = image_files[n_train + n_valid:]


def process_files(image_list, dest_images_dir, dest_labels_dir):
    for image_file in tqdm(image_list, desc=f"Processing {os.path.basename(dest_images_dir)}"):
        src_image_path = os.path.join(images_dir, image_file)
        src_label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')

        dest_image_path = os.path.join(dest_images_dir, image_file)
        dest_label_path = os.path.join(dest_labels_dir, os.path.splitext(image_file)[0] + '.txt')

        try:
            # Attempt to open and resize the image
            with Image.open(src_image_path) as img:
                img = img.resize((400, 400))
                img.save(dest_image_path)

            # Copy the label file if it exists
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dest_label_path)
            else:
                print(f"Warning: Label file missing for {image_file}")

            # Remove the original image and label files
            os.remove(src_image_path)
            if os.path.exists(src_label_path):
                os.remove(src_label_path)

        except UnidentifiedImageError:
            print(f"Error: Cannot identify image file '{src_image_path}'. Skipping.")
        except PermissionError as e:
            print(f"PermissionError: {e}. Skipping file '{src_image_path}'.")
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}. Skipping.")
        except Exception as e:
            print(f"Unexpected error with file '{src_image_path}': {e}. Skipping.")


# Process splits
process_files(train_files, train_images_dir, train_labels_dir)
process_files(valid_files, valid_images_dir, valid_labels_dir)
process_files(test_files, test_images_dir, test_labels_dir)
