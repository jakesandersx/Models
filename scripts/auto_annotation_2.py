import os
import logging
from typing import List, Dict
import torch
import shutil
from ultralytics import YOLO
from tqdm import tqdm
import random

logging.basicConfig(
    filename="error_log.txt",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class YOLOAnnotator:
    def __init__(
            self,
            model_path: str,
            image_dir: str,
            output_dir: str,
            annotations_dir: str,
            classes: List[str] = ["periodical cicada", "blue eye", "fungus", "red eye", "wing"],
            confidence_threshold: float = 0.8,
            max_images: int = 10000
    ):
        for path in [image_dir, output_dir, annotations_dir]:
            os.makedirs(path, exist_ok=True)

        self.model_path = model_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.annotations_dir = annotations_dir
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.max_images = max_images

        self.device = self._get_device()

        self.model = self._load_model()

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _load_model(self) -> YOLO:
        try:
            model = YOLO(self.model_path)
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def annotate_images(self):
        # Get all valid image files
        all_images = [
            img for img in os.listdir(self.image_dir)
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ]

        # Randomly select up to max_images
        images_to_process = random.sample(all_images, min(len(all_images), self.max_images))

        for image_file in tqdm(images_to_process, desc="Annotating images"):
            self._process_single_image(image_file)

        print(
            f"Annotation process complete. Processed {len(images_to_process)} images. Check error_log.txt for any issues.")

    def _process_single_image(self, image_file: str):
        try:
            image_path = os.path.join(self.image_dir, image_file)

            results = self.model.predict(
                source=image_path,
                save=False,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )

            annotations = self._extract_annotations(results)

            if self._is_periodical_cicada_detected(annotations):
                shutil.copy(image_path, os.path.join(self.output_dir, image_file))
                self._save_annotations(image_file, annotations)

        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")

    def _extract_annotations(self, results) -> List[str]:
        annotations = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                xywhn = box.xywhn[0].tolist()
                conf = float(box.conf)

                if 0 <= cls < len(self.classes) and conf >= self.confidence_threshold:
                    annotations.append(f"{cls} {xywhn[0]} {xywhn[1]} {xywhn[2]} {xywhn[3]} {conf}")

        return annotations

    def _is_periodical_cicada_detected(self, annotations: List[str]) -> bool:
        for annotation in annotations:
            cls_id = int(annotation.split()[0])
            if self.classes[cls_id] == "periodical cicada":
                return True
        return False

    def _save_annotations(self, image_file: str, annotations: List[str]):
        output_file = os.path.join(
            self.annotations_dir,
            os.path.splitext(image_file)[0] + '.txt'
        )

        with open(output_file, 'w') as f:
            f.write('\n'.join(annotations))


def main():
    annotator = YOLOAnnotator(
        model_path="C:/Users/MSJAIUser/PycharmProjects/Senior Research Project/cicada_yolo/runs/detect/train11/weights/best.pt",
        image_dir="G:/photos",
        output_dir="C:/Users/MSJAIUser/PycharmProjects/Senior Research Project/cicada_yolo/new_labeled_images2/images",
        annotations_dir="C:/Users/MSJAIUser/PycharmProjects/Senior Research Project/cicada_yolo/new_labeled_images2/labels",
        max_images=10000
    )
    annotator.annotate_images()


if __name__ == "__main__":
    main()