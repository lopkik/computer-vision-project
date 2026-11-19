import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training_set"
TRAIN_DIR = ROOT / "training_set" / "train"
TRAIN_ANNOTATIONS_PATH = TRAIN_DIR / "_annotations.coco.json"
DEFAULT_IMAGE = (
    "2013-04-15_12_50_07_jpg.rf.9bc1ed74da7af68359989f03255b0c07.jpg"
)
EMPTY_CODE = 1
OCCUPIED_CODE = 2
WORDS = 32
DESCRIPTORS_PER_SPACE = 64
MAX_VOCAB_DESCRIPTORS = 20_000
NEIGHBORS = 5


def load_dataset() -> tuple[dict[str, dict], dict[int, list[dict]]]:
    dataset = json.loads(TRAIN_ANNOTATIONS_PATH.read_text())
    image_by_name = {
        image["file_name"]: image
        for image in dataset["images"]
        if (DATA_DIR / image["file_name"]).is_file()
    }
    annotations_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        if annotation["category_id"] == EMPTY_CODE or annotation["category_id"] == OCCUPIED_CODE:
            annotations_by_image_id[annotation["image_id"]].append(annotation)
    return image_by_name, annotations_by_image_id


def crop_space(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    x, y, width, height = bbox
    left = max(0, int(x))
    top = max(0, int(y))
    right = min(image.shape[1], int(np.ceil(x + width)))
    bottom = min(image.shape[0], int(np.ceil(y + height)))
    return image[top:bottom, left:right]


def extract_descriptors(
    sift: cv.SIFT, image: np.ndarray, bbox: list[float], rng: np.random.Generator
) -> np.ndarray | None:
    crop = crop_space(image, bbox)
    if crop.size == 0:
        return None

    _, descriptors = sift.detectAndCompute(crop, None)
    if descriptors is None or len(descriptors) == 0:
        return None
    if len(descriptors) > DESCRIPTORS_PER_SPACE:
        indices = rng.choice(len(descriptors), DESCRIPTORS_PER_SPACE, replace=False)
        descriptors = descriptors[indices]
    return descriptors.astype(np.float32, copy=False)


# def build_histogram(descriptors: np.ndarray | None, vocabulary: np.ndarray) -> np.ndarray:
#     histogram = np.zeros(len(vocabulary), dtype=np.float32)
#     if descriptors is None:
#         return histogram

#     distances = np.sum(
#         (descriptors[:, np.newaxis, :] - vocabulary[np.newaxis, :, :]) ** 2,
#         axis=2,
#     )
#     word_ids = np.argmin(distances, axis=1)
#     histogram = np.bincount(word_ids, minlength=len(vocabulary)).astype(np.float32)
#     histogram /= histogram.sum()
#     return np.sqrt(histogram)


def predict_occupancy(
    feature: np.ndarray, training_features: np.ndarray, training_labels: np.ndarray
) -> int:
    distances = np.linalg.norm(training_features - feature, axis=1)
    class_scores = np.zeros(3, dtype=np.float64)
    for label in (EMPTY_CODE, OCCUPIED_CODE):
        class_distances = distances[training_labels == label]
        neighbor_count = min(NEIGHBORS, len(class_distances))
        nearest = np.partition(class_distances, neighbor_count - 1)[:neighbor_count]
        class_scores[label] = np.mean(1.0 / np.maximum(nearest, 1e-6))

    predicted = int(np.argmax(class_scores[EMPTY_CODE : OCCUPIED_CODE + 1]) + EMPTY_CODE)
    return predicted


def run() -> None:
    image_by_name, annotations_by_image = load_dataset()
    print(f"Loaded {len(image_by_name)} images and {sum(len(v) for v in annotations_by_image.values())} annotations.")

    rng = np.random.default_rng(7)
    sift = cv.SIFT.create()
    samples: list[tuple[np.ndarray, int]] = []
    vocabulary_descriptors: list[np.ndarray] = []

    for name, image_record in list(image_by_name.items())[:10]:
        target_annotations = annotations_by_image[image_record["id"]]
        if not target_annotations:
            raise ValueError(f"No parking-space annotations found for {name}")
        
        image = cv.imread(str(TRAIN_DIR / name), cv.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Could not read training image: {TRAIN_DIR / name}")

        for annotation in target_annotations:
            descriptors = extract_descriptors(sift, image, annotation["bbox"], rng)
            if descriptors is None:
                continue
            samples.append((descriptors, annotation["category_id"]))
            vocabulary_descriptors.append(descriptors)

        correct = 0
        predicted_occupied = 0
        actual_occupied = 0
        for annotation in target_annotations:
            descriptors = extract_descriptors(sift, image, annotation["bbox"], rng)
            # feature = build_histogram(descriptors, vocabulary)
            prediction = OCCUPIED_CODE  # Placeholder for prediction logic
            # prediction = predict_occupancy(
            #     feature, training_features, training_labels
            # )
            actual = annotation["category_id"]
            correct += prediction == actual
            predicted_occupied += prediction == OCCUPIED_CODE
            actual_occupied += actual == OCCUPIED_CODE
    
    
        print(f"Image: {name}")
        print(f"Predicted occupied spaces: {predicted_occupied}")
        print(f"Annotated occupied spaces: {actual_occupied}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
