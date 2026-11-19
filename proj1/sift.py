import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training_set"
TRAIN_DIR = DATA_DIR / "train"
EMPTY_CODE = 1
OCCUPIED_CODE = 2
WORDS = 32
DESCRIPTORS_PER_SPACE = 64
MAX_VOCAB_DESCRIPTORS = 20_000
NEIGHBORS = 5
DEFAULT_MAX_SAMPLES_PER_CLASS = 1_500


def load_dataset(
    split: str,
) -> tuple[dict[str, dict], dict[int, list[dict]]]:
    split_dir = DATA_DIR / split
    annotations_path = split_dir / "_annotations.coco.json"
    dataset = json.loads(annotations_path.read_text())
    image_by_name = {
        image["file_name"]: image
        for image in dataset["images"]
        if (split_dir / image["file_name"]).is_file()
    }
    annotations_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        if annotation["category_id"] in (EMPTY_CODE, OCCUPIED_CODE):
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


def select_training_annotations(
    annotations_by_image_id: dict[int, list[dict]],
    max_samples_per_class: int,
    rng: np.random.Generator,
) -> list[tuple[int, dict]]:
    selected: list[tuple[int, dict]] = []
    for label in (EMPTY_CODE, OCCUPIED_CODE):
        samples = [
            (image_id, annotation)
            for image_id, annotations in annotations_by_image_id.items()
            for annotation in annotations
            if annotation["category_id"] == label
        ]
        if not samples:
            raise ValueError(f"No training samples found for category {label}")
        count = min(max_samples_per_class, len(samples))
        indices = rng.choice(len(samples), size=count, replace=False)
        selected.extend(samples[int(index)] for index in indices)
    return sorted(selected, key=lambda sample: sample[0])


def build_vocabulary(
    descriptors: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    if not descriptors:
        raise ValueError("SIFT found no descriptors in the selected training spaces")
    all_descriptors = np.concatenate(descriptors, axis=0)
    if len(all_descriptors) < WORDS:
        raise ValueError(
            f"Need at least {WORDS} SIFT descriptors to build the vocabulary; "
            f"found {len(all_descriptors)}"
        )
    if len(all_descriptors) > MAX_VOCAB_DESCRIPTORS:
        indices = rng.choice(
            len(all_descriptors), size=MAX_VOCAB_DESCRIPTORS, replace=False
        )
        all_descriptors = all_descriptors[indices]

    cv.setRNGSeed(7)
    _, _, vocabulary = cv.kmeans(
        all_descriptors,
        WORDS,
        np.zeros((len(all_descriptors), 1), dtype=np.int32),
        (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.1),
        3,
        cv.KMEANS_PP_CENTERS,
    )
    return vocabulary


def build_histogram(
    descriptors: np.ndarray | None, vocabulary: np.ndarray
) -> np.ndarray:
    histogram = np.zeros(len(vocabulary), dtype=np.float32)
    if descriptors is None or len(descriptors) == 0:
        return histogram

    squared_distances = (
        np.sum(descriptors * descriptors, axis=1, keepdims=True)
        + np.sum(vocabulary * vocabulary, axis=1)[np.newaxis, :]
        - 2 * descriptors @ vocabulary.T
    )
    word_ids = np.argmin(squared_distances, axis=1)
    histogram = np.bincount(word_ids, minlength=len(vocabulary)).astype(np.float32)
    histogram /= histogram.sum()
    return np.sqrt(histogram)


def predict_occupancy(
    feature: np.ndarray, training_features: np.ndarray, training_labels: np.ndarray
) -> int:
    if len(training_features) == 0 or len(training_features) != len(training_labels):
        raise ValueError("Training features and labels must be non-empty and aligned")

    distances = np.linalg.norm(training_features - feature, axis=1)
    neighbor_count = min(NEIGHBORS, len(distances))
    nearest_indices = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
    scores = {
        label: float(
            np.sum(
                1.0
                / np.maximum(
                    distances[nearest_indices][training_labels[nearest_indices] == label],
                    1e-6,
                )
            )
        )
        for label in (EMPTY_CODE, OCCUPIED_CODE)
    }
    if scores[EMPTY_CODE] >= scores[OCCUPIED_CODE]:
        return EMPTY_CODE
    return OCCUPIED_CODE


def train_classifier(
    image_by_name: dict[str, dict],
    annotations_by_image_id: dict[int, list[dict]],
    max_samples_per_class: int,
    rng: np.random.Generator,
) -> tuple[cv.SIFT, np.ndarray, np.ndarray, np.ndarray]:
    sift = cv.SIFT.create()
    image_by_id = {record["id"]: record for record in image_by_name.values()}
    selected = select_training_annotations(
        annotations_by_image_id, max_samples_per_class, rng
    )
    print(
        f"Extracting SIFT features from {len(selected)} balanced training spaces "
        f"({len(selected) // 2} per class)"
    )

    samples: list[tuple[np.ndarray | None, int]] = []
    vocabulary_descriptors: list[np.ndarray] = []
    current_image_id: int | None = None
    image: np.ndarray | None = None
    for index, (image_id, annotation) in enumerate(selected, start=1):
        if image_id != current_image_id:
            image_record = image_by_id[image_id]
            image_path = TRAIN_DIR / image_record["file_name"]
            image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
            if image is None:
                raise OSError(f"Could not read training image: {image_path}")
            current_image_id = image_id

        if image is None:
            raise RuntimeError("Training image was not loaded")
        descriptors = extract_descriptors(sift, image, annotation["bbox"], rng)
        samples.append((descriptors, annotation["category_id"]))
        if descriptors is not None:
            vocabulary_descriptors.append(descriptors)
        if index % 500 == 0 or index == len(selected):
            print(f"  processed {index}/{len(selected)} training spaces")

    vocabulary = build_vocabulary(vocabulary_descriptors, rng)
    training_features = np.vstack(
        [build_histogram(descriptors, vocabulary) for descriptors, _ in samples]
    )
    training_labels = np.array([label for _, label in samples], dtype=np.int32)
    return sift, vocabulary, training_features, training_labels


def predict_image(
    image_path: Path,
    annotations: list[dict],
    sift: cv.SIFT,
    vocabulary: np.ndarray,
    training_features: np.ndarray,
    training_labels: np.ndarray,
) -> tuple[int, int]:
    img = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Could not read training image: {image_path}")
    if not annotations:
        raise ValueError(f"No parking-space boxes found for {image_path.name}")

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rng = np.random.default_rng(7)
    correct = 0
    for annotation in annotations:
        descriptors = extract_descriptors(sift, img_gray, annotation["bbox"], rng)
        feature = build_histogram(descriptors, vocabulary)
        prediction = predict_occupancy(feature, training_features, training_labels)
        correct += prediction == annotation["category_id"]
    return correct, len(annotations)


def evaluate_dataset(
    image_dir: Path,
    image_by_name: dict[str, dict],
    annotations_by_image_id: dict[int, list[dict]],
    sift: cv.SIFT,
    vocabulary: np.ndarray,
    training_features: np.ndarray,
    training_labels: np.ndarray,
) -> tuple[int, int]:
    correct = 0
    total = 0
    image_count = len(image_by_name)
    for index, (image_name, image_record) in enumerate(
        image_by_name.items(), start=1
    ):
        annotations = annotations_by_image_id.get(image_record["id"], [])
        if annotations:
            image_correct, image_total = predict_image(
                image_dir / image_name,
                annotations,
                sift,
                vocabulary,
                training_features,
                training_labels,
            )
            correct += image_correct
            total += image_total
        if index % 100 == 0 or index == image_count:
            print(f"  evaluated {index}/{image_count} training images")

    if total == 0:
        raise ValueError(f"No parking-space annotations found in {image_dir}")
    return correct, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SIFT occupancy predictions on the full training split."
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_CLASS,
        help="Maximum number of training crops per class (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.max_samples_per_class < 1:
        parser.error("--max-samples-per-class must be at least 1")
    return args


def run() -> None:
    args = parse_args()
    train_images, train_annotations = load_dataset("train")
    if not train_images:
        raise ValueError("No readable images found in training_set/train")

    rng = np.random.default_rng(7)
    sift, vocabulary, training_features, training_labels = train_classifier(
        train_images,
        train_annotations,
        args.max_samples_per_class,
        rng,
    )
    correct, total = evaluate_dataset(
        TRAIN_DIR,
        train_images,
        train_annotations,
        sift,
        vocabulary,
        training_features,
        training_labels,
    )

    accuracy = correct / total
    print(f"Training-set accuracy: {correct}/{total} ({accuracy:.1%})")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
