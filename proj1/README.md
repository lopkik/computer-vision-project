# Mini Project 1 (Conventional Algorithm)

This project explores conventional computer-vision methods for parking-space
analysis. The SIFT occupancy demo learns a Bag-of-Visual-Words vocabulary from
balanced, labeled parking-space crops in `training_set/train`, then classifies
all annotated parking spaces in `training_set/train` with a nearest-neighbor
classifier. The COCO annotations provide the parking-space locations; SIFT
classifies each crop as empty or occupied.

## Run the occupancy demo

Install the Python dependencies if needed:

```bash
python -m pip install opencv-python numpy
```

From the repository root, run:

```bash
python proj1/sift.py
```

The script prints accuracy aggregated across all annotated spaces in the
training split. It does not generate an output image. Since this evaluation
includes the examples used to train the classifier, this is training accuracy,
not a measure of held-out performance.

Training is balanced between empty and occupied spaces and capped at 1,500
crops per class by default. Adjust the training sample cap with
`--max-samples-per-class`.
