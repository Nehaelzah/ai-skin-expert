# AI Skin Expert — Image Classification Component

An educational CNN image-classification component extracted from the AI Skin Expert university project. The portfolio version focuses on the reproducible machine-learning workflow and deliberately excludes the original patient-facing web application, patient images, database, and trained model files.

## Important disclaimer

This repository is for educational and research experimentation only. It is not a medical device, must not be used for clinical diagnosis, and must not inform treatment decisions.

## What is included

- `src/cnn_training.py` — configurable CNN training script using image-directory class labels.
- `src/cnn_predict.py` — local prediction CLI that uses the saved class mapping.
- `data/` — dataset guidance only; no images or metadata are provided.

The model architecture retains the original three convolutional blocks (32, 64, and 128 filters) and dense classifier, while the public version corrects the original local-path assumptions and requires separate training and validation directories.

## Training

Install dependencies from `requirements.txt`, organise authorised images in separate directory trees with matching class-folder names, then run:

```bash
python src/cnn_training.py --train-dir /path/to/train --validation-dir /path/to/validation --output-model models/skin_classifier.h5
```

The script saves both the model and a `.classes.json` class mapping. Models, images, and run outputs are ignored by Git.

## Prediction

```bash
python src/cnn_predict.py --model models/skin_classifier.h5 --image /path/to/image.jpg
```

## Scope of this public release

The original Flask/MySQL prototype is not published because it requires a separate security overhaul before public release. The historical training script also used the same directory for training and validation, so no validation-performance claim is made here.

## Author

Neha Elsa Renji — Master of Data Science student, The University of Queensland.
