# Classification of Alzheimer's Disease Subjects from MRI using Deep Convolutional Neural Networks

TensorFlow/Keras implementation of the convolutional neural network architectures presented in the paper:

**Classification of Alzheimer's Disease Subjects from MRI using Deep Convolutional Neural Networks**

## Task

Four-class classification of brain MRI images into:

- `MildDemented`
- `ModerateDemented`
- `NonDemented`
- `VeryMildDemented`

## Dataset

The paper uses the public Kaggle dataset:

```text
https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images
```

Expected directory structure:

```text
Alzheimer_s Dataset/
├── train/
│   ├── MildDemented/
│   ├── ModerateDemented/
│   ├── NonDemented/
│   └── VeryMildDemented/
└── test/
    ├── MildDemented/
    ├── ModerateDemented/
    ├── NonDemented/
    └── VeryMildDemented/
```

The publication reports 6,400 MRI images:

| Class | Total | Training | Testing |
|---|---:|---:|---:|
| Mild Demented | 896 | 717 | 179 |
| Moderate Demented | 64 | 52 | 12 |
| Non Demented | 3,200 | 2,560 | 640 |
| Very Mild Demented | 2,240 | 1,792 | 448 |
| **Total** | **6,400** | **5,121** | **1,279** |

## CNN Architectures

### Architecture 1

```text
(Conv2D ×2 → BatchNormalization → MaxPooling2D → Dropout) ×2
→ GlobalAveragePooling2D
→ Flatten
→ Dense(256)
→ Dropout
→ Softmax Output
```

### Architecture 2

```text
((Conv2D → BatchNormalization) ×2 → MaxPooling2D → Dropout) ×2
→ GlobalAveragePooling2D
→ Flatten
→ Dense(256)
→ Dropout
→ Softmax Output
```

### Architecture 3

```text
(Conv2D ×3 → BatchNormalization → MaxPooling2D → Dropout) ×2
→ Conv2D ×2
→ BatchNormalization
→ MaxPooling2D
→ Dropout
→ GlobalAveragePooling2D
→ Flatten
→ Dense(256)
→ Dropout
→ Softmax Output
```

## Implementation Details

The implementation follows the methodology presented in the paper and uses the following configuration:

- Input size: `224 × 224 × 3`
- Four-class softmax classification
- Convolution kernel: `3 × 3`
- Filter progression: `32 → 64 → 128`
- Activation: ReLU
- Optimizer: Adam
- Initial learning rate: `1e-3`
- Loss: Sparse categorical cross-entropy
- Block dropout: `0.25`
- Dense dropout: `0.50`
- Default epochs: `20`
- Training augmentation
- Class weighting to reduce the effect of class imbalance
- Random seed: `42`

These values can be changed from the command line or in `src/models.py`.

## Project Structure

```text
Alzheimers-Disease-MRI-CNN/
├── README.md
├── LICENSE
├── requirements.txt
├── train.py
├── evaluate.py
├── .gitignore
├── outputs/
└── src/
    ├── __init__.py
    ├── data.py
    ├── models.py
    └── utils.py
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Architecture 1:

```bash
python train.py \
  --data-dir "path/to/Alzheimer_s Dataset" \
  --architecture 1 \
  --batch-size 128 \
  --epochs 20
```

Architecture 2:

```bash
python train.py \
  --data-dir "path/to/Alzheimer_s Dataset" \
  --architecture 2 \
  --batch-size 128 \
  --epochs 20
```

Architecture 3:

```bash
python train.py \
  --data-dir "path/to/Alzheimer_s Dataset" \
  --architecture 3 \
  --batch-size 128 \
  --epochs 20
```

The publication evaluates batch sizes:

```text
128, 256
```

## Evaluation

```bash
python evaluate.py \
  --data-dir "path/to/Alzheimer_s Dataset" \
  --model-path "outputs/architecture_2/best_model.keras"
```

The evaluation script produces:

- Loss
- Accuracy
- Confusion matrix
- Classification report
- Per-class precision, recall, and F1-score

## Published Results

For batch size 128 and 20 epochs, the publication reports:

| Architecture | Loss | Accuracy |
|---|---:|---:|
| Architecture 1 | 0.1389 | 95.00% |
| Architecture 2 | 0.1098 | 95.90% |
| Architecture 3 | 0.1354 | 94.75% |

Architecture 2 achieved the strongest reported accuracy for this experimental setting.

Results may vary depending on the software environment, preprocessing pipeline, random initialization, hyperparameter configuration, and hardware platform.

## Citation

If you use this implementation in your research, please cite the original paper.

```bibtex
@inproceedings{DBLP:conf/nids/Papadimitriou0M23,
  author       = {Orestis Papadimitriou and
                  Athanasios Kanavos and
                  Phivos Mylonas and
                  Manolis Maragoudakis},
  editor       = {Katerina Kabassi and
                  Phivos Mylonas and
                  Jaime Caro},
  title        = {Classification of Alzheimer's Disease Subjects from {MRI} Using Deep
                  Convolutional Neural Networks},
  booktitle    = {Novel {\&} Intelligent Digital Systems: Proceedings of the 3rd
                  International Conference (NiDS 2023) - Volume 2, Athens, Greece, 28-29
                  September 2023},
  series       = {Lecture Notes in Networks and Systems},
  volume       = {784},
  pages        = {277--286},
  publisher    = {Springer},
  year         = {2023},
  url          = {https://doi.org/10.1007/978-3-031-44146-2\_28},
  doi          = {10.1007/978-3-031-44146-2\_28},
  timestamp    = {Mon, 03 Mar 2025 21:19:19 +0100},
  biburl       = {https://dblp.org/rec/conf/nids/Papadimitriou0M23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## License

This project is released under the MIT License.
