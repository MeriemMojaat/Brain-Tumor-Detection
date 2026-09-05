# Deep Learning for Brain Tumor Detection in X-ray Scans

This project uses deep learning to classify brain MRI/X-ray images as **Healthy** or **Brain Tumor**. It compares a custom-built Convolutional Neural Network (CNN) against a transfer-learning approach using **VGG16**, and includes data exploration, augmentation, training, evaluation, and hyperparameter tuning.

## 📖 Overview

The notebook (`BrainTumorVF.ipynb`) walks through the full pipeline for a binary medical image classification task:

1. **About the Data** – background on brain tumors and the dataset.
2. **Imports & Setup** – TensorFlow/Keras and supporting libraries.
3. **Data Loading & Preprocessing** – reading images, resizing to 224x224, counting samples per class, and inspecting image size distribution.
4. **Data Visualization** – class balance and sample image display.
5. **Data Processing** – train/test split and data augmentation (rotation, shift, shear, zoom, flip).
6. **Custom CNN Model** – a 3-block convolutional network trained from scratch with class-weighting to handle imbalance.
7. **Pretrained CNN Model (VGG16)** – transfer learning using VGG16 as a frozen feature extractor with a custom classification head, plus later fine-tuning of the top layers.
8. **Model Comparison** – classification reports, confusion matrices, and accuracy/loss curves for both models.
9. **Hyperparameter Tuning** – learning rate/batch size search with Keras Tuner, plus manual/grid-search tuning experiments for the CNN.

## 🗂 Dataset

- **Classes:** `Healthy`, `Brain Tumor`
- **Expected folder structure:**
  ```
  Brain Tumor Data Set/
  ├── Healthy/
  └── Brain Tumor/
  ```
- Images are resized to **224x224** pixels (required input size for VGG16).

> ⚠️ **Note:** The notebook currently references local Windows paths (e.g. `C:\Users\yosrc\Downloads\...`) for the dataset and the VGG16 pretrained weights file (`vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5`). Update these paths to match your own environment before running.

## 🛠 Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- OpenCV (`cv2`)
- Matplotlib
- Seaborn
- scikit-learn
- Keras Tuner (`pip install keras-tuner`)

Install the core dependencies with:

```bash
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn keras-tuner
```

## ▶️ How to Run

1. Download/organize the dataset into the `Healthy` / `Brain Tumor` folder structure described above.
2. Download the VGG16 no-top weights file, or let Keras fetch ImageNet weights automatically.
3. Update the dataset and weights file paths in the notebook to match your local setup.
4. Open `BrainTumorVF.ipynb` in Jupyter Notebook / JupyterLab and run the cells sequentially.

## 📊 Models

| Model | Approach |
|---|---|
| Custom CNN | 3 convolutional blocks (32 → 64 → 128 filters) + dense layers, trained from scratch with RMSprop and class weighting |
| VGG16 (Transfer Learning) | Frozen VGG16 base + GlobalAveragePooling + Dense head, later fine-tuned on the top layers with a low learning rate |

Both models are evaluated using classification reports (precision, recall, F1-score) and confusion matrices, and their training/validation curves are compared.

## 📌 Notes

- `EarlyStopping` is used during training to prevent overfitting.
- Class weighting is applied to address any imbalance between the Healthy and Brain Tumor classes.
- Hyperparameter tuning (learning rate, batch size, filters, dropout rate) is explored via Keras Tuner and manual/grid-search experiments toward the end of the notebook.

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not a certified medical diagnostic tool and should not be used for actual clinical decision-making.
