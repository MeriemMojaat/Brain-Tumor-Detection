# 🎙️ Speech Emotion Recognition (SER) with Generative Augmentation

This project implements and benchmarks multiple deep learning approaches for **Speech Emotion Recognition (SER)**, following a CRISP-DM workflow. It goes beyond a standard classifier by using **generative models** (Diffusion, VAE, CVAE, CGAN) to synthesize emotion-preserving speech audio and augment the training data, then evaluates whether this augmentation improves recognition performance.

## 📖 Overview

The notebook (`emotion.ipynb`) benchmarks its implementation against a reference research paper on emotion-aware speech enhancement, and follows the CRISP-DM methodology:

1. **Business Understanding** – goal of reconstructing/generating emotion-preserving mel-spectrograms to improve downstream SER accuracy.
2. **Data Understanding** – exploring the **RAVDESS** and **EmoDB** datasets: duration analysis, pitch/formant analysis, and class distribution.
3. **Data Preparation** – parsing filenames for emotion labels, converting audio to mel-spectrograms, normalization, augmentation (pitch shift, time stretch, pre-emphasis, noise reduction), and train/validation splitting.
4. **Modeling** – several architectures are implemented and compared:
   - **Diffusion Model (U-Net based)** – simplified DDPM-style model used to generate augmented spectrograms.
   - **ResNet-50 based SER classifier** – baseline emotion classifier, benchmarked against the reference paper's reported 98.31% accuracy.
   - **CNN-LSTM hybrid** – exploratory model for capturing temporal dependencies.
   - **VAE / Conditional VAE (CVAE)** – generative models for emotion-conditioned mel-spectrogram reconstruction and synthesis.
   - **CGAN** – CNN-based Conditional GAN for generating emotion-conditioned spectrograms.
5. **Evaluation** – models are compared using Accuracy, F1 Score, Weighted/Unweighted Accuracy (WA/UA), confusion matrices, and training/validation loss curves. A baseline ResNet (real data only) is compared against a version trained on real + diffusion-augmented data.
6. **Final Comparison & Conclusion** – a summary of all models' relative strengths and a discussion of how generative augmentation impacts SER performance.

## 🗂 Datasets

- **RAVDESS** – emotion parsed from filename segments (neutral, calm, happy, sad, angry, fearful, disgust, surprised).
- **EmoDB** – emotion parsed from a character code in the filename (anger, boredom, disgust, fear, happiness, sadness, neutral).
- Audio is converted to **mel-spectrograms** (`n_mels=80–128`, `n_fft=1024–2048`, `hop_length=256–512`), normalized, and padded/truncated to a fixed duration.

> ⚠️ **Note:** The notebook references Kaggle-specific paths (`/kaggle/input/...`, `/kaggle/working/...`) and local Windows image paths (e.g. `C:\Users\ettey\OneDrive\Desktop\dl_emotion\...` for architecture diagrams). Update these to match your environment before running outside Kaggle.

## 🛠 Requirements

- Python 3.x
- PyTorch (`torch`, `torchvision`, `torchaudio`)
- librosa
- NumPy, pandas
- Matplotlib, Seaborn, Plotly
- scikit-learn
- tqdm
- soundfile

Install the core dependencies with:

```bash
pip install torch torchvision torchaudio librosa numpy pandas matplotlib seaborn plotly scikit-learn tqdm soundfile
```

## ▶️ How to Run

1. Download the **RAVDESS** and **EmoDB** datasets and update the paths (`EMODB_PATH`, `RAVDESS_PATH`, or Kaggle-style paths) to match your local setup.
2. Run the filename-parsing cells to generate metadata CSVs for both datasets.
3. Run the preprocessing cells to convert audio into mel-spectrograms and build train/validation splits.
4. Run the modeling sections in order — Diffusion, ResNet SER classifier, VAE/CVAE, CGAN — as each later section may build on datasets/objects created earlier.
5. Review the evaluation and final comparison sections for accuracy/F1/UA metrics across all models.

## 📊 Results Summary (from the notebook)

| Model | Accuracy | F1 Score | UA (Unweighted Acc.) |
|---|---|---|---|
| Baseline ResNet (Real Only) | 0.63 | 0.65 | 0.67 |
| ResNet + Diffusion Augmented | 0.80 | 0.79 | 0.81 |

Training on real + diffusion-augmented data notably improved classifier performance over real data alone.

## 🎯 Key Takeaways

- Generative augmentation (Diffusion, CVAE, CGAN) is an effective strategy for enriching limited emotional speech datasets.
- The ResNet-50 classifier was used as a consistent evaluation backbone across experiments for fair benchmarking.
- CVAE achieved high-quality emotion-conditioned spectrogram reconstructions; CGAN produced sharp, emotion-faithful spectrograms validated by classifier accuracy.
- The CNN-LSTM hybrid underperformed, likely due to limited training and a simple architecture.

## ⚠️ Notes
- This project is for **educational and research purposes**.
