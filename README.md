# Traffic-Sign Recognition & Auditory Guidance for Elderly Users

This project builds a **lightweight traffic-sign classifier** and a simple **audio prompt** pipeline designed to assist elderly or low-vision users. It trains a compact **TinyCNN** on the **GTSRB** dataset (43 classes), produces **paper-ready figures**, and optionally speaks predictions using **Text-to-Speech (TTS)**. A minimal **real-time** script is included to combine an (optional) YOLOv8-nano detector + the classifier + voice on a webcam feed.

<img width="40" height="40" alt="Screenshot 2025-08-04 at 21 55 36" src="https://github.com/user-attachments/assets/849bc6af-cdfa-4937-bef1-2bacd4ac696a" />

<img width="40" height="40" alt="Screenshot 2025-08-04 at 21 55 49" src="https://github.com/user-attachments/assets/0f684dbf-50a7-4c23-8f65-30049912ad84" />

<img width="40" height="40" alt="Screenshot 2025-08-04 at 21 56 16" src="https://github.com/user-attachments/assets/ce8f8cd1-acad-4586-9bbb-c13910860b9f" />




---

## System Overview

1. **Input**: an image (or live video frame).  
2. **(Optional) Detect**: YOLOv8-nano finds sign boxes in the frame.  
3. **Classify**: cropped signs are resized (48×48) and classified by a TinyCNN into one of 43 GTSRB classes.  
4. **Speak**: the predicted label is mapped to a short phrase (e.g., “**Stop sign**” / “**Speed limit 50 kilometers per hour**”) and spoken via TTS.

The full pipeline is optimized for simplicity and runs smoothly in **Google Colab** or locally on **CPU/GPU**.

---

## Key Features

- **Beginner-friendly training**: one Colab notebook from setup → training → testing → saving weights.  
- **Lightweight model**: TinyCNN (~0.6M params) reaches **~80%+** test accuracy on GTSRB in minutes.  
- **Paper assets**: automatic 3×3 **prediction grid** and accuracy prints for reports.  
- **Auditory output**: optional **gTTS** (notebook) or **pyttsx3** (local, offline) to speak sign names.  
- **Real-time option**: demo script that connects (optional) YOLOv8-nano + TinyCNN + TTS to a webcam.

---

## Dataset

- **GTSRB** (German Traffic Sign Recognition Benchmark): 43 classes.  
  TorchVision can download it automatically inside the notebook.  
- The notebook includes safe transforms (resize, light augmentation) and a 90/10 train/val split.

---




---

## Software Architecture

- **Language**: Python 3.10+  
- **Core ML**: PyTorch (`torch`), TorchVision (`torchvision`)  
- **Visualization**: Matplotlib  
- **TTS**: `gTTS` (Colab/desktop) and `pyttsx3` (offline for local real-time)  
- **(Optional) Detection**: Ultralytics YOLOv8-nano  
- **Demo UI**: Gradio (optional)

---


### Run in **Google Colab** (recommended for first run)

1. Open the notebook: `notebooks/traffic_Sign_Recognition_For_Elderly.ipynb`.  
2. **Runtime → Change runtime type → GPU** (optional but faster).  
3. Run the cells top-to-bottom. The notebook will:
   - install matching PyTorch/TorchVision wheels,  
   - download **GTSRB**,  
   - train **TinyCNN**,  
   - evaluate on the test set,  
   - save **`weights/tinycnn_fp32.pt`**,  
   - create **`figures/sample_predictions_grid.png`**,  
   - (optional) speak predictions with **gTTS**.


---

