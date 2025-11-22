# 🏭 Vision-Based Quality Inspection Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-red)
![ONNX](https://img.shields.io/badge/Inference-ONNX_Runtime-lightgrey)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

> **An end-to-end automated defect detection system capable of identifying manufacturing anomalies with 97% accuracy and <40ms latency.**

---

## 📺 Demo
*(Replace this line with your GIF! e.g., `![Demo Preview](demo.gif)`) - Shows the system classifying "Good" vs "Defective" parts in real-time.*

---

## 📖 Project Overview
In high-speed manufacturing, manual visual inspection is error-prone and slow. This project implements a computer vision pipeline to automate quality control.

Using **Transfer Learning (ResNet50)** on the MVTec AD dataset, the model detects subtle surface defects (scratches, cracks). The pipeline is optimized for edge deployment using **ONNX Runtime**, achieving a **3x speedup** over standard PyTorch inference, and is fully containerized via **Docker**.

### 🚀 Key Features
* **High Accuracy:** Achieved **96.4% validation accuracy** using transfer learning and weighted loss functions to handle class imbalance.
* **Edge Optimization:** Reduced inference latency from **120ms** (PyTorch) to **~35ms** (ONNX) on CPU.
* **Robustness:** Implemented data augmentation (Rotation, Color Jitter) to handle lighting variations on the factory floor.
* **Deployable:** Fully containerized application with X11 forwarding support for visual debugging inside Docker.

---

## 📊 Performance Metrics

| Metric | PyTorch (Baseline) | ONNX Runtime (Optimized) |
| :--- | :--- | :--- |
| **Model Size** | 98 MB | 98 MB |
| **Inference Time (CPU)** | ~120 ms / frame | **~35 ms / frame** |
| **FPS** | ~8 FPS | **~28 FPS** |
| **Validation Accuracy** | - | **96.42%** |

---

## 🛠️ Tech Stack

* **Data Engineering:** Python, OpenCV, NumPy
* **Model Training:** PyTorch, Torchvision (ResNet50 backbone)
* **Inference Engine:** ONNX Runtime (ORT), OpenCV
* **Deployment:** Docker, Shell Scripting
* **Dataset:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (Hazelnut/Bottle categories)

---

## 📂 Repository Structure

```text
Vision-Quality-Inspection/
├── dataset/                 # Raw images (GitIgnored)
├── models/
│   ├── defect_model.pth     # Trained PyTorch Weights
│   └── defect_detector.onnx # Optimized ONNX Model
├── src/
│   ├── prepare_data.py      # ETL pipeline for MVTec data
│   ├── train.py             # ResNet50 training script
│   ├── export_onnx.py       # Model conversion script
│   └── inference.py         # Real-time production loop
├── Dockerfile               # Multi-stage build definition
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```
# ⚡ Getting Started

This guide helps you set up and run the Vision Quality Inspection project using either a **local Python environment** or a **Docker container**.

---

## 🚀 Option A: Local Installation (Python)

### 1. Clone the repository
```bash
git clone https://github.com/MananxRobin/vision-quality-inspection.git
cd vision-quality-inspection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data & Train
Download the MVTec AD dataset (e.g., Bottle) and extract it.

Then run the pipeline:
```bash
python src/prepare_data.py  # Structure the data
python src/train.py         # Train the model
python src/export_onnx.py   # Convert to ONNX
```

### 4. Run Inference
```bash
python src/inference.py
```

## 🐳 Option B: Docker Deployment (Recommended)
Simulate a production environment using Docker.

### 1. Build the Image
```bash
docker build -t vision-quality .
```

### 2. Run with Camera Access
(Linux / Windows / macOS with X11)

Note:
macOS users need XQuartz installed and must 

run: xhost +localhost

```bash
docker run --rm -it \
  -e DISPLAY=host.docker.internal:0 \
  --device /dev/video0:/dev/video0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  vision-quality
```
## 🧠 Technical Decisions & Trade-offs
### 1. Why ResNet50?

While lighter models like MobileNet exist, ResNet50 provided the necessary depth to capture subtle textural defects (scratches/dents) that shallower networks missed. The residual connections prevent vanishing gradients during fine-tuning.

### 2. Handling Class Imbalance

Manufacturing data is inherently imbalanced (mostly "Good" parts).

Solution: I implemented WeightedRandomSampler and passed calculated class weights to the CrossEntropyLoss function (Weight 4.0 for Defects) to penalize false negatives heavily.

### 3. ONNX vs. PyTorch in Production

PyTorch relies on a dynamic computation graph which adds overhead. Converting to ONNX allowed for static graph optimizations (Constant Folding, Node Fusion), resulting in a 60% reduction in latency, enabling the system to keep up with conveyor belt speeds.

