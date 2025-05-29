# Wavelet-Enhanced UNet for Image Segmentation

![Wavelet-UNet](https://img.shields.io/badge/model-wavelet--unet-blue)  
*A PyTorch-based implementation of a UNet image segmentation model enhanced with wavelet-transformed feature channels.*

## 🧠 Overview

This project proposes a wavelet-enhanced UNet architecture for image segmentation tasks, specifically targeting improvements in **boundary precision** and **contour clarity**. By incorporating **wavelet decomposition** (Haar DWT) to extract multi-scale edge and texture features, the model gains a stronger understanding of object shapes and structural details in complex images.

This approach is particularly beneficial in tasks where fine-grained boundary detection is crucial, such as:

- 🏥 Medical image segmentation  
- 🏙️ Building footprint extraction from aerial/satellite imagery  
- 🚗 Scene parsing for autonomous driving  
- 📡 Signal or radio map analysis  

---

## 📌 Key Features

- 🔍 **Wavelet Feature Fusion**  
  Uses a single-level 2D Haar Discrete Wavelet Transform (DWT) to extract four subbands:  
  - **LL**: Low-frequency (approximate image structure)  
  - **LH**, **HL**, **HH**: High-frequency bands capturing horizontal, vertical, and diagonal edges  

- 🧩 **Multi-channel Input Strategy**  
  The original grayscale image is concatenated with the three high-frequency subbands (LH, HL, HH) to form a **5-channel composite input**, enabling the network to receive both semantic and structural information.

- 🧠 **UNet Architecture**  
  Based on the original UNet with encoder-decoder symmetry and skip connections for precise feature reconstruction. No major structural change is required—only the input layer is adapted for 5-channel input.

- 🛠️ **Simple Integration**  
  Minimal code changes are needed to adopt this technique into standard UNet pipelines. The wavelet transform is applied in a preprocessing step.

- 📈 **Strong Performance**  
  The model significantly outperforms the vanilla UNet in edge-sensitive segmentation metrics, especially in contour clarity.

---

## 📊 Quantitative Results

Evaluated on the **RadioMapSeer** dataset using standard image quality metrics:

| Metric   | Score   |
|----------|---------|
| **NMSE** | 0.0135  |
| **RMSE** | 0.0536  |
| **SSIM** | 0.9820  |
| **PSNR** | 26.01 dB |

These results confirm that the wavelet-enhanced UNet is more accurate and robust, especially in preserving fine details and edges.

---

## 🧪 Dataset: RadioMapSeer

The RadioMapSeer dataset provides high-resolution signal distribution maps and is commonly used in:

- Wireless environment modeling  
- Indoor/outdoor radio propagation analysis  
- Spatial structure segmentation tasks

This dataset is ideal for testing algorithms that benefit from spatial pattern recognition and edge preservation.

**Preprocessing Pipeline**:

1. Normalize image to [0, 1]
2. Apply 1-level 2D Haar wavelet transform
3. Upsample LH, HL, HH subbands to original size
4. Concatenate with original image as 5-channel input

---

## 🏗️ Model Architecture

The enhanced model retains the classical UNet structure with these changes:

- **Input**: 5-channel image (original + 3 high-frequency wavelet bands)
- **No changes to convolutional structure** except input layer adaptation
- **End-to-end trainable** using standard losses (e.g., MSE, Dice)
- **Enhanced performance** on edge reconstruction due to explicit multi-scale feature input

---

## 🏃 Training Highlights

- **Loss Function**: Mean Squared Error (MSE) or Dice Loss  
- **Optimizer**: Adam (LR = 0.001)  
- **Training Epochs**: 100  
- **Batch Size**: 16  
- **Device**: NVIDIA RTX 4090 GPU  
- **Precision**: Mixed precision training (AMP) for efficiency  
- **Stability**: Adversarial Weight Perturbation (AWP) to improve generalization  

---

## 🖼️ Example Results

<div align="center">
  <img src="visual/gt_0_0.png" alt="Wavelet UNet Results" width="600"/>
</div>

Above are the prediction results showing the ground-truth labels and corresponding output from the wavelet-enhanced UNet. The predicted contours are sharp and structurally accurate.

---

## 🔧 Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- NumPy  
- OpenCV  
- Matplotlib  
- PyWavelets  

To install dependencies:

```bash
pip install -r requirements.txt
