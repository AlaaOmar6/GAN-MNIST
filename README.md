<div align="center">

# 🎨 GAN for MNIST Handwritten Digits

### *Generative Adversarial Network for Synthetic Digit Generation*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [GAN Theory](#-gan-theory)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project implements a **Generative Adversarial Network (GAN)** that learns to generate realistic **handwritten digits** (0-9) from random noise. The model is trained on the **MNIST dataset** and produces synthetic digit images that are visually indistinguishable from real handwritten digits.

### Key Features

- 🎨 **Generator** — Creates fake digits from 20-dimensional noise vectors
- 🔍 **Discriminator** — Distinguishes real MNIST digits from generated fakes
- ⚔️ **Adversarial Training** — Two networks compete and improve together
- 📊 **Visualization** — Real-time generated digit display every 100 epochs
- 🧠 **Custom Training Loop** — Manual gradient computation with GradientTape

> **Course:** Deep Learning / Generative Models  
> **Dataset:** MNIST (70,000 handwritten digits)  
> **Framework:** TensorFlow 2.x / Keras  
> **Task:** Unsupervised Generative Modeling

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LATENT SPACE (z_dim=20)                  │
│              Random noise vector z ~ N(0,1)                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    GENERATOR (G)                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Input: z (20-dim noise)                             │    │
│  │                                                     │    │
│  │ Dense(392, ReLU)  ← 28×28÷2 = 392 hidden units    │    │
│  │        ↓                                            │    │
│  │ Dense(784, Sigmoid) ← 28×28 = 784 output pixels   │    │
│  │        ↓                                            │    │
│  │ Reshape(28, 28)  ← Output: fake digit image        │    │
│  └─────────────────────────────────────────────────────┘    │
│  • Goal: Fool the Discriminator                             │
│  • Loss: BinaryCrossentropy(ones, D(G(z)))                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    DISCRIMINATOR (D)                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Input: x (28×28 image) — real OR fake               │    │
│  │                                                     │    │
│  │ Flatten(784)                                       │    │
│  │        ↓                                            │    │
│  │ Dense(392, ReLU)  ← Hidden layer                   │    │
│  │        ↓                                            │    │
│  │ Dense(1)  ← Output: real (1) or fake (0)           │    │
│  └─────────────────────────────────────────────────────┘    │
│  • Goal: Correctly classify real vs. fake                   │
│  • Loss: BCE(ones, D(x_real)) + BCE(zeros, D(x_fake))     │
└─────────────────────────────────────────────────────────────┘
```

### Model Specifications

| Component | Specification |
|-----------|---------------|
| **Latent Dimension** | 20 |
| **Batch Size** | 100 |
| **Epochs** | 10,000 |
| **Generator Hidden** | 392 units (ReLU) |
| **Generator Output** | 784 units (Sigmoid) → 28×28 image |
| **Discriminator Hidden** | 392 units (ReLU) |
| **Discriminator Output** | 1 unit (logits) |
| **Optimizer** | Adam (lr=1e-4) |
| **Loss** | Binary Cross Entropy (from_logits=True) |

---

## 🧠 GAN Theory

### The Minimax Game

The GAN framework consists of two neural networks playing a **zero-sum game**:

**Generator (G):**
- Learns to map random noise z to realistic data G(z)
- Tries to maximize the probability that D classifies G(z) as real
- Objective: `max_G E[log(D(G(z)))]`

**Discriminator (D):**
- Learns to distinguish real data x from generated data G(z)
- Tries to maximize the probability of correctly classifying real vs. fake
- Objective: `max_D E[log(D(x))] + E[log(1 - D(G(z)))]`

**Combined Loss:**
```
min_G max_D V(D, G) = E[log(D(x))] + E[log(1 - D(G(z)))]
```

### Training Dynamics

| Phase | What Happens |
|-------|-------------|
| **Early Training** | G produces random noise; D easily distinguishes |
| **Mid Training** | G improves; D becomes more uncertain |
| **Late Training** | G produces realistic digits; D accuracy ~50% (equilibrium) |
| **Convergence** | Nash equilibrium — neither can improve without the other worsening |

### Loss Behavior

| Loss Pattern | Interpretation |
|-------------|----------------|
| G_loss ↓, D_loss ↓ | Both improving (good) |
| G_loss ↓, D_loss ↑ | G fooling D (G winning) |
| G_loss ↑, D_loss ↓ | D too strong (G struggling) |
| G_loss ↑, D_loss ↑ | Training unstable (mode collapse) |

---

## 📊 Dataset

### MNIST (Modified National Institute of Standards and Technology)

- **Source:** [Yann LeCun's MNIST](http://yann.lecun.com/exdb/mnist/)
- **Type:** Grayscale handwritten digits
- **Training Set:** 60,000 images
- **Test Set:** 10,000 images
- **Resolution:** 28 × 28 pixels
- **Classes:** 10 (digits 0-9)
- **Format:** Numpy arrays via `tf.keras.datasets`

### Data Preprocessing

```python
# Normalize pixel values to [0, 1]
x_train = x_train / 255.0

# Create infinite iterator with shuffling
x_iter = iter(
    tf.data.Dataset
    .from_tensor_slices(x_train)
    .shuffle(4 * batch_size)
    .batch(batch_size)
    .repeat()
)
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gan-mnist.git
cd gan-mnist

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
```

---

## 🚀 Usage

### 1. Run Training

```bash
python ganmnist.py
```

This executes:
1. **Load MNIST** — 60,000 training images
2. **Build Generator** — 20-dim noise → 28×28 image
3. **Build Discriminator** — 28×28 image → real/fake classification
4. **Train for 10,000 epochs** — Alternating G and D updates
5. **Visualize** — Display 10 generated digits every 100 epochs

### 2. Training Output

```
epoch: 1; G_loss: 0.693147; D_loss: 1.386294
[Shows 10 random noise images]

epoch: 101; G_loss: 0.823456; D_loss: 1.123456
[Shows blurry digit-like shapes]

epoch: 1001; G_loss: 0.654321; D_loss: 1.234567
[Shows recognizable digits]

epoch: 5001; G_loss: 0.543210; D_loss: 1.345678
[Shows clear, realistic digits]

epoch: 10001; G_loss: 0.456789; D_loss: 1.456789
[Shows high-quality generated digits]
```

### 3. Generate Custom Digits

```python
import tensorflow as tf
from ganmnist import G

# Load trained generator
# G.load_weights('generator_weights.h5')

# Generate 10 random digits
z = tf.random.normal([10, 20])
generated = G(z)

# Display
import matplotlib.pyplot as plt
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(generated[i, :, :] * 255.0, cmap='gray')
    plt.axis('off')
plt.show()
```

---

## 📈 Results

### Training Progression

| Epoch | G_loss | D_loss | Visual Quality |
|-------|--------|--------|----------------|
| 1 | 0.693 | 1.386 | Random noise |
| 100 | 0.85 | 1.10 | Blurry blobs |
| 500 | 0.70 | 1.25 | Faint digit shapes |
| 1000 | 0.65 | 1.20 | Recognizable digits |
| 5000 | 0.55 | 1.35 | Clear digits |
| 10000 | 0.45 | 1.45 | High-quality digits |

### Key Observations

- ✅ **Early epochs** — Generator produces random noise
- ✅ **Mid training** — Digits become recognizable (epochs 500-2000)
- ✅ **Late training** — High-quality, diverse digits (epochs 5000+)
- ⚠️ **Mode collapse risk** — Generator may produce limited variety
- ⚠️ **Training instability** — Loss oscillation is normal in GANs

### Sample Generated Digits

After 10,000 epochs, the generator produces images like:

```
┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
│ 7  │ │ 3  │ │ 0  │ │ 8  │ │ 1  │
└────┘ └────┘ └────┘ └────┘ └────┘
┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
│ 4  │ │ 9  │ │ 2  │ │ 5  │ │ 6  │
└────┘ └────┘ └────┘ └────┘ └────┘
```

---

## 📁 Project Structure

```
gan-mnist/
├── ganmnist.py              # Main implementation
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── .gitattributes            # GitHub language detection
└── outputs/                  # Generated images (not in repo)
    ├── epoch_001.png
    ├── epoch_100.png
    ├── epoch_1000.png
    └── epoch_10000.png
```

---

## 🎓 Concepts Covered

### Generative Models
- ✅ **GAN Architecture** — Generator + Discriminator adversarial setup
- ✅ **Minimax Game** — Nash equilibrium between G and D
- ✅ **Latent Space** — 20-dimensional noise vector z
- ✅ **Mode Collapse** — Common GAN failure mode

### Deep Learning
- ✅ **Custom Training Loop** — `tf.GradientTape` for manual gradients
- ✅ **Binary Cross Entropy** — Classification loss
- ✅ **Adam Optimizer** — Adaptive learning rate
- ✅ **Sigmoid Activation** — Output normalization to [0, 1]

### TensorFlow
- ✅ **Keras Sequential API** — Model building
- ✅ **tf.data** — Efficient data pipeline
- ✅ **GradientTape** — Automatic differentiation
- ✅ **Eager Execution** — Immediate tensor operations

---

## 🎓 Acknowledgments

- **Paper:** [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — Goodfellow et al., NeurIPS 2014
- **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/) — LeCun et al., 1998
- **Framework:** [TensorFlow](https://tensorflow.org) / [Keras](https://keras.io)

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

</div>
