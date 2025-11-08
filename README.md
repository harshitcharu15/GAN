# Conditional GAN on CelebA (Blond Hair Attribute)

This project implements a **Conditional Generative Adversarial Network (cGAN)** to generate human face images conditioned on the **Blond Hair** attribute from the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).  
The implementation is inspired by David Foster’s *Generative Deep Learning (2nd Edition)* and based on the official example [`03_cgan/cgan.ipynb`](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/04_gan/03_cgan/cgan.ipynb).

---

## 🧠 Project Overview

A **cGAN** extends the standard GAN architecture by introducing an auxiliary variable *y* that conditions both the generator and the discriminator, enabling direct control over the generated output (Mirza & Osindero, 2014).

In this experiment:
- The **generator** receives a random noise vector *z* concatenated with a binary label *y* (Blond Hair = 1, Not Blond = 0).
- The **critic/discriminator** takes the input image concatenated with a label mask channel corresponding to *y*.
- The model learns to produce realistic facial images with or without blond hair depending on the condition.

---

## ⚙️ Architecture

**Generator**
- Input: Noise vector (128D) + Label (1D)
- Layers: Transposed Convolutions (DCGAN-style)
- Output: 64×64 RGB image
- Activation: ReLU + Tanh

**Critic / Discriminator**
- Input: Image (3×64×64) + Label Mask (1×64×64)
- Layers: Convolutions with InstanceNorm
- Output: Single real/fake scalar
- Activation: LeakyReLU

**Loss Function**
- Wasserstein GAN with Gradient Penalty (WGAN-GP)
- Gradient penalty λ = 10  
- Critic trained 5× per generator step

---

## 🧩 Dataset

**CelebA: Large-scale Face Attributes Dataset**

- 200,000+ face images, each with 40 annotated binary attributes
- For this project, only the **Blond_Hair** attribute is used
- Automatically downloaded using `torchvision.datasets.CelebA`

---

## 🧪 Training Details

| Parameter | Value |
|------------|--------|
| Image Size | 64×64 |
| Batch Size | 128 |
| Epochs | 20 |
| Learning Rate | 2×10⁻⁴ |
| Optimizer | Adam (β₁ = 0.5, β₂ = 0.999) |
| Gradient Penalty | 10 |
| Critic Updates per Step | 5 |

During training, sample grids are saved every few hundred steps under the `samples/` directory.  
Images are grouped in two rows — **top: Not Blond**, **bottom: Blond**.

---

## 💻 Code Structure
