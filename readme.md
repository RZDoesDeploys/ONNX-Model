# Running ONNX Image Classification Models on Raspberry Pi

This guide describes how to use ONNX image classification models on a Raspberry Pi using three approaches:

- `javascript/`: Run ONNX models entirely using Node.js (tested on 32-bit Raspberry Pi OS).
- `python/`: Run inference using Python (for ARM-supported Raspberry Pi OS).
- `python-js/`: Hybrid approach using Python for image preprocessing and Node.js for model inference.

---

## Requirements

Install the necessary dependencies for your selected approach. For all approaches, ensure you have:

## 1. JavaScript-only (Folder: `javascript/`)

This setup uses **Node.js** and the `onnxruntime-node` and `sharp` package to run inference.

### Tested on:
- Raspberry Pi OS (32-bit) using Node.js 22 [Download Link](https://www.raspberrypi.com/software/operating-systems/#raspberry-pi-desktop)

## 2. Python-only (Folder: `python/`)

This setup uses **Python** and the `onnxruntime` and `Pillow` package to run inference.

### Tested on:
- Raspberry PI ARM64 using Python 3.11.11

## 2. Python-JS (Folder: `python-js/`)

Use this only when you are getting error in doing cmake build of package components.

This setup uses **Python** for image processing and converting it into numpy array and **Node.js**  to run inference.

### Tested on:
- Raspios Buster ARM64 [Download Link](https://downloads.raspberrypi.com/raspios_arm64/images/raspios_arm64-2020-08-24/)