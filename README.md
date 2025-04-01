# 📌 Anomaly Detection for Line Clearance in Industrial Systems

Welcome to the **Line-Clearance** repository! 🎯 This project is the result of my master thesis in **Artificial Intelligence**, where I explored the **study, creation, and potential deployment of an AI-based Line Clearance system** for industrial applications.

⚠️ Note: This repository contains the core elements of my work. Preliminary and failed tests have been excluded to keep things clean and relevant.

---

## 🔍 What is Line Clearance?

Line clearance is a critical procedure in industrial settings, ensuring that production equipment and workspaces are free from residual materials, documents, or contaminants **before switching to a new batch**.

This process consists of **three main stages**:

✅ **Clearing** – Removing leftover materials, labels, and packaging from the previous batch.  
🧼 **Cleaning** – Disinfecting and drying all surfaces and equipment.  
👀 **Checking** – Performing a thorough inspection before restarting production.  

In industries like **pharmaceuticals**, this procedure is **essential** to prevent contamination. However, **manual inspection isn't foolproof**—errors happen. That’s where AI comes in!

The goal is to **automate the checking phase** using an **Anomaly Detection system**, making inspections faster, more accurate, and less prone to human error. 🚀

---

## 🛠 How to Use This Project

This repository includes implementations of **two powerful Anomaly Detection models**:
- 🟢 **DDAD** (Denoising Diffusion Anomaly Detection)
- 🔵 **DRAEM** (Discriminatively trained Reconstruction Anomaly Embedding Model)

📂 You’ll find detailed usage instructions inside the `README` files of their respective directories.

---

## 🔬 Workflow Overview

Here’s a high-level breakdown of the steps followed to implement and evaluate the models.

### 📸 1. Data Acquisition
This project was developed in an industrial setting using a real machine. **All dataset images were captured on-site**, ensuring real-world applicability.

📌 **Note:** The dataset is not included in this repository, but you can test the models on publicly available benchmarks like [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

### 🏷️ 2. Data Labeling
Anomalous images were labeled using the **[VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/)**. The labels were saved in `.csv` format and converted into 2D masks for evaluation.

### 🎭 3. Masking

Since some images contained **high detail or motion blur**, the models tended to generate **false positives**. To counteract this, **heavy masking** was applied where part of the anomaly could be masked.

**Why?** 🤔 The goal wasn’t perfect segmentation but **helping the user locate the anomaly**. Even if some anomaly parts were masked, as long as the AI highlighted the defective areas, it was considered a success.

#### ⏳ When is Masking Applied?
Two approaches were tested:
  - 1️⃣ **Before Anomaly Map Computation** – Masking occurs before any analysis, reducing unwanted noise.  
  - 2️⃣ **After Anomaly Map Computation** – Masked regions are set to `0` in the anomaly map to prevent false detections.

### 🔄 4. Image Registration

To simulate real-world conditions, some dataset images were captured **with slight shifts in camera angles**. This mimics an operator accidentally moving the camera.

**Image registration** was applied to **align all images** to a common reference frame, improving model consistency. It can be applied using the `image_registration.py` file, which will save the modified images in a copy of the dataset.

### 🎚️ 5. Threshold Selection (Finding the Sweet Spot)

Thresholding is **crucial** in anomaly detection! Too low, and **false positives explode**; too high, and **real anomalies are missed**. 🚦

Following the **[MVTec AD paper](https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf)**, three thresholding methods were tested:
- **Maximum Threshold** 
- **p-Quantile Threshold** 
- **k-Sigma Threshold** 

### 📊 6. Evaluation Metrics

To rigorously assess the models, multiple **performance metrics** were used:
- **Accuracy**
- **AUROC (Area Under the Receiver Operating Characteristic Curve)** – Image & Pixel Level
- **PRO (Per-Region Overlap)**
- **AP (Average Precision)**
- **IoU (Intersection over Union)**

---

### ⭐ Special files
This repo also has some special files, they ease the train/evaluation of multiple networks, namely:
  - **train_all.py** 📈 - Automates training across all dataset categories for both networks, updating the .yaml file and saving results automatically.
  - **detect_all.py** 📊 - Evaluates trained models for both networks across different techniques (masking, image registration, etc.), storing evaluation metrics, graphs, and curves. ⚠️ Be mindful of storage space!
  - **compare_networks.py** 🏆 - Compares the two networks by analyzing their ability on finding the anomaly in the images, stores all the information in the directory `networks_comparison`.

## 🚀 Conclusion

This project demonstrates how AI can **enhance industrial processes** by automating defect detection, reducing human errors, and improving efficiency. With further refinement, such **Anomaly Detection systems** could play a **crucial role in modern manufacturing**! 🏭🤖

💡 *Want to dive deeper? Check out the code and start experimenting!*

📩 Feel free to reach out for discussions or feedback.

