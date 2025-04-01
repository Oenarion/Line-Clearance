# 🔍 Anomaly Detection with Conditioned Denoising Diffusion Models

This repository contains an implementation of [DDAD](https://arxiv.org/abs/2305.15956), based on the official [GitHub repository](https://github.com/arimousa/DDAD).

![Framework](images/DDAD_Framework.png)

## 📦 Installation
To set up the required dependencies, simply run:

```bash
pip install -r requirements.txt
```

## 🚀 Training & Evaluation
The entire training and evaluation process is managed within **main.py**, offering four key functionalities:

### 🏋️ Train the Denoising UNet
```bash
python main.py --train True
```

### 🎯 Fine-Tune the Feature Extractor
```bash
python main.py --domain_adaptation True
```

### 📊 Compute Optimal Thresholds
```bash
python main.py --threshold True
```

### 🧐 Evaluate & Test the Model
```bash
python main.py --detection True
```

🔹 **Configuration:** All hyperparameters, dataset paths and other useful parameters must be defined in the `config.yaml` file. Helpful comments are included for most parameters to guide you through its structure.

## 📝 Special Scripts
Beyond the main training and evaluation process, additional scripts were created to streamline experimentation:

- **train_all.py** 📈 - Automates training across all dataset categories, updating the `.yaml` file and saving results automatically. A `training_times.txt` file logs the training duration.
- **evaluate_checkpoints.py** 📊 - Evaluates trained models across different techniques (masking, image registration, etc.), storing evaluation metrics, graphs, and curves. ⚠️ Be mindful of storage space!
- **check_best_model.py** 🏆 - Analyzes logs to determine the best-performing model based on evaluation metrics and creates a summary report.

🔹 **Important:** These scripts rely on predefined parameters at the start of each file, update them as needed to suit your setup. 
🔹 **Important:** The order of execution matters, as each step depends on the previous one's outputs.
## 📂 Dataset Structure
This project was tested on a **custom dataset** (not included in this repository). However, it can be used with benchmark datasets like [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/) or any dataset structured as follows:

```bash
Name_of_Dataset
|-- Category
|-----|----- ground_truth
|-----|----- test
|-----|--------|------ good
|-----|--------|------ ... (other anomalous categories)
|-----|----- train
|-----|--------|------ good
```
The **train** set should contain only **nominal samples** (`good`), while the **test** set should include both nominal and anomalous samples.

## 📖 Citation
If you use this work, please cite the following paper:

```bibtex
@article{mousakhan2023anomaly,
  title={Anomaly Detection with Conditioned Denoising Diffusion Models},
  author={Mousakhan, Arian and Brox, Thomas and Tayyub, Jawad},
  journal={arXiv preprint arXiv:2305.15956},
  year={2023}
}
```
