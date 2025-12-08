# App Usage Detection Using Encrypted LTE/5G Traffic

> Cellular traffic fingerprinting attacks, in which an unprivileged adversary passively monitors encrypted wireless channels to infer user activities, introduce significant privacy risks by giving attackers the ability to track user behaviors, infer sensitive activities, and profile victims without authorization. Although such attacks have been discussed for LTE and 5G, many existing studies rely on idealized assumptions that fall short when faced with the complexities of real-world practical scenarios.
> In this paper, we present the first practical traffic fingerprinting attack leveraging a Man-in-the-Middle (MITM) Relay in an operational cellular network. Implemented with open-source software, our attack allows a passive adversary to identify user applications with up to 99.02% accuracy, even under noisy conditions. We evaluate our method using 40 applications across five categories on multiple COTS user equipment (UE). Our approach further demonstrates the ability to infer fine-grained user activities such as browsing, messaging, and video streaming under practical constraints, including partial traffic knowledge and app version drift. The attack also achieves cross-device and cross-network transferability, and it remains robust in open-world scenarios where only a subset of application traffic is known to the adversary.
> We additionally propose a novel traffic regularization-based defense tailored specifically for cellular networks. This defense operates as an optional, backward-compatible security layer integrated seamlessly into the existing cellular protocol stack, effectively balancing security strength with practical considerations such as latency and bandwidth overhead.

This repository contains the implementation of the **mobile application fingerprinting attack model** proposed in the following paper:

```bibtex
@inproceedings{wang2026whatapp,
  title     = {What-App? App Usage Detection Using Encrypted LTE/5G Traffic},
  author    = {Jinjin, WANG and Zishuai, Cheng and Mihai, Ordean and Baojiang, Cui},
  booktitle = {Proceedings on Privacy Enhancing Technologies (PoPETs)},
  volume    = {2026},
  issue     = {1},
  year      = {2026}
}
```

## About

This repository provides the implementation of our cellular traffic fingerprinting attack model. The main objective of this project is to identify mobile application usage from encrypted LTE/5G traffic without requiring privileged network access. The attack leverages traffic metadata collected from the PDCP and RLC layers, extracted from cellular network deployments using an MITM relay setup.

Our implementation contains the full pipeline for data preprocessing, model training, evaluation, and reproducibility.

## Included Models

### CNN-based Classifier

We provide the implementation of the convolutional neural network (CNN) model used in the paper, including:

* Training data construction from encrypted traffic traces
* Model training and testing scripts

### Traditional Baseline Models
Other models are also provided for comparation:
* Support Vector Machine (SVM)
* k-Nearest Neighbors (KNN)
* Multi-Layer Perceptron (MLP)

## Dataset
The dataset used in our paper is available at: https://doi.org/10.5281/zenodo.17722145

## Usage

### 1. Download and extract the dataset
```bash
wget -O /tmp/whatapp-pdcp-dataset.tar.gz \
"https://zenodo.org/records/17722145/files/whatapp-pdcp-dataset.tar.gz?download=1" && \
mkdir -p /dataset/raw && \
tar -xzf /tmp/whatapp-pdcp-dataset.tar.gz -C /dataset/raw && \
rm /tmp/whatapp-pdcp-dataset.tar.gz
```

### 2. Run the CNN Model

To train and evaluate the CNN-based fingerprinting classifier:
```bash
cd cnn
./run_cnn.sh /dataset # /dataset is the dataset root directory
```

### 3. Run Baseline Models
For SVM, KNN, or MLP models:

```bash
cd compared_models
python3 run_models.py --input [DATASET_DIR] --model [MODEL_TYPE]
```

Where:
`DATASET_DIR` is the dataset root directory, and `MODEL_TYPE` ∈ {svm, knn, mlp}

Example:
```bash
python3 run_models.py --input "/dataset" --model svm
```