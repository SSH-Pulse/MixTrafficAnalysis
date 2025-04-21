# MixTrafficGen

**MixTrafficGen** is a mixed traffic generation tool designed to facilitate research and reproducibility in the field of tunneled traffic analysis. This repository provides the implementation of our proposed algorithm, along with utilities for feature extraction and dataset generation.

🔗 **Related paper**: _[Insert the title of your paper here]_  
📄 **Citation**: _[Add BibTeX citation if available]_

---

## 🚀 Features

- Implementation of **Algorithm 1** as described in our paper.
- Utilities to generate tunneled mixed traffic with **fine-grained packet-level labels**.
- Traffic feature extraction tools for downstream analysis and classification.
- Example configurations and usage scripts.

## 🚀 Dataset
- https://drive.google.com/drive/folders/1NMGLJ12LbhFJbVLc8550gjw-Y62RNNJH?usp=sharing

---

## 📦 Installation

```bash
git clone https://github.com/SSH-Pulse/MixTrafficGen.git
cd MixTrafficGen
pip install -r requirements.txt

MixTrafficGen/
├── algorithm/              # Implementation of Algorithm 1
├── feature_extraction/     # Tools for packet-level feature extraction
├── data/                   # Sample PCAP files or generated traffic
├── configs/                # Configuration files for traffic generation
├── docs/                   # Documentation and usage examples
├── mix_traffic_gen.py      # Main script for traffic generation
├── extract_features.py     # Script for feature extraction
└── README.md               # This file

