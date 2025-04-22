To modify the README for the `MixTrafficGen` repository, I would suggest updating it to reflect the correct tool usage and script structure while integrating the information from your provided code (`pcap_mix_for.py`). Here's an updated version of the README that incorporates the script details:

---

# MixTrafficGen

**MixTrafficGen** is a mixed traffic generation tool designed to facilitate research and reproducibility in the field of tunneled traffic analysis. This repository provides the implementation of our proposed algorithm, along with utilities for feature extraction and dataset generation.

## 🚀 Features

- Implementation of **Algorithm 1** as described in our paper.
- Utilities to generate tunneled mixed traffic with **fine-grained packet-level labels**.
- Traffic feature extraction tools for downstream analysis and classification.
- Example configurations and usage scripts.

## 🚀 Dataset
- [Dataset Download Link](https://drive.google.com/drive/folders/1NMGLJ12LbhFJbVLc8550gjw-Y62RNNJH?usp=sharing)

## 📦 Installation

```bash
git clone https://github.com/SSH-Pulse/MixTrafficGen.git
cd MixTrafficGen
pip install -r requirements.txt
```

## 📝 Directory Structure

```
├── pcap_mix_for.py         # Script for merging and mixing pcap files
└── README.md               # This file
```

## 🛠 Usage

The primary script for merging and mixing pcap files is `pcap_mix_for.py`. It combines two pcap files, ensuring packets are merged based on their timestamp order, with special handling for SSH packets.


## 📜 Notes

- **SSH Handling**: The tool provides special handling for SSH traffic. If an `ssh.pcap` file is present in the input directories, the packets will be merged using the `twopcapmix_ssh` method, which includes tracking the mix rates.
- **Feature Extraction**: Packet-level features such as packet length and arrival time delta are extracted for analysis and classification tasks.
- **Mixing Strategy**: The tool merges packets based on timestamp order to create realistic mixed traffic scenarios.

