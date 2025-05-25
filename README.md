# Domain-Adaptation-Pipeline
<p align="center">
  <img src="diagram.png" alt="System Diagram" width="500"/>
</p>

### Checklist
- [x] Mesh creation node – **written**, **tested**, **integrated**, **documented**
- [x] RayTracing node – **written**, **tested**, **integrated**, **documented**
- [x] Habitat bridge env – **written**, **tested**, **integrated**, **documented**
- [x] Deeplab node labels – **written**, **tested**, **integrated**, **documented**
- [ ] Deeplab node finetune – **written**, _not tested_, **integrated**, **documented**
- [ ] Control node – **written**, **tested**, **integrated**, _not documented_
- [ ] Final test – _not done_


# Description

This project provides a ROS-based pipeline for unsupervised continual domain adaptation in 3D environments using ScanNet data. It generates a 3D mesh, extracts pseudo-labels from the mesh, refines them using SAM (Segment Anything Model), and prepares the data for DeepLabV3-based semantic segmentation finetuning.

> **Paper & Code References:**
> - _Unsupervised Continual Semantic Adaptation through Neural Rendering_  
>   Zhizheng Liu, Francesco Milano, Jonas Frey, Roland Siegwart, Hermann Blum, Cesar Cadena  
> - [Jonas Frey's Kimera Interfacer](https://github.com/JonasFrey96/Kimera-Interfacer)
> - [ETHZ-ASL DeepLabV3 model](https://www.research-collection.ethz.ch/handle/20.500.11850/637142)

---

## 📁 Repository Structure

```
IO_pipeline/
├── Pipeline/
│   └── Output_kimera_mesh/
├── PseudoLabels/
└── Scannet_DB/
    └── scans/
        ├── scene0000_00/
        │   ├── color/
        │   ├── deeplab_labels/
        │   ├── deeplab_labels_colored/
        │   ├── depth/
        │   ├── intrinsic/
        |   ...
```

---

## 🧩 Features

- ROS 1 Noetic integration (Ubuntu 20.04)
- DeepLabV3 segmentation with automatic preprocessing
- Label generation via ray casting and SAM
- Mocked control node to simulate and test pipeline behavior
- Output pseudo-labels are stored in ScanNet-like structure
- Designed for GPU-based acceleration (tested on RTX 3060)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/aislabunimi/domain.adaptation.3D
cd domain.adaptation.3D/catkin_ws
catkin build
source devel/setup.bash
```

### 2. Install Dependencies

- Ubuntu 20.04
- ROS Noetic
- Python >= 3.13.2
- PyTorch 2.4.1 + CUDA 12.1
- [ScanNet](http://www.scan-net.org/) dataset (placed inside `IO_pipeline/Scannet_DB/scans/`)

### 3. Preprocess Input Labels

Create DeepLab predictions for NYU40-compatible format:

```bash
python generate_all_labels.py --scenes 0-5
```

You can find the script to download the scannet database and the GTs: [ros_visual_datasets](https://github.com/micheleantonazzi/ros_visual_datasets)

### 4. Launch the Pipeline

Use the mock control node to simulate full pipeline:

```bash
roslaunch control_node start_mock.launch scene_number:=0005_00
```

Modify `start_mock.launch` to change:
- Scene
- Paths
- Voxel size
- Input/Output folders

---

## ⚙️ Components

### 🔧 Control Node (Mock)

Launches the full pipeline with the following responsibilities:
- Loads ScanNet RGBD data
- Runs Kimera meshing and saves mesh/serialization
- Generates raycast-based pseudo-labels
- Applies SAM segmentation for refinement
- Outputs stored as: `sam_{voxel_size}` inside the scene folder

### 🧠 Model Checkpoint

- Using [ETHZ-ASL DeepLabV3](https://www.research-collection.ethz.ch/handle/20.500.11850/637142/best-epoch143-step175536.ckpt) converted to `.pth`.

---

## 📦 Output Description

For each scene, you will get:
- `deeplab_labels/` — Raw labels from DeepLab
- `deeplab_labels_colored/` — RGB visualization
- `pseudo_labels_{voxel_size}/` — Pseudo-labels from raycasting
- `sam_{voxel_size}/` — SAM-refined labels

---

## 🧪 Future Work

- Finetuning DeepLabV3 automatically on the generated labels (pipeline-ready)
- Continuous online adaptation loop (WIP)

---

## 📝 Citation

If you use this codebase, please cite:

```bibtex
@article{liu2024unsupervised,
  title={Unsupervised Continual Semantic Adaptation through Neural Rendering},
  author={Liu, Zhizheng and Milano, Francesco and Frey, Jonas and Siegwart, Roland and Blum, Hermann and Cadena, Cesar},
  journal={arXiv preprint arXiv:2403.01309},
  year={2024}
}
```

---

## 🔗 Related Repos

- [kimera_interfacer](https://github.com/JonasFrey/kimera_interfacer)
- [ros_visual_datasets](https://github.com/micheleantonazzi/ros_visual_datasets)

---

## 💬 Contact

For issues or questions, feel free to open an issue or contact the maintainers.
