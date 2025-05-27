# 3D Domain Adaptation Pipeline

## Overview

This repository provides a ROS-based pipeline for unsupervised continual domain adaptation in 3D environments using ScanNet data. It includes 3D mesh reconstruction, pseudo-label generation using ray casting, label refinement via Segment Anything Model (SAM), and preparation for DeepLabV3-based semantic segmentation fine-tuning.

- 3D mesh reconstruction using **Kimera**
- Pseudo-label generation via **ray casting**
- Label refinement using **Segment Anything Model (SAM)**
- Semantic segmentation fine-tuning with **DeepLabV3**


# Installation & Setup Guide

## System Requirements

- **Operating System**: Ubuntu 20.04 (required due to ROS Noetic)
- **ROS**: Noetic
- **GPU**: CUDA-enabled GPU (tested on NVIDIA RTX 3060)
- **Python**: 3.8 (via Conda environment recommended)


## ROS Noetic Installation

Follow official instructions to install ROS Noetic:  
https://wiki.ros.org/noetic/Installation/Ubuntu

## ScanNet Dataset

We use the ScanNet dataset for this project. To download the ScanNet data and the corresponding NYU40 labels, please use the helper repository:

https://github.com/micheleantonazzi/ros_visual_datasets

To simplify the integration with this repository and avoid modifying the launch files, we recommend recreating the following directory structure on your system while downloading the ScanNet dataset. This structure ensures minimal changes to the configuration files:

```
Domain_Adaptation_Pipeline/
├── IO_pipeline/
│   ├── Pipeline/
│   │   └── Output_kimera_mesh/
│   ├── PseudoLabels/
│   └── Scannet_DB/
│       └── scans/
│           ├── scene0000_00/
│           │   ├── color/
│           │   ├── deeplab_labels/
│           │   ├── deeplab_labels_colored/
│           │   ├── depth/
│           │   ├── intrinsic/
│           │   └── ...
│
└── domain.adaptation.3D(this repo)/
```

This setup provides a clean starting point and organizes the data consistently with the pipeline expectations. Further instructions on how to adjust paths in the launch files will follow in the appropriate sections.

## Conda Environment Setup

We recommend using a clean Conda environment to prevent conflicts between base and project-specific dependencies.

```bash
conda create -n domainadapt3d python=3.8
conda activate domainadapt3d
```

## Install Python Dependencies

Install the required packages inside the Conda environment:

```bash
# Conda packages
conda install -c conda-forge imageio pyyaml opencv scipy open3d
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Pip packages (within the conda env)
pip install catkin_pkg rospkg trimesh embreex ultralytics
```

Outside the Conda environment:

```bash
pip3 install empy==3.3.4
sudo apt install pykdl
```

Then install Catkin tools:

```bash
sudo apt install python3-catkin-tools python3-osrf-pycommon
```

## Clone the Project

```bash
git clone --recurse-submodules https://github.com/aislabunimi/domain.adaptation.3D
cd domain.adaptation.3D/catkin_ws
```

Add the project to your Python path:

```bash
export PYTHONPATH=/path/to/domain.adaptation.3D/catkin_ws/src:$PYTHONPATH
```

## Modifications to Kimera Semantics

Modify the following files:

### In `kimera_semantics_ros/src/semantic_tsdf_server.cpp`

Replace constructor definitions with:

```cpp
namespace kimera {

SemanticTsdfServer::SemanticTsdfServer(const ros::NodeHandle& nh,
                                       const ros::NodeHandle& nh_private,
                                       bool verbose)
    : SemanticTsdfServer(nh,
                         nh_private,
                         vxb::getTsdfMapConfigFromRosParam(nh_private),
                         vxb::getTsdfIntegratorConfigFromRosParam(nh_private),
                         vxb::getMeshIntegratorConfigFromRosParam(nh_private),
                         verbose) {}

SemanticTsdfServer::SemanticTsdfServer(
    const ros::NodeHandle& nh,
    const ros::NodeHandle& nh_private,
    const vxb::TsdfMap::Config& config,
    const vxb::TsdfIntegratorBase::Config& integrator_config,
    const vxb::MeshIntegratorConfig& mesh_config,
    bool verbose)
    : vxb::TsdfServer(nh, nh_private, config, integrator_config, mesh_config),
      semantic_config_(getSemanticTsdfIntegratorConfigFromRosParam(nh_private)),
      semantic_layer_(nullptr) {

  verbose_ = verbose;

  semantic_layer_.reset(new vxb::Layer<SemanticVoxel>(
      config.tsdf_voxel_size, config.tsdf_voxels_per_side));

  tsdf_integrator_ =
      SemanticTsdfIntegratorFactory::create(
        getSemanticTsdfIntegratorTypeFromRosParam(nh_private),
        integrator_config,
        semantic_config_,
        tsdf_map_->getTsdfLayerPtr(),
        semantic_layer_.get());

  CHECK(tsdf_integrator_);
}
```

### In `kimera_semantics_ros/include/kimera_semantics_ros/semantic_tsdf_server.h`

Replace main content with:

```cpp
#include "kimera_semantics_ros/semantic_tsdf_server.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "kimera_semantics");

  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  kimera::SemanticTsdfServer node(nh, nh_private);

  ros::spin();

  return EXIT_SUCCESS;
}
```

## Build ROS Packages

```bash
catkin build kimera_interfacer
catkin build control_node
catkin build label_generator_ros

source devel/setup.bash
```

Create NYU40-compatible DeepLabV3 predictions:

```bash
cd ../TestScripts
python GenerateAllLabels.py
```

You can add '--scene=00002' and '--base_path=path/to/Domain_Adaptation_Pipeline' args to specify the scene number and the correct path.

## Run the Full Pipeline

Before launching the full pipeline modify `catkin_ws/src/control_node/launch/start_mock.launch` to configure:

- Scene number
- Input/output paths
- Voxel size

To simulate a full scene pipeline:

```bash
roslaunch control_node start_mock.launch
```

## Notes on Protobuf

ROS is not compatible with recent versions of `protoc`. If you have a newer version installed:

```bash
sudo apt remove libprotobuf-dev protobuf-compiler
```

Install version 3.15.8 manually:

```bash
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.15.8/protobuf-all-3.15.8.tar.gz
tar -xzf protobuf-all-3.15.8.tar.gz
cd protobuf-3.15.8
./configure
make -j$(nproc)
sudo make install
sudo ldconfig
```

Verify:

```bash
protoc --version
```

Also, check your Conda base environment for `libprotobuf` conflicts. Rebuild proto files in Kimera if needed:

```bash
cd catkin_ws/src/kimera_semantics_ros/include/proto/
protoc --cpp_out=. semantic_map.proto
```

## Troubleshooting

- Ensure `numpy` is not imported from the base Conda env to avoid conflicts.
- Check `PYTHONPATH` ordering if packages are not found.
- Always `source devel/setup.bash` after building or modifying any code.


> **Paper & Code References:**
> - _Unsupervised Continual Semantic Adaptation through Neural Rendering_  
>   Zhizheng Liu, Francesco Milano, Jonas Frey, Roland Siegwart, Hermann Blum, Cesar Cadena  
> - [Jonas Frey's Kimera Interfacer](https://github.com/JonasFrey96/Kimera-Interfacer)
> - [ETHZ-ASL DeepLabV3 model](https://www.research-collection.ethz.ch/handle/20.500.11850/637142)


## Citation

If you use this codebase, cite the following work:

```bibtex
@article{liu2024unsupervised,
  title={Unsupervised Continual Semantic Adaptation through Neural Rendering},
  author={Liu, Zhizheng and Milano, Francesco and Frey, Jonas and Siegwart, Roland and Blum, Hermann and Cadena, Cesar},
  journal={arXiv preprint arXiv:2403.01309},
  year={2024}
}
```