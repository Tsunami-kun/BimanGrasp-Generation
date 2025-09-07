# BimanGrasp-Dataset
This is the official repository for the open-source implementation of the BimanGrasp Algorithm for synthesizing bimanual grasps on 3D objects of our RA-L and ICRA 2025 paper:

<p align="center">
  <h2 align="center">Bimanual Grasp Synthesis for Dexterous Robot Hands</h2>


<p align="center">
    <strong>Yanming Shao</strong></a>
    ·
    <strong>Chenxi Xiao*</strong>
 </p>
 
<h3 align="center">RA-L 24' | presented at ICRA 25'</h3>

<p align="center">
    <a href="https://arxiv.org/abs/2411.15903">
      <img src='https://img.shields.io/badge/Paper-green?style=for-the-badge&logo=adobeacrobatreader&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
</p>

# BimanGrasp Generation

BimanGrasp is a differentiable optimization framework for generating stable bimanual dexterous grasps on diverse objects. The system leverages force closure estimation and SDF computing to produce physics-plausible bimanual grasps. 

---

## Installation

### Option 1: Automatic Installation (Recommended)

The easiest way for installation is to run the shell script:

```bash
bash install.sh
```

This will create a conda env named **bimangrasp**, installing **PyTorch 2.1.0** with **CUDA 11.8** support, **PyTorch3D (v0.7.8)**, and third-party dependencies (**TorchSDF** and **pytorch\_kinematics**).

### Option 2: Manual Installation

You can install everything step by step.

1. **Create and activate Conda environment**

   ```bash
   conda create -n bimangrasp python=3.8 -y
   conda activate bimangrasp
   ```

2. **Install PyTorch (CUDA 11.8 support)**

   ```bash
   conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

3. **Install PyTorch3D**

   ```bash
   pip install https://github.com/facebookresearch/pytorch3d/archive/refs/tags/V0.7.8.tar.gz
   ```

4. **Install other dependencies**

   ```bash
   conda install -c conda-forge transforms3d trimesh plotly rtree -y
   pip install urdf_parser_py scipy networkx tensorboard six
   ```

5. **Build and install TorchSDF**

   ```bash
   cd thirdparty/TorchSDF
   bash install.sh
   cd ../..
   ```

6. **Install pytorch\_kinematics**

   ```bash
   cd thirdparty/pytorch_kinematics
   pip install -e .
   cd ../..
   ```

---

## Usage

```bash
# Generate bimanual grasps
python main.py

# Visualize results
python visualization.py --object_code <object_name> --num <grasp_index>
```

While BimanGrasp-Generation is able to work with any 3D object mesh, this repository contains a mini demo on 5 randomly sample objects. To prepare your own objects, you could follow the asset processing script and instructions by DexGraspNet: https://github.com/PKU-EPIC/DexGraspNet/tree/main/asset_process

## Demo Visualization

This demo visualizations on the 5 objects are tested on an A40 GPU without cherry picking.


| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets\figs\Breyer_Horse_Of_The_Year_2015_0_screenshot.png" width="100%"> | <img src="assets\figs\Breyer_Horse_Of_The_Year_2015_1_screenshot.png" width="100%"> | <img src="assets\figs\Breyer_Horse_Of_The_Year_2015_2_screenshot.png" width="100%"> | <img src="assets\figs\Breyer_Horse_Of_The_Year_2015_3_screenshot.png" width="100%"> |
| <img src="assets\figs\Cole_Hardware_Dishtowel_Multicolors_0_screenshot.png" width="100%"> | <img src="assets\figs\Cole_Hardware_Dishtowel_Multicolors_1_screenshot.png" width="100%"> | <img src="assets\figs\Cole_Hardware_Dishtowel_Multicolors_2_screenshot.png" width="100%"> | <img src="assets\figs\Cole_Hardware_Dishtowel_Multicolors_3_screenshot.png" width="100%"> |
| <img src="assets\figs\Curver_Storage_Bin_Black_Small_0_screenshot.png" width="100%"> | <img src="assets\figs\Curver_Storage_Bin_Black_Small_1_screenshot.png" width="100%"> | <img src="assets\figs\Curver_Storage_Bin_Black_Small_2_screenshot.png" width="100%"> | <img src="assets\figs\Curver_Storage_Bin_Black_Small_3_screenshot.png" width="100%"> |
| <img src="assets\figs\Hasbro_Monopoly_Hotels_Game_0_screenshot.png" width="100%"> | <img src="assets\figs\Hasbro_Monopoly_Hotels_Game_1_screenshot.png" width="100%"> | <img src="assets\figs\Hasbro_Monopoly_Hotels_Game_2_screenshot.png" width="100%"> | <img src="assets\figs\Hasbro_Monopoly_Hotels_Game_3_screenshot.png" width="100%"> |
| <img src="assets\figs\Schleich_S_Bayala_Unicorn_70432_0_screenshot.png" width="100%"> | <img src="assets\figs\Schleich_S_Bayala_Unicorn_70432_1_screenshot.png" width="100%"> | <img src="assets\figs\Schleich_S_Bayala_Unicorn_70432_2_screenshot.png" width="100%"> | <img src="assets\figs\Schleich_S_Bayala_Unicorn_70432_3_screenshot.png" width="100%"> |


## To-Do

- [ ] Release validation code for both simulation and real-world validation.

## Acknowledgments

We would like to express our gratitude to the authors of the following repository, from which we referenced code:

* [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet/tree/main)

## Dataset Repo

Our released BimanGrasp-Dataset is in this repo: [[BimanGrasp-Dataset](https://github.com/Tsunami-kun/BimanGrasp-Dataset)].

## License
The Project is under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (LICENSE.md).

## Citation

If you find this code useful, please consider citing:

```bibtex
@article{shao2024bimanual,
  title={Bimanual grasp synthesis for dexterous robot hands},
  author={Shao, Yanming and Xiao, Chenxi},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```

## License
The Project is under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (LICENSE.md).
