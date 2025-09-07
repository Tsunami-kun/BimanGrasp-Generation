# BimanGrasp-Dataset
This is the official repository for the BimanGrasp-Dataset release of our paper


<p align="center">
  <h2 align="center">Bimanual Grasp Synthesis for Dexterous Robot Hands</h2>


<p align="center">
    <strong>Yanming Shao</strong></a>
    ·
    <strong>Chenxi Xiao*</strong>
 </p>
 
<h3 align="center">RA-L 24' | Transferred to ICRA 25'</h3>

<p align="center">
    <a href="https://arxiv.org/abs/2411.15903">
      <img src='https://img.shields.io/badge/Paper-green?style=for-the-badge&logo=adobeacrobatreader&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
</p>

# BimanGrasp Generation

BimanGrasp is a differentiable optimization framework for generating stable bimanual dexterous grasps on diverse objects. The system leverages force closure estimation and SDF computing to produce physics-plausible bimanual grasps. 

## Quick Installation

### Setup

This project requires Conda and a CUDA-enabled GPU. The installation script has been tested on a desktop with an NVIDIA RTX 4090 and a server with an NVIDIA A40.

```bash
bash install.sh
```

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
