<!-- ##### Table of contents
1. [Installation](##-Installation-:hammer:)
2. [Data Preparation](#Data_Preparation) 
3. [Run the code](#Run_the_code)
4. [Acknowledgments](#Acknowledgments) -->

# DiverseDream: Diverse Text-to-3D Synthesis with Augmented Text Embedding [ECCV'24](https://eccv.ecva.net/)
[Uy Dieu Tran](), [Minh Luu](), [Phong Ha Nguyen](https://phongnhhn.info/), [Khoi Nguyen](https://www.khoinguyen.org), and [Binh-Son Hua](https://sonhua.github.io/)

<a href="https://diversedream.github.io/"><img src="https://img.shields.io/badge/Project Page-diversedream.github.io-blue?style=for-the-badge"></a>
<a href="https://arxiv.org/abs/2312.02192"><img src="https://img.shields.io/badge/arxiv-2312.02192-red?style=for-the-badge"></a>

<p align="center">
<img alt="diversedream" src="assets/teaser.gif" width="100%">
</p>

> **Abstract**: 
Text-to-3D synthesis has recently emerged as a new approach to sampling 3D models by adopting pretrained text-to-image models as guiding visual priors. An intriguing but underexplored problem with existing text-to-3D methods is that 3D models obtained from the sampling-by-optimization procedure tend to have mode collapses, and hence poor diversity in their results. In this paper, we provide an analysis and identify potential causes of such a limited diversity, which motivates us to devise a new method that considers the joint generation of different 3D models from the same text prompt. We propose to use augmented text prompts via textual inversion of reference images to diversify the joint generation. We show that our method leads to improved diversity in text-to-3D synthesis qualitatively and quantitatively.

![pipeline](assets/pipeline.png)

Details of the model architecture and experimental results can be found in [our paper](https://arxiv.org/abs/2312.02192):
```bibtext
@inproceedings{DiverseDream,
      title={Diverse Text-to-3D Synthesis with Augmented Text Embedding}, 
      author={Uy Dieu Tran, Minh Luu, Phong Ha Nguyen, Khoi Nguyen, Binh-Son Hua},
      year={2024},
      booktitle={ECCV},
}
```
**Please CITE** our paper whenever this repository is used to help produce published results or incorporated into other software.

## Installation
Please refer to installation guide of [HiPer](https://github.com/HiPer0/HiPer) for textual inversion and [Threestudio](https://github.com/threestudio-project/threestudio) for generating 3D object

## Training
```.bash
# Textual Inversion
cd HiPer
sh script.sh
cd ..

# Text to 3D
sh run.sh
```
## Evaluation
```.bash
cd evaluate
sh eval.sh
```

## Acknowledgements
Thanks for all contributor of public repositories [Threestudio](https://github.com/threestudio-project/threestudio), [HiPer](https://github.com/HiPer0/HiPer)