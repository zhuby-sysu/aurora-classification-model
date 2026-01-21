# Aurora Image Classification Model

This repository contains the code used to train and evaluate the deep learning model for auroral image classification as described in the paper:  
"High-Accuracy Aurora Image Classification with Swin Transformer and Active Learning" (submitted to Earth and Space Science).

This repository implements an **active learning framework** for classifying aurora images using a **Swin Transformer** backbone. The goal is to achieve competitive performance with significantly fewer labeled samples compared to full supervision.

## 📦 Dependencies

The code requires the following Python packages:

- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision
- timm (for Swin Transformer)
- scikit-learn
- matplotlib, seaborn
- tqdm
- Pillow (PIL)
- scipy
We recommend using a virtual environment (e.g., `venv` or `conda`) to manage dependencies.


# Citation
title={Swin transformer: Hierarchical vision transformer using shifted windows},
author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
year={2021}

title={Auroral image classification with deep neural networks},
author={Kvammen, Andreas and Wickstr{\o}m, Kristoffer and McKay, Derek and Partamies, Noora},
journal={Journal of Geophysical Research: Space Physics},
year={2020}
