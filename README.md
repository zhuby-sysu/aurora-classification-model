# Aurora Image Classification Model

This repository contains the code used to train, evaluate, and interpret the deep learning model for auroral image classification as described in the paper:  
"High-Accuracy Aurora Image Classification with Swin Transformer and Active Learning" (submitted to Earth and Space Science).

This specific implementation provides a robust, fully supervised training pipeline using a **Swin Transformer** backbone. It incorporates **5-Fold Cross-Validation** for reliable performance estimation and **Grad-CAM** for model interpretability, allowing researchers to visualize which parts of the aurora image drive the model's predictions.

## 📦 Dependencies

The code requires the following Python packages:

- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision
- timm (for Swin Transformer)
- scikit-learn
- pandas
- opencv-python (cv2)
- matplotlib, seaborn
- tqdm
- Pillow (PIL)
- scipy

We recommend using a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

## 🌟 Key Features

- ✅ **State-of-the-art backbone**: Swin-Tiny Transformer adapted for 128×128 image inputs.
- ✅ **Robust Evaluation Framework**: Implements 5-Fold Cross-Validation to ensure reliable and generalized performance metrics across the dataset.
- ✅ **Model Interpretability (Grad-CAM)**: Automatically generates attention heatmaps overlaying original images, visualizing the specific features the model uses to classify aurora types.
- ✅ **Comprehensive Visualization & Reporting**: 
  - Multi-class ROC curves with AUC scores.
  - Detailed Confusion Matrices.
  - Convergence/Training curves (Loss & Accuracy) across folds.
  - Pandas-driven tabular reporting for per-class Precision, Recall, and F1-Score.
- ✅ **Full reproducibility**: Fixed random seeds and deterministic CuDNN settings to ensure consistent results across runs.

## 💻 System Requirements

- Python ≥ 3.8
- Optional but highly recommended: NVIDIA GPU with CUDA support for accelerated training and heatmap generation.

## 📚 Citation

title={Swin transformer: Hierarchical vision transformer using shifted windows},
author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
year={2021}

title={Auroral image classification with deep neural networks},
author={Kvammen, Andreas and Wickstr{\o}m, Kristoffer and McKay, Derek and Partamies, Noora},
journal={Journal of Geophysical Research: Space Physics},
year={2020}
