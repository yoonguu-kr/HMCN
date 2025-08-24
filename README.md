# HMCN: Hierarchical Multi-scale Contrastive Network for Brain Image Registration

This repository contains the implementation of **HMCN (Hierarchical Multi-scale Contrastive Network)**, a deep learning framework for deformable 3D brain MRI registration.  
The network builds upon multi-resolution strategies, probabilistic modeling, and contrastive feature learning to improve alignment accuracy while maintaining smooth and diffeomorphic transformations.

---

## 🔑 Key Features
- **Multi-level registration**: Three-stage hierarchical structure (lvl1 → lvl2 → lvl3) for coarse-to-fine registration  
- **Loss functions**:
  - Local normalized cross-correlation (NCC) and multi-resolution NCC
  - Smoothness loss and Jacobian determinant regularization
  - Contrastive loss for robust feature representation
  - Variational KL divergence loss for probabilistic modeling
- **Probabilistic model** for learning mean and variance of deformation fields
- **Data augmentation**: random flip, rotation, crop, Gaussian noise
- **Evaluation metrics**: Dice score, Hausdorff distance, ASSD, precision, recall, sensitivity, specificity

---

## 📂 Repository Structure
```
├── HMCN.py                # Model architectures, transforms, loss functions
├── Train_HMCN.py          # Training pipeline (lvl1, lvl2, lvl3)
├── config.py              # Configuration file (dataset paths, hyperparameters)
├── Data_augmentation.py   # Data augmentation methods
├── Functions.py           # Utility functions (dataloader, metrics, save/load, etc.)
```

---

## ⚙️ Configuration
Edit `config.py` to set dataset paths, GPU configuration, and training parameters.  
Example (OASIS dataset):
```python
_C.DATASET.DATA_PATH = 'YOUR DATASET PATH'
_C.DATASET.DATA_PATH_IMGS = _C.DATASET.DATA_PATH + 'specific imgs path'
_C.DATASET.DATA_PATH_LABELS = _C.DATASET.DATA_PATH + 'specific imgs label path'
```

Key solver parameters:
- `LEARNING_RATE = 1e-5`
- `CHECKPOINT = 5000`
- `hyp_loss_smooth = 3.5`
- `hyp_loss_CL = 10`
- `hyp_loss_kl = 1e-3`
- `FREEZE_STEP = 2000`

---

## 🚀 Training
Run the training script for hierarchical levels:

```bash
python Train_HMCN.py
```

By default, training starts at **level 1**, and progressively incorporates pretrained models for level 2 and level 3.  
Checkpoints and logs are automatically saved in the `Result/` directory.

---

## 📊 Evaluation
The framework supports multi-class Dice score and other segmentation-based metrics.  
Example Dice calculation:
```python
from Functions import dice_multi
dice_score, dice_ratio = dice_multi(pred_volume, gt_volume)
```

---

## 📚 Reference
This code is inspired by and extends upon **LapIRN (Laplacian Pyramid Networks for Unsupervised Deformable Image Registration)** by Tony M.  
Original LapIRN repository: [https://github.com/cwmok/LapIRN](https://github.com/cwmok/LapIRN)

If you use this code, please also cite the LapIRN work.

---

## 🙌 Acknowledgments
- **LapIRN (Tony M.)** for the foundational hierarchical registration framework.  
- **MedPy** and **PyTorch** for image metrics and deep learning utilities.  
