# MultilabelResNet

This project is a **multilabel image classification model** using PyTorch and ResNet18.  
It predicts **4 attributes per image** from a dataset of 972 images, handling missing labels (`NA`) and class imbalance.

---

## Live Demo
Try it online: [Hugging Face Space](https://huggingface.co/spaces/AmitSharma99/multilabelResNet)  
Upload an image to see predicted attributes.

---

## Dataset
- 972 images in `images/`
- Labels in `labels.txt`
- Each image has 4 attributes: `1 = present`, `0 = absent`, `NA = unknown`

Example:  

---

## Approach
- **Model:** ResNet18 pretrained on ImageNet  
- **Loss:** BCEWithLogitsLoss with masking for `NA` and pos_weight for imbalance  
- **Training:** Fine-tuned; handles missing labels with masks  

---

