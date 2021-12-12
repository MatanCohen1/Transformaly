# Transformaly - Exploit Vision Transformer Power for Anomaly Detection

Official PyTorch implementation of [**"Transformaly - Exploit Vision Transformer Power for Anomaly Detection"**](https://arxiv.org/abs/2112.04185).

We Introduces *Transformaly* - a transformer based architecture that uses pretrained Vision Transformer and teacher-student training for semantic anomaly detection.
In our setting, a pretrained teacher network is used to train a student network on the normal training samples. 
Since the student network is trained only on normal samples, it is expected to deviate from the teacher network in abnormal cases. This difference can serve as a complementary representation to the
pre-trained feature vector. Our method - *Transformaly* - exploits a pre-trained Vision Transformer to extract both feature vectors: the pre-trained (agnostic) features and
the teacher-student (fine-tuned) features. We report stateof-the-art AUROC results in both the common unimodal setting, where one class is considered normal and the rest are
considered abnormal, and the multimodal setting, where all classes but one are considered normal, and just one class is considered abnormal.

This Repository is based on the Vision transformer [**Pytorch implementation**](https://github.com/lukemelas/PyTorch-Pretrained-ViT) of Luke Melas.

### Setup
```
cd <path-to-Transformaly-directory>
git clone https://github.com/MatanCohen1/Transformaly.git
cd Transformaly
conda create -n transformalyenv python=3.7
conda activate transformalyenv
conda install --file requirements.txt
```

### Unimodal Training And Evaluation  
```
--dataset cifar10 --data_path ./data --epochs 30 --batch_size 32 --eval_every 5 --unimodal
--dataset cifar10 --batch_size 32 --data_path ./data --whitening_threshold 0.9 --unimodal
```

### Multimodal Training And Evaluation  
```
--dataset cifar10 --data_path ./data --epochs 30 --batch_size 32 --eval_every 5 
--dataset cifar10 --batch_size 32 --data_path ./data --whitening_threshold 0.9 
```