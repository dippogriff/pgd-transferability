# pgd-transferability
Investigates adversarial attacks (PGD) on various models (a simple shallow CNN, ResNet18, ViT, Llama-Vision) using the CIFAR-10 dataset and analyzes the transferability of adversarial examples.

#### To clone the repo and set up the environment:
```bash
conda create -n pgd_transfer python=3.12
conda activate pgd_transfer
git clone https://github.com/dippogriff/pgd-transferability.git
cd pgd-transferability
pip install -r requirements.txt
```
Open attack_transfer_demo.ipynb for the usage of the code and visualizations showing the effectiveness and transferability of PGD across the different models.

Notes: 
1) If you want to retrain any of the the first 3 models, then please delete model_a.pth, model_b.pth, and/or model_c.pth. If not, the notebook will use the existing trained models.
2) I opted to download Llama-vision locally for this experiment even though it takes up a lot of storage space and is quite tedious to run the inference on one mediocre GPU. So simply don't run the notebook past the first heatmap with the first 3 simpler models if you don't want to wait around, or change the code to use the huggingface hosted model.
