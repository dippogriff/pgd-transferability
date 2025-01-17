# pgd-transferability
Investigates adversarial attacks (PGD) on various models (CNN, ResNet18, ViT, Llama-Vision) using the CIFAR-10 dataset and analyzes the transferability of adversarial examples.

#### To clone the repo and set up the environment:
```bash
conda create -n pgd_transfer python=3.12
conda activate pgd_transfer
git clone https://github.com/dippogriff/pgd-transferability.git
cd pgd-transferability
pip install -r requirements.txt
```
Open attack_transfer_demo.ipynb for the usage of the code and visualizations showing the effectiveness and transferability of PGD across the different models.
