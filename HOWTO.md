## Setup
Install the libraries used by the project. 
```bash
pip install -r requirements.txt
```

Substitute Dataset Generation -- download cifar100
```bash
python knowledge_distill_dataset.py
```
Dataset Reduction:
```bash
python data_compression.py
```
Backdoor Injection:
```bash
python poison_model.py
```