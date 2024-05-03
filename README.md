# qlora-finetune
NYU Tandon Deep Learning Final Project Repository\

This repository contains the Qlora fine-tuning code for the Pythia models (2.8B, 6.9B, and 12B) using the Alpaca dataset.

This repo adapts the qlora training code from [https://github.com/artidoro/qlora](https://github.com/artidoro/qlora) and [https://github.com/tloen/alpaca-lora](https://github.com/tloen/alpaca-lora). 

### Group Members
Ali Umut Kaypak | Aashray Pola | Mohnish Bangaru \
{ak10531,ap8130,mb9628}@nyu.edu

### Repo Structure
Training code is in /src/train.py. It takes the model id, lora rank, lora alpha, number of training epoch and output directory, and train the model. 
```bash
python /src/train.py --model-id {model_id} --lora-rank {lora_rank} --number-train-epoch {number_of_training_epoch} --output-dir {output_dir}
```
Available model ids are: ['EleutherAI/pythia-12b', 'EleutherAI/pythia-2.8b', 'EleutherAI/pythia-6.9b'] 

Inference notebook for the fine-tuned models is in /src/inference.ipynb. It computes the test loss for the fine-tuned model and show the difference between intruction following capabilities of the base and the fine-tuned model on some examples. Note that model id for the base model and the lora adapter config file must match in this notebook. 

Qlora header files we trained are in /models and stored by following the convention alpaca_{model_name}_{lora_rank}, e.g alpaca_pythia-6.9b_8.

We store alpaca input structure in /templates/alpaca.json.

### Installation
We tested our code with Python 3.11 \ 
After installing the Pytorch by following the steps in [link](https://pytorch.org/get-started/locally/), install the requirements as follows.
```bash
pip install -U -r requirements.txt
```

### Results
| Model Name | Lora Rank  | Test Loss |
|:----------:|:----------:|:----------:|
pythia 12b | 8  | 1.2049 |
pythia 6.9b | 8 | |
pythia 2.8b | 8 | |

