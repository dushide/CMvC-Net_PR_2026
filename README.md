## CMvC-Net

CMvC-Net is a framework that unfolds a three-component multi-view reconstruction into a network. This repository contains the code and datasets used for the experiments in our paper.

### Datasets

Download the datasets from <https://drive.google.com/drive/folders/1lsbbjSlvksqIgL48QfDul0zpgghlxRPT?usp=drive_link> and place the `.mat` files in `data/`.

### Requirements

- Python 3.10.18
- torch 2.5.1 (CUDA 11.8)
- numpy 2.2.6
- scipy 1.15.2
- scikit-learn 1.7.1
- tqdm 4.67.3
- h5py 3.14.0

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended. Add `--no-cuda` to train on CPU.

### Run

```bash
python run.py
```

One dataset:

```bash
python run.py --dataset_override Animals
```

### Contact

If you have any questions, please feel free to contact dushidems@gmail.com. Thanks.
