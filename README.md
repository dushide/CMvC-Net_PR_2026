## CMvC-Net

CMvC-Net is a framework that unfolds a three-component multi-view reconstruction into a fixed-depth network. Unconstrained blocks use learnable momentum, and the two Stiefel factors use a closed-form SVD Procrustes update. 

### Datasets

Download the datasets from <https://drive.google.com/drive/folders/1lsbbjSlvksqIgL48QfDul0zpgghlxRPT?usp=drive_link> and place the `.mat` files in `data/`.

### Requirements

- Python 3
- torch
- numpy
- scipy
- scikit-learn
- tqdm
- h5py

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
