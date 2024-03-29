# Enhancing domain generalization in the AI-based analysis of chest radiographs with federated learning



Overview
------

* This is the official repository of the paper [**Enhancing domain generalization in the AI-based analysis of chest radiographs with federated learning**](https://www.nature.com/articles/s41598-023-49956-8).
* Pre-print version: [**https://doi.org/10.48550/arXiv.2310.00757**](https://doi.org/10.48550/arXiv.2310.00757).


Introduction
------
...

### Prerequisites

The software is developed in **Python 3.9**. For deep learning, the **PyTorch 2.0** framework is used.



Main Python modules required for the software can be installed from ./requirements:

```
$ conda env create -f requirements.yaml
$ conda activate fldo
```

**Note:** This might take a few minutes.


Code structure
---

Our source code for federated learning as well as training and evaluation of the deep neural networks, image analysis and preprocessing, and data augmentation are available here.

1. Everything can be run from *./main_fldo.py*. 
* The data preprocessing parameters, directories, hyper-parameters, and model parameters can be modified from *./configs/config.yaml*.
* Also, you should first choose an `experiment` name (if you are starting a new experiment) for training, in which all the evaluation and loss value statistics, tensorboard events, and model & checkpoints will be stored. Furthermore, a `config.yaml` file will be created for each experiment storing all the information needed.
* For testing, just load the experiment its model you need.

2. The rest of the files:
* *./data/* directory contains all the data preprocessing, augmentation, and loading files.
* *./Train_Valid_fldo.py* contains the training and validation processes.
* *./Prediction_fldo.py* all the prediction and testing processes.

------
### In case you use this repository, please cite the original paper:

Tayebi Arasteh, S., Kuhl, C., Saehn, MJ. et al. *Enhancing domain generalization in the AI-based analysis of chest radiographs with federated learning*. Sci Rep 13, 22576 (2023). [https://doi.org/10.1038/s41598-023-49956-8](https://doi.org/10.1038/s41598-023-49956-8).

### BibTex

    @article {fldo2023,
      author = {Tayebi Arasteh, Soroosh and Kuhl, Christiane and Saehn, Marwin-Jonathan and Isfort, Peter and Truhn, Daniel and Nebelung, Sven},
      title = {Enhancing domain generalization in the AI-based analysis of chest radiographs with federated learning},
      year = {2023},
      doi = {10.1038/s41598-023-49956-8},
      publisher = {Nature Portfolio},
      journal = {Scientific Reports}
    }
