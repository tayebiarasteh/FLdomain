# Mind the Gap: Federated Learning Broadens Domain Generalization in Diagnostic AI Models



Overview
------

* This is the official repository of the paper **Mind the Gap: Federated Learning Broadens Domain Generalization in Diagnostic AI Models**.


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

S. Tayebi Arasteh et al. Mind the Gap: Federated Learning Broadens Domain Generalization in Diagnostic AI Models (2023).

### BibTex

    @article {fldo2023,
      author = {Tayebi Arasteh, Soroosh and others},
      title = {Mind the Gap: Federated Learning Broadens Domain Generalization in Diagnostic AI Models},
      year = {2023},
    }
