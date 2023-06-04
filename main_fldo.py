"""
Created on May 4, 2023.
main_fldo.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms, models
import timm
import numpy as np
from sklearn import metrics
# from mne.stats import fdr_correction

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_fldo import Training
from Prediction_fldo import Prediction
from data.data_provider import vindr_data_loader_2D, chexpert_data_loader_2D, mimic_data_loader_2D, cxr14_data_loader_2D, padchest_data_loader_2D

import warnings
warnings.filterwarnings('ignore')




def main_train_central_2D(global_config_path="", valid=False, resume=False, augment=False, experiment_name='name', dataset_name='vindr',
                          pretrained=False, vit=False, dinov2=True, image_size=224, batch_size=30, lr=1e-5):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/FLdomain/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        augment: bool
            if we want to have data augmentation during training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    if dataset_name == 'vindr':
        train_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        train_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        train_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        train_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        train_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    weight = train_dataset.pos_weight()
    label_names = train_dataset.chosen_labels

    if valid:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    else:
        valid_loader = None

    # Changeable network parameters
    if vit:
        if dinov2:
            model = load_pretrained_dinov2(num_classes=len(weight))
        else:
            model = load_pretrained_timm_model(num_classes=len(weight), pretrained=pretrained, imgsize=image_size)
    else:
        model = load_pretrained_timm_model(num_classes=len(weight), model_name='resnet50d', pretrained=pretrained)

    loss_function = BCEWithLogitsLoss

    model_info = params['Network']
    model_info['lr'] = lr
    model_info['batch_size'] = batch_size
    params['Network'] = model_info
    write_config(params, cfg_path, sort_keys=True)

    if vit:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                      weight_decay=float(params['Network']['weight_decay']))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr),
                                     weight_decay=float(params['Network']['weight_decay']),
                                     amsgrad=params['Network']['amsgrad'])

    trainer = Training(cfg_path, resume=resume, label_names=label_names)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader, num_epochs=params['Network']['num_epochs'])



def main_train_federated(global_config_path="", valid=False, resume=False, augment=False, experiment_name='name', train_sites=['vindr', 'cxr14'], pretrained=True, vit=False, dinov2=True, image_size=224, batch_size=30, lr=1e-5):
    """

        Parameters
        ----------
        global_config_path: str
            always global_config_path="FLdomain/config/config.yaml"

        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    train_loader = []
    valid_loader = []
    weight_loader = []
    loss_function_loader = []
    label_names_loader = []

    for dataset_name in train_sites:

        if dataset_name == 'vindr':
            train_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'chexpert':
            train_dataset_model = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'mimic':
            train_dataset_model = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'cxr14':
            train_dataset_model = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'padchest':
            train_dataset_model = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)

        model_info = params['Network']
        model_info['lr'] = lr
        model_info['batch_size'] = batch_size
        params['Network'] = model_info
        write_config(params, cfg_path, sort_keys=True)

        weight_model = train_dataset_model.pos_weight()
        label_names_model = train_dataset_model.chosen_labels

        loss_function_model = BCEWithLogitsLoss

        train_loader_model = torch.utils.data.DataLoader(dataset=train_dataset_model,
                                                         batch_size=batch_size,
                                                         pin_memory=True, drop_last=True, shuffle=True, num_workers=10)

        train_loader.append(train_loader_model)
        weight_loader.append(weight_model)
        loss_function_loader.append(loss_function_model)
        label_names_loader.append(label_names_model)

        valid_loader_model = torch.utils.data.DataLoader(dataset=valid_dataset_model, batch_size=batch_size,
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
        valid_loader.append(valid_loader_model)

    # Changeable network parameters for the global network
    if vit:
        if dinov2:
            model = load_pretrained_dinov2(num_classes=len(weight_model))
        else:
            model = load_pretrained_timm_model(num_classes=len(weight_model), pretrained=pretrained, imgsize=image_size)
    else:
        model = load_pretrained_timm_model(num_classes=len(weight_model), model_name='resnet50d', pretrained=pretrained)

    trainer = Training(cfg_path, resume=resume, label_names_loader=label_names_loader)

    if resume == True:
        trainer.load_checkpoints(model=model, loss_function_loader=loss_function_loader, weight_loader=weight_loader)
    else:
        trainer.setup_models(model=model, loss_function_loader=loss_function_loader, weight_loader=weight_loader)
    trainer.training_setup_conventional_federated(train_loader=train_loader, valid_loader=valid_loader, vit=vit)



def main_train_federated_validall(global_config_path="",
                  resume=False, augment=False, experiment_name='name', train_sites=['vindr', 'cxr14'], pretrained=True, vit=False, dinov2=True, image_size=224, batch_size=30, lr=1e-5):
    """

        Parameters
        ----------
        global_config_path: str
            always global_config_path="FLdomain/config/config.yaml"

        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    train_loader = []
    valid_loader = []
    weight_loader = []
    loss_function_loader = []
    label_names_loader = []

    for dataset_name in train_sites:

        if dataset_name == 'vindr':
            train_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        elif dataset_name == 'chexpert':
            train_dataset_model = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        elif dataset_name == 'mimic':
            train_dataset_model = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        elif dataset_name == 'cxr14':
            train_dataset_model = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        elif dataset_name == 'padchest':
            train_dataset_model = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)

        model_info = params['Network']
        model_info['lr'] = lr
        model_info['batch_size'] = batch_size
        params['Network'] = model_info
        write_config(params, cfg_path, sort_keys=True)

        weight_model = train_dataset_model.pos_weight()
        loss_function_model = BCEWithLogitsLoss
        train_loader_model = torch.utils.data.DataLoader(dataset=train_dataset_model,
                                                         batch_size=batch_size,
                                                         pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
        train_loader.append(train_loader_model)
        weight_loader.append(weight_model)
        loss_function_loader.append(loss_function_model)

    valid_dataset_vindr = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    label_names_vindr = valid_dataset_vindr.chosen_labels
    label_names_loader.append(label_names_vindr)
    valid_loader_vindr = torch.utils.data.DataLoader(dataset=valid_dataset_vindr, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    valid_loader.append(valid_loader_vindr)

    valid_dataset_cxr14 = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    label_names_cxr14 = valid_dataset_cxr14.chosen_labels
    label_names_loader.append(label_names_cxr14)
    valid_loader_cxr14 = torch.utils.data.DataLoader(dataset=valid_dataset_cxr14, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    valid_loader.append(valid_loader_cxr14)

    valid_dataset_chexpert = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    label_names_chexpert = valid_dataset_chexpert.chosen_labels
    label_names_loader.append(label_names_chexpert)
    valid_loader_chexpert = torch.utils.data.DataLoader(dataset=valid_dataset_chexpert, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    valid_loader.append(valid_loader_chexpert)

    valid_dataset_mimic = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    label_names_mimic = valid_dataset_mimic.chosen_labels
    label_names_loader.append(label_names_mimic)
    valid_loader_mimic = torch.utils.data.DataLoader(dataset=valid_dataset_mimic, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    valid_loader.append(valid_loader_mimic)

    valid_dataset_padchest = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    label_names_padchest = valid_dataset_padchest.chosen_labels
    label_names_loader.append(label_names_padchest)
    valid_loader_padchest = torch.utils.data.DataLoader(dataset=valid_dataset_padchest, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    valid_loader.append(valid_loader_padchest)

    # Changeable network parameters for the global network
    if vit:
        if dinov2:
            model = load_pretrained_dinov2(num_classes=len(weight_model))
        else:
            model = load_pretrained_timm_model(num_classes=len(weight_model), pretrained=pretrained, imgsize=image_size)
    else:
        model = load_pretrained_timm_model(num_classes=len(weight_model), model_name='resnet50d', pretrained=pretrained)

    trainer = Training(cfg_path, resume=resume, label_names_loader=label_names_loader)

    if resume == True:
        trainer.load_checkpoints(model=model, loss_function_loader=loss_function_loader, weight_loader=weight_loader)
    else:
        trainer.setup_models(model=model, loss_function_loader=loss_function_loader, weight_loader=weight_loader)
    trainer.training_setup_conventional_federated(train_loader=train_loader, valid_loader=valid_loader, vit=vit)


def main_train_federated_validone(global_config_path="",
                  resume=False, augment=False, experiment_name='name', train_site='cxr14', pretrained=True, vit=False, dinov2=True, image_size=224, batch_size=30, lr=1e-5):
    """

        Parameters
        ----------
        global_config_path: str
            always global_config_path="FLdomain/config/config.yaml"

        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    train_loader = []
    valid_loader = []
    weight_loader = []
    loss_function_loader = []
    label_names_loader = []

    for idx in range(4):

        if train_site == 'chexpert':
            train_dataset_model = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size, site_num=idx+1)
        elif train_site == 'mimic':
            train_dataset_model = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size, site_num=idx+1)
        elif train_site == 'cxr14':
            train_dataset_model = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size, site_num=idx+1)
        elif train_site == 'padchest':
            train_dataset_model = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size, site_num=idx+1)

        model_info = params['Network']
        model_info['lr'] = lr
        model_info['batch_size'] = batch_size
        params['Network'] = model_info
        write_config(params, cfg_path, sort_keys=True)

        weight_model = train_dataset_model.pos_weight()
        loss_function_model = BCEWithLogitsLoss
        train_loader_model = torch.utils.data.DataLoader(dataset=train_dataset_model,
                                                         batch_size=batch_size,
                                                         pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
        train_loader.append(train_loader_model)
        weight_loader.append(weight_model)
        loss_function_loader.append(loss_function_model)

    valid_dataset_vindr = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    label_names_vindr = valid_dataset_vindr.chosen_labels
    label_names_loader.append(label_names_vindr)
    valid_loader_vindr = torch.utils.data.DataLoader(dataset=valid_dataset_vindr, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    valid_loader.append(valid_loader_vindr)

    valid_dataset_cxr14 = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    label_names_cxr14 = valid_dataset_cxr14.chosen_labels
    label_names_loader.append(label_names_cxr14)
    valid_loader_cxr14 = torch.utils.data.DataLoader(dataset=valid_dataset_cxr14, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    valid_loader.append(valid_loader_cxr14)

    valid_dataset_chexpert = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    label_names_chexpert = valid_dataset_chexpert.chosen_labels
    label_names_loader.append(label_names_chexpert)
    valid_loader_chexpert = torch.utils.data.DataLoader(dataset=valid_dataset_chexpert, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    valid_loader.append(valid_loader_chexpert)

    valid_dataset_mimic = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    label_names_mimic = valid_dataset_mimic.chosen_labels
    label_names_loader.append(label_names_mimic)
    valid_loader_mimic = torch.utils.data.DataLoader(dataset=valid_dataset_mimic, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    valid_loader.append(valid_loader_mimic)

    valid_dataset_padchest = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    label_names_padchest = valid_dataset_padchest.chosen_labels
    label_names_loader.append(label_names_padchest)
    valid_loader_padchest = torch.utils.data.DataLoader(dataset=valid_dataset_padchest, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    valid_loader.append(valid_loader_padchest)

    # Changeable network parameters for the global network
    if vit:
        if dinov2:
            model = load_pretrained_dinov2(num_classes=len(weight_model))
        else:
            model = load_pretrained_timm_model(num_classes=len(weight_model), pretrained=pretrained, imgsize=image_size)
    else:
        model = load_pretrained_timm_model(num_classes=len(weight_model), model_name='resnet50d', pretrained=pretrained)

    trainer = Training(cfg_path, resume=resume, label_names_loader=label_names_loader)

    if resume == True:
        trainer.load_checkpoints(model=model, loss_function_loader=loss_function_loader, weight_loader=weight_loader)
    else:
        trainer.setup_models(model=model, loss_function_loader=loss_function_loader, weight_loader=weight_loader)
    trainer.training_setup_conventional_federated(train_loader=train_loader, valid_loader=valid_loader, vit=vit)




def main_test_central_2D_pvalue_out_of_bootstrap(global_config_path="",
                                                 experiment_name1='central_exp_for_test', experiment_name2='central_exp_for_test',
                                                 experiment1_epoch_num=100, experiment2_epoch_num=100, dataset_name='vindr', vit=False, dinov2=False, image_size=224):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params1 = open_experiment(experiment_name1, global_config_path)
    cfg_path1 = params1['cfg_path']

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        test_dataset = padchest_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    # Changeable network parameters for the global network
    if vit:
        if dinov2:
            model1 = load_pretrained_dinov2(num_classes=len(weight))
        else:
            model1 = load_pretrained_timm_model(num_classes=len(weight), imgsize=image_size)
    else:
        model1 = load_pretrained_timm_model(num_classes=len(weight), model_name='resnet50d')

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params1['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    index_list = []
    for counter in range(1000):
        index_list.append(np.random.choice(len(test_dataset), len(test_dataset)))

    # Initialize prediction 1
    predictor1 = Prediction(cfg_path1, label_names)
    predictor1.setup_model(model=model1, epoch_num=experiment1_epoch_num)
    pred_array1, target_array1 = predictor1.predict_only(test_loader)
    AUC_list1 = predictor1.bootstrapper(pred_array1.cpu().numpy(), target_array1.int().cpu().numpy(), index_list, dataset_name)

    # Changeable network parameters
    if vit:
        if dinov2:
            model2 = load_pretrained_dinov2(num_classes=len(weight))
        else:
            model2 = load_pretrained_timm_model(num_classes=len(weight), imgsize=image_size)
    else:
        model2 = load_pretrained_timm_model(num_classes=len(weight), model_name='resnet50d')

    # Initialize prediction 2
    params2 = open_experiment(experiment_name2, global_config_path)
    cfg_path2 = params2['cfg_path']
    predictor2 = Prediction(cfg_path2, label_names)
    predictor2.setup_model(model=model2, epoch_num=experiment2_epoch_num)
    pred_array2, target_array2 = predictor2.predict_only(test_loader)
    AUC_list2 = predictor2.bootstrapper(pred_array2.cpu().numpy(), target_array2.int().cpu().numpy(), index_list, dataset_name)

    print('individual labels p-values:\n')
    for idx, pathology in enumerate(label_names):
        counter = AUC_list1[:, idx] > AUC_list2[:, idx]
        ratio1 = (len(counter) - counter.sum()) / len(counter)

        reject_fdr, ratio1 = fdr_correction(ratio1, alpha=0.05, method='indep')

        if ratio1 <= 0.05:
            print(f'\t{pathology} p-value: {ratio1}; model 1 significantly higher AUC than model 2')
        else:
            counter = AUC_list2[:, idx] > AUC_list1[:, idx]
            ratio2 = (len(counter) - counter.sum()) / len(counter)

            reject_fdr, ratio2 = fdr_correction(ratio2, alpha=0.05, method='indep')

            if ratio2 <= 0.05:
                print(f'\t{pathology} p-value: {ratio2}; model 2 significantly higher AUC than model 1')
            else:
                print(f'\t{pathology} p-value: {ratio1}; models NOT significantly different for this label')

    print('\nAvg AUC of labels p-values:\n')
    avgAUC_list1 = AUC_list1.mean(1)
    avgAUC_list2 = AUC_list2.mean(1)
    counter = avgAUC_list1 > avgAUC_list2
    ratio1 = (len(counter) - counter.sum()) / len(counter)

    reject_fdr, ratio1 = fdr_correction(ratio1, alpha=0.05, method='indep')

    if ratio1 <= 0.05:
        print(f'\tp-value: {ratio1}; model 1 significantly higher AUC than model 2 on average')
    else:
        counter = avgAUC_list2 > avgAUC_list1
        ratio2 = (len(counter) - counter.sum()) / len(counter)

        reject_fdr, ratio2 = fdr_correction(ratio2, alpha=0.05, method='indep')

        if ratio2 <= 0.05:
            print(f'\tp-value: {ratio2}; model 2 significantly higher AUC than model 1 on average')
        else:
            print(f'\tp-value: {ratio1}; models NOT significantly different on average for all labels')


    msg = f'\n\nindividual labels p-values:\n'
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        counter = AUC_list1[:, idx] > AUC_list2[:, idx]
        ratio1 = (len(counter) - counter.sum()) / len(counter)

        reject_fdr, ratio1 = fdr_correction(ratio1, alpha=0.05, method='indep')

        if ratio1 <= 0.05:
            msg = f'\t{pathology} p-value: {ratio1}; model 1 significantly higher AUC than model 2'
        else:
            counter = AUC_list2[:, idx] > AUC_list1[:, idx]
            ratio2 = (len(counter) - counter.sum()) / len(counter)

            reject_fdr, ratio2 = fdr_correction(ratio2, alpha=0.05, method='indep')

            if ratio2 <= 0.05:
                msg = f'\t{pathology} p-value: {ratio2}; model 2 significantly higher AUC than model 1'
            else:
                msg = f'\t{pathology} p-value: {ratio1}; models NOT significantly different for this label'

        with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
            f.write(msg)
        with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
            f.write(msg)


    msg = f'\n\nAvg AUC of labels p-values:\n'
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    avgAUC_list1 = AUC_list1.mean(1)
    avgAUC_list2 = AUC_list2.mean(1)
    counter = avgAUC_list1 > avgAUC_list2
    ratio1 = (len(counter) - counter.sum()) / len(counter)

    reject_fdr, ratio1 = fdr_correction(ratio1, alpha=0.05, method='indep')

    if ratio1 <= 0.05:
        msg = f'\tp-value: {ratio1}; model 1 significantly higher AUC than model 2 on average'
    else:
        counter = avgAUC_list2 > avgAUC_list1
        ratio2 = (len(counter) - counter.sum()) / len(counter)

        reject_fdr, ratio2 = fdr_correction(ratio2, alpha=0.05, method='indep')

        if ratio2 <= 0.05:
            msg = f'\tp-value: {ratio2}; model 2 significantly higher AUC than model 1 on average'
        else:
            msg = f'\tp-value: {ratio1}; models NOT significantly different on average for all labels'

    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)


def load_pretrained_timm_model(num_classes=2, model_name='vit_base_patch16_224', pretrained=False, imgsize=512):
    # Load a pre-trained model from config file

    if model_name == 'resnet50d':
        model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)

    else:
        model = timm.create_model(model_name, num_classes=num_classes, img_size=imgsize, pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = True

    return model


def load_pretrained_dinov2(num_classes=2):
    # Load a pre-trained model from config file

    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.head = torch.nn.Linear(in_features=768, out_features=num_classes)
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    for param in model.parameters():
        param.requires_grad = True

    return model






if __name__ == '__main__':

