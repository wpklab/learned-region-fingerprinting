import copy # someone on stack overflow says copy is built into python, and thus requires no installation.
import os
import shutil # fine
import time # fine
from PIL import Image # seems fine
import wandb
import pandas as pd
import random

# from connector_work.transfg import VisionTransformer, CONFIGS

# from connector_work.S3N import MultiSmoothLoss

# from connector_work.pmg import load_model, PMG, jigsaw_generator

# from transfg import VisionTransformer, CONFIGS

# from S3N import MultiSmoothLoss

# from pmg import load_model, PMG, jigsaw_generator


import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
import numpy as np

import timm
import torch.distributed as dist

import matplotlib.pyplot as plt

# from collections import Counter

torch.hub.set_dir('data/Models')

os.environ["HUGGINGFACE_HUB_CACHE"] = "data/Models"

def set_parameter_requires_grad(model, feature_extracting, freeze_layers):
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False #I think this line was freezing a lot of parameters
    else:
        for param in model.parameters():
            param.requires_grad = True

# To initialize the model
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, freeze_layers=True, classifier_layer_config=0, input_size=448):
    if model_name == "resnet18":
        model_ft = models.resnet18(weights='DEFAULT')
    if model_name == "resnet34":
        model_ft = models.resnet34(weights='DEFAULT')
    if model_name == "resnet50":
        model_ft = models.resnet50(weights='DEFAULT')
    if model_name == "resnet101":
        model_ft = models.resnet101(weights='DEFAULT')
    if model_name == "resnet152":
        model_ft = models.resnet152(weights='DEFAULT')
    if model_name == "convnext_base":
        model_ft = models.convnext_base(weights='DEFAULT')
    if model_name == "wideresnet50":
        model_ft = models.wide_resnet50_2(weights='DEFAULT')
    if model_name == "efficientnetv2_s":
        model_ft = models.efficientnet_v2_s(weights='DEFAULT')
    if model_name == "efficientnetv2_m":
        model_ft = models.efficientnet_v2_m(weights='DEFAULT')
    if model_name == "swin_v2_t":
        model_ft = models.swin_v2_t(weights='DEFAULT')
    if model_name == "swin_v2_s":
        model_ft = models.swin_v2_s(weights='DEFAULT')
    if model_name == 'convnext_small':
        model_ft = models.convnext_small(weights='DEFAULT')
    if model_name == 'convnext_tiny':
        model_ft = models.convnext_small(weights='DEFAULT')
    if model_name == 'maxvit':
        model_ft = models.maxvit_t(weights='DEFAULT')
    if model_name == 'mobilenet_large':
        model_ft = models.mobilenet_v3_large(weights='DEFAULT')
    if model_name == 'mobilenet_small':
        model_ft = models.mobilenet_v3_small(weights='DEFAULT')
    if model_name == 'vit_b_16':
        #model_ft = models.vit_b_16(weights='DEFAULT')
        # model_ft = timm.create_model('vit_base_patch16_384', pretrained=True, img_size=448, num_classes=num_classes)
        model_ft = timm.create_model('vit_base_patch16_384', pretrained=True, img_size=input_size, num_classes=num_classes)
    if model_name == 'vit_b_32':
        model_ft = models.vit_b_32(weights='DEFAULT')
    if model_name == 'vit_l_16':
        model_ft = models.vit_l_16(weights='DEFAULT')
    if model_name == 'vit_l_32':
        model_ft = models.vit_l_32(weights='DEFAULT')
    if model_name == 'vit_h_14':
        model_ft = models.vit_h_14(weights='DEFAULT')
    if model_name == 'poolformer_m36':
        model_ft = timm.create_model('poolformer_m36.sail_in1k', pretrained=True, num_classes=num_classes)
    if model_name == 'poolformer_m48':
        model_ft = timm.create_model('poolformer_m48', pretrained=True, num_classes=num_classes)
    if model_name == 'poolformer_s36':
        model_ft = timm.create_model('poolformer_s36', pretrained=True, num_classes=num_classes)
    if model_name == 'efficient_vit_m5':
        model_ft = timm.create_model('efficientvit_m5', pretrained=True, num_classes=num_classes)
    if model_name == 'inception_next_base':
        model_ft = timm.create_model('inception_next_base', pretrained=True, num_classes=num_classes)
    if model_name == 'convnextv2_base':
        model_ft = timm.create_model('convnextv2_base.fcmae', pretrained=True, num_classes=num_classes)
    # if model_name == 'transfg':
    #     model_ft = VisionTransformer(CONFIGS['ViT-B_16'], img_size=448, zero_head=False, num_classes=num_classes, smoothing_value=0.0)
    # if model_name == 'pmg_50':
    #     model_ft = load_model(model_name='resnet50_pmg', img_size=448, num_classes=num_classes, pretrain=True, require_grad=True)
    # if model_name == 'pmg_101':
    #     model_ft = load_model(model_name='resnet101_pmg', img_size=448, num_classes=num_classes, pretrain=True, require_grad=True)
    # if model_name == 'apcnn':
    #     from hawkeye_models import MODEL
    #     config = {'num_classes': num_classes}
    #     model_ft = MODEL.get('APCNN')(config)
    # if model_name == 's3n':
    #     from S3N import MODEL
    #     config = {'num_classes': num_classes} #does nothing
    #     model_ft = MODEL.get('S3N')(config)
    if model_name == 'efficientnetv2_l':
        model_ft = timm.create_model('efficientnetv2_l', num_classes=num_classes)
    if model_name == 'efficientnetv2_xl':
        model_ft = timm.create_model('efficientnetv2_xl', num_classes=num_classes)


    set_parameter_requires_grad(model_ft, feature_extract, freeze_layers)

    if model_name == "convnext_base":
        sequential_layers = nn.Sequential(
            nn.LayerNorm((1024, 1, 1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1)
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers
    
    elif model_name == "convnext_small":
        sequential_layers = nn.Sequential(
            nn.LayerNorm((768, 1, 1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers
    
    elif model_name == "convnext_tiny":
        sequential_layers = nn.Sequential(
            nn.LayerNorm((768, 1, 1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, 384),
                                            nn.ReLU(),
                                            nn.Linear(384, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers
    
    elif model_name == "maxvit":
        sequential_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True),
            nn.Linear(512, 512),
            nn.Tanh(),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(512, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(512, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(512, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)

        model_ft.classifier = sequential_layers
    
    elif model_name == "mobilenet_large":
        sequential_layers = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardshrink(),
            nn.Dropout(p=0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1280, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers

    elif model_name == "mobilenet_small":
        sequential_layers = nn.Sequential(
            nn.Linear(576, 1024, bias=True),
            nn.Hardshrink(),
            nn.Dropout(p=0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)

        model_ft.classifier = sequential_layers

    elif model_name == "efficientnetv2_s":
        sequential_layers = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1280, 640),
                                            nn.ReLU(),
                                            nn.Linear(640, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers

    elif model_name == "efficientnetv2_m":
        
        sequential_layers = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            #classifier_layers=nn.Sequential(nn.Linear(1280, num_classes))
            classifier_layers=nn.Sequential(nn.Linear(1280, 640),
                                            nn.ReLU(),
                                            nn.Linear(640, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers

    elif model_name == 'swin_v2_t':
        n_inputs = model_ft.head.in_features
        sequential_layers = nn.Sequential()

        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, int(n_inputs/2)),
                                            nn.ReLU(),
                                            nn.Linear(int(n_inputs/2), num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.head = sequential_layers
    
    elif model_name == 'swin_v2_s':
        n_inputs = model_ft.head.in_features
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.head = sequential_layers

    elif model_name == 'vit_b_16':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            # classifier_layers=nn.Sequential(nn.Linear(768, 512),
            #                                 nn.ReLU(),
            #                                 nn.Linear(512, 256),
            #                                 nn.ReLU(),
            #                                 nn.Linear(256, 128),
            #                                 nn.ReLU(),
            #                                 nn.Linear(128, num_classes))
            classifier_layers=nn.Sequential(nn.Linear(768, 384),
                                            nn.ReLU(),
                                            nn.Linear(384, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_b_32':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_l_16':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_l_32':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_h_14':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1280, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'efficient_vit_m5':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=model_ft.head
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(384, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(384, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'inception_next_base':
        sequential_layers = model_ft.head
        model_ft.head = sequential_layers

    elif model_name == 'convnextv2_base':
        sequential_layers = nn.Sequential(nn.Linear(1024, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, num_classes))
        model_ft.head.fc = sequential_layers

    elif (model_name == 'poolformer_m36') | (model_name == 'poolformer_m48') | (model_name == 'poolformer_s36'):
        sequential_layers = model_ft.head
        model_ft.head = sequential_layers

    elif model_name == 'transfg':
        pass

    elif model_name == 'pmg_50':
        pass

    elif model_name == 'pmg_101':
        pass

    elif model_name == 'apcnn':
        pass
    elif model_name == 's3n':
        pass

    elif model_name == 'efficientnetv2_l':
        pass

    elif model_name == 'efficientnetv2_xl':
        pass

    else:       
        num_ftrs = model_ft.fc.in_features
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(num_ftrs, int(num_ftrs/2)),
                                            nn.ReLU(),
                                            nn.Linear(int(num_ftrs/2), num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(num_ftrs, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(num_ftrs, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.fc = sequential_layers

    # Try not adding softmax layer        
    # if (classifier_layer_config != 1) & (model_name != 'poolformer_m36') & (model_name != 'poolformer_m48') & (model_name != 'poolformer_s36'):
    #     sequential_layers.append(nn.LogSoftmax(dim=1))
    #     model_ft.fc = sequential_layers

    input_size = 448

    return model_ft, input_size

def gather_tensors(tensor):
    gather_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=0)

def gather_indices(indices):
    indices_tensor = torch.tensor(indices, dtype=torch.int64).cuda()
    gather_list = [torch.zeros_like(indices_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, indices_tensor)
    return torch.cat(gather_list, dim=0)

class EarlyStopping:
    def __init__(self, patience=4, verbose=False, delta=0.00001):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# To train the model
def train_model(model, model_name, dataloaders, image_datasets, criterion, optimizer, batch_size, class_names, data_dir, test_samples, device1, rank, scheduler, n_train_samples, n_val_samples, val_sampler, sigma, num_epochs, jigsaw=False, train_sampler=None, log_interval=5, use_dps=False):

    if use_dps:
        from fingerprint_proposal import adjust_sigma
    early_stopping = EarlyStopping(patience=50, verbose=True)
    # model.to(device1)
    model.cuda()

    since = time.time()
    val_acc_history = []
    train_loss_history = []
    best_acc = 0.0
    best_loss = np.inf
    #Smooth_Loss = MultiSmoothLoss(config={'num_classes': len(class_names) - 1}) #Dummy Config File

    #Overwrite batch size for augmentation
    #batch_size = 24
    softmax = nn.LogSoftmax(dim=1)
    
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(num_epochs):
        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if rank == 0:
            epoch_since = time.time()
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 30)
            print('lr {}'.format(scheduler.get_last_lr()))
            if use_dps:
                print("Sigma: ", model.module[0][0].TOPK.sigma)
        running_loss = 0.
        running_corrects = 0.
        model.train()
        phase = 'train'
        
        for batch_id, (inputs, labels) in enumerate(dataloaders[phase], start=epoch * len(dataloaders[phase])):
            # new_labels = torch.tensor([true_labels[int(label)] for label in labels])
            # inputs, labels = inputs.to(device1), new_labels.to(device1)
            
            # if batch_id > 1: For Testing
            #     break

            # labels = labels.view(-1, 1)
            # inputs, labels = inputs.to(device1), labels.to(device1)
            inputs, labels = inputs.cuda(), labels.cuda()

            # labels = torch.ones(inputs.size()[0], device=device1) * labels
            # labels = labels.long()
            
            # labels = torch.Tensor(labels).to(device1).long()
            labels = torch.Tensor(labels).cuda().long()
            
            if use_dps:
                adjust_sigma(0, num_epochs, sigma, model.module[0][0], dataloaders[phase], batch_id+1)
            

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Clip gradients to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)  # Step the optimizer
            scaler.update()
            running_loss += loss.item() * (inputs.size(0))
            running_corrects += (torch.sum(preds.view(-1) == labels.view(-1)).item())

            if rank == 0:
                
                if batch_id % log_interval == 0: #Remove log interval
                    num_samples_processed = (batch_id + 1 - (epoch * len(dataloaders[phase]))) * batch_size
                    print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}\tAcc: {}/{}".format(
                        epoch,
                        num_samples_processed,
                        len(dataloaders['train'].dataset),
                        100. * num_samples_processed / len(dataloaders['train'].dataset),
                        running_loss / num_samples_processed,
                        int(running_corrects),
                        num_samples_processed
                    ))

        
        
        if rank == 0:
            epoch_loss = running_loss / n_train_samples
            epoch_acc = running_corrects / n_train_samples
            time_elapsed = time.time() - epoch_since
            print("Training Epoch compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
            print("{} Loss: {} Acc: {}".format('Train', epoch_loss, epoch_acc))
        
        ##Log Validation Statistics
        if (epoch % 5 == 0) | (epoch == num_epochs-1): #Remove log interval
            val_loss, val_acc, pred_array_final, label_array_final = test_model(model, model_name, dataloaders['val'], image_datasets, criterion, epoch, class_names, data_dir, test_samples, n_val_samples, val_sampler, device1, rank, epoch_end=True)
            # The code `epoch_loss` appears to be a variable or placeholder in Python. It is not
            # performing any specific action in the provided snippet.
            epoch_loss = running_loss / n_train_samples
            epoch_acc = running_corrects / n_train_samples
            
            world_size = dist.get_world_size()

            # Assuming val_loss and val_acc are lists of validation losses and accuracies
            val_loss_tensor = torch.tensor(val_loss).cuda()
            
            if val_loss_tensor.dim() == 0:
                val_loss_tensor = val_loss_tensor.unsqueeze(0)

            val_loss_tensor = gather_tensors(val_loss_tensor)
            
            val_loss_tensor = val_loss_tensor[~torch.isnan(val_loss_tensor)]
            
            if len(val_loss_tensor) > 0:
                val_loss = torch.mean(val_loss_tensor).item()
            else:
                val_loss = 100.0  # or some default value

            pred_array_final = gather_tensors(pred_array_final)
            label_array_final = gather_tensors(label_array_final)
            dist.barrier()
            
            val_acc = torch.sum(pred_array_final == label_array_final).item() / len(label_array_final)
            
            val_acc_history.append(val_acc)
            train_loss_history.append(epoch_loss)
            
            indices = list(val_sampler)
            all_indices = gather_indices(indices)
            
            early_stopping(val_acc, model) ## Early stoppage based on validation loss
            
            if rank == 0:
                print(f"Averaged Validation Loss: {val_loss:.4f}")
                print(f"Averaged Validation Accuracy: {val_acc:.4f}")
                
            if early_stopping.early_stop:
                print("Early stopping")
                return model, val_acc_history, train_loss_history, pred_array_best, label_array_best, best_loss, best_acc
            
            if val_acc > best_acc:
            # if epoch_loss < best_loss:
                best_acc = val_acc
                best_loss = val_loss
                
                pred_array_best = pred_array_final
                label_array_best = label_array_final
                
                if rank == 0:
                    best_model_wts = copy.deepcopy(model.state_dict())

                    wandb.log({'max_acc': best_acc})

                    wandb.log({"conf_mat_" : wandb.plot.confusion_matrix( 
                        preds=np.array(pred_array_final.cpu()), y_true=np.array(label_array_final.cpu()), class_names=np.array(class_names))})
                    #class_names = Work on adding some kind of distinction based on location in the part
                    
                    # test_loss, test_acc, _, _ = test_model(model, model_name, dataloaders['test'], image_datasets, criterion, epoch, class_names, data_dir, test_samples, device1, epoch_end=True)
                    # wandb.log({'test_acc': test_acc, 'test_loss': test_loss})
                    
                    
                    ##Hardcoded in for now
                    torch.save(model.state_dict(), 'data/Models/aims_connector_test_' + wandb.run.id + '_dpp.pth')

                    df = pd.DataFrame(columns=['img', 'label', 'pred'])
                    df.img = [image_datasets['val'].imgs[idx] for idx in all_indices]
                    # df = pd.DataFrame(columns=['label', 'pred'])

                    df.pred = pred_array_best.cpu()
                    df.label = label_array_best.cpu()
                    
                    csv_path = data_dir + "/csv_outputs/ddp_model_" + str(wandb.run.id) + '_' + str(epoch) + ".csv"

                    df.to_csv(csv_path, index=False)
                
            if rank == 0:
                print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))

                wandb.log({'epoch_train': epoch, 'train acc': epoch_acc, 'train loss': epoch_loss, 'val acc': val_acc, 'val loss': val_loss})

            scheduler.step()

    if rank == 0:
        time_elapsed = time.time() - since
        print("Training compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
        print("Best val Acc: {}".format(best_acc))
        

        model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_loss_history, pred_array_best, label_array_best, best_loss, best_acc

# To test the model during training


##### Functions Added to predict around voids #####
def mode_and_filter_raw(array, class_names, device1, filter_vals=None):
    # array = array.to(device1)
    array = array.cuda()
    #filter_vals = filter_vals.to(device1)
    #array[:, :, filter_vals] = 0 ## Zero empty class contributions
    # result = torch.mean(array, dim=1)
    result = torch.sum(array, dim=1)
    # pred = result
    max_val, pred = result.max(dim=1)
    
    #Remove unknowns for now
    #pred[max_val <= array.size()[1]*0.3] = len(class_names) - 1 ## Set to unknown class if the majority class has more than 30% of the votes
    return pred

def visualize_map(model, sample_imgs, labels, output_dir, device1, class_names):
    # forward pass
    with torch.amp.autocast('cuda'):
        with torch.autograd.set_grad_enabled(False):
            outputs = model.module[0][0].scorer(transforms.Resize((650, 850))(sample_imgs))

    if outputs.dim() == 4:
        outputs = outputs.permute((0, 2, 3, 1))
        
    outputs = outputs.cpu().numpy()
    labels = labels.cpu().numpy()

    # save the visualization images in output_dir/xxx
    for i in range(outputs.shape[0]):
        # rescale to 0-1 range, for matplotlib
        curr_map = (outputs[i] - np.min(outputs[i])) / (np.max(outputs[i]) - np.min(outputs[i]))
        plt.figure(frameon=False)
        plt.imshow(curr_map, cmap='plasma')
        plt.axis('off')
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, str(class_names[int(labels[i])] + '_id_' +  str(i) + '_Device_' + str(device1))))
        plt.close("all")
        
    indicators = model.module[0][0].indicators
                
    indicators_resized = torch.nn.functional.interpolate(indicators, size=(sample_imgs.size(2), sample_imgs.size(3)), mode='bilinear', align_corners=False)
    square_size = 448  # Size of the square to be filled
    for i in range(indicators_resized.size(0)):  # Iterate over batch size
        for j in range(indicators_resized.size(1)):  # Iterate over the number of indicators
            # Get the current indicator
            indicator = indicators_resized[i, j]

            # Find the center point of the indicator
            center = torch.nonzero(indicator == indicator.max(), as_tuple=False)[0]
            center_y, center_x = center[0].item(), center[1].item()

            # Calculate the top-left corner of the patch size square
            top_left_y = max(center_y - square_size // 2, 0)
            top_left_x = max(center_x - square_size // 2, 0)

            # Ensure the square does not exceed the image boundaries
            bottom_right_y = min(top_left_y + square_size, indicator.size(0))
            bottom_right_x = min(top_left_x + square_size, indicator.size(1))

            ## Consant mask
            indicators_resized[i, j, top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1
            
    indicator_images = indicators_resized.sum(dim=1).cpu().numpy()
    
    mean = [0.2606, 0.2654, 0.2964]
    std =[0.1024, 0.1033, 0.1126]
    
    for image_id in range(indicator_images.shape[0]):
        # Prepare base image
        plot_image = sample_imgs[image_id].cpu().numpy().transpose(1, 2, 0)
        plot_image = (np.clip(plot_image * std[0] + mean[0], 0, 1) * 255).astype(np.uint8)
        
        # Process indicator image
        indicator_np = indicator_images[image_id]
        indicator_np = (indicator_np - indicator_np.min()) / (indicator_np.max() - indicator_np.min() + 1e-8)
        # Convert to float32 for transparency overlay
        indicator_np = indicator_np.astype(np.float32)
        
        # Create an alpha mask; ensure it has the same size and values between 0 and 1
        alpha_mask = np.where(indicator_np > 0.05, 0.5, 0.0).astype(np.float32)
        
        # Plot and overlay
        plt.figure(figsize=(10, 10))
        plt.imshow(plot_image)
        plt.imshow(indicator_np, cmap='plasma', alpha=alpha_mask) 
        plt.axis('off')
        plt.savefig(output_dir + f'/test_indicator_{image_id}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

def test_model(model, model_name, dataloaders, image_datasets, criterion, epoch, class_names, data_dir, test_samples, n_val_samples, val_sampler, device1, rank, epoch_end=False):
    
    # model.to(device1)
    model.cuda()

    running_loss = 0.
    running_corrects = 0.
    running_report_corrects = 0.
    model.eval()
    since = time.time()
    # pred_array = torch.zeros(0, device=device1)
    # label_array = torch.zeros(0, device=device1)
    # pred_part_array = torch.zeros(0, device=device1)
    # label_part_array = torch.zeros(0, device=device1)
    # output_array = torch.zeros(0, device=device1)
    pred_array = torch.zeros(0).cuda()
    label_array = torch.zeros(0).cuda()
    pred_part_array = torch.zeros(0).cuda()
    label_part_array = torch.zeros(0).cuda()
    output_array = torch.zeros(0).cuda()
    softmax = nn.LogSoftmax(dim=1)
    CELoss = nn.CrossEntropyLoss()
    #Smooth_Loss = MultiSmoothLoss(config={'num_classes': len(class_names) - 1}) #Dummy Config File

    # true_labels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    if rank == 0:
        print('Testing model')  
        
    # model[0][0].TOPK.sigma = 1e-5 ## Set sigma to 0 for testing

    for batch_id, (inputs, labels) in enumerate(dataloaders):    #Added this line to evaluate model performance for multiple batches
        # inputs, labels = inputs.to(device1), labels.to(device1)
        
        # if batch_id > 1: For testing
        #     break
        
        inputs, labels = inputs.cuda(), labels.cuda()
        # inputs = inputs.squeeze()
        #inputs = inputs.view(-1, 3, 448, 448)
        # labels = torch.ones(inputs.size()[0], device=device1) * labels
        # labels = labels.long()
        
        label_plot = labels

        # labels = torch.Tensor(labels).to(device1).long()
        labels = torch.Tensor(labels).cuda().long()
        # labels = labels.view(-1, 1)
        

        with torch.amp.autocast('cuda'):  # Enable mixed precision
            with torch.autograd.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                outputs = softmax(outputs)
        _, preds = torch.max(outputs, 1)
        # preds = outputs
        running_loss += loss.item() * (inputs.size(0))
        # if (torch.mode(preds.view(-1), 0)[0] == torch.mode(labels.view(-1), 0)[0]):
        #     running_corrects += 1
        
        # Add running correct count for batch-level accuracy reporting
        batch_corrects = torch.sum(preds.view(-1) == labels.view(-1)).item()
        running_report_corrects += batch_corrects

        # Add the requested logging
        if rank == 0:
            if batch_id % 10 == 0:
                num_samples_processed = (batch_id + 1) * inputs.size(0)
                print("Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {}/{}".format(
                    epoch,
                    num_samples_processed,
                    n_val_samples,
                    100. * num_samples_processed / n_val_samples,
                    running_loss / num_samples_processed,
                    int(running_report_corrects),
                    num_samples_processed
                ))
        

        pred_array = torch.cat((pred_array, preds.view(-1)), 0)
        label_array = torch.cat((label_array, labels.view(-1)), 0)
        output_array = torch.cat((output_array, outputs.view(-1)), 0)
        
        # if batch_id == 0:
        #     image_model_path = data_dir + "/image_outputs/aims_" + str(os.environ["SLURM_JOBID"]) + '/'
        #     if not os.path.exists(image_model_path):
        #         os.makedirs(image_model_path, exist_ok=True)
        #     image_path = data_dir + "/image_outputs/aims_" + str(os.environ["SLURM_JOBID"]) + '/' + str(epoch) + '/'
        #     if not os.path.exists(image_path):
        #         os.makedirs(image_path, exist_ok=True)
        #     visualize_map(model, inputs, label_plot, image_path, device1, class_names)
        
    # pred_img_array = pred_array.view(-1, test_samples)
    # label_img_array = label_array.view(-1, test_samples)
    # output_img_array = output_array.view(-1, test_samples)
    
    pred_img_array = pred_array.view(-1, test_samples)
    label_img_array = label_array.view(-1, test_samples)
    output_img_array = output_array.view(-1, test_samples, len(class_names))
    
    # pred_img_array = pred_array.view(-1)
    # label_img_array = label_array.view(-1)
    # output_img_array = output_array.view(-1, len(class_names)-1)
    
    ## Filter out voids
    #pred_img_array_mode = mode_and_filter(pred_img_array, torch.tensor([np.where(class_names == 'Empty')[0][0]]), class_names).to(device1)
    ## Try raw voting scheme
    #pred_img_array_mode = mode_and_filter_raw(output_img_array, torch.tensor([np.where(class_names == 'Empty')[0][0]]), class_names).to(device1)
    # pred_img_array_mode = mode_and_filter_raw(output_img_array, class_names, device1).to(device1)
    pred_img_array_mode = mode_and_filter_raw(output_img_array, class_names, device1).cuda()
    #label_img_array_mode = mode_and_filter_raw(label_img_array, torch.tensor([np.where(class_names == 'Empty')[0][0]]), class_names).to(device1)
    label_img_array_mode, _ = torch.mode(label_img_array, 1)
    # label_img_array_mode = label_img_array_mode.to(device1)
    label_img_array_mode = label_img_array_mode.cuda()
    
    running_corrects += torch.sum(pred_img_array_mode == label_img_array_mode).detach().cpu().numpy()
    pred_part_array = torch.cat((pred_part_array, pred_img_array_mode), 0)
    label_part_array = torch.cat((label_part_array, label_img_array_mode), 0)

    # running_corrects += torch.sum(torch.mode(pred_img_array, 1)[0] == torch.mode(label_img_array, 1)[0]).detach().cpu().numpy()
    # pred_part_array = torch.cat((pred_part_array, torch.mode(pred_img_array, 1)[0].to(dtype=torch.long, device=device1)), 0)
    # label_part_array = torch.cat((label_part_array, torch.mode(label_img_array, 1)[0].to(dtype=torch.long, device=device1)), 0)


    epoch_loss = running_loss / n_val_samples

    # epoch_acc = np.sqrt((running_corrects))
    epoch_acc = (running_corrects) / n_val_samples
    
    time_elapsed = time.time() - since
    
    if rank == 0:
        # print('Full Part Accuracy: {}'.format(epoch_acc))

        ## Accuracy for known parts
        # num_known_parts = pred_part_array[pred_part_array != len(class_names) - 1].size()[0]
        # num_correct_known_parts = torch.sum(pred_part_array[pred_part_array != len(class_names) - 1] == label_part_array[pred_part_array != len(class_names) - 1]).detach().cpu().numpy()
        # epoch_acc = num_correct_known_parts / num_known_parts

        print("Testing compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
        print("{} Loss: {} Acc: {}".format('val', epoch_loss, epoch_acc))
    # wandb.log({"conf_mat_" : wandb.plot.confusion_matrix( 
    # #     #preds=np.array(pred_array.cpu()), y_true=np.array(label_array.cpu()), class_names=['M2', 'M3'])})
    #     preds=np.array(pred_array.cpu()), y_true=np.array(label_array.cpu()), class_names=class_names)})
    # if epoch_end:
    # df = pd.DataFrame(columns=['img', 'label', 'pred'])
    # df.img = (np.repeat(np.array(image_datasets['val'].imgs)[:, 0], test_samples))

    # df.pred = pred_array.cpu()
    # df.label = label_array.cpu()
    # print('Outputting csv file')
    # pd.DataFrame(df).to_csv(data_dir + "csv_outputs/" + wandb.run.name +'_'+ wandb.run.id + ".csv")
    # #pd.DataFrame(df).to_csv(data_dir + "csv_outputs/test.csv")
    # print('Outputting csv file complete')

    # model.load_state_dict(best_model_wts)

    return epoch_loss, epoch_acc, pred_part_array, label_part_array
