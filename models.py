import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics.ranking import roc_auc_score

import torch
import torch.nn as nn
import torchvision.models as models

import resnet_wider
import densenet



def ClassificationNet(arch_name, num_class, conv=None, weight=None, activation=None):
    if weight is None:
        weight = "none"

    if conv is None:
        try:
            model = resnet_wider.__dict__[arch_name](sobel=False)
        except:
            model = models.__dict__[arch_name](pretrained=False)
    else:
        if arch_name.lower().startswith("resnet"):
            model = resnet_wider.__dict__[arch_name + "_layerwise"](conv, sobel=False)
        elif arch_name.lower().startswith("densenet"):
            model = densenet.__dict__[arch_name + "_layerwise"](conv)

    if arch_name.lower().startswith("resnet"):
        kernelCount = model.fc.in_features
        if activation is None:
            model.fc = nn.Linear(kernelCount, num_class)
        elif activation == "Sigmoid":
            model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())


        # init the fc layer
        if activation is None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()
    elif arch_name.lower().startswith("densenet"):
        kernelCount = model.classifier.in_features
        if activation is None:
            model.classifier = nn.Linear(kernelCount, num_class)
        elif activation == "Sigmoid":
            model.classifier = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())

        # init the classifier layer
        if activation is None:
            model.classifier.weight.data.normal_(mean=0.0, std=0.01)
            model.classifier.bias.data.zero_()
        else:
            model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)
            model.classifier[0].bias.data.zero_()

    def _weight_loading_check(_arch_name, _activation, _msg):
        if len(_msg.missing_keys) != 0:
            if _arch_name.lower().startswith("resnet"):
                if _activation is None:
                    assert set(_msg.missing_keys) == {"fc.weight", "fc.bias"}
                else:
                    assert set(_msg.missing_keys) == {"fc.0.weight", "fc.0.bias"}
            elif _arch_name.lower().startswith("densenet"):
                if _activation is None:
                    assert set(_msg.missing_keys) == {"classifier.weight", "classifier.bias"}
                else:
                    assert set(_msg.missing_keys) == {"classifier.0.weight", "classifier.0.bias"}

    if weight.lower() == "imagenet":
        pretrained_model = models.__dict__[arch_name](pretrained=True)
        state_dict = pretrained_model.state_dict()

        # delete fc layer
        for k in list(state_dict.keys()):
            if k.startswith('fc') or k.startswith('classifier'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        _weight_loading_check(arch_name, activation, msg)
        print("=> loaded supervised ImageNet pre-trained model")
    elif os.path.isfile(weight):
        checkpoint = torch.load(weight, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("module.encoder_q.", ""): v for k, v in state_dict.items()}

        for k in list(state_dict.keys()):
            if k.startswith('fc') or k.startswith('classifier') or k.startswith('projection_head') or k.startswith('prototypes'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        _weight_loading_check(arch_name, activation, msg)
        print("=> loaded pre-trained model '{}'".format(weight))
        print("missing keys:", msg.missing_keys)


    # reinitialize fc layer again
    if arch_name.lower().startswith("resnet"):
        if activation is None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()
    elif arch_name.lower().startswith("densenet"):
        if activation is None:
            model.classifier.weight.data.normal_(mean=0.0, std=0.01)
            model.classifier.bias.data.zero_()
        else:
            model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)

    return model


def build_classification_model(args):
    if args.init.lower() =="random" or args.init.lower() =="imagenet":
        model = ClassificationNet(args.model_name.lower(), args.num_class, weight=args.init,
                              activation=args.activate)

    else:
        model = ClassificationNet(args.model_name.lower(), args.num_class, weight=args.proxy_dir,
                              activation=args.activate)


    return model

def save_checkpoint(state,filename='model'):

    torch.save( state,filename + '.pth.tar')




