import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import resnet_wider
import densenet
import timm
import segmentation_models_pytorch as smp
from swin_transformer import UperNet_swin,load_swin_pretrained,SwinTransformer
from convnext import UperNet_convnext
from timm.utils import NativeScaler

def ClassificationNet(arch_name, num_class, conv=None, weight=None, activation=None,img_size=224):
    if weight is None:
        weight = "none"
    if arch_name.lower().startswith("resnet") or arch_name.lower().startswith("densenet"):
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
    elif arch_name.lower().startswith("convnext"):
            model = timm.create_model(arch_name, num_classes=num_class, pretrained=False)
    elif arch_name.lower().startswith("swin"):
        if arch_name.lower() == "swin_base":
            model = SwinTransformer(img_size=img_size, num_classes=num_class,
                                    patch_size=4, window_size=7, embed_dim=128, depths=[2, 2, 18, 2],
                                    num_heads=[4, 8, 16, 32])
        elif arch_name.lower() == "swin_tiny":
            model = SwinTransformer(img_size=img_size, num_classes=num_class,
                                    patch_size=4, window_size=7, embed_dim=96, depths=[2, 2, 6, 2 ],
                                    num_heads=[ 3, 6, 12, 24 ])
        elif arch_name.lower() == "swin_small":
            model = SwinTransformer(img_size=img_size, num_classes=num_class,
                                    patch_size=4, window_size=7, embed_dim=96, depths=[ 2, 2, 18, 2 ],
                                    num_heads=[ 3, 6, 12, 24 ])

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

    if weight.lower() == "imagenet" and arch_name.lower().startswith("resnet"):
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
        if arch_name.lower().startswith("convnext") and "mimic" in weight.lower():
            state_dict = torch.load(weight, map_location='cpu')
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}
            for k in list(state_dict.keys()):
                if k.startswith('head'):
                    del state_dict[k]
            msg = model.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(weight))
            print("missing keys:", msg.missing_keys)
        elif arch_name.lower().startswith("convnext") and ("random" not in weight.lower()):
            import re
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "teacher" in state_dict:
                state_dict = state_dict["teacher"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "student" in state_dict:
                state_dict = state_dict["student"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            # convert original convnext imagenet weights to timm compatible format
            out_dict = {}
            for k, v in state_dict.items():
                k = k.replace('downsample_layers.0.', 'stem.')
                k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
                k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
                k = k.replace('dwconv', 'conv_dw')
                k = k.replace('pwconv', 'mlp.fc')
                if 'grn' in k:
                    k = k.replace('grn.beta', 'mlp.grn.bias')
                    k = k.replace('grn.gamma', 'mlp.grn.weight')
                    v = v.reshape(v.shape[-1])
                k = k.replace('head.', 'head.fc.')
                if k.startswith('norm.'):
                    k = k.replace('norm', 'head.norm')
                if v.ndim == 2 and 'head' not in k:
                    model_shape = model.state_dict()[k].shape
                    v = v.reshape(model_shape)
                out_dict[k] = v
            for k in list(out_dict.keys()):
                if k.startswith('head'):
                    del out_dict[k]
            msg = model.load_state_dict(out_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(weight))
            print("missing keys:", msg.missing_keys)
        elif arch_name.lower().startswith("swin"):
            state_dict = torch.load(weight, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in state_dict:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del state_dict[k]
            msg = model.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(weight))
            print("missing keys:", msg.missing_keys)
        elif "radimagenet" in weight.lower():
            state_dict = torch.load(weight, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = {k.replace("backbone.0", "conv1"): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.1", "bn1"): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.4", "layer1"): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.5", "layer2"): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.6", "layer3"): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.7", "layer4"): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(weight))
            print("missing keys:", msg.missing_keys)
        else:
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
                              activation=args.activate,img_size=args.img_size)

    else:
        model = ClassificationNet(args.model_name.lower(), args.num_class, weight=args.proxy_dir,
                              activation=args.activate,img_size=args.img_size)

    return model

def build_segmentation_model(args,log_writter):
    if args.arch == "swin_upernet":
        model = UperNet_swin(args.backbone,img_size=args.img_size, num_classes=args.num_classes)
        if args.proxy_dir is not None:
            print("Loading pretrained weights", file=log_writter)
            checkpoint = torch.load(args.proxy_dir, map_location='cpu')
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "student" in checkpoint:
                checkpoint = checkpoint["student"]
            elif "model" in checkpoint:
                checkpoint = checkpoint["model"]
            checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
            checkpoint_model = {k.replace("encoder.", ""): v for k, v in checkpoint_model.items()}
            checkpoint_model = {k.replace("encoder_q.", ""): v for k, v in checkpoint_model.items()}
            checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            load_swin_pretrained(checkpoint_model, model.backbone, log_writter)

    elif args.arch == "convnext_upernet":
        model = UperNet_convnext(args.backbone,img_size=args.img_size, num_classes=args.num_classes)
        if args.proxy_dir is not None:
            print("Loading  pretrained weights", file=log_writter)
            checkpoint = torch.load(args.proxy_dir, map_location='cpu')
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint:
                checkpoint = checkpoint["model"]
            checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            import re
            out_dict = {}
            for k, v in checkpoint_model.items():
                k = k.replace('stem.','downsample_layers.0.')
                k = re.sub(r'stages.([0-9]+).blocks.([0-9]+)', r'stages.\1.\2', k)
                k = re.sub(r'stages.([0-9]+).downsample.([0-9]+)', r'downsample_layers.\1.\2', k)
                k = k.replace('conv_dw','dwconv' )
                k = k.replace('mlp.fc','pwconv')
                if 'grn' in k:
                    k = k.replace('mlp.grn.bias','grn.beta' )
                    k = k.replace('mlp.grn.weight','grn.gamma')
                    v = v.reshape(v.shape[-1])
                k = k.replace('head.fc.','head.', )
                if v.ndim == 2 and 'head' not in k:
                    model_shape = model.backbone.state_dict()[k].shape
                    v = v.reshape(model_shape)
                out_dict[k] = v

            for k in list(out_dict.keys()):
                if k.startswith('head'):
                    del out_dict[k]
            msg = model.backbone.load_state_dict(out_dict, strict=False)
            print('Loaded with msg: {}'.format(msg), file=log_writter)
    elif args.arch == "unet":
        if args.backbone == "resnet50":
            if args.init.lower() == "imagenet":
              model = smp.Unet(args.backbone, encoder_weights=args.init)
            else:
              model = smp.Unet(args.backbone, encoder_weights=args.proxy_dir)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=0, momentum=0.9, nesterov=False)
    if torch.cuda.is_available():
            model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
            model = model.cuda()
            cudnn.benchmark = True
    loss_scaler = NativeScaler()
    return model, optimizer,loss_scaler

def save_checkpoint(state,filename='model'):
    torch.save( state,filename + '.pth.tar')




