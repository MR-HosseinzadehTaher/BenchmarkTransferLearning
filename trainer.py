from utils import MetricLogger, ProgressLogger
from models import ClassificationNet
import time
import torch
from tqdm import tqdm
import math
from swin_transformer import UperNet_swin
from convnext import UperNet_convnext
import segmentation_models_pytorch as smp
import torch.backends.cudnn as cudnn

def train_one_epoch_classification(data_loader_train, device, model, criterion, optimizer, epoch, apply_activation=False):
  batch_time = MetricLogger('Time', ':6.3f')
  losses = MetricLogger('Loss', ':.4e')
  progress = ProgressLogger(
    len(data_loader_train),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))
  model.train()
  end = time.time()
  for i, (samples, targets) in enumerate(data_loader_train):
    samples, targets = samples.float().to(device), targets.float().to(device)
    outputs = model(samples)
    if apply_activation:
        outputs = torch.sigmoid(outputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.update(loss.item(), samples.size(0))
    batch_time.update(time.time() - end)
    end = time.time()
    if i % 50 == 0:
      progress.display(i)

def train_one_epoch_segmentation(model,train_loader, optimizer, loss_scaler, epoch,criterion,log_writter):
    model.train(True)
    batch_time = MetricLogger('Time', ':6.3f')
    data_time = MetricLogger('Data', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    criterion = criterion
    end = time.time()
    for idx, (img,mask) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = img.shape[0]
        img = img.float().cuda(non_blocking=True)
        mask = mask.float().cuda(non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = torch.sigmoid(model(img))
            if outputs.size()[-1] != mask.size()[-1]:
                outputs = F.interpolate(outputs, size=args.img_size, mode='bilinear')
            loss = criterion(mask, outputs)
        if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), file=log_writter)
                sys.exit(1)
            # update metric
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=None,
                    parameters=model.parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'lr {lr}\t'
                  'Total loss {ttloss.val:.5f} ({ttloss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, lr=optimizer.param_groups[0]['lr'], ttloss=losses), file=log_writter)
            log_writter.flush()

    return losses.avg

def evaluate_classification(data_loader_val, device, model, criterion, apply_activation=False):
  model.eval()
  with torch.no_grad():
    batch_time = MetricLogger('Time', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_val),
      [batch_time, losses], prefix='Val: ')
    end = time.time()
    for i, (samples, targets) in enumerate(data_loader_val):
      samples, targets = samples.float().to(device), targets.float().to(device)
      outputs = model(samples)
      if apply_activation:
          outputs = torch.sigmoid(outputs)
      loss = criterion(outputs, targets)
      losses.update(loss.item(), samples.size(0))
      batch_time.update(time.time() - end)
      end = time.time()
      if i % 50 == 0:
        progress.display(i)
  return losses.avg

def evaluate_segmentation(model, val_loader, epoch,criterion, log_writter):
    model.eval()
    losses = MetricLogger('Loss', ':.4e')
    criterion = criterion
    with torch.no_grad():
        for idx, (img, mask) in enumerate(val_loader):
            bsz = img.shape[0]
            img = img.float().cuda(non_blocking=True)
            mask = mask.float().cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = torch.sigmoid(model(img))
                if outputs.size()[-1] != mask.size()[-1]:
                    outputs = F.interpolate(outputs, size=args.img_size, mode='bilinear')
                loss = criterion(mask, outputs)
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), file=log_writter)
                sys.exit(1)
                # update metric
            losses.update(loss.item(), bsz)
            torch.cuda.synchronize()
            if (idx + 1) % 10 == 0:
                print('Evaluation: [{0}][{1}/{2}]\t'
                      'Total loss {ttloss.val:.5f} ({ttloss.avg:.5f})'.format(
                    epoch, idx + 1, len(val_loader), ttloss=losses), file=log_writter)
                log_writter.flush()
    return losses.avg


def test_classification(checkpoint, data_loader_test, device, args):
  model = ClassificationNet(args.model_name.lower(), args.num_class, activation=args.activate)
  print(model)
  modelCheckpoint = torch.load(checkpoint)
  state_dict = modelCheckpoint['state_dict']
  for k in list(state_dict.keys()):
    if k.startswith('module.'):
      state_dict[k[len("module."):]] = state_dict[k]
      del state_dict[k]
  msg = model.load_state_dict(state_dict)
  assert len(msg.missing_keys) == 0
  print("=> loaded pre-trained model '{}'".format(checkpoint))
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)
  model.eval()
  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()
  with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      targets = targets.cuda()
      y_test = torch.cat((y_test, targets), 0)
      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()
      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())
      out = model(varInput)
      if "convnext" in args.model_name.lower() or "swin" in args.model_name.lower() or "vit" in args.model_name.lower():
        out = torch.sigmoid(out)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)
  return y_test, p_test

def test_segmentation(args, model_save_path,data_loader_test,log_writter):
    print("testing....", file=log_writter)
    if args.arch == "swin_upernet":
        model = UperNet_swin(args.backbone,img_size=args.img_size, num_classes=args.num_classes)
    elif args.arch == "convnext_upernet":
        model = UperNet_convnext(args.backbone,img_size=args.img_size, num_classes=args.num_classes)
    elif args.arch == "unet":
        if args.backbone == "resnet50":
              model = smp.Unet(args.backbone, encoder_weights=None)

    checkpoint = torch.load(model_save_path, map_location='cpu')
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint_model)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        model = model.cuda()
        cudnn.benchmark = True
    model.eval()
    with torch.no_grad():
        test_p = None
        test_y = None
        for idx, (img, mask) in enumerate(data_loader_test):
            bsz = img.shape[0]
            img = img.float().cuda(non_blocking=True)
            mask = mask.float().cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = torch.sigmoid(model(img))
                if outputs.size()[-1] != mask.size()[-1]:
                    outputs = F.interpolate(outputs, size=args.img_size, mode='bilinear')
                outputs = outputs.cpu().detach()
                mask = mask.cpu().detach()
                if test_p is None and test_y is None:
                    test_p = outputs
                    test_y = mask
                else:
                    test_p = torch.cat((test_p, outputs), 0)
                    test_y = torch.cat((test_y, mask), 0)
                torch.cuda.empty_cache()
                if (idx + 1) % 20 == 0:
                    print("Testing Step[{}/{}] ".format(idx + 1, len(data_loader_test)), file=log_writter)
                    log_writter.flush()

        print("Done testing iteration!", file=log_writter)
        log_writter.flush()
        test_p = test_p.numpy()
        test_y = test_y.numpy()
        test_y = test_y.reshape(test_p.shape)
        return test_y, test_p
