from utils import MetricLogger, ProgressLogger
from models import ClassificationNet
import time
import torch
from tqdm import tqdm


def train_one_epoch(data_loader_train, device,model, criterion, optimizer, epoch):
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

    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.update(loss.item(), samples.size(0))
    batch_time.update(time.time() - end)
    end = time.time()

    if i % 50 == 0:
      progress.display(i)


def evaluate(data_loader_val, device, model, criterion):
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

      loss = criterion(outputs, targets)

      losses.update(loss.item(), samples.size(0))
      losses.update(loss.item(), samples.size(0))
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

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
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test

def test_segmentation(model, model_save_path,data_loader_test, device,log_writter):
    print("testing....", file=log_writter)
    checkpoint = torch.load(model_save_path)
    state_dict = checkpoint["state_dict"]
    for k in list(state_dict.keys()):
      if k.startswith('module.'):
        state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(device)
    with torch.no_grad():
        test_p = None
        test_y = None
        model.eval()
        for batch_ndx, (x_t, y_t) in enumerate(tqdm(data_loader_test)):
            x_t, y_t = x_t.float().to(device), y_t.float().to(device)
            pred_t = model(x_t)
            if test_p is None and test_y is None:
                test_p = pred_t
                test_y = y_t
            else:
                test_p = torch.cat((test_p, pred_t), 0)
                test_y = torch.cat((test_y, y_t), 0)

            if (batch_ndx + 1) % 5 == 0:
                print("Testing Step[{}]: ".format(batch_ndx + 1) , file=log_writter)
                log_writter.flush()

        print("Done testing iteration!", file=log_writter)
        log_writter.flush()

    test_p = test_p.cpu().detach().numpy()
    test_y = test_y.cpu().detach().numpy()

    return test_y, test_p


