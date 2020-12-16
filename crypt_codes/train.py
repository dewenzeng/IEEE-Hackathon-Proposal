import U_net_Model
import myconfig
import os
import torch
import torch.nn.functional as F
from covid19_dataset import COVID19Dataset
from lr_scheduler import LR_Scheduler
from metrics import SegmentationMetric
from utils import *

def train_fn(data_loader, model, criterion, epoch, optimizer, scheduler, args):
    model.train()
    losses = AverageMeter()
    for batch_idx, tup in enumerate(data_loader):
        img, label = tup
        image_var = img.float().to(args.device)
        label = label.long().to(args.device)
        scheduler(optimizer, batch_idx, epoch)
        x_out = model(image_var)
        loss = criterion(x_out, label)
        losses.update(loss.item(), image_var.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Training epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, lr:{optimizer.param_groups[0]['lr']:.6f}, loss:{losses.avg:.4f}")
    return losses.avg

def test_fn(data_loader, model, epoch, args):
    model.eval()
    metric_val = SegmentationMetric(args.classes)
    metric_val.reset()
    with torch.no_grad():
        for batch_idx, tup in enumerate(data_loader):
            img, label = tup
            # measure data loading time
            image_var = img.float().to(args.device)
            label = label.long().to(args.device)
            x_out = model(image_var)
            x_out = F.softmax(x_out, dim=1)
            metric_val.update(label.long(), x_out)
            pixAcc, mIoU, Dice = metric_val.get()
            print(f"Testing epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, mean Dice:{Dice}")
    return Dice

args = myconfig.get_config()
args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
# define model
model=U_net_Model.UNet(in_channels=args.input_channels,out_channels=args.classes,init_features=args.initial_filter_size)
if args.resume:
    print('loading from saved model ' + args.pretrained_model_path)
    model.load_state_dict(torch.load(args.pretrained_model_path)['state_dict'])

model.to(args.device)

# load dataset
train_dataset = COVID19Dataset(data_dir=args.train_dir, direction='X')
test_dataset = COVID19Dataset(data_dir=args.test_dir, direction='X')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader))

print('start training...')
for epoch in range(args.epochs):
    train_loss = train_fn(train_loader, model, criterion, epoch, optimizer, scheduler, args)
    test_dice = test_fn(test_loader, model, epoch, args)
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    save_dict = {"state_dict": model.state_dict()}
    torch.save(save_dict, os.path.join(args.results_dir, "latest.pth"))

print('training finished...')

