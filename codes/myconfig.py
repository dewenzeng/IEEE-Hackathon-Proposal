import argparse
import os

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--num_works", type=int, default=8)
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./saved_model', help='results dir')

# Data
parser.add_argument("--train_dir", type=str, default="../dataset/train/")
parser.add_argument("--test_dir", type=str, default="../dataset/test/")
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--seed', type=int, default=12345)

# Model
parser.add_argument("--input_channels", type=int, default=5)
parser.add_argument("--initial_filter_size", type=int, default=16)
parser.add_argument("--classes", type=int, default=2)

# Train
parser.add_argument("--resume", default=False, action='store_true')
parser.add_argument("--pretrained_model_path", type=str, default="../pretrained_model/model_infection/best_model-X.pth")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
parser.add_argument("--lr_scheduler", type=str, default='cos')

# Test config
parser.add_argument("--model_infection_dir", type=str, default="/afs/crc.nd.edu/user/d/dzeng2/code/covid19/pretrained_model/model_infection/")
parser.add_argument("--model_lung_dir", type=str, default="/afs/crc.nd.edu/user/d/dzeng2/code/covid19/pretrained_model/model_lung/")
parser.add_argument("--test_data_dir", type=str, default="/afs/crc.nd.edu/user/d/dzeng2/code/covid19/dataset/test/")

def save_args(obj, defaults, kwargs):
    for k,v in defaults.iteritems():
        if k in kwargs: v = kwargs[k]
        setattr(obj, k, v)

def get_config():
    config = parser.parse_args()
    return config
