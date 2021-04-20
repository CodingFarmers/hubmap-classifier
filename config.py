import argparse
import os

parser = argparse.ArgumentParser()

# augmentations
parser.add_argument('--vflip', type=float, default=0.4)
parser.add_argument('--hflip', type=float, default=0.3)
parser.add_argument('--sharpen', type=float, default=0.1)
parser.add_argument('--ssr', type=float, default=0.2)
parser.add_argument('--channelshuffle', type=float, default=0.2)
parser.add_argument('--oneofcgb', type=float, default=0.1)
parser.add_argument('--mixup', type=float, default=0)

# data root 
parser.add_argument('--data-root', type=str, default='hubmap-256/')

# model 
parser.add_argument('--model', type=str, default='tf_efficientnet_b0_ns')

# mixup
# parser.add_argument('--mixup', type=bool, default=False)
parser.add_argument('--alpha', type=float, default=0.2)


# hyper 
parser.add_argument('--thersh', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-6)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--accum_iter', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--random-search', type=bool, default=False)
parser.add_argument('--verbose', type=int, default=1)

# other params
parser.add_argument('--nfold', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=str, default='2,3')


# lr range test
parser.add_argument('--find-lr', action='store_true')
parser.add_argument('--start-lr', type=float, default=1e-4)
parser.add_argument('--end-lr', type=float, default=10)

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
