import argparse

parser = argparse.ArgumentParser(description='Pruning filters for efficient ConvNets')

parser.add_argument('--model', typ=str, default='ResNet34',
                    help='Model to prune: "ResNet34"|"ResNet50"|"MobileNet"')
parser.add_argument('--data', type=str, default='',
                    help='Path to imagenet validation data')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--in_channel', type=int, default=3,
                    help='input channel (grey scale or RGB) default=3')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 20)')
parser.add_argument('-v', default='A', type=str,
                    help='version of the pruned model')

args = parser.parse_args()