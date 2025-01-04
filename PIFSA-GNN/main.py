import argparse
from PIFSA_GNN import train

parser = argparse.ArgumentParser()

# movie20
# parser.add_argument('--dataset', type=str, default='movie20')
# parser.add_argument('--aggregator', type=str, default='sum')
# parser.add_argument('--n_epochs', type=int, default=10)
# parser.add_argument('--neighbor_sample_size', type=int, default=8)
# parser.add_argument('--dim', type=int, default=16)
# parser.add_argument('--n_iter', type=int, default=1)
# parser.add_argument('--batch_size', type=int, default=4096)
# parser.add_argument('--l2_weight', type=float, default=1e-6)
# parser.add_argument('--lr', type=float, default=2e-2)
# args = parser.parse_args(['--l2_weight', '1e-6'])

# restaurant
# parser.add_argument('--dataset', type=str, default='restaurant')
# parser.add_argument('--aggregator', type=str, default='sum')
# parser.add_argument('--n_epochs', type=int, default=5)
# parser.add_argument('--neighbor_sample_size', type=int, default=32)
# parser.add_argument('--dim', type=int, default=16)
# parser.add_argument('--n_iter', type=int, default=1)
# parser.add_argument('--batch_size', type=int, default=4096)
# parser.add_argument('--l2_weight', type=float, default=5e-6)
# parser.add_argument('--lr', type=float, default=2e-2)
# args = parser.parse_args(['--l2_weight','5e-6'])

# music
parser.add_argument('--dataset', type=str, default='music')
parser.add_argument('--aggregator', type=str, default='concat')
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--neighbor_sample_size', type=int, default=8)
parser.add_argument('--dim', type=int, default=32)
parser.add_argument('--n_iter', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--l2_weight', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=5e-4)
args = parser.parse_args(['--l2_weight', '1e-4 '])

if __name__ == '__main__':

    train(args)