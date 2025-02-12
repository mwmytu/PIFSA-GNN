import argparse
import numpy as np
from time import time
from data_loader import load_data
from train import train
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(556)

parser = argparse.ArgumentParser()


# Movie-20m
# parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
# parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=16, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
# parser.add_argument('--ls_weight', type=float, default=1.0, help='weight of LS regularization')
# parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')

# Movie-1m
# parser.add_argument('--dataset', type=str, default='movie1m', help='which dataset to use')
# parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
# parser.add_argument('--ls_weight', type=float, default=0.1, help='weight of LS regularization')
# parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')

# Restaurant
# parser.add_argument('--dataset', type=str, default='restaurant', help='which dataset to use')
# parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
# parser.add_argument('--ls_weight', type=float, default=0.5, help='weight of LS regularization')
# parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')

# Music
# parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
# parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
# parser.add_argument('--ls_weight', type=float, default=0.1, help='weight of LS regularization')
# parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')

# Yelp
# parser.add_argument('--dataset', type=str, default='yelp', help='which dataset to use')
# parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
# parser.add_argument('--ls_weight', type=float, default=0.1, help='weight of LS regularization')
# parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')

# Book
# parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
# parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
# parser.add_argument('--ls_weight', type=float, default=0.5, help='weight of LS regularization')
# parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')

show_loss = False
show_time = False
show_topk = False

t = time()

args = parser.parse_args()
data = load_data(args)
train(args, data, show_loss, show_topk)

if show_time:
    print('time used: %d s' % (time() - t))
