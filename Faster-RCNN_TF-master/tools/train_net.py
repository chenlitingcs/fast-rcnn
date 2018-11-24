#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import numpy as np
import sys
import pdb

def parse_args():
    """
    Parse(翻译：解析） input(下面函数调用） arguments
    """                                                                                '''device代表是cpu或者gpu,id代表u的代号
    
    
    slover代表模型的配置文件 iters代表模型的最大可迭代次数，默认7000次,  weights代表预训练权重文件的路径！！'''
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',      #cfg代表可选的配置文件
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',     #!imdb代表训练的数据集
                        help='dataset to train on',
                        default='kitti_train', type=str)
    parser.add_argument('--rand', dest='randomize',    #rand代表是否使用不同的随机数种子生成随机数
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name', #--network代表网络名称，一般具有固定的形式。常以'_train'结尾
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',         # --set的功能见下
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':#依旧不忘的从main函数开始看起，这个之前都忘记了
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        #import the 具体的函数，就不用再声明了,核心获取imdb roidb 数据
    imdb = get_imdb(args.imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)


    output_dir = get_output_dir(imdb, None)
    print('Output will be saved to `{:s}`'.format(output_dir))

    #对于三大，无愧于心是最好的解释，多活一段时间更好了，享福的日子不知道是否能到来了，哎
    #如果能活着，尽一点孝心也是好的
    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print(device_name)

    network = get_network(args.network_name)
    print('Use network `{:s}` in training'.format(args.network_name))

    train_net(network, imdb, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
