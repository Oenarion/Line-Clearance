import torch
import numpy as np
import os
import argparse
from train import *
from test import *
from threshold_selection import *
from omegaconf import OmegaConf


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"


def train(config):
    torch.manual_seed(42)
    np.random.seed(42)
    train_on_device(config)


def detection(config):
    test(config)

def threshold_computation(config):
    threshold(config)


def parse_args():
    cmdline_parser = argparse.ArgumentParser('DRAEM')    
    cmdline_parser.add_argument('-cfg', '--config', 
                                default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
                                help='config file')
    cmdline_parser.add_argument('--train', 
                                default= False, 
                                help='Train DRAEM model')
    cmdline_parser.add_argument('--detection', 
                                default= False, 
                                help='Detection anomalies')
    cmdline_parser.add_argument('--threshold',
                                default=False,
                                help='Computes the optimal threshold based on nominal data of the validation set')
    args, _ = cmdline_parser.parse_known_args()
    return args


    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    config = OmegaConf.load(args.config)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if args.train:
        print('Training...')
        train(config)
    if args.threshold:
        print('Computing thresholds...')
        threshold_computation(config)
    if args.detection:
        print('Detecting Anomalies...')
        detection(config)
