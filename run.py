import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.tools import string_split


def main():
    parser = argparse.ArgumentParser(description='TSMAE')
    parser.add_argument('--mode', default='finetune', type=str, help='Name of model to train, options: [pretrain, finetune, Transformer]')
    parser.add_argument('--data', type=str, required=False, default='ETTh1',
                        help='name of dataset')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='MTPNet_Linear', help='model name, options: [MTPNet, MTPNet_Linear]')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

    parser.add_argument('--root_path', type=str, default='./data/datasets/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='location of the data file')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length for encoder, look back window')
    parser.add_argument('--label_len', type=int, default=96, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length, horizon')
    parser.add_argument('--features', type=str, default='M', choices=['S', 'M'],
                        help='features S is univariate, M is multivariate')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--data_split', type=str, default='0.6,0.2,0.2',
                        help='train/val/test split, can be ratio or number')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--data_dim', type=int, default=7, help='Number of dimensions of the MTS data (D)')
    parser.add_argument('--embed_dim', type=int, default=8, help='encoder input size')
    parser.add_argument('--decoder_embed_dim', type=int, default=8, help='encoder input size')


    parser.add_argument('--n_heads', type=int, default=4, help='number of multihead attention')
    parser.add_argument('--encoder_depth', type=int, default=2, help='batch size of train input data')
    parser.add_argument('--decoder_depth', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of MLP in transformer')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--H_depth', type=int, default=1, help='The depth of hierarchical transformer')
    parser.add_argument('--patch_size', type=str, default='12, 24', help='patch sizes of hierarchical architecture')
    parser.add_argument('--trend_patch_size', type=str, default='24', help='patch sizes of Trend hierarchical architecture')
    parser.add_argument('--moving_avg', type=str, default='13, 17', help='window size of moving average')

    parser.add_argument('--seed', type=int, default=2023, help='Random Seed')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--loss', type=str, default='L1', help='dropout')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0', help='multiple gpu')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--lradj', type=int, default=3, help='adjust learning rate')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--load_pretrained_model', type=bool, default=False, help='flag for wether load encoder from pretrained model')
    parser.add_argument('--ala_type', type=str, default='False', help='ablation study type')


    args = parser.parse_args()
    args.mode = 'finetune'

    args.patch_size = [int(i) for i in args.patch_size.split(', ')]
    args.trend_patch_size = [int(i) for i in args.trend_patch_size.split(', ')]
    args.H_depth = len(args.patch_size)
    args.moving_avg = [int(i) for i in args.moving_avg.split(', ')]
    args.lradj = 'CosineAnnealing'
    args.desktop_idx = 5

    # Model's parameters
    args.decoder_embed_dim = args.embed_dim
    args.d_ff = args.embed_dim*args.n_heads if args.d_ff == None else args.d_ff
    args.output_attention = False

    # Optimization's parameters
    args.use_amp = False
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'ETTh1': {'data': 'ETT-data/ETTh1.csv', 'data_dim': 7, 'split': [12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24]},
        'ETTh2': {'data': 'ETT-data/ETTh2.csv', 'data_dim': 7, 'split': [12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24]},
        'ETTm1': {'data': 'ETT-data/ETTm1.csv', 'data_dim': 7, 'split': [4 * 12 * 30 * 24, 4 * 4 * 30 * 24, 4 * 4 * 30 * 24]},
        'ETTm2': {'data': 'ETT-data/ETTm2.csv', 'data_dim': 7, 'split': [4 * 12 * 30 * 24, 4 * 4 * 30 * 24, 4 * 4 * 30 * 24]},
        'electricity': {'data': 'STEE/electricity.csv', 'data_dim': 321, 'split': [0.7, 0.1, 0.2]},
        'Weather': {'data': 'STEE/weather.csv', 'data_dim': 21, 'split': [0.7, 0.1, 0.2]},
        'ILI': {'data': 'national_illness.csv', 'data_dim': 7, 'split': [0.7, 0.1, 0.2]},
        'Traffic': {'data': 'traffic.csv', 'data_dim': 862, 'split': [0.7, 0.1, 0.2]},
        'Exchange': {'data': 'STEE/exchange_rate.csv', 'data_dim': 8, 'split': [0.7, 0.1, 0.2]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.data_dim = data_info['data_dim']
        args.data_split = data_info['split']
    else:
        args.data_split = string_split(args.data_split)

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_segl{}_dm{}_Hlvl{}_nh{}_el{}_dl{}_df{}'.format(
                args.mode,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.patch_size,
                args.embed_dim,
                args.H_depth,
                args.n_heads,
                args.encoder_depth,
                args.decoder_depth,
                1)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_segl{}_dm{}_Hlvl{}_nh{}_el{}_dl{}_df{}'.format(
            args.mode,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.patch_size,
            args.embed_dim,
            args.H_depth,
            args.n_heads,
            args.encoder_depth,
            args.decoder_depth,
            1)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    print('Finished Training')