from utils import runner
from utils import parser, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings(action='ignore')


def main():
    # args parse
    args = parser.get_args()
    # set cuda gpu
    args.use_gpu = torch.cuda.is_available() 
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # set logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    
    # define the tensorboard writer
    train_writer = None
    if not args.test:
        train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
    # set config
    config = get_config(args, logger=logger)
    config.dataset.train.others.bs = config.total_bs
    
    # log
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    
    # set random seed
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed, deterministic=args.deterministic) # seed + rank, for augmentation
    
    runner.run_net(args, config, train_writer)




if __name__ == '__main__':
    main()
