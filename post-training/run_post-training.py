import argparse
import math
import os
import random
import time
import torch
import numpy as np
import collections

from loguru import logger
from sklearn import metrics

import shutil

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from transformers import (
    get_constant_schedule, 
    get_constant_schedule_with_warmup, 
    get_cosine_schedule_with_warmup, 
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from transformers import SplinterConfig, SplinterForPreTraining, AutoConfig
from torch.utils.data import IterableDataset
import pickle
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./path_of_splinter_model')

def parse_args():
    parser = argparse.ArgumentParser(description="BERT for Sequence Classification.")
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="Node rank for distributed training."
    )
    parser.add_argument(
        "--init_from_checkpoint",
        action='store_true',
        help="whether to load checkpoint from disk"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action='store_true',
        help="whether to load checkpoint from disk"
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.1,
        type=float,
        help='warmup ratio for scheduler'
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="./path_of_splinter_model",
        type=str,
        help="The pretrained huggingface model name or path."
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        required=True,
        help="The full path of train_data_file."
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--scheduler",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        default="linear",
        help="The name of the scheduler to use."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--warmup_steps",
        default=20000,
        type=int,
        help="Warmup steps over the training process."
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for initialization."
    )
    parser.add_argument(
        "--save_steps",
        default=100,
        type=int,
        help="The interval steps to save checkpoints."
    )
    parser.add_argument(
        "--logging_steps",
        default=20,
        type=int,
        help="The interval steps to logging."
    )
    parser.add_argument(
        "--mask",
        action='store_true',
        help="whether to add mask loss"
    )
    parser.add_argument(
        "--mask_ratio",
        default=15,
        type=int,
        help="The interval steps to logging."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--saved_dir",
        default="./pytorch_train_output",
        type=str,
        help="The output directory where the model checkpoints will be written."
    )
    args = parser.parse_args()
    return args


def get_dataset(train_data_file):
    files = []
    train_datas = train_data_file.strip(' ').split(':')
    for train_data in train_datas:
        dataset_name, dataset_number = train_data.strip(' ').split(',')
        an_start_number, an_end_number = dataset_number.split('_')[0].split('-')
        for i in range(int(an_start_number), int(an_end_number)+1):
            files.append('pretraining_data/answerable/{0}/pretrain_data_{1}.pkl'.format(dataset_name, i))
        if len(dataset_number.split('_')) == 2:
            ua_start_number, ua_end_number = dataset_number.split('_')[1].split('-')
        else:
            ua_start_number, ua_end_number = 1, 0
        for i in range(int(ua_start_number), int(ua_end_number)+1):
            files.append('pretraining_data/unanswerable/{0}/pretrain_data_{1}.pkl'.format(dataset_name, i))
    total_features = []
    for file in files:
        f = open(file, 'rb')
        features = pickle.load(f)
        total_features.extend(features)
        f.close()
    input_ids = torch.tensor([f[0] for f in total_features], dtype=torch.long)
    attention_mask = torch.tensor([f[1] for f in total_features], dtype=torch.long)
    token_type_ids = torch.tensor([f[2] for f in total_features], dtype=torch.long)
    start_position = torch.tensor([f[3] for f in total_features], dtype=torch.long)
    end_position = torch.tensor([f[4] for f in total_features], dtype=torch.long)
    trainDataset = TensorDataset(input_ids, attention_mask, token_type_ids, start_position, end_position)
    
    return trainDataset
    
def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    valid = ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
    if name not in valid:
        raise ValueError("name {} not in valid scheduler type: {}".format(name, valid))
    
    if name == 'constant':
        return get_constant_schedule(optimizer)
    
    # All other schedulers require num_warmup_steps
    if num_warmup_steps is None:
        raise ValueError("{} scheduler requires num_warmup_steps, please provide that argument.".format(name))
    
    if name == 'constant_with_warmup':
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
    
    # All other schedulers require num_training_steps
    if num_training_steps is None:
        raise ValueError("{} scheduler requires num_training_steps, please provide that argument.".format(name))
    
    if name == 'cosine':
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    
    if name == 'cosine_with_restarts':
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    
    if name == 'polynomial':
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_pretrained_model(pretrained_model_name_or_path):
    origin_state_dict = torch.load(os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin'))
    new_state_dict = collections.OrderedDict()
    for name in origin_state_dict.keys():
        if 'bert' in name:
            new_state_dict[str(name).replace('bert', 'splinter')] = origin_state_dict[name]
        elif 'cls.span_predictions' in name:
            new_name = str(name).replace('cls.span_predictions', 'splinter_qass')
            if '_classifier' in name:
                new_name = new_name + '.weight'
            new_state_dict[new_name] = origin_state_dict[name]
        else:
            new_state_dict[name] = origin_state_dict[name]
    return new_state_dict

def train(args):
    # check 
    if (
            os.path.exists(args.saved_dir)
            and os.listdir(args.saved_dir)
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    
    if args.overwrite_output_dir and os.path.isdir(args.saved_dir) and (args.local_rank == -1 or args.local_rank == 0):
        shutil.rmtree(args.saved_dir)
    if args.local_rank == -1 or args.local_rank == 0:
        os.mkdir(args.saved_dir)

    # set summarywriter
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(args.saved_dir)
    # set device
    n_gpu = torch.cuda.device_count()
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

    # set seed
    set_seed(args.seed)
    
    # build dataloader
    
    train_dataset = get_dataset(args.train_data_file)
    if args.local_rank == -1:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    else:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            pin_memory=True
        )
    
    # load pretrained model
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    config = SplinterConfig()
    config.vocab_size = 28996
    model = SplinterForPreTraining(config)

    if args.init_from_checkpoint:
        model.load_state_dict(load_pretrained_model(args.pretrained_model_name_or_path), strict=False)
        logger.info("initializing weights from: {}".format(args.pretrained_model_name_or_path))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(device)
    if n_gpu > 1:
        if args.local_rank == -1:
            model = nn.DataParallel(model)
        else:
            model = DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank,
                find_unused_parameters=True
            )
    
    # preparation before training
    num_training_steps = len(train_dataloader) * args.epochs
    warmup_steps = math.ceil(num_training_steps * args.warmup_ratio)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-6)
    scheduler = get_scheduler(
        name=args.scheduler, optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    logger.info("******************* Model train hyper-parameters ******************************")
    logger.info("\tusing scheduler: {}".format(args.scheduler))
    logger.info("\twarmup_steps: {}".format(warmup_steps))
    logger.info("\ttrain_total_steps: {}".format(num_training_steps))
    logger.info("\ttrain_total_steps: {}".format(num_training_steps))
    
    global_step = 0
    os.makedirs(args.saved_dir, exist_ok=True)
    
    # begin training
    model.train()
    tic_train = time.time()
    logger.info("***** Running training *****")
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_dataloader, start=1):
            model.zero_grad()

            input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)
            
            loss = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                start_positions = start_positions,
                end_positions = end_positions,
            ).loss
            if n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            if loss.item() > 1000000000.0:
                optimizer.zero_grad()
            else:
                optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            if args.local_rank in [-1, 0]:
                # print(global_step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                if loss.item() < 10000000000.0:
                    writer.add_scalar("loss", loss.item(), global_step)

            if global_step % args.logging_steps == 0 and (args.local_rank == -1 or args.local_rank == 0):
                time_diff = time.time() - tic_train
                logger.info("global step: %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                            % (global_step, epoch, step, loss, args.logging_steps / time_diff))
                tic_train = time.time()
            
            if global_step % args.save_steps == 0 and (args.local_rank == -1 or args.local_rank == 0):
                saved_model_file = os.path.join(args.saved_dir, "pytorch_model_{}_{}_{}_{:.4f}.bin".format(global_step, epoch, step, loss.item()))
                print(saved_model_file)
                if n_gpu > 1:
                    torch.save(model.module.state_dict(), saved_model_file)
                else:
                    torch.save(model.state_dict(), saved_model_file)
                tic_train = time.time()

if __name__ == "__main__":
    args = parse_args()
    train(args)
