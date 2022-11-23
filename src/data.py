import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from CodeBERT.CodeBERT.code2nl.run import Example, convert_examples_to_features


def get_absolute_path(p):
    return Path(__file__).parent.parent / p


def load_dataset(path):
    path = get_absolute_path(path)
    frame = pd.read_feather(path)
    frame['length'] = frame.method.apply(len)
    frame = frame.query('length<10000')
    examples = []
    upper = re.compile('[A-Z]')
    for idx, source, name in zip(frame.index, frame.method, frame.name):
        target = ' '.join(name_to_sequence(name, upper))
        ex = Example(idx=idx, source=source, target=target)
        examples.append(ex)

    return examples


def name_to_sequence(name, pattern):
    prev = 0
    seq = []
    for match in pattern.finditer(name):
        seq.append(name[prev].lower() + name[prev + 1:match.start()])
        prev = match.start()

    seq.append(name[prev].lower() + name[prev + 1:len(name)])
    return seq


def convert_examples_to_dataset(tokenizer, train_args, examples, stage):
    train_features = convert_examples_to_features(examples, tokenizer, train_args, stage=stage)
    all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
    dataset = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
    return dataset
