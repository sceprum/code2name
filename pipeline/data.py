import torch
from torch.utils.data import TensorDataset

from CodeBERT.CodeBERT.code2nl.run import convert_examples_to_features


def convert_examples_to_dataset(tokenizer, train_args, examples, stage):
    train_features = convert_examples_to_features(examples, tokenizer, train_args, stage=stage)
    all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
    dataset = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
    return dataset
