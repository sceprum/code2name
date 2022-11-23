import os
import random

import numpy as np
import torch
import transformers
from torch.optim import AdamW
from torch.utils.data import SequentialSampler, DataLoader, TensorDataset
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig, AdamW

from CodeBERT.CodeBERT.code2nl import bleu
from CodeBERT.CodeBERT.code2nl.model import Seq2Seq
from torch import nn

from src.data import load_dataset, convert_examples_to_dataset


def load_model(name, train_args, cache_dir=None, model_class=RobertaModel, config_class=RobertaConfig,
               weights_path=None, logger=None):
    tokenizer_class = RobertaTokenizer
    model_class = _convert_to_class(model_class)
    config_class = _convert_to_class(config_class)

    config = config_class.from_pretrained(name, cache_dir=cache_dir)
    tokenizer = tokenizer_class.from_pretrained(name, cache_dir=cache_dir)
    encoder = model_class.from_pretrained(name, config=config, cache_dir=cache_dir)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=train_args.beam_size, max_length=train_args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    if weights_path is not None:
        if logger is not None:
            logger.info("reload model from {}".format(weights_path))
        model.load_state_dict(torch.load(weights_path))

    return model, tokenizer


def _convert_to_class(arg):
    return getattr(transformers, arg) if isinstance(arg, str) else arg


def decode_predicts(predicts, tokenizer):
    predictions = []
    for pred_ids in predicts:
        pred_ids = pred_ids[:np.argmax(pred_ids == 0)]  # Remove elements after first zero
        text = tokenizer.decode(pred_ids, clean_up_tokenization_spaces=False)
        predictions.append(text)
    return predictions


def compute_blue(args, blue_examples, logger, predictions):
    with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
            os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
        for i, ref, gold in enumerate(zip(blue_examples, predictions)):
            predictions.append(str(gold.idx) + '\t' + ref)
            f.write(str(gold.idx) + '\t' + ref + '\n')
            f1.write(str(gold.idx) + '\t' + gold.target + '\n')
    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
    dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
    logger.info("  " + "*" * 20)
    return dev_bleu


def select_subset_to_estimate_bleu(eval_dataset, train_args, examples):
    subset_count = min(1000, len(eval_dataset))
    all_ids = list(np.arange(len(eval_dataset)))
    ids = random.sample(all_ids, subset_count)
    eval_data = TensorDataset(*eval_dataset[ids])
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=train_args.eval_batch_size)
    return eval_dataloader, eval_data, [examples[i] for i in ids]


def evaluate_model(device, eval_dataloader, model):
    model.eval()
    eval_loss, tokens_num = 0, 0
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask, target_ids, target_mask = batch

        with torch.no_grad():
            _, loss, num = model(source_ids=source_ids, source_mask=source_mask,
                                 target_ids=target_ids, target_mask=target_mask)
        eval_loss += loss.sum().item()
        tokens_num += num.sum().item()
    model.train()
    eval_loss = eval_loss / tokens_num
    return eval_loss


def collect_predicts(device, eval_dataloader, model):
    # Start Evaling model
    model.eval()
    predicts = []
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch

            y_pred = model(source_ids=source_ids, source_mask=source_mask)
            predicts.append(y_pred.cpu().numpy())
    predicts = np.vstack(predicts)
    return predicts


def make_step(loss, optimizer, scheduler):
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()


def log_training_info(logger, num_train_optimization_steps, train_args, train_data):
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Batch size = %d", train_args.train_batch_size)
    logger.info("  Num epoch = %d",
                num_train_optimization_steps * train_args.train_batch_size // len(train_data))


def create_optimizer(model, train_args):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': train_args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=train_args.learning_rate, eps=train_args.adam_epsilon)
    return optimizer


def get_dataloader(tokenizer, train_args, train_path, sampler_class, stage):
    train_examples = load_dataset(train_path)
    train_data = convert_examples_to_dataset(tokenizer, train_args, train_examples, stage=stage)
    train_sampler = sampler_class(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler,
        batch_size=train_args.train_batch_size)
    return train_dataloader, train_data, train_examples
