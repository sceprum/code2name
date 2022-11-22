import os

from pipeline.data import convert_examples_to_dataset
from pipeline.utils import create_logger

# TODO: remove
os.environ['TRANSFORMERS_OFFLINE'] = 'TRUE'

from torch.utils.tensorboard import SummaryWriter

from src.pipeline import load_model

import numpy as np
from itertools import cycle
import random
from types import SimpleNamespace
import hydra
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from CodeBERT.CodeBERT.code2nl import bleu
from CodeBERT.CodeBERT.code2nl.run import convert_examples_to_features, set_seed
from src.data import load_dataset


@hydra.main(config_path='../config', config_name='train')
def train(config):
    logger = create_logger()
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    args = SimpleNamespace(**config)
    training_conf = config['training']
    train_args = SimpleNamespace(**training_conf, n_gpu=1)
    # Set seed
    set_seed(train_args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    model, tokenizer = load_model(**args.model, train_args=train_args, cache_dir=args.cache_dir)

    model.to(device)

    writer = SummaryWriter(log_dir='tb-events')

    data_paths = args.data
    if args.do_train:
        # Prepare training data loader
        train_path = data_paths['train']
        train_examples = load_dataset(train_path)
        train_data = convert_examples_to_dataset(tokenizer, train_args, train_examples, stage='train')

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler,
            batch_size=train_args.train_batch_size // train_args.gradient_accumulation_steps)

        num_train_optimization_steps = train_args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': train_args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=train_args.learning_rate, eps=train_args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", train_args.train_batch_size)
        logger.info("  Num epoch = %d",
                    num_train_optimization_steps * train_args.train_batch_size // len(train_examples))

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)
        eval_flag = True
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                               target_mask=target_mask)

            writer.add_scalar('Loss/train', loss, step)

            # if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
            if train_args.gradient_accumulation_steps > 1:
                loss = loss / train_args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss = round(tr_loss * train_args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % train_args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            if args.do_eval and ((global_step + 1) % train_args.eval_steps == 0) and eval_flag:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = load_dataset(data_paths['val'])
                    eval_data = convert_examples_to_dataset(tokenizer, train_args, eval_examples, stage='dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=train_args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", train_args.eval_batch_size)

                # Start Evaling model
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
                # Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}

                writer.add_scalar('Loss/train_', train_loss, step)
                writer.add_scalar('Loss/val', eval_loss, step)

                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                # save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss < best_loss:
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    logger.info("  " + "*" * 20)
                    best_loss = eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                    # Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = load_dataset(data_paths['val'])
                    eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, train_args, stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=train_args.eval_batch_size)

                model.eval()
                p = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask = batch
                    with torch.no_grad():
                        preds = model(source_ids=source_ids, source_mask=source_mask)
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions = []
                with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
                        os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
                    for ref, gold in zip(p, eval_examples):
                        predictions.append(str(gold.idx) + '\t' + ref)
                        f.write(str(gold.idx) + '\t' + ref + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target + '\n')

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
                dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  " + "*" * 20)
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_test:
        files = []
        if data_paths['val'] is not None:
            files.append(data_paths['val'])
        if data_paths['test'] is not None:
            files.append(data_paths['test'])
        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = load_dataset(file)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, train_args, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids, all_source_mask)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=train_args.eval_batch_size)

            model.eval()
            p = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    preds = model(source_ids=source_ids, source_mask=source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)
            model.train()
            predictions = []
            with open(os.path.join(args.output_dir, "test_{}.output".format(str(idx))), 'w') as f, open(
                    os.path.join(args.output_dir, "test_{}.gold".format(str(idx))), 'w') as f1:
                for ref, gold in zip(p, eval_examples):
                    predictions.append(str(gold.idx) + '\t' + ref)
                    f.write(str(gold.idx) + '\t' + ref + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target + '\n')

            (goldMap, predictionMap) = bleu.computeMaps(predictions,
                                                        os.path.join(args.output_dir, "test_{}.gold".format(idx)))
            dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            logger.info("  " + "*" * 20)


if __name__ == '__main__':
    train()
