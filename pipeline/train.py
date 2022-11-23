import os
from src.logging import log_metrics, save_best_model, save_last_checkpoint
from pipeline.utils import create_logger
from torch.utils.tensorboard import SummaryWriter
from src.pipeline import load_model, evaluate_model, make_step, log_training_info, create_optimizer, get_dataloader

import numpy as np
from itertools import cycle
from types import SimpleNamespace
import hydra
import torch
from torch.utils.data import SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from CodeBERT.CodeBERT.code2nl.run import set_seed


class TrainingPipeline:
    """
    This class implements refactored version of code2nl training pipeline.
    """

    # Is still looks cluttered (but much better than original code) and should be replaced to LightningModule-based
    # pipeline from src/lightning_pipeline.py in the future

    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(log_dir='tb-events')
        self.args = SimpleNamespace(**self.config)
        training_conf = self.config['training']
        self.train_args = SimpleNamespace(**training_conf, n_gpu=1)
        set_seed(self.train_args)
        self.best_bleu = 0
        self.best_loss = 1e6

    def run(self):
        logger = create_logger()
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_type)

        # Set seed
        # make dir if output_dir not exist
        if os.path.exists(self.args.output_dir) is False:
            os.makedirs(self.args.output_dir)

        model, tokenizer = load_model(**self.args.model, train_args=self.train_args, cache_dir=self.args.cache_dir)
        model.to(device)

        data_paths = self.args.data
        eval_dataloader, eval_dataset, eval_examples = get_dataloader(
            tokenizer, self.train_args, data_paths['val'], SequentialSampler, stage='dev')

        # Prepare training data loader
        train_dataloader, train_dataset, _ = get_dataloader(tokenizer, self.train_args, data_paths['train'],
                                                            RandomSampler,
                                                            stage='train')
        train_dataloader = cycle(train_dataloader)

        num_train_optimization_steps = self.train_args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer = create_optimizer(model, self.train_args)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.train_args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        log_training_info(logger, num_train_optimization_steps, self.train_args, train_dataset)

        # nb_tr_steps, train_loss_sum, best_bleu, = 0, 0, 0
        nb_tr_steps = 0
        train_loss_sum = 0
        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)

        for global_step, step in enumerate(bar):
            model.train()
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                               target_mask=target_mask)

            self.writer.add_scalar('Loss/train', loss, step)

            # if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
            train_loss_sum += loss.item()
            nb_tr_steps += 1
            train_loss = train_loss_sum / nb_tr_steps
            train_loss_for_print = round(train_loss, 4)
            bar.set_description("loss {}".format(train_loss_for_print))
            make_step(loss, optimizer, scheduler)

            if (global_step + 1) % self.train_args.eval_steps == 0:
                # Eval model with dev dataset
                eval_loss = self.make_eval_step(device, eval_dataloader, global_step, logger, model)
                log_metrics(eval_loss, global_step, logger, train_loss)
                self.writer.add_scalar('Loss/train_', train_loss, step)
                self.writer.add_scalar('Loss/val', eval_loss, step)
                nb_tr_steps = 0
                train_loss_sum = 0

    def make_eval_step(self, device, eval_dataloader, global_step, logger, model):

        logger.info("\n***** Running evaluation *****")
        eval_loss = evaluate_model(device, eval_dataloader, model)
        logger.info("  " + "*" * 20)

        save_last_checkpoint(self.args, model, global_step)
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            save_best_model(self.args, logger, model)
            logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
        return eval_loss


@hydra.main(config_path='../config', config_name='train')
def train(config):
    TrainingPipeline(config).run()


if __name__ == '__main__':
    train()
