from types import SimpleNamespace

from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler

from src.data import load_dataset, convert_examples_to_dataset
from src.pipeline import load_model


class Pipeline(LightningModule):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = SimpleNamespace(**config)
        self.data_paths = self.args.data
        training_conf = config['training']
        self.train_args = SimpleNamespace(**training_conf, n_gpu=1)

        self.model, self.tokenizer = load_model(
            **self.args.model, train_args=self.train_args, cache_dir=self.args.cache_dir)

    def train_dataloader(self):
        train_path = self.data_paths['train']
        train_examples = load_dataset(train_path)
        train_data = convert_examples_to_dataset(self.tokenizer, self.train_args, train_examples, stage='train')

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler,
            batch_size=self.train_args.train_batch_size)

        return train_dataloader

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        eval_examples = load_dataset(self.data_paths['val'])
        eval_data = convert_examples_to_dataset(self.tokenizer, self.train_args, eval_examples, stage='dev')
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.train_args.eval_batch_size)
        return eval_dataloader

    def predict_dataloader(self):
        pass

    def training_step(self, batch, batch_id):
        source_ids, source_mask, target_ids, target_mask = batch
        loss, _, _ = self.model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                                target_mask=target_mask)
        return loss

    def validation_step(self, batch, batch_id):
        source_ids, source_mask, target_ids, target_mask = batch
        _, loss, num = self.model(source_ids=source_ids, source_mask=source_mask,
                                  target_ids=target_ids, target_mask=target_mask)
        return loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.train_args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.train_args.learning_rate, eps=self.train_args.adam_epsilon)

        return optimizer
