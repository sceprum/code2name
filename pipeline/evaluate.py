import numpy as np
import re
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import SequentialSampler

from src.pipeline import load_model, get_train_args, get_dataloader, get_device, collect_predicts


def decode_ids(tokenizer, pred: np.array):
    zero_pos = np.argmax(pred == 0)
    if zero_pos > 0:
        pred = pred[:zero_pos]
    text = tokenizer.decode(pred, clean_up_tokenization_spaces=False)
    return text


def get_f1_score(ground_truth, predicted):
    tp = 0
    fp = 0
    fn = 0

    for true, pred in zip(ground_truth, predicted):
        pred_tokens = set(pred.split())
        true_tokens = set(true.split())
        tp += len(pred_tokens.intersection(true_tokens))
        fn += len(true_tokens.difference(pred_tokens))
        fp += len(pred_tokens.difference(true_tokens))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * recall * precision / (precision + recall)

    return f1_score


def evaluate(splits, experiment_dir=None, output_dir=None):
    config = OmegaConf.load('config/train.yaml')
    if experiment_dir is None:
        experiment_dir = get_last_experiment_dir(config)
        if experiment_dir is None:
            print('Failed to found directory with experiment. Please specify it using command-line arguments')
            return
    else:
        experiment_dir = Path(experiment_dir)

    model_path = experiment_dir / 'artifacts/checkpoint-best-ppl/pytorch_model.bin'
    if output_dir is None:
        output_dir = experiment_dir / 'predictions'
        output_dir.mkdir(exist_ok=True)

    train_args = get_train_args(config)

    data_paths = config['data']
    model, tokenizer = load_model(config['model']['name'], train_args, weights_path=model_path)
    device = get_device()

    for split in splits:
        if split == 'validation':
            split = 'val'

        pred_frame = create_predictions_frame(data_paths, device, model, split, tokenizer, train_args)
        f1_score = get_f1_score(pred_frame.true, pred_frame.predicted)

        print(f'Split:', split, '; F1 score:', f1_score)

        if output_dir:
            pred_frame.to_csv(output_dir / f'{split}_predicts.csv', index=False)


def create_predictions_frame(data_paths, device, model, split, tokenizer, train_args):
    dataloader, _, examples = get_dataloader(
        tokenizer, train_args, data_paths[split], SequentialSampler, stage='test')
    model = model.to(device)
    predicted = collect_predicts(device, dataloader, model, first_predict_only=True, show_progress=True)
    prediction = pd.DataFrame(
        get_prediction_records(examples, predicted, tokenizer),
        columns=['id', 'true', 'predicted']
    )
    return prediction


def get_prediction_records(examples, predicted, tokenizer):
    for pred, example in zip(predicted, examples):
        pred = decode_ids(tokenizer, pred)
        yield example.idx, example.target, pred


def get_last_experiment_dir(config):
    run_config = config.get('hydra', {}).get('run', {})
    # We don't want not call custom resolver in OmegaConf which is not registered by hydra in this script
    run_dir = eval(str(run_config)).get('dir', None)
    if run_dir is None:
        return None
    runs_dir = Path(run_dir).parent
    pat = re.compile(r'[0-9\-]+')
    subdirs = [d for d in runs_dir.glob('*') if (d.is_dir() and pat.fullmatch(d.name))]
    subdirs.sort()
    if len(subdirs) == 0:
        return None
    last_experiment_dir = subdirs[-1]
    return last_experiment_dir


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--experiment', help='Path to the experiment directory')
    parser.add_argument('--splits', help='Splits to evaluate on', default=['validation', 'test'], nargs='+')
    parser.add_argument('--output-dir', help='Path to save the results')
    args = parser.parse_args()
    evaluate(args.splits, args.experiment, output_dir=args.output_dir)
