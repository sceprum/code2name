import os

import numpy as np
import torch


def log_metrics(eval_loss, global_step, logger, train_loss):
    result = {'eval_ppl': round(np.exp(eval_loss), 5),
              'global_step': global_step + 1,
              'train_loss': round(train_loss, 5)}
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))


def save_best_model(args, logger, model):
    logger.info("  " + "*" * 20)
    # Save best checkpoint for best ppl
    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)


def save_last_checkpoint(args, model, step):
    last_output_dir = os.path.join(args.output_dir, f'checkpoint-last')
    if not os.path.exists(last_output_dir):
        os.makedirs(last_output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(last_output_dir, f"pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)


def save_best_model_by_blue(args, logger, model):
    logger.info("  " + "*" * 20)
    # Save best checkpoint for best bleu
    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
