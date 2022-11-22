import torch
import transformers
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig

from CodeBERT.CodeBERT.code2nl.model import Seq2Seq
from torch import nn


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
