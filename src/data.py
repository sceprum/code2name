import re

import pandas as pd

from CodeBERT.CodeBERT.code2nl.run import Example


def load_dataset(path):
    frame = pd.read_feather(path)
    frame['length'] = frame.method.apply(len)
    frame = frame.query('length<10000')
    examples = []
    upper = re.compile('[A-Z]')
    for idx, source, name in zip(frame.index, frame.method, frame.name):
        target = ' ' .join(name_to_sequence(name, upper))
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
