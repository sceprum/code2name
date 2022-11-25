# Prediction of Java method name by it's body

To solve the problem it is reasonable to select a pre-trained model and fine-tune it on some dataset.
[CodeBERT](https://huggingface.co/microsoft/codebert-base) looks like a good choice. 

To train and estimate quality of the model we will use datasets, presented in 
[code2seq paper](https://arxiv.org/abs/1808.01400). 
Data used in this paper is available in two forms: source files and AST trees stored in .c2s files. CodeBERT accepts
source code as input so some kind of preprocessing is required to prepare data for the training.
Basically, we have the following options:
- manually parse files with source code 
- find some tool that are designed to parse a source code and use it to extract methods and their names
- use some another tool to unparse existing AST and reproduce the source code of the method

To train a model we need to find a declaration of a method and define the limits of it's body.
It may seem as pretty straightforward task, but there are still some corner-cases like presence of commented-out code, 
symbols in strings and so on. This option is used in this repository, and it looks like
reasonable solution for the demo purposes. But to achieve the better quality of the datasets it is better to replace 
the parsing functionality with some specialized software like [astminer](https://github.com/JetBrains-Research/astminer)
or [javaparser](https://github.com/javaparser/javaparser).

## Training a model

### Download submodules
```
git submodule init
git submodule update
```

### Dataset download

Download datasets
```bash
mkdir -p data/raw/
cd data/raw
wget https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz
tar -xvzf java-large.tar.gz
rm java-large.tar.gz
cd ../..
```


### Methods extraction

```bash
python -m pip install -r requirements.txt
mkdir data/preprocessed
python -m pipeline.preprocess -i data/raw/java-small -o data/preprocessed/java-small --splits test validation training
```
This command extracts methods from the source files and saves extracted information in Apache Arrow format

### Train model

```
python -m pipeline.train
```

This command loads configuration for training that are located in `config/train.yaml` and trains the model using
parameters specified in configuration. The loss curves and models are saved in the `experiments` directory. Each
new run creates a new subdirectory to store training artifacts. 

## Evaluate performance of the trained model
```
python -m pipeline.evaluate --splits val test
```
Evaluation script finds the best model (selected by validation loss during the training stage) and evaluates it on 
data from the splits. The names of the splits are passed through command-line interface. The command line arguments
also accepts a path to some other experiment directory (`-e` argument) and path to save the target method name for
each example and predicted value. Script saves the outputs in the same experiment directory if no other value was provided.


The command also prints F1 scores for each split. The expected metrics are presented in the table below

| Metric | validation | test |
| --- | --- | -- |
| F1 score | 0.528 | 0.519 |

As it is reported in original paper, the presented model achieves F1 score of 0.43 on java-small dataset. So 
metrics in the table seem good enough.  Difference of the model's performance on validation and test
splits doesn't look like a problem, because in java-small dataset these splits consists of a single project each and
structure and complexity of a code may differ.







