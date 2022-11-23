# Prediction of Java method name by method body

## Data and preprocessing

### Approach to data parsing

To train and estimate quality of the model we can use datasets, presented in code2seq paper (reference).
Data used in this paper available in two forms: source files and AST trees, stored in .c2s files. Since we want to
provide code of the method as input to the model, we have the following options:
- manually parse files with source code 
- use some tool that are designed to build AST to extract methods from the source code
- use some another tool to unparse existing AST and reproduce the source code of the method

In the first case we basically need to find a declaration of a method and define the limits of it's body.
It may seem as pretty straightforward task, but there are still some corner-cases like presence of commented-out code, 
symbols in strings and so on.
So it is still looks like reasonable solution for the demo purposes, but it is better to replace the parsing 
functionality with some specialized software like [astminer](https://github.com/JetBrains-Research/astminer) or 
[javaparser](https://github.com/javaparser/javaparser).

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
python -m pipeline.preprocess -i data/raw/java-small -o data/preprocessed/java-small --splits test validation training```
```
This command extracts methods from the source files and saves extracted information in Apache Arrow format

### Train model

```
python -m pipeline.train
```

This command loads configuration for training that are located in `config/train.yaml` and trains the model using
parameters specified in configuration. The loss curves and models are saved in the `experiments` directory. Each
new run creates a new subdirectory to store training artifacts. 

### Evaluate performance of the trained model
```
python -m pipeline.evaluate --splits val test
```
Evaluation script finds the best model (selected by validation loss during the training stage) and evaluates it on 
data from the splits. The names of the splits are passed through command-line interface. The command line arguments
also accepts a path to some other experiment directory (`-e` argument) and path to save the target method name for
each example and predicted value. Script saves the outputs in the same experiment dir if no other value was provided to


The command also prints F1 scores for each split. 






