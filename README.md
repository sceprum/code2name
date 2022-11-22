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
It may seem as pretty straightforward task, but there are still some corner-cases like some markup in comments, strings, etc.
So it is still looks like reasonable solution for the demo purposes, but it is better to replace the parsing 
functionality with some specialized solution like [astminer](https://github.com/JetBrains-Research/astminer) or 
[javaparser](https://github.com/javaparser/javaparser).

## Instructions

### Download submodules
```
git submodule init
git submodule update
```

### Preprocess datasets

Download datasets
```bash
mkdir data/raw/
cd data/raw
wget 
```

### Run preprocessing

```bash
python -m pipeline.preprocess -i data/raw/java-small -o data/preprocessed/java-small --splits test validation training```
```
