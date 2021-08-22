# Deep Variational Bayes Filters

A PyTorch implementation of the Deep Variational Bayes Filters [paper](https://arxiv.org/abs/1605.06432). The official TensorFlow implementation can be found [here](https://github.com/Jgmorton/vbf) 

## Installation
To run in a Docker container, run the following commands. 
```
make docker-build
make run-gpu
```
This will automatically install all required packages and set up a testing environment. 

## Training a model
To train a model, run `python train.py [model-name]`. This will train a model with default parameters. Run `python train.py --help` for a full list of arguments. The script saves models, parameters, and logging information in the `results` folder. 

## Implementing your own transition model
One of the features of the Deep Variational Bayes Filters model is that you can implement custom latent space transition models. You will find an abstract class `TransitionModel` in `dvbf/bayes_filter.py`, which defines how a transition model should be implemented.

