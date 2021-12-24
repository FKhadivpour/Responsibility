# Responsibility #

## What is Responsibility? ##
Explainable Artificial Intelligence (XAI) methods are intended to help human users better understand and trust the decision making of an AI agent, often applied to machine learning (ML) models. Responsibility is a novel XAI approach that identifies the most responsible training instance for a particular decision. This instance can then be presented as an explanation: ``this is what I (the AI) learned that led me to do that``. 

## Requirements ##
  
   * Python 3.6.12
   * Tensorflow 2.4.1
   * Numpy 1.21.2
  
## Usage ##

### train.py: ### Run this file to train a machine learning model with default configuration stated in ``get_default_config`` function.  
### calc_responsible_samples.py: ### Run this file to find the most responsible training samples for ``test_sample_num`` number of test samples from each class.
### plot_samples.py: ### Run this file to plot each sample and their associated responsible training samples.
   
