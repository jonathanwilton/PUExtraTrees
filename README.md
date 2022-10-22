# Positive-Unlabeled Learning using Random Forests via Recursive Greedy Risk Minimization

We propose new random forest algorithms for PU-learning that recursively and greedily minimise PU-data based estimators of the expected risk. Unbiased (uPU) and nonnegative (nnPU) risk estimators are both supported with either one of the quadratic or logistic loss. Using the quadratic loss and logistic loss are equivalent to using the Gini and entropy impurities in traditional (PN) random forests.

Paper: https://arxiv.org/pdf/2210.08461

## How to use PU ET
A minimal working example usage of PU ET is found in ```run_puet_simple.py```. Alternatively, ```run_puet.py``` demonstrates how to make use of more functionality. The implementation also supports PN learning, with example given in ```run_pnet.py```.

## Requirements
The implementation was created with these packages available. Correct functionality may be achieved with previous versions of packages but this is not tested. 
```
numpy '1.21.2'
scipy '1.7.1'
joblib '1.1.0'
tree.py
trees.py
```
