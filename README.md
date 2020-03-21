# DGA domain detection based on machine learning and deep learning

DGA (Domain Generation Algorithm) is a technology that uses random characters to generate a C&C domain name, thereby evading the blacklist detection of a domain name. This project will compare deep learning bigram neural networks with machine learning decision tree models to help us detect DGA.

## data set

- DGA: https://data.netlab.360.com/feeds/dga/dga.txt
- Normal: https://www.kaggle.com/cheedcheed/top1m

## Machine learning

- Feature set: Shannon entropy, proportion of vowels, proportion of numbers, proportion of repeated letters, proportion of consecutive numbers, 2-gram, 3-gram, whether it is a top-level domain
- Training using decision tree models

## Deep learning

- Bigistic-based logistic regression classifier
- The data set is divided into 0.05 for final test evaluation
- k-fold cross validation with folds = 10
- Each training train iterates 10 times with 128 samples as a batch
- Tested a normal sample and a DGA sample separately

## Results:

The bigram-based logistic regression classifier has higher accuracy and recall than the decision tree model, which are: 0.981000 and 0.988417, and the decision tree model are: 0.932400 and 0.960980, respectively.
