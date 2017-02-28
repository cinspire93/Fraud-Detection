# Fraud Detection Case Study

## Objective:
The objective of this case study is fraud detection for an online ticket sales company. We are given a relatively large dataset (100k+ rows and ~50 features). Due to the sensitive nature of the data, I cannot show snippets of it like I did the DotA 2 dataset. However, I hope that methodology can still show through.

## Case Study:
### Preprocessing
The dataset is rather well-behaved, with a tiny percentage of missing values. The hardest part in the initialization of the dataset was dealing with json files (since the data is in that format). The target variable, or fraudulent account, was created during the initialization as well. Train test split took place last in the preprocessing step. All preliminary data clean up is done with the data_cleanup.py within the src folder.

### Methodology
The biggest issue with the dataset is imbalance. There are way more non-fraudulent accounts than there are fraudulent ones. This causes two possible issues: First, machine leanring models tend to maximize accuracy instead of recall. However, recall is our main concern because missing a fraud is very costly. Our model's result may not align well with our business goal. Second, cross validation could potentially be biased when one or more of the folds do not contain any fraudulent accounts. In that case our model will be super accurate, which again deviates from our business goal. We use oversampling techniques to combat the former problem and StratifiedShuffleSplit to deal with the latter.

### Results
