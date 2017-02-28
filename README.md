# Fraud Detection Case Study

## Objective:
The objective of this case study is fraud detection for an online ticket sales company. We are given a relatively large dataset (100k+ rows and ~50 features). Due to the sensitive nature of the data, I cannot show snippets of it like I did the DotA 2 dataset. However, I hope that methodology can still show through.

## Case Study:
### Preprocessing
The dataset is rather well-behaved, with a tiny percentage of missing values. The hardest part in the initialization of the dataset was dealing with json files (since the data is in that format). The target variable, or fraudulent account, was created during the initialization as well. Train test split took place last in the preprocessing step. All preliminary data clean up is done with the data_cleanup.py within the src folder.

### Methodology
The biggest issue with the dataset was imbalance. There were way more non-fraudulent accounts than there were fraudulent ones. This caused two problems: First, machine leanring models tend to maximize accuracy instead of recall. However, recall was our main concern because missing a fraud would be very costly. Our model's result may not align well with our business goal. Second, cross validation could potentially be biased when one or more of the folds did not contain any fraudulent accounts. In that case our model would be super accurate, which would deviate from our business goal. We used oversampling techniques to combat the former problem and StratifiedShuffleSplit to deal with the latter. The Classifiers.py file addressed both issues.

### Results
![ROC curve](/img/ROC_curves.png)
The pipeline can be seen in test_models.py. The groundwork is laid so that I only have to provide a list of models that I want to try and the pipeline will go through every single one of them automatically. Since the main focus was on improving boosting and random forest classifiers, I took a very liberal approach to feature engineering. In the end, the random forest classifier model achieved a 93% recall while maintaining an 88% precision. The most important features included payout ratio(number of payouts divided by number of orders) and whether an account had sold any tickets before.

It is important to note, however, that recall and precision is very much changeable up to the user. User could change the threshold parameter in the Classifier.py file that modifies the fraud decision boundary based on the user's preference. If the cost of checking in with a user for fraudulent activity is relatively affordable, then lowering the threshold to get a higher recall at the cost of precision is not a bad option. Conversely, raising the threshold is also reasonable if low precision comes with too much cost.
![Profit curve](/img/profit_curves.png)
The cost-benefit matrix was constructed from a loss perspective, meaning that only the cost of every action is taken into account. For example, if the model predicted fraud, and there was a fraud, we lose the money from checking in with the account user. On the other hand, if we failed to catch a fraud, that will be reflected as a huge loss. The key insight from the profit curve image was that we did not have to predict every account to be a fraud. In fact, the sweet spot stayed at around 93 percent because it actually maximized our profit.
