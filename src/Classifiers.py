from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

class Classifiers(object):
    '''
    Classifier object for fitting, storing, and comparing multiple model outputs.
    '''
    def __init__(self, classifier_list):
        self.classifiers = classifier_list
        self.classifier_names = [est.__class__.__name__ for est in self.classifiers]

    def train(self, X, y):
    '''
    Trains each model using X and y, the method train_test_splits within the method,
    thus no need to do it before we use the Classifiers class
    '''
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.25, random_state=42)
        for clf in self.classifiers:
            sm = SMOTE(ratio=0.43, random_state=42)
            X_res, y_res = sm.fit_sample(X_train, y_train)
            clf.fit(X_res, y_res)
        self._X_test = X_test
        self._y_test = y_test

    def cross_validate(self, X, y, threshold=0.1):
        '''
        Apart from the general cross_validation, we added a new parameter
        that gives us more freedom in deciding whether a particular datapoint is
        a fraud, hence the threshold
        '''
        for clf in self.classifiers:
            # Make sure that every fold contains a certain amount of observation from the minority class
            sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
            sss.get_n_splits(X, y)
            precisions = []
            recalls = []
            f1s = []
            print("\n____________{}____________".format(clf.__class__.__name__))
            for train_index, test_index in sss.split(X, y):
                # print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                ## SMOTE
                ## Chose ratio that basically provides a 1:2 minority:majority
                ## propotion in the resampled dataset
                sm = SMOTE(ratio=0.43, random_state=42)
                X_res, y_res = sm.fit_sample(X_train, y_train)
                ### fit the classifier using training set, and test on validation set
                clf.fit(X_res, y_res)
                # Resampling never happens on the test set
                predicted_probs = clf.predict_proba(X_test)[:, 1]
                predictions = (predicted_probs > threshold).astype(int)
                precisions.append(precision_score(y_test, predictions))
                recalls.append(recall_score(y_test, predictions))
                f1s.append(f1_score(y_test, predictions))
                print(confusion_matrix(y_test, predictions))
            # Since we only care about catching fraud, we should pay attention to
            # recall, precision and f1 scores
            print("Recall: {:.3%}".format(np.mean(recalls)))
            print("Precision: {:.3%}".format(np.mean(precisions)))
            print("F1: {:.3%}".format(np.mean(f1s)))


    def plot_roc_curve(self):
        fig, ax = plt.subplots()
        for name, clf in zip(self.classifier_names, self.classifiers):
            predict_probas = clf.predict_proba(self._X_test)[:,1]
            fpr, tpr, thresholds = roc_curve(self._y_test, predict_probas, pos_label=1)
            roc_auc = auc(x=fpr, y=tpr)
            ax.plot(fpr, tpr, label='{} (AUC = {:.2f})'.format(name, roc_auc), lw=5, color='#3a474d')
        # 45 degree line
        x_diag = np.linspace(0, 1.0, 20)
        ax.plot(x_diag, x_diag, color='grey', ls='--')
        ax.legend(loc='best', fontsize=25)
        ax.set_title('ROC Curve', fontsize=40, weight='bold')
        ax.set_ylabel('True Positive Rate', size=30, weight='bold')
        ax.set_xlabel('False Positive Rate', size=30, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=20)
        fig.set_size_inches(15, 10)
        fig.savefig('ROC_curves.png', dpi=100)


    def plot_profit(self, cb):
        fig, ax = plt.subplots()
        percentages = np.linspace(0, 100, len(self._y_test) + 1)
        for name, clf in zip(self.classifier_names, self.classifiers):
            probabilities = clf.predict_proba(self._X_test)[:,1]
            thresholds = sorted(probabilities)
            thresholds.append(1.0)
            profits = []
            for threshold in thresholds:
                y_predict = probabilities >= threshold
                confusion_mat = confusion_matrix(self._y_test, y_predict)
                profit = np.sum(confusion_mat * cb) / float(len(self._y_test))
                profits.append(profit)
            ax.plot(percentages, profits, label=name, lw=5, color='#3a474d')
        ax.legend(loc='best', fontsize=25)
        ax.set_title('Profit Curve', fontsize=40, weight='bold')
        ax.set_ylabel('Profit', size=30, weight='bold')
        ax.set_xlabel('Proportion of Test Instances', size=30, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=20)
        fig.set_size_inches(15, 10)
        fig.savefig('profit_curves.png')
