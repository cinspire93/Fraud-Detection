from Classifiers import Classifiers
from feature_format import feature_engineering
from imblearn.over_sampling import SMOTE
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

CLF_PICKLE_FILENAME = "our_classifier.pkl"

def dump_classifier(clf):
    with open(CLF_PICKLE_FILENAME, "wb") as clf_outfile:
        pickle.dump(clf, clf_outfile)

if __name__ == '__main__':
    ## get training features
    with open('data/X_train.json') as f:
        data = json.load(f)
    X, feature_names = feature_engineering(data)

    ## get training labels
    with open('data/y_train.json') as f:
        labels = json.load(f)
    df_y = pd.DataFrame(labels)
    # get y
    y = df_y.pop('fraud').values

    '''INITIAL MODEL SELECTION'''
    models = [LogisticRegression(), RandomForestClassifier(),
        AdaBoostClassifier(), GradientBoostingClassifier()]
    clf_init = Classifiers(models)
    clf_init.cross_validate(X, y, threshold=0.1)
    for model in tuned_models:
        cross_validate(model, X, y)

    '''TUNING MODEL COMPARISON'''
    tuned_models = [RandomForestClassifier(n_estimators=150, criterion='entropy', n_jobs=-1)
                    GradientBoostingClassifier(n_estimators=250, learning_rate=0.01, max_features='sqrt', max_depth=3)]
    clf = Classifiers(tuned_models)
    clf.cross_validate(X, y, threshold=0.1)
    clf.train(X, y)

    '''FEATURE IMPORTANCES'''
    print(feature_names[np.argsort(clf.classifiers[0].feature_importances_)])

    '''FINAL STEPS'''
    # First we must plot our ROC and profit curves
    cb = np.array([[0, -20],[-100, -20]])
    tuned_model = [RandomForestClassifier(n_estimators=150, criterion='entropy', n_jobs=-1)]
    clf = Classifiers(tuned_models)
    clf.train(X, y)
    clf.plot_roc_curve()
    clf.plot_profit(cb)

    # Second, we train our final model, and dump the model into a pickle file
    sm = SMOTE(ratio=0.43, random_state=42)
    X_res, y_res = sm.fit_sample(X, y)
    final_model = RandomForestClassifier(n_estimators=150, criterion='entropy', n_jobs=-1)
    final_model.fit(X_res, y_res)
    dump_classifier(final_model)
