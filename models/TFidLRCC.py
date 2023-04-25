import os


import numpy as np

import torch
from datasets import Dataset

from sklearn.metrics import accuracy_score


torch.cuda.empty_cache()


from util.data import load_transposed_data
from util.evaluate import evaluate_results

os.environ["WANDB_DISABLED"] = "true"



from util.evaluate import convert_2d_numpy_array_to_list, words_array_to_array

# 4. TfidfVectorizer-LogisticRegression(BinaryRelavance)
# TFidLRCC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline




class TFidLRCC():
    def __init__(self, path, parameters):
        self.path = path
        self.parameters = parameters
    
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
        }

    def run_pipeline(self):


        train, y_train, test, y_test, num_classes = load_transposed_data(self.path)

        tfidf_model = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=self.parameters['max_features'],sublinear_tf=True, min_df=5, ngram_range=self.parameters['ngram_range'], stop_words='english') ),
            ('classifier', ClassifierChain(LogisticRegression(solver='lbfgs', max_iter=10000)))
        ])

        tfidf_model.fit(train["text"], y_train)

        predictions = tfidf_model.predict(test["text"])

        new_predictions = []
        for item in predictions:
            new_predictions.append(int(item))
        predictions = new_predictions

        new_list = []
        for item in y_test:
            new_list.append(int(item[0]))
        y_test = new_list

        df = evaluate_results("TFidLRCC", predictions, y_test, num_classes)
        df['Dataset'] = self.path
        df['Config'] = self.parameters['config_version']

        return df, predictions, y_test
