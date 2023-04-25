import os


import numpy as np

import torch
from datasets import Dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


from util.data import load_transposed_data
from util.evaluate import convert_2d_numpy_array_to_list, words_array_to_array

torch.cuda.empty_cache()


from util.evaluate import evaluate_results

os.environ["WANDB_DISABLED"] = "true"



# 2. Tfidf-RF
# TFidRF

class TFidRF():
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
        batch_size = 8


        # data = pd.read_csv(self.path, engine='python')

        train, y_train, test, y_test, num_classes = load_transposed_data(self.path)


        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.pipeline import Pipeline


        bow_model = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=self.parameters['max_features'],sublinear_tf=True, min_df=5, ngram_range=self.parameters['ngram_range'], stop_words='english') ),
            ('classifier',  RandomForestClassifier(n_estimators=self.parameters['n_estimators']))
        ])

        bow_model.fit(train["text"], y_train)

        predictions = bow_model.predict(test["text"])


        # print("Predictions:")
        
        new_predictions = []
        for item in predictions:
            new_predictions.append(int(item))
        predictions = new_predictions
        
        # print(predictions)


        # print("Y Test")

        new_list = []
        for item in y_test:
            new_list.append(int(item[0]))

        y_test = new_list

        # print(y_test)


        # y_test=words_array_to_array(y_test)



        df = evaluate_results("TfidRF", predictions, y_test, num_classes)

        # debug
        print('TfidRF done')        


        df['Dataset'] = self.path
        df['Config'] = self.parameters['config_version']
        # return float("{0:.4f}".format(res)), float("{0:.4f}".format(t))
        
        return df, predictions, y_test

