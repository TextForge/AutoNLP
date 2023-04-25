import fasttext
import numpy as np
import pandas as pd
from AutoNLP.util.data import load_data_for_ft
from AutoNLP.util.evaluate import evaluate_results

class Fasttext():
    def __init__(self, dataset_name, train_df, test_df, parameters, ft_file):
        self.dataset_name = dataset_name
        self.train_df = train_df
        self.test_df = test_df
        self.parameters = parameters
        self.ft_file = ft_file

    def get_size(self, ft_file):
        obj = {
            "crawl-300d-2M-subword": 300,
            "crawl-300d-2M": 300,
            'fasttext-wiki-news-subwords-300' : 300,
            'glove-twitter-25' : 25,
            'glove-twitter-50' : 50,
            'glove-twitter-100' : 100,
            'glove-twitter-200' : 200,
            'glove-wiki-gigaword-50' : 50,
            'glove-wiki-gigaword-100' : 100,
            'glove-wiki-gigaword-200' : 200,
            'glove-wiki-gigaword-300' : 300,
            "wiki-news-300d-1M-subword" : 300,
            "wiki-news-300d-1M" : 300,
            'word2vec-ruscorpora-300' : 300
        }
        return obj[ft_file]

    def run_pipeline(self):
        train = self.train_df
        test = self.test_df
        FASTTEXT_FILE = self.ft_file

        with open("data.train", "w", encoding='utf-8', errors='ignore') as f:
            for index, row in train.iterrows():
                text = row['text']
                label = row['label']
                f.write(f"__label__{label} {text}\n")

        print(FASTTEXT_FILE)

        model = fasttext.train_supervised(
        
            input="data.train",
            epoch=self.parameters['epochs'],
            lr=self.parameters['learning_rate'],
            # wordNgrams=self.parameters['wordNgrams'],
            verbose=self.parameters['verbose'],
            # minCount=self.parameters['minCount'],
            dim=self.get_size(FASTTEXT_FILE),
            loss="softmax",
            bucket=2000000,
            thread=8,
            lrUpdateRate=self.parameters['lrUpdateRate'],
            t=self.parameters['t'],
            pretrainedVectors="models/word_vectors/"+FASTTEXT_FILE+".vec",
            # autotuneValidationFile="data.train",
            autotuneDuration=300, # How long to autotune for (in seconds I think)
            autotunePredictions=1,
            autotuneModelSize="1M",
            autotuneMetric="f1",
        )

        print("OK")

        def predict_label(text):
            label = model.predict(text)
            return label[0][0].split("__label__")[1]
            
        y_test = test['label'].values

        predictions = test['text'].apply(predict_label).values

        #convert y_test and predictions to int
        y_test = y_test.astype(str)
        predictions = predictions.astype(str)

        df = evaluate_results("Fasttext", y_test, predictions)
        # debug
        print('Fasttext done')

        
        df['Dataset'] = self.dataset_name
        return df, y_test, predictions
            