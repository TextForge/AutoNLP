from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from AutoNLP.util.evaluate import evaluate_results

#? 1. BOW(CountVectorizer)-RF

class BOWrf():
    def __init__(self, dataset_name, train_df, test_df, parameters):
        self.dataset_name = dataset_name
        self.train_df = train_df
        self.test_df = test_df
        self.parameters = parameters

    def run_pipeline(self):

        train = self.train_df
        test = self.test_df

        y_train = train['label']
        y_test = test['label']

        bow_model = Pipeline([
            ('vectorizer', CountVectorizer(analyzer = "word", ngram_range = (1,1), binary = True)),
            ('classifier', RandomForestClassifier(n_estimators=self.parameters['n_estimators']))
        ])
        
        bow_model.fit(train["text"], y_train)
        predictions = bow_model.predict(test["text"])

        df = evaluate_results("BOWrf", y_test, predictions)

        print('BOWrf done')
        df['Dataset'] = self.dataset_name

        return df, y_test, predictions