from util.evaluate import evaluate_results
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 4. TfidfVectorizer-LogisticRegression

class TFidLR():
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

        tfidf_model = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=self.parameters['max_features'],sublinear_tf=True, min_df=5, ngram_range=self.parameters['ngram_range'], stop_words='english') ),
            ('classifier',  LogisticRegression(solver='lbfgs', max_iter=10000))
        ])

        tfidf_model.fit(train["text"], y_train)
        predictions = tfidf_model.predict(test["text"])

        df = evaluate_results("TfidfLR", y_test, predictions)

        print('TfidfLR done')
        df['Dataset'] = self.dataset_name

        return df, y_test, predictions