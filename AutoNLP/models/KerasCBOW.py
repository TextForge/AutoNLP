import holoviews as hv
from sklearn.metrics import accuracy_score

hv.extension('bokeh')
from AutoNLP.util.data import load_transposed_data
from AutoNLP.util.evaluate import (convert_2d_numpy_array_to_list, evaluate_results,
                           words_array_to_array)







class KerasCBOW():

    def __init__(self, dataset_name, train_df, test_df, parameters):
        self.dataset_name = dataset_name
        self.train_df = train_df
        self.test_df = test_df
        self.parameters = parameters
    
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
        }

    def run_pipeline(self):

        import holoviews as hv
        hv.extension('bokeh')
        import numpy as np
        import tensorflow as tf



        

        train, y_train, test, y_test, le = load_transposed_data(self.path)
        df = pd.DataFrame()
        # number of columns in train
        num_cols = len(train.columns) - 1

        from keras.preprocessing.text import Tokenizer

        maxwords = 15000
        tokenizer = Tokenizer(num_words = maxwords)

        tokenizer.fit_on_texts(train["text"])

        X_train = tokenizer.texts_to_sequences(train["text"])

        # X_train[0]
        # tokenizer.word_index

        X_test = tokenizer.texts_to_sequences(test["text"])


        hv.BoxWhisker([len(text) for text in X_train]).opts(invert_axes=True, width=800, tools=['hover'])

        from keras.preprocessing.sequence import pad_sequences 
        maxsequence = 120
        X_train = pad_sequences(X_train, maxlen=maxsequence)
        X_test = pad_sequences(X_test, maxlen=maxsequence)


        from keras.layers import GlobalAveragePooling1D
        from keras.layers.core import Activation, Dense
        from keras.layers.embeddings import Embedding
        from keras.models import Sequential

        cbow_model = Sequential()
        cbow_model.add(Embedding(input_dim=maxwords, input_length=maxsequence, output_dim=64))
        cbow_model.add(GlobalAveragePooling1D())
        cbow_model.add(Dense(100, activation='relu'))
        cbow_model.add(Dense(num_cols))
        cbow_model.add(Activation('sigmoid'))
        cbow_model.summary()

        cbow_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
        cbow_model.fit(X_train, y_train, batch_size=8, epochs=10)

        cbow_preds = cbow_model.predict(X_test)



        cbow_preds = convert_2d_numpy_array_to_list(cbow_preds)
        print("##########################")
        y_test = words_array_to_array(y_test)



        df = evaluate_results("KerasCBOW", cbow_preds, y_test, (len(list(le.classes_))))
        df['Dataset'] = self.path
        df['Config'] = self.parameters['config_version']
        print('KerasCBOW done')
        return df,cbow_preds, y_test
