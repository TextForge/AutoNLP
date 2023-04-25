from util.evaluate import (convert_2d_numpy_array_to_list, evaluate_results,
                           words_array_to_array)
from util.data import load_transposed_data
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import io

import holoviews as hv
import numpy as np
from sklearn.metrics import accuracy_score

hv.extension('bokeh')
#import OneHotEncoder

# from toxic import load_transposed_data


def load_vectors_words(fname, words):
    """Loads embeddings from a FastText file. Only loads embeddings for the given dictionary of words"""
    data = {}

    fin = io.open(fname, 'r', encoding='utf-7', newline='\n', errors='ignore')

    i = 0
    next(fin)  # Skip first line, just contains embeddings size data
    for line in fin:
        # print(line)
        # print(i)
        i += 1

        tokens = line.rstrip().split(' ')
        word = tokens[0]
        # print(tokens)
        # print(word)
        if word in words:
            data[word] = np.array(list(map(float, tokens[1:])))
        # break;
    return data


class KerasFT_GRU():

    def __init__(self, path, parameters, ft_file):
        self.path = path
        self.parameters = parameters
        self.ft_file = ft_file

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

        # from toxic import load_transposed_data

        train, y_train, test, y_test, le = load_transposed_data(self.path)

        df = pd.DataFrame()

        FASTTEXT_FILE = self.ft_file

        FASTTEXT_FILE = "word_vectors/"+FASTTEXT_FILE+".vec"

        # number of columns in train
        num_cols = len(train.columns) - 1
        print(num_cols)

        from keras.preprocessing.text import Tokenizer

        maxwords = 15000
        tokenizer = Tokenizer(num_words=maxwords)

        tokenizer.fit_on_texts(train["text"])

        X_train = tokenizer.texts_to_sequences(train["text"])

        X_train[0]

        tokenizer.word_index

        X_test = tokenizer.texts_to_sequences(test["text"])

        hv.BoxWhisker([len(text) for text in X_train]).opts(
            invert_axes=True, width=800, tools=['hover'])

        from keras.preprocessing.sequence import pad_sequences
        maxsequence = 120
        X_train = pad_sequences(X_train, maxlen=maxsequence)
        X_test = pad_sequences(X_test, maxlen=maxsequence)

        from keras.layers import GlobalAveragePooling1D
        from keras.layers.core import Activation, Dense
        from keras.layers.embeddings import Embedding
        from keras.models import Sequential

        # cbow_model = Sequential()
        # cbow_model.add(Embedding(input_dim=maxwords, input_length=maxsequence, output_dim=64))
        # cbow_model.add(GlobalAveragePooling1D())
        # cbow_model.add(Dense(100, activation='relu'))
        # cbow_model.add(Dense(num_cols))
        # cbow_model.add(Activation('sigmoid'))
        # cbow_model.summary()
        # cbow_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
        # cbow_model.fit(X_train, y_train, batch_size=32, epochs=20)
        # cbow_preds = cbow_model.predict(X_test)
        # print(cbow_preds)
        # np.save(str(dataset)+"cbow_preds", cbow_preds)
        # df = evaluate_results("CBOW", convert_2d_numpy_array_to_list(cbow_preds), convert_2d_numpy_array_to_list(y_test), (len(list(le.classes_))))
        #
        #
        #

        # debug
        # print('CBOW done')

        # df = pd.concat([df, df], ignore_index=True)

        #  [markdown]
        # ## CBoW + GRU mixing

        from keras.layers import (GRU, Bidirectional, CuDNNGRU, Dense,
                                  GlobalAveragePooling1D, GlobalMaxPool1D,
                                  Input, SpatialDropout1D, concatenate)
        from keras.layers.embeddings import Embedding
        from keras.models import Model

        # cbow_gru_model = Sequential()
        # cbow_gru_model.add(Embedding(input_dim=maxwords, input_length=maxsequence, output_dim=64))
        # # cbow_gru_model.add(Bidirectional(CuDNNGRU(80, return_sequences=True)))
        # cbow_gru_model.add(Bidirectional(GRU(80, return_sequences=True)))
        # cbow_gru_model.add(GlobalAveragePooling1D())
        # cbow_gru_model.add(Dense(num_cols, activation="sigmoid"))
        # cbow_gru_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
        # cbow_gru_model.summary()
        # cbow_gru_model.fit(X_train, y_train, batch_size=32, epochs=10)
        # cbow_gru_preds =cbow_gru_model.predict(X_test)
        # np.save(str(dataset)+"cbow_gru_preds", cbow_gru_preds)
        # df = evaluate_results("CBOW GRU", convert_2d_numpy_array_to_list(cbow_gru_preds), convert_2d_numpy_array_to_list(y_test), (len(list(le.classes_))))
        #
        #
        #

        # debug
        # print('CBOW GRU done')

        # df = pd.concat([df, df], ignore_index=True)

        #  [markdown]
        # ## FastText embeddings

        import io

        import numpy as np
        from regex import E

        # FASTTEXT_FILE = "crawl-300d-2M-subword.vec"
        # FASTTEXT_FILE = "wiki-news-300d-1M-subword.vec"
        # FASTTEXT_FILE = "crawl-300d-2M.vec"
        # FASTTEXT_FILE = "wiki-news-300d-1M.vec"

        embeddings = load_vectors_words(FASTTEXT_FILE, tokenizer.word_index)

        def create_embedding_matrix(embeddings, tokenizer):
            """Creates a weight matrix for an Embedding layer using an embeddings dictionary and a Tokenizer"""

            # Compute mean and standard deviation for embeddings
            all_embs = np.stack(embeddings.values())

            emb_mean = all_embs.mean()
            emb_std = all_embs.std()

            # print(np.mean(list(all_embs)))

            # emb_mean = 10
            # emb_std = 10

            # print(iter(len(embeddings.values())))

            embedding_size = len(next(iter(embeddings.values())))
            # embedding_size = 300

            embedding_matrix = np.random.normal(
                emb_mean, emb_std, (tokenizer.num_words, embedding_size))

            for word, i in tokenizer.word_index.items():
                if i >= tokenizer.num_words:
                    break
                embedding_vector = embeddings.get(word)

                if embedding_vector is not None:
                    try:
                        embedding_matrix[i] = embedding_vector
                    except:
                        print(word)

            return embedding_matrix

        embedding_matrix = create_embedding_matrix(embeddings, tokenizer)

        embedding_matrix.shape

        pretrained = Embedding(maxwords, embedding_matrix.shape[1], weights=[
                               embedding_matrix], trainable=False)

        from keras.layers import (CuDNNGRU, Dense, GlobalAveragePooling1D,
                                  GlobalMaxPool1D, Input, SpatialDropout1D,
                                  concatenate)
        from keras.layers.embeddings import Embedding
        from keras.models import Model

        inp = Input(shape=(maxsequence, ))
        x = Embedding(maxwords, embedding_matrix.shape[1], weights=[
                      embedding_matrix], trainable=False)(inp)
        x = GlobalAveragePooling1D()(x)
        x = Dense(100, activation='relu')(x)
        outp = Dense(num_cols, activation="sigmoid")(x)

        # fasttext_model = Model(inputs=inp, outputs=outp)

        # fasttext_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
        # fasttext_model.summary()

        # fasttext_model.fit(X_train, y_train, batch_size=32, epochs=100)

        # fasttext_preds = fasttext_model.predict(X_test)
        # np.save(str(dataset)+"fasttext_preds", fasttext_preds)

        # df = evaluate_results("Fasttext DL", convert_2d_numpy_array_to_list(fasttext_preds), convert_2d_numpy_array_to_list(y_test), (len(list(le.classes_))))
        #
        #
        #
        # df['embedding'] = FASTTEXT_FILE

        #

        # # debug
        # print('Fasttext DL')

        #  [markdown]
        # ## FastText embeddings + GRU mixing

        from keras.layers import (Bidirectional, CuDNNGRU, Dense,
                                  GlobalAveragePooling1D, GlobalMaxPool1D,
                                  Input, SpatialDropout1D, concatenate)
        from keras.layers.embeddings import Embedding
        from keras.models import Model

        fasttext_gru_model = Sequential()

        fasttext_gru_model.add(Embedding(maxwords, embedding_matrix.shape[1], weights=[
                               embedding_matrix], trainable=False))

        # fasttext_gru_model.add(Bidirectional(CuDNNGRU(units=80, return_sequences=True)))
        fasttext_gru_model.add(Bidirectional(
            GRU(units=80, return_sequences=True)))

        fasttext_gru_model.add(GlobalAveragePooling1D())
        fasttext_gru_model.add(Dense(num_cols, activation="sigmoid"))

        fasttext_gru_model.compile(
            loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
        fasttext_gru_model.summary()

        fasttext_gru_model.fit(X_train, y_train, batch_size=32, epochs=10)

        fasttext_gru_preds = fasttext_gru_model.predict(X_test)

        print(fasttext_gru_preds)

        # np.save(str(dataset)+"fasttext_gru_preds", fasttext_gru_preds)

        y_test = words_array_to_array(y_test)

        df = evaluate_results("KerasFT_GRU", convert_2d_numpy_array_to_list(
            fasttext_gru_preds), y_test, (len(list(le.classes_))))

        df['embedding'] = FASTTEXT_FILE

        df['Dataset'] = self.path
        df['Config'] = self.parameters['config_version']

        # debug
        print('Fasttext DL GRU')

        return df, convert_2d_numpy_array_to_list(fasttext_gru_preds), y_test
