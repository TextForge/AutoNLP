from email import header

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



def load_transposed_data(dataset_path):

    print("Dataset Path:", dataset_path)
    # Datasets - Train\ade_corpus_v2-Ade_corpus_v2_classification-v0.csv

    #replace -v1,-v2,-v3,-v4,-v5,-v6,-v7,-v8,-v9,-v10 with v0
    test_data = dataset_path.replace("train_", "test_")
    
    #replace "Datasets - Train\" wih "Datasets - Test\"
    test_data = test_data.replace("Datasets - Train", "Datasets - Test")

    # test_data = test_data[:test_data.rfind("-v")] + ".csv"
    


    print(test_data)
    test_data = pd.read_csv(test_data)
    




    """Loads the toxic classification problem data, decimating the training data"""
    # Load data CSVs
    train_data = pd.read_csv(dataset_path)
    train_data = train_data.dropna()

    #change first column name
    train_data.rename(columns={ 0: 'text'}, inplace=True)
    train_data.rename(columns={ 1: 'label'}, inplace=True)
    
    test_data.rename(columns={ 0: 'text'}, inplace=True)
    test_data.rename(columns={ 1: 'label'}, inplace=True)


    num_classes = len(train_data['label'].unique())



    train_data['label'] = train_data['label'].apply(lambda x: w2n.word_to_num(x))
    test_data['label'] = test_data['label'].apply(lambda x: w2n.word_to_num(x))




    y_train = train_data.drop("text", axis=1).values
    y_test = test_data.drop("text", axis=1).values


    print("Train Size", train_data.shape[0])
    print("Test Size", test_data.shape[0])

    return train_data, y_train, test_data, y_test, num_classes




def load_data(dataset_path):

    test_data = dataset_path.replace("train_", "test_")
    
    
    #replace "Datasets - Train\" wih "Datasets - Test\"
    test_data = test_data.replace("Datasets - Train", "Datasets - Test")
    test_data = test_data[:test_data.rfind("-v")] + ".csv"

    test_data = pd.read_csv(test_data)

    train_data = pd.read_csv(dataset_path, encoding='latin-1', header=None)
    train_data = train_data.dropna()
    
    #change first column name
    train_data.rename(columns={ 0: 'text'}, inplace=True)
    train_data.rename(columns={ 1: 'label'}, inplace=True)

    test_data.rename(columns={ 0: 'text'}, inplace=True)
    test_data.rename(columns={ 1: 'label'}, inplace=True) 

    train_data = train_data[train_data.text.notnull()]
    

    # get the first 20000 rows
    # data = data.iloc[:20000, :]

    #split data into train and test\
    # text_training, text_testing, training_labels, testing_labels = train_test_split(data['text'], data['label'], test_size=test_size, shuffle=False)
    
    
    text_training = train_data['text']
    text_testing = test_data['text']
    
    training_labels= train_data['label']
    testing_labels = test_data['label']

    # return X_train, X_test, y_train, y_test


    train = pd.DataFrame({'text': text_training, 'label': training_labels})
    test = pd.DataFrame({'text': text_testing, 'label': testing_labels})


    
    train["text"] = train["text"].astype(str)
    test["text"] = test["text"].astype(str)


    data = pd.concat([train, test], axis=0, ignore_index=True)


    print("Train Size", train_data.shape[0])
    print("Test Size", test_data.shape[0])

    return train, test, text_training, text_testing, training_labels, testing_labels, data


from word2number import w2n


def load_data_for_og(dataset_path):
    


    test_data = dataset_path.replace("train_", "test_")
    

    #replace "Datasets - Train\" wih "Datasets - Test\"
    test_data = test_data.replace("Datasets - Train", "Datasets - Test")
    test_data = test_data[:test_data.rfind("-v")] + ".csv"
    test_data = pd.read_csv(test_data)

    # train_data = pd.read_csv(dataset_path, encoding='latin-1', header=None)
    train_data = pd.read_csv(dataset_path, encoding='latin-1')
    train_data = train_data.dropna()
    
    #change first column name
    train_data.rename(columns={ 0: 'text'}, inplace=True)
    train_data.rename(columns={ 1: 'label'}, inplace=True)

    test_data.rename(columns={ 0: 'text'}, inplace=True)
    test_data.rename(columns={ 1: 'label'}, inplace=True) 

    train_data = train_data[train_data.text.notnull()]
    


    
    train_data['text'] = train_data['text'].str.replace('\d+', '', regex =True)
    test_data['text'] = test_data['text'].str.replace('\d+', '', regex =True)
    

    # X_train, X_test = train_test_split(data, test_size=0.5, random_state=42)

    data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    #number of unique classes in label
    num_classes = len(data['label'].unique())


    #convert words to numbers
    train_data['label'] = train_data['label'].apply(lambda x: w2n.word_to_num(x))
    test_data['label'] = test_data['label'].apply(lambda x: w2n.word_to_num(x))






    return train_data, test_data, num_classes
        

def load_data_for_ft(dataset_path):




    test_data = dataset_path.replace("train_", "test_")
    
    #replace "Datasets - Train\" wih "Datasets - Test\"
    test_data = test_data.replace("Datasets - Train", "Datasets - Test")
    test_data = test_data[:test_data.rfind("-v")] + ".csv"
    test_data = pd.read_csv(test_data)

    train_data = pd.read_csv(dataset_path, encoding='latin-1', header=None)
    train_data = train_data.dropna()
    
    #change first column name
    train_data.rename(columns={ 0: 'text'}, inplace=True)
    train_data.rename(columns={ 1: 'label'}, inplace=True)

    test_data.rename(columns={ 0: 'text'}, inplace=True)
    test_data.rename(columns={ 1: 'label'}, inplace=True) 

    train_data = train_data[train_data.text.notnull()]
    




    
    train_data['text'] = train_data['text'].str.replace('\d+', '', regex =True)
    test_data['text'] = test_data['text'].str.replace('\d+', '', regex =True)


    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    train_data['text'] = train_data['text'].str.replace('\d+', '', regex =True)
    train_data['text'] = train_data['text'].str.replace('\n', '', regex =True)
    train_data.dropna(inplace=True)


    test_data['text'] = test_data['text'].str.replace('\d+', '', regex =True)
    test_data['text'] = test_data['text'].str.replace('\n', '', regex =True)
    test_data.dropna(inplace=True)



    #find number of  unique values in label column
    num_classes = len(test_data['label'].unique())
    


    print("Train Size", train_data.shape[0])
    print("Test Size", test_data.shape[0])
    return train_data, test_data, num_classes

#! Optional Text Cleaning

# from nltk.corpus import stopwords
# ", ".join(stopwords.words('english'))
# STOPWORDS = set(stopwords.words('english'))

# def remove_stopwords(text):
#     """custom function to remove the stopwords"""
#     return " ".join([word for word in str(text).split() if word not in STOPWORDS])


# train['text']=train['text'].apply(lambda x:remove_stopwords(x))
# test['text']=test['text'].apply(lambda x:remove_stopwords(x))

# stop_words = ['a', 'an', 'the']

# # Basic cleansing
# def cleansing(text):
#     tokens = text.split(' ')
#     tokens = [w.lower() for w in tokens]
#     tokens = [w for w in tokens if w not in stop_words]
#     return ' '.join(tokens)

# # All-in-one preprocess
# def preprocess_x(x):
#     processed_x = [cleansing(text) for text in x]
#     return processed_x

# train['text_new']=train['text'].apply(lambda x:preprocess_x(x))
# test['text_new']=test['text'].apply(lambda x:preprocess_x(x))