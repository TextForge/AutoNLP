from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import numpy as np
import pandas as pd

    
def convert_predictions_list_to_2d_numpy_array(predictions_list, class_count):
    try:

        #get the max value in predictions_list
        # max_temp = class_count+1
        max_temp =  190

        out = []
        for item in predictions_list:
            arr = []
            #prediction_list to numpy array
        
            for i in range(max_temp):
                arr.append(0)
                #add 1 to the index of the predicted class
                # arr = np.append(arr, 0)

            arr[item] = 1
            out.append(arr)
        out = np.array(out)
        return out
    except Exception as error:
        print(f"ERROR (convert_predictions_list_to_2d_numpy_array): {error}")

        #print the number of classes
        print(f"Number of classes: {class_count}")
        print(f"Max Temp: {max_temp}")
        #print max and min values of y_test
        print(f"Max value of y_test: {max(predictions_list)}")
        print(f"Min value of y_test: {min(predictions_list)}")

        return None


def convert_2d_numpy_array_to_list(predictions_array):
    try:
        out = []
        for item in predictions_array:
            out.append(np.argmax(item))
        return out
    except Exception as error:
        print(f"ERROR (convert_2d_numpy_array_to_list): {error}")
        print(predictions_array)
        return None


def words_array_to_array(array):

    from word2number import w2n
    #create array with length of (len(list(le.classes_)))
    
    # print(y_test[0][0])
    # print(w2n.word_to_num(y_test[0][0]))

    array_transformed = []
    for item in array:
        try:
            array_transformed.append(w2n.word_to_num(item[0]))
        except Exception as e:
            print("##########################")
            print("##########################")
            print("##########################")
            print("##########################")
            print("##########################")
            print(item[0])

    array = array_transformed

    return array


def safe_execute(default, exception, function, *args):
    try:
        return function(*args)
    except Exception as error:
        print(f"ERROR ({exception}): {error}")
        return default


import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_results(model_name, y_test, preds):
    print(f"Evaluating results for {model_name}...")
    
    if preds.shape != y_test.shape:
        print(f"ERROR: The shapes of preds ({preds.shape}) and y_test ({y_test.shape}) are not the same.")
        return None
    
    df = pd.DataFrame({
        'Model Name': [model_name],
        'F1 macro': [f1_score(y_test, preds, average='macro')],
        'F1 micro': [f1_score(y_test, preds, average='micro')],
        'F1 weighted': [f1_score(y_test, preds, average='weighted')],
        'Precision macro': [precision_score(y_test, preds, average='macro')],
        'Precision micro': [precision_score(y_test, preds, average='micro')],
        'Precision weighted': [precision_score(y_test, preds, average='weighted')],
        'Recall macro': [recall_score(y_test, preds, average='macro')],
        'Recall micro': [recall_score(y_test, preds, average='micro')],
        'Recall weighted': [recall_score(y_test, preds, average='weighted')],
        'Accuracy': [accuracy_score(y_test, preds)],
        'AUC macro': [0],
        'AUC weighted': [0],
    })
    
    try:
        df['AUC macro'] = roc_auc_score(y_test, preds, multi_class='ovr', average='macro')
    except:
        pass

    try:
        df['AUC weighted'] = roc_auc_score(y_test, preds, multi_class='ovr', average='weighted')
    except:
        pass
    
    return df
