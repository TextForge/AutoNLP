import os

import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from AutoNLP.util.data import load_transposed_data
from AutoNLP.util.evaluate import convert_2d_numpy_array_to_list

torch.cuda.empty_cache()


from AutoNLP.util.evaluate import evaluate_results, words_array_to_array

os.environ["WANDB_DISABLED"] = "true"


#? 3. HashingVectorizer model
#? To account for n-grams we can also try a HashingVectorizer transformation


class HVOvRC():
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
        batch_size = 8


        train, y_train, test, y_test, num_classes = load_transposed_data(self.path)


        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.pipeline import Pipeline
      
        
        
        bow_model = Pipeline([
            ('vectorizer', HashingVectorizer(analyzer = "word", ngram_range = (1,3), binary = True)),
            ('classifier', OneVsRestClassifier(CalibratedClassifierCV(LinearSVC()), n_jobs=1))
        ])

       

        bow_model.fit(train["text"], y_train)

        predictions = bow_model.predict_proba(test["text"])

        

        #print all unique classes in the 'label' column

       
       
        new_list = []
        for item in y_test:
            new_list.append(int(item[0]))

        y_test = new_list

        print(y_test)

        print("***********")
        # print(predictions)

        print(convert_2d_numpy_array_to_list(predictions))



        
        # y_test=words_array_to_array(y_test)




        # df = evaluate_results("HVOvRC", predictions.label_ids, preds, (len(list(le.classes_))))

        df = evaluate_results("HVOvRC", convert_2d_numpy_array_to_list(predictions), y_test, num_classes)

        # debug
        print('HVOvRC done')

        
        # return float("{0:.4f}".format(res)), float("{0:.4f}".format(t))
        
        df['Dataset'] = self.path
        df['Config'] = self.parameters['config_version']
        return df, convert_2d_numpy_array_to_list(predictions), y_test




