import os

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (AlbertTokenizerFast,
                          AutoModelForSequenceClassification, Trainer,
                          TrainingArguments)
from util.data import load_data_for_og
from util.evaluate import evaluate_results
        
from sklearn.preprocessing import LabelEncoder


os.environ["WANDB_DISABLED"] = "true"

class Albert():
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

        train = self.train_df
        test = self.test_df

        X_train, X_test = train, test

        le = LabelEncoder()
        le.fit(X_train['label'])
        X_train['label'] = le.transform(X_train['label'])
        X_test['label'] = le.transform(X_test['label'])

        #get number of unique classes
        num_classes = len(X_train['label'].unique())

        model_name = "albert-base-v2"

        # max sequence length for each document/sentence sample
        max_length = 120

        tokenizer = AlbertTokenizerFast.from_pretrained(model_name)

        def preprocess_function(examples):
            return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=max_length)

        dataset_train = Dataset.from_pandas(X_train)
        dataset_train = dataset_train.map(preprocess_function, batched=True)

        dataset_test = Dataset.from_pandas(X_test)
        dataset_test = dataset_test.map(preprocess_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=le).to("cuda")

        training_args = TrainingArguments(
            output_dir='./albert_results',          # output directory
            num_train_epochs=self.parameters['epochs'],  # total number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            per_device_eval_batch_size=batch_size*2,   # batch size for evaluation
            warmup_steps=self.parameters['warmup_steps'],
            learning_rate=self.parameters['learning_rate'],
            adam_beta1=self.parameters['adam_beta1'],
            adam_beta2=self.parameters['adam_beta2'],
            do_eval=False,
        )
        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=dataset_train,         # training dataset
            eval_dataset=dataset_test,          # evaluation dataset
            compute_metrics=self.compute_metrics,     # the callback that computes metrics of interest
        )
        trainer.train()

        predictions = trainer.predict(dataset_test)
        preds = np.argmax(predictions.predictions, axis=-1)

        df = evaluate_results("Albert", predictions.label_ids, preds)
        # debug
        print('Albert done')
        
        
        df['Dataset'] = self.dataset_name
        return df, predictions.label_ids, preds

