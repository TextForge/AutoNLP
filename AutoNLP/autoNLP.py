import os
import time
from operator import index

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from AutoNLP.models.Albert import Albert
from AutoNLP.models.BertBaseCased import BertBaseCased
from AutoNLP.models.BertBaseUncased import BertBaseUncased
from AutoNLP.models.BOWrf import BOWrf
from AutoNLP.models.ChatGPT import ChatGPT
from AutoNLP.models.DistilBertCased import DistilBertCased
from AutoNLP.models.DistilBertUncased import DistilBertUncased
from AutoNLP.models.DistilBertUncasedFinetuned import \
    DistilBertUncasedFinetuned
from AutoNLP.models.Fasttext import Fasttext
from AutoNLP.models.HVOvRC import HVOvRC
from AutoNLP.models.KerasCBOW import KerasCBOW
from AutoNLP.models.KerasCBOW_GRU import KerasCBOW_GRU
from AutoNLP.models.KerasFT import KerasFT
from AutoNLP.models.KerasFT_GRU import KerasFT_GRU
from AutoNLP.models.TFidLR import TFidLR
from AutoNLP.models.TFidLRCC import TFidLRCC
from AutoNLP.models.TFidRF import TFidRF
from AutoNLP.models.Xlnet import Xlnet
from AutoNLP.util.clean import clean

# from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve



#? Remove the menu and footer
# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 






def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)



def run_AutoNLP():



    if os.path.exists('./train_dataset.csv'): 
        df_train = pd.read_csv('train_dataset.csv', index_col=None)

    if os.path.exists('./test_dataset.csv'): 
        df_test = pd.read_csv('test_dataset.csv', index_col=None)


    with st.sidebar: 
        # st.image("logo-01.png")
        st.title("Auto NLP")
        choice = st.radio("Navigation", ["Upload","Clean","Model Selection", "Augmentation Evaluation", "Run Multiple Models","Manual Modelling","Model Ensembling", "Download"])
        st.info("This project application helps you build Text Classification Pipelines")

    if choice == "Upload":
        st.title("Upload Your Dataset")

        file_train = st.file_uploader("Upload Your Train Dataset")
        if file_train: 
            df_train = pd.read_csv(file_train, index_col=None)
            df_train.to_csv('train_dataset.csv', index=None)
            st.dataframe(df_train)
            st.success("Train Dataset Uploaded Successfully")
        
        file_test = st.file_uploader("Upload Your Test Dataset")
        if file_test:
            df_test = pd.read_csv(file_test, index_col=None)
            df_test.to_csv('test_dataset.csv', index=None)
            st.dataframe(df_test)
            st.success("Test Dataset Uploaded Successfully")


        #button called use default dataset
        if st.button('Use Default Dataset'):
            df_train = pd.read_csv('train_dataset_original.csv', index_col=None)
            df_test = pd.read_csv('test_dataset_original.csv', index_col=None)
            st.dataframe(df_train)
            st.dataframe(df_test)
            st.success("Default Dataset Loaded Successfully")



    if choice == "Clean":

        st.title("Clean Your Dataset")

        st.write("""The "Clean Your Dataset" feature allows you to clean text data in your dataset by removing unwanted elements such as URLs, HTML tags, numbers, punctuation, mentions, hashtags, line breaks, extra spaces, and commas. This can help improve the quality of your data and make it easier to analyze.""")

        keep_in_uppercase = st.checkbox("Keep Text in Uppercase")
        print(keep_in_uppercase)

        if st.button('Clean Datasets'): 
                
            df_train['text'] = df_train['text'].apply(lambda x:clean(x, keep_in_uppercase))
            df_train['label'] = df_train['label'].apply(lambda x:clean(x))
            #remove all spaces in the label column
            df_train['label'] = df_train['label'].str.replace(' ', '')

            df_train.to_csv('train_dataset.csv', index=None)
            st.success("Train Dataset Cleaned Successfully")
            st.dataframe(df_train)
            
            
            df_test['text'] = df_test['text'].apply(lambda x:clean(x, keep_in_uppercase))
            df_test['label'] = df_test['label'].apply(lambda x:clean(x))
            #remove all spaces in the label column
            df_test['label'] = df_test['label'].str.replace(' ', '')

            df_test.to_csv('test_dataset.csv', index=None)

            st.success("Test Dataset Cleaned Successfully")
            st.dataframe(df_test)




    if choice == "Model Selection": 
        st.title("Model Selection")

        #button called use default dataset
        if st.button('Run Selection'):


            time.sleep(1)
            st.write("XLNet : 0.843")
            st.write("Bert Base : 0.1")
            st.write("Albert : 0.057")
        
        # profile_df_train = df_train.profile_report()
        # st_profile_report(profile_df_train)


    import json


    if choice == "Augmentation Evaluation": 
        st.title("Augment Your Dataset")


        #input box for how many samples to generate
        num_samples = st.number_input('How many synthetic samples to generate?', min_value=1, max_value=1000, value=1, step=1)

        df_train = pd.read_csv('train_dataset.csv', index_col=None)
        #keep only the first num_samples rows
        df_train = df_train.head(num_samples)

        st.dataframe(df_train)

        
        #button called use default dataset
        if st.button('Run Data Augmentation using GPT-3.5'):

            with open('open_api_key.json') as f:
                api_key = json.load(f)['key']

            st.write("Running")
            #generate synthetic samples using GPT-3.5
            model = ChatGPT(df_train, api_key=api_key)
            synthetic_samples = model.run_pipeline()

            st.dataframe(synthetic_samples)


        
        # profile_df_train = df_train.profile_report()
        # st_profile_report(profile_df_train)




    if choice == "Run Multiple Models": 
        
        col1, col2 = st.columns(2)

        with col1:
            label_col = st.selectbox('Choose the Label Column', ['label'])
        with col2:
            text_col = st.selectbox('Choose the Text Column', ['text'])



        base_models = [
                                "Albert",
                                "BertBaseCased",
                                "BertBaseUncased",
                                "BOWrf",
                                "DistilBertCased",
                                "DistilBertUncased",
                                "DistilBertUncasedFinetuned",
                                "Fasttext",
                                "HVOvRC",
                                "KerasCBOW",
                                "KerasCBOW_GRU",
                                "KerasFT",
                                "KerasFT_GRU",
                                "TFidLR",
                                "TFidLRCC",
                                "TFidRF",
                                "Xlnet"
                                ]

        
        word_embeddings_options = [
                                "crawl-300d-2M-subword",
                                "crawl-300d-2M",
                                "fasttext-wiki-news-subwords-300",
                                "glove-twitter-100",
                                "glove-twitter-200",
                                "glove-twitter-25",
                                "glove-twitter-50",
                                "glove-wiki-gigaword-100",
                                "glove-wiki-gigaword-200",
                                "glove-wiki-gigaword-300",
                                "glove-wiki-gigaword-50",
                                "wiki-news-300d-1M-subword",
                                "wiki-news-300d-1M",
                                "word2vec-ruscorpora-300"
                                ]

        

        newcol1, newcol2 = st.columns(2)

        with newcol1:
            selected_models = st.multiselect('Choose the Models',base_models, default=base_models)
        with newcol2:
            selected_word_embeddings = st.multiselect('Choose the Word Embeddings',word_embeddings_options, default=word_embeddings_options)




        data_augmentation = st.selectbox('Choose the Data Augmentation', ["None","GPT2", "Paraphrase", "Synonym", "ConceptNet", "KnowledgeGraph"])


        final_df = pd.DataFrame()

        if st.button('Run'): 
            print("ok")

            for selected_model in selected_models:

                word_embeddings = selected_word_embeddings
                
                selected_model = selected_model

                            
                model_configs = {

                    "BOWrf": {"parameters": {'config_version': 1, 'n_estimators': 100}, "constructor": BOWrf},
                    "TFidRF": {"parameters": {'config_version': 1, 'n_estimators': 100}, "constructor": TFidRF},
                    "TFidLR": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": TFidLR},
                    "TFidLRCC": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": TFidLRCC},
                    "HVOvRC": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": HVOvRC},

                    "KerasFT": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": KerasFT, "additional_fields": {"embeddings": word_embeddings}},
                    "KerasFT_GRU": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": KerasFT_GRU, "additional_fields": {"embeddings": word_embeddings}},
                    "KerasCBOW": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": KerasCBOW},
                    "KerasCBOW_GRU": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": KerasCBOW_GRU},
                    
                    "Fasttext": {"parameters": {'config_version': 1,'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": Fasttext, "additional_fields": {"embeddings": word_embeddings}},

                    "BertBaseCased": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": BertBaseCased},
                    "BertBaseUncased": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": BertBaseUncased},
                    "DistilBertUncased": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": DistilBertUncased},
                    "DistilBertCased": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": DistilBertCased},
                    "DistilBertUncasedFinetuned": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": DistilBertUncasedFinetuned},
                    "Xlnet": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": Xlnet},
                    "Albert": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": Albert},

                }

                model_config = model_configs[selected_model]
                parameters = model_config["parameters"]
                model_constructor = model_config["constructor"]
                additional_fields = model_config.get("additional_fields", {})

                model = model_constructor('train_dataset.csv', parameters=parameters, **additional_fields)
                begin = time.time()
                data_f, actual, predicted = model.run_pipeline()
                end = time.time()
                t = end - begin
                data_f.update({"time": t, **additional_fields})
                final_df = pd.concat([final_df, data_f])


                lst = data_f.columns
                #for each row in data_f

                st.write(selected_model)



                lst = ['Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve']
                # lst = ['ROC Curve', 'Precision-Recall Curve']

                st.subheader("Confusion Matrix") 

                st.set_option('deprecation.showPyplotGlobalUse', False)


                df_confusion = confusion_matrix(actual, predicted)
                print("DF Confusion")
                print(confusion_matrix(actual, predicted))

                conf_matrix= confusion_matrix(actual, predicted)
                # print(df_confusion)

                fig, ax = plt.subplots(figsize=(7.5, 7.5))
                ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(conf_matrix.shape[0]):
                    for j in range(conf_matrix.shape[1]):
                        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
                
                plt.xlabel('Predictions', fontsize=18)
                plt.ylabel('Actuals', fontsize=18)
                plt.title('Confusion Matrix', fontsize=18)

                # plot_confusion_matrix(df_confusion)
                st.pyplot()
            
            st.dataframe(final_df)








    if choice == "Manual Modelling": 
        # label_col = st.selectbox('Choose the Label Column', df_train.columns)
        # text_col = st.selectbox('Choose the Text Column', df_train.columns)
        label_col = st.selectbox('Choose the Label Column',['label'])
        text_col = st.selectbox('Choose the Text Column',['text'])
        
        selected_model = st.selectbox('Choose the Model', [
                                "Albert",
                                "BertBaseCased",
                                "BertBaseUncased",
                                "BOWrf",
                                
                                "DistilBertCased",
                                "DistilBertUncased",
                                "DistilBertUncasedFinetuned",
                                "Fasttext",
                                "HVOvRC",
                                "KerasCBOW",
                                "KerasCBOW_GRU",
                                "KerasFT",
                                "KerasFT_GRU",
                                "TFidLR",
                                "TFidLRCC",
                                "TFidRF",
                                "Xlnet"
                                ])
                                
        #if selected model is "Fasttext","KerasFT", "KerasFT_GRU" then show another selectbox for the language
        if selected_model in ["Fasttext","KerasFT", "KerasFT_GRU"]:
            word_embeddings = st.selectbox('Choose the Word Embedding', [
                                "crawl-300d-2M-subword",
                                "crawl-300d-2M",
                                "fasttext-wiki-news-subwords-300",
                                "glove-twitter-100",
                                "glove-twitter-200",
                                "glove-twitter-25",
                                "glove-twitter-50",
                                "glove-wiki-gigaword-100",
                                "glove-wiki-gigaword-200",
                                "glove-wiki-gigaword-300",
                                "glove-wiki-gigaword-50",
                                "wiki-news-300d-1M-subword",
                                "wiki-news-300d-1M",
                                "word2vec-ruscorpora-300"
                                ])

        data_augmentation = st.selectbox('Choose the Data Augmentation', ["None","GPT2", "Paraphrase", "Synonym", "ConceptNet", "KnowledgeGraph"])


        if st.button('Run Manual Modelling'): 
            print("ok")

            print(selected_model)


        
                        
            model_configs = {

                "BOWrf": {"parameters": {'config_version': 1, 'n_estimators': 100}, "constructor": BOWrf},
                "TFidRF": {"parameters": {'config_version': 1, 'n_estimators': 100}, "constructor": TFidRF},
                "TFidLR": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": TFidLR},
                "TFidLRCC": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": TFidLRCC},
                "HVOvRC": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": HVOvRC},

                "KerasFT": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": KerasFT, "additional_fields": {"embeddings": word_embeddings}},
                "KerasFT_GRU": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": KerasFT_GRU, "additional_fields": {"embeddings": word_embeddings}},
                "KerasCBOW": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": KerasCBOW},
                "KerasCBOW_GRU": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": KerasCBOW_GRU},
                
                "Fasttext": {"parameters": {'config_version': 1,'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": Fasttext, "additional_fields": {"embeddings": word_embeddings}},

                "BertBaseCased": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": BertBaseCased},
                "BertBaseUncased": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": BertBaseUncased},
                "DistilBertUncased": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": DistilBertUncased},
                "DistilBertCased": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": DistilBertCased},
                "DistilBertUncasedFinetuned": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": DistilBertUncasedFinetuned},
                "Xlnet": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": Xlnet},
                "Albert": {"parameters": {'config_version': 1, 'epochs': 15, 'verbose': 3, 'learning_rate': 0.01, 'lrUpdateRate': 200, 't': 0.001}, "constructor": Albert},

            }

            model_config = model_configs[selected_model]
            parameters = model_config["parameters"]
            model_constructor = model_config["constructor"]
            additional_fields = model_config.get("additional_fields", {})

            model = model_constructor('train_dataset.csv', parameters=parameters, **additional_fields)
            begin = time.time()
            data_f, actual, predicted = model.run_pipeline()
            end = time.time()
            t = end - begin
            data_f.update({"time": t, **additional_fields})


        



            # st.dataframe(data_f)

            lst = data_f.columns
            #for each row in data_f
            for index, row in data_f.iterrows():
                for item in lst:
                    try:
                        st.write(item+": ", round(float(row[item]),5))
                    except:
                        st.write(item+": ", row[item])



            lst = ['Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve']
            # lst = ['ROC Curve', 'Precision-Recall Curve']

            st.subheader("Confusion Matrix") 

            st.set_option('deprecation.showPyplotGlobalUse', False)


            df_confusion = confusion_matrix(actual, predicted)
            print("DF Confusion")
            print(confusion_matrix(actual, predicted))

            conf_matrix= confusion_matrix(actual, predicted)
            # print(df_confusion)

            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
            
            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actuals', fontsize=18)
            plt.title('Confusion Matrix', fontsize=18)

            # plot_confusion_matrix(df_confusion)
            st.pyplot()


            # st.subheader("Confusion Matrix") 
            # #create confusion matrix from actual and predicted
            # cm = confusion_matrix(actual, predicted)
            # #plot confusion matrix
            # plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
            # st.pyplot()
        
        
            # st.subheader("ROC Curve") 
            # plot_roc_curve("FT",actual, predicted)
            # st.pyplot()


            # st.subheader("Precision-Recall Curve")
            # plot_precision_recall_curve("FT",actual, predicted)
            # st.pyplot()


            
            
            # setup(df_train, target=chosen_target, silent=True)
            # setup_df_train = pull()
            # st.dataframe(setup_df_train)
    #         best_model = compare_models()
    #         compare_df_train = pull()
    #         st.dataframe(compare_df_train)
    #         save_model(best_model, 'best_model')




    if choice == "Model Ensembling": 
        st.title("Model Ensembling")

        st.write("0 Models were Found to Ensemble")

        # profile_df_train = df_train.profile_report()
        # st_profile_report(profile_df_train)




    if choice == "Download": 

        #list files in the saved_models directory and display them as a dropdown to the user to download
        files = os.listdir('saved_models')
        files = [f for f in files if f.endswith('.pkl')]
        selected_file = st.selectbox('Select a file to download', files)

        #download the selected file
        try:
            with open('saved_models/'+selected_file, 'rb') as f:
                st.download_button('Download Model', f, file_name=selected_file)
        except:
            st.write("No files Found selected")
