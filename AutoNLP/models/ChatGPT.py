from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from util.evaluate import evaluate_results

import pandas as pd
import openai
import re

def preprocess_text(text):
    text = str(text)
    text = text.lower()  # convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove non-english letters
    return text.strip()


class ChatGPT():
    def __init__(self, train_df, api_key):
        self.train_df = train_df
        self.api_key = api_key

        openai.api_key = self.api_key

    def run_pipeline(self):

        new_prompt = """
        For the following Sentence please give me 10 rephrased answers for each example i give you.
        Also try to make the answers as different as possible from each other.
        """

        synthetic_data = pd.DataFrame(columns=['orignal_text','synthetic_text', 'label'])
        
        # loop through the data and generate synthetic examples
        for i, row in self.train_df.iterrows():

            prompt = new_prompt + row["text"]
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": prompt}],
            )

            answer = chat_completion.choices[0].message.content.strip().splitlines()

            answer = pd.DataFrame(answer, columns=["synthetic_text"]).dropna()
            answer["synthetic_text"] = answer["synthetic_text"].str.replace('^[0-9]+\. ', '', regex=True).apply(preprocess_text)
            answer["label"] = row["label"]
            answer['orignal_text'] = row["text"]
            #remove rows where the text is less than 10 characters
            answer = answer[answer["synthetic_text"].str.len() > 3]

            print(answer.shape)

            synthetic_data = pd.concat([synthetic_data, answer], ignore_index=True)
        

        return synthetic_data