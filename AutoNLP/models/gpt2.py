from gpt.generate import run_generator
from gpt.train import run_trainer
import pandas as pd

class GPT2():

    def __init__(self, path, parameters):
        self.path = path
        self.parameters = parameters
        
        df = pd.read_csv('ag_news-v2.csv')
        df = df[['label', 'text']]
        df.to_csv('text_aug_data/ag_news-v2', sep='\t', index=False, header=False)

    def train(
        self,
        epoch=3,
        warmup=300,
        model_name='ag_news-v2',
        data_file='text_aug_data/ag_news-v2',
        batch=64,
        learning_rate=3e-5,
        max_len=200
        ):

        run_trainer(
            epoch=epoch,
            warmup=warmup,
            model_name=model_name,
            data_file=data_file,
            batch=batch,
            learning_rate=learning_rate,
            max_len=max_len
            )


    def generate(
        self,
        model_name='ag_news-v2.pt',
        sentences=10,
        label='zero'
        ):


        return run_generator(
            model_name=model_name,
            sentences=sentences,
            label=label
        )

