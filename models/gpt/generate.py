
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F

from util import clean

def run_generator(
    model_name='mymodel.pt',
    sentences=5,
    label='zero'
):

    # Specify the device to use (e.g. 'cpu' or 'cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the fine-tuned model and tokenizer
    # model = GPT2LMHeadModel.from_pretrained('ag_news-v2.pt')
    # tokenizer = GPT2Tokenizer.from_pretrained('ag_news-v2.pt')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model_path = 'ag_news-v2.pt'
    model.load_state_dict(torch.load(model_path))

    model.to(device)


    # Set the label to use as a prompt for text generation

    # Encode the label as a tensor and move it to the specified device
    input_ids = torch.tensor(tokenizer.encode(label)).unsqueeze(0).to(device)


    import numpy as np


    def choose_from_top_k_top_n(probs, k=50, p=0.8):
        ind = np.argpartition(probs, -k)[-k:]
        top_prob = probs[ind]
        top_prob = {i: top_prob[idx] for idx, i in enumerate(ind)}
        sorted_top_prob = {k: v for k, v in sorted(
            top_prob.items(), key=lambda item: item[1], reverse=True)}

        t = 0
        f = []
        pr = []
        for k, v in sorted_top_prob.items():
            t += v
            f.append(k)
            pr.append(v)
            if t >= p:
                break
        top_prob = pr / np.sum(pr)
        token_id = np.random.choice(f, 1, p=top_prob)

        return int(token_id)


    arr = []

    # Generate text
    with torch.no_grad():
        for idx in range(sentences):
            finished = False
            cur_ids = torch.tensor(tokenizer.encode(label)).unsqueeze(0).to(device)

            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]

                softmax_logits = torch.softmax(logits[0,-1], dim=0)

                if i < 5:
                    n = 10
                else:
                    n = 5

                next_token_id = choose_from_top_k_top_n(softmax_logits.to('cpu').numpy()) #top-k-top-n sampling
                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    finished = True
                    break

            if finished:	          
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                arr.append(output_text)
                # print (output_text)
            else:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                arr.append(output_text)
                # print (output_text)

        # Decode the generated text
        generated_text = tokenizer.decode(input_ids[0, :].tolist())


    #save arr to csv
    


    #seperate the items in arr to label and text    

    #remove the label from the text
    for i in range(len(arr)):
        arr[i] = arr[i].replace(label, '')
    
    #remove <|endoftext|> from the text
    for i in range(len(arr)):
        arr[i] = arr[i].replace('<|endoftext|>', '')

    #apply clean(text, keep_upper=False) to the text
    for i in range(len(arr)):
        arr[i] = clean(arr[i], keep_upper=True)

    df = pd.DataFrame(arr)
    df['label'] = label

    return df