import re 

def lower(text):
    low_text= text.lower()
    return low_text

def remove_urls(text):
    url_remove = re.compile(r'https?://\S+|www\.\S+')
    return url_remove.sub(r'', text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

# Lower casing
def lower(text):
    low_text= text.lower()
    return low_text

# Number removal
def remove_num(text):
    remove= re.sub(r'\d+', '', text)
    return remove

def punct_remove(text):
    punct = re.sub(r"[^\w\s\d]","", text)
    return punct

#Remove mentions and hashtags
def remove_mention(x):
    text=re.sub(r'@\w+','',x)
    return text

def remove_hash(x):
    text=re.sub(r'#\w+','',x)
    return text

def remove_space(text):
    space_remove = re.sub(r"\s+"," ",text).strip()
    return space_remove

def remove_commar(text):
    comma_remove = re.sub(r"\,"," ",text).strip()
    return comma_remove

def remove_linebreaks(text):
    line_remove = re.sub(r"\n"," ",text).strip()
    return line_remove

def remove_double_space(text):
    double_space_remove = re.sub(r"\s+"," ",text).strip()
    return double_space_remove




def clean(text, keep_upper=False):
    text = str(text)

    if not keep_upper:
        text = lower(text)
        
    text = remove_mention(text)
    text = remove_hash(text)
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_num(text)
    text = punct_remove(text)
    text = remove_space(text)
    text = remove_commar(text)
    text = remove_linebreaks(text)
    text = remove_double_space(text)

    return text



