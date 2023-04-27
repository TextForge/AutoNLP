# AutoNLP README

AutoNLP is a Python package that provides a simple user interface for natural language processing (NLP) tasks such as text classification, named entity recognition, and sentiment analysis. This README file provides instructions on how to use AutoNLP on GitHub.

## Installation

To use AutoNLP, follow these steps:

1. Clone the AutoNLP repository by running the following command in your terminal:

```
git clone https://github.com/TextForge/AutoNLP.git
```

2. Navigate to the cloned directory by running the following command:

```
cd AutoNLP
```

3. Install the package using pip by running the following command:

```
pip install .
```

4. Create a file called `app.py` in the `AutoNLP` directory.

5. Add the following code to `app.py`:

```python
import streamlit as st
from AutoNLP import autoNLP

autoNLP.run_AutoNLP()
```

6. Save `app.py` and run the following command in your terminal to launch the app:

```
streamlit run app.py
```

## Using GPT-3.5

If you want to use the GPT-3.5 model, you need to create a file called `open_api_key.json` in your home directory. 

Follow these steps:

1. Create a new file called `open_api_key.json` in your home directory.

2. Add your OpenAI API key to the file in the following format:

```json
{
    "openai_api_key": "YOUR_API_KEY"
}
```

3. Save `open_api_key.json` and run the app using the instructions above.

## Conclusion

You should now be able to use AutoNLP on GitHub by following these instructions. If you have any questions or encounter any issues, please refer to the documentation or open an issue on the GitHub repository.