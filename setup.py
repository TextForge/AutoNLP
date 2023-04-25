from setuptools import setup, find_packages

setup(
    name="AutoNLP",
    version="0.0.5",
    description="TextForge AutoNLP - Automated Text Classification Tool",
    packages=find_packages(include=['AutoNLP', 'AutoNLP.*']),
    install_requires=[
        # "numpy",
        # "pandas",
        # "textstat",
        # "scipy",
        # "nltk",
        # "scikit-learn",
        # "textblob"
    ]
)
