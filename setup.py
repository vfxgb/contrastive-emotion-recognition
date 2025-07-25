from setuptools import setup, find_packages

setup(
    name="contrastive_emotion",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch==2.6.0",
        "numpy==1.24.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "mamba-ssm==2.2.4",
        "einops==0.8.1",
        "transformers==4.50.2",
        "matplotlib",
        "spacy==3.8.4",
        "pyodbc==5.2.0",
        "pydantic<2.0",
        "pandas",
        "tensorflow"
    ],
)
