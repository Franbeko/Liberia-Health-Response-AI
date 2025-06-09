# Liberia-Health-Response-AI


# How to run?
### STEPS:

Clone the repository

```bash
project repo: https://github.com/
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n liberiahealthresponseai python=3.10 -y
```

```bash
conda activate liberiahealthresponseai
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone and openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- Langchain
- Flask
- GPT
- Pinecone