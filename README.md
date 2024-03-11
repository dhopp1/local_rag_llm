# local_llm
Create and run a local LLM with RAG. Adaptation of [this](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval.html) original article. Works well in conjunction with the [nlp_pipeline](https://github.com/dhopp1/nlp_pipeline) library which you can use to convert your PDFs and websites to the .txt files the library uses.

# Installation
In addition to the libraries in the `requirements.txt`, [Postgres SQL](https://www.postgresql.org/) and [pgvector](https://github.com/pgvector/pgvector) need to be installed on your system.

# Quick usage
If you want to use RAG, first you have to have Postgres running. E.g., on Mac, `brew services start postgresql` from the command line. You then need to make sure you have created the users/roles you will enter later on in Postgres. E.g., `psql postgres` from the command line, then in SQL:

```SQL
CREATE ROLE <user> WITH LOGIN PASSWORD '<password>';ALTER ROLE <user> SUPERUSER;
```

Now you are ready to use the library in Python. 

## RAG example

```python
from local_rag_llm import local_llm

# instantiate the model
model = local_llm.local_llm(
	llm_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf", # the URL of the particular LLM you want to use. If you have the model locally you don't need to pass this
	llm_path = llm_path, # path where the local LLM file is stored or will be downloaded to
	redownload_llm = True, # whether or not to redownload the LLM file
	text_path = text_path, # either a directory where your .txt files are stored, or a list of absolute paths to the .txt files 
	metadata_path = "metadata.csv", # optional in case your .txt files have more metadata about them
	hf_token = None, # hugging face API token. If "HF_AUTH" is in your environment, you don't need to pass	n_gpu_layers = 0, # number of GPU layers, 0 for CPU	temperature = 0.0, # 0-1, 0 = more conservative, 1 = more random/creative	max_new_tokens = 512, # length of new responses, equal to words more or less	context_window = 3900 # working memory of the model in tokens, model-dependent but max is usually around 4k
)

# setup the Postgres database
model.setup_db(
	user = "<user>",
	password = "<password>"
)

# populate the database with your own documents
model.populate_db(
	chunk_size = 1024 # number of tokens/words to split your documents into
)

# get a response from the model
response = model.gen_response(
	prompt = "prompt",
	similarity_top_k = 4 # number of documents/chunks to return/query alongside the prompt
)

response["response"] # the text response of the model
response["supporting_text_01"] # the text of the chunks the response is largely based on plus its metadata

# you can also convert tabular CSVs in long format into LLM readable ones
from local_rag_llm.db_setup import convert_csv

convert_csv(
	csv_path = "gdp.csv", # CSV to convert
	txt_out_path = "texts/gdp.txt", # where to write the text file to, e.g. the same as your 'text_path'
	sentence_template = "GDP in {country} was {GDP} in {year}" # template sentence, where {col_name} will be replaced with the values in the CSV
)
```

## non-RAG example
A non-RAG model is simpler to set up. The library will infer that the model is non-RAG if you pass nothing for the `text_path` parameter.

```python
from local_rag_llm import local_llm

# instantiate the model
model = local_llm.local_llm(
	llm_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
	llm_path = llm_path,
	redownload_llm = True,
	hf_token = None,	n_gpu_layers = 0,	temperature = 0.0,	max_new_tokens = 512,	context_window = 3900
)

# get a response from the model
response = model.gen_response(
	prompt = "prompt"
)

response # the text of the model's response

```