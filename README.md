# local_llm
Create and run a local LLM with RAG. Adaptation of [this](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval.html) original article. Works well in conjunction with the [nlp_pipeline](https://github.com/dhopp1/nlp_pipeline) library which you can use to convert your PDFs and websites to the .txt files the library uses. It also handles .csv data files.

# Installation
In addition to the libraries in the `requirements.txt`, [Postgres SQL](https://www.postgresql.org/) and [pgvector](https://github.com/pgvector/pgvector) need to be installed on your system.

# Quick usage
If you want to use RAG, first you have to have Postgres running. E.g., on Mac, `brew services start postgresql` from the command line, on Windows `pg_ctl -D "C:\Program Files\PostgreSQL\16\data" start`, depending on where you installed Postgres. You then need to make sure you have created the users/roles you will enter later on in Postgres. E.g., `psql postgres` from the command line (`psql -U postgres` in Windows), then in SQL:

```SQL
CREATE ROLE <user> WITH LOGIN PASSWORD '<password>';ALTER ROLE <user> SUPERUSER;
```

Now you are ready to use the library in Python. 

## RAG example

```python
from local_rag_llm.model_setup import instantiate_llm
from local_rag_llm.local_llm import local_llm

# instantiate the LLM
llm = instantiate_llm(    llm_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf", # the URL of the particular LLM you want to use. If you have the model locally you don't need to pass this
	llm_path = llm_path, # path where the local LLM file is stored or will be downloaded to
	redownload_llm = True, # whether or not to redownload the LLM file
	n_gpu_layers = 0, # number of GPU layers, 0 for CPU, e.g., 100 if you have a GPU
	context_window = 3900, # working memory of the model in tokens, model-dependent)

# instantiate the model
model = local_llm(
	text_path = text_path, # either a directory where your .txt files are stored, or a list of absolute paths to the .txt files 
	metadata_path = "metadata.csv", # optional in case your .txt files have more metadata about them
	hf_token = None, # hugging face API token. If "HF_AUTH" is in your environment, you don't need to pass	temperature = 0.0, # 0-1, 0 = more conservative, 1 = more random/creative	max_new_tokens = 512, # length of new responses, equal to words more or less
	memory_limit = 2048, # if using a chat engine, memory limit of the chat engine
	system_prompt = "You are a chatbot." # priming context of the chatbot
)

# setup the Postgres database
model.setup_db(
	user = "<user>",
	password = "<password>",
	table_name = "texts",
	clear_database = False, # whether or not to clear out the vector database
)

# populate the database with your own documents
# you can skip this step if you had 'clear_database' = False in the previous step and the vector db has previously been populated
model.populate_db(
	chunk_size = 1024 # number of tokens/words to split your documents into
)

# get a response from the model
response = model.gen_response(
	prompt = "prompt",
	llm = llm,
	similarity_top_k = 4, # number of documents/chunks to return/query alongside the prompt
	use_chat_engine = True, # whether or not to use the chat engine, i.e., have a short-term memory of your chat history
	reset_chat_engine = False, # if using a chat engine, whether or not to reset its memory
	chat_mode = "context" # "context" for first searching the vector db with the user's query, putting that information in the context prompt format, then answering based on that and the user's chat history. "condense_plus_context" for condensing the conversation and last user query into a question, searching the vector db with that, then pass the context plus that query to the LLM. If you plan on having follow-up questions query the vector db, "condense_plus_context" is recommended
)

response["response"] # the text response of the model
response["supporting_text_01"] # the text of the chunks the response is largely based on plus its metadata

# if you set streaming=True in .gen_response, response["response"] will be the streaming agent, not the text response
response["response"].print_response_stream() # to generate it the first time
response["response"].response # to access it after it's been generated
```

## Transferring a vector database
You can transfer a vector database for easy backup and portability.

```python
from local_rag_llm.db_setup import pg_dump, pg_restore

# save a vector database to file
pg_dump(
	host = "<host>" # e.g., "localhost"
	port = <port>, # e.g., 5432
	user = "<user>",
	password = "<password>",
	db_name = "<name of db to dump",
	filename = "<filename.sql>",
)

# restore a vector database
pg_restore(
	host = "<host>" # e.g., "localhost"
	port = <port>, # e.g., 5432
	user = "<user>",
	password = "<password>",
	db_name = "<name of newly restored db",
	filename = "<filename.sql>",
)
```

## non-RAG example
A non-RAG model is simpler to set up. The library will infer that the model is non-RAG if you pass nothing for the `text_path` parameter.

```python
from local_rag_llm.model_setup import instantiate_llm
from local_rag_llm import local_llm

# instantiate the LLM
llm = instantiate_llm(    llm_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf", # the URL of the particular LLM you want to use. If you have the model locally you don't need to pass this
	llm_path = llm_path, # path where the local LLM file is stored or will be downloaded to
	redownload_llm = True, # whether or not to redownload the LLM file
	n_gpu_layers = 0 # number of GPU layers, 0 for CPU, e.g., 100 if you have a GPU)

# instantiate the model
model = local_llm.local_llm(
	hf_token = None,	temperature = 0.0,	max_new_tokens = 512,
)

# get a response from the model
response = model.gen_response(
	prompt = "prompt",
	llm = llm
)

response["response"] # the text of the model's response

```

## Docker usage
### CPU only and CUDA
If only using the CPU or an Nvidia GPU, you can do run everything with Docker.

- Download the `docker-compose.yml` and `Dockerfile` (for CPU-only) or `Dockerfile-gpu` (for GPU) files
- Edit the `HF_TOKEN` to your API token
- Change the volume mappings to your desired local directories
- Navigate to the directory you saved the `.yml` file and run `docker compose up`. If you don't have a GPU, make the following edits to the `docker-compose.yml` file: 
	- Change `dockerfile: Dockerfile-gpu` line to be `dockerfile: Dockerfile`
	- Delete the `deploy:` line and everything below it
- Check the name of the `local_rag_llm` image (not the postgres one) with `docker ps -a`
- Run your desired script with `docker exec -t <image name from previous step> python /app/script.py`, being sure to use the container's directory structure in the script. The `setup_db()` call in your script should look like:

```py
model.setup_db(    host = "postgres",    port = "5432",	user = "postgres",	password = "secret",    db_name = "vector_db",    table_name = "desired_table_name",)
```

### Apple silicon
If you are using Apple silicon, you won't be able to run everything in Docker because of the lack of MPS drivers. You can still use the pgvector image however.

- Install torch with the mps backend enabled with `pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`
- Install the `requirements.txt` file as normal with `pip install -r requirements.txt`, as well as `pip install local_rag_llm`
- You can check if you successfully installed torch with the MPS backend enabled by running `torch.backends.mps.is_available()` in Python
- Download the `docker-compose.yml` and `Dockerfile` files
- Comment out the `environment:` and `volumes:` lines
- Navigate to the directory you saved the `.yml` file and run `docker compose start postgres`
- You can now instantiate your LLM with `local_rag_llm.model_setup.instantiate_llm`, passing `n_gpu_layers = 100` or the desired amount
- The `setup_db()` call in your script should look like:

```py
model.setup_db(  	host = "localhost",   	port = "5432",	user = "postgres",	password = "secret",   	db_name = "vector_db",  	table_name = "desired_table_name",)
```
