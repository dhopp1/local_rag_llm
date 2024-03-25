from importlib import import_module
import torch
import os
import pandas as pd


class local_llm:
    """Primary class of the library, use and manage the LLM, your RAG documents, and their metadata
    parameters:
        :embedding_model_id: str: ID of the Hugginf Face embedding model
        :llm_url: str: URL for initial download of the LLM
        :llm_path: str: path of the LLM locally, if available
        :redownload_llm: bool: whether or not to force redownloading of the LLM model
        :text_path: str: path of the directory containing the .txt files for RAG, or a list of absolute paths of individual .txt files
        :metadata_path: str: path of the metadata CSV file. The CSV must at least have the two columns, "text_id", and "file_path", containing a unique identifier for the .txt file and the location of the file. Any date columns in the metadata should be in 'YYYY-MM-DD' format
        :n_gpu_layers: str: number of GPU layers to use, set '0' for cpu
        :hf_token: str: Hugging Face authorization token
        :temperature: float: number between 0 and 1, 0 = more conservative/less creative, 1 = more random/creative
        :max_new_tokens: int: limit of how many tokens to produce for an answer
        :context_window: int: 'working memory' of the LLM, varies by model
        :memory_limit: int: if using a chat engine, memory limit of the chat engine
        :system_prompt: str: prompt for initialization of the chatbot
    """

    def __init__(
        self,
        embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
        llm_url="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        llm_path=None,
        redownload_llm=True,
        text_path=None,
        metadata_path=None,
        hf_token=None,
        n_gpu_layers=0,
        temperature=0.0,
        max_new_tokens=512,
        context_window=3900,
        memory_limit=2048,
        system_prompt="",
    ):
        self.model_setup = import_module("local_rag_llm.model_setup")
        self.db_setup = import_module("local_rag_llm.db_setup")

        self.embedding_model_id = embedding_model_id
        self.llm_url = llm_url
        self.llm_path = llm_path
        self.text_path = text_path
        self.metadata_path = metadata_path
        if metadata_path is None:
            if type(text_path) != list:
                self.metadata = pd.DataFrame(
                    {
                        "file_path": [
                            os.path.abspath(f"{text_path}{x}")
                            for x in os.listdir(text_path)
                        ]
                    }
                )
            else:
                self.metadata = pd.DataFrame(
                    {"file_path": [os.path.abspath(x) for x in text_path]}
                )
        if hf_token != None:
            os.environ["HF_TOKEN"] = hf_token
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.context_window = context_window
        self.memory_limit = memory_limit
        self.system_prompt = system_prompt

        if torch.cuda.is_available():
            self.device = f"cuda:{torch.cuda.current_device()}"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.n_gpu_layers = n_gpu_layers
        self.chat_engine = None

        self.llm = self.model_setup.instantiate_model(
            text_path=self.text_path,
            llm_url=self.llm_url,
            llm_path=self.llm_path,
            redownload_llm=redownload_llm,
            temperature=self.temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            n_gpu_layers=self.n_gpu_layers,
        )

        self.embed_model = self.db_setup.setup_embeddings(self.embedding_model_id)
        self.vector_store = None

        # handling set up of text files and metadata csv
        if self.text_path is not None:
            if self.metadata_path is not None:
                self.metadata = pd.read_csv(metadata_path, encoding="latin1")

    def setup_db(
        self,
        db_name="vector_db",
        host="localhost",
        password="password",
        port="5432",
        user="user1",
        embed_dim=384,
        table_name="texts",
        clear_database=True,
        clear_table=True,
    ):
        """Create the Postgres database for storing document vectors. Most of these values can be left as the default, but Postgres must be installed and running on the computer and the provided user role must have been created directly in Postgres via e.g.:
            CREATE ROLE <user> WITH LOGIN PASSWORD '<password>';
            ALTER ROLE <user> SUPERUSER;
        parameters:
            :db_name: str: name of the database
            :host: str: usually 'localhost' is fine
            :password: str: password for the role
            :port: str: 5432 is the default port for Postgres
            :user: str: username
            :embed_dim: int: number of embedding dimensions
            :table_name: str: name of the table within the db.
            :clear_database: bool: whether or not to clear the existing database
            :clear_table: bool: whether or not to clear the existing table
        """
        self.vector_store, self.db_connection, self.db_name = self.db_setup.setup_db(
            db_name=db_name,
            host=host,
            password=password,
            port=port,
            user=user,
            embed_dim=embed_dim,
            table_name=table_name,
            clear_database=clear_database,
            clear_table=clear_table,
        )

    def close_connection(self):
        query = f"""SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = '{self.db_name}'
AND pid <> pg_backend_pid();"""
        with self.db_connection.cursor() as c:
            c.execute(query)
        self.db_connection.close()

    def populate_db(
        self,
        chunk_overlap=200,
        chunk_size=1024,
        paragraph_separator="\n\n\n",
        separator=" ",
        quiet=True,
    ):
        """Populate the vector database
        parameters:
            :chunk_overlap: int: overlap of each chunk when splitting
            :chunk_size: int: chunk size of the vectors
            :paragraph_separator: str: separator between paragraphs
            :separator: str: token/word separator
            :quiet: bool: whether or not to print progress
        """
        self.db_setup.populate_db(
            self.vector_store,
            self.embed_model,
            self.text_path,
            self.metadata,
            chunk_overlap,
            chunk_size,
            paragraph_separator,
            separator,
            quiet,
        )

    def gen_response(
        self,
        prompt,
        similarity_top_k=4,
        use_chat_engine=False,
        reset_chat_engine=False,
        streaming=False,
    ):
        """Generate a response to a prompt
        parameters:
            :prompt: str: what you're asking the LLM
            :similarity_top_k: int: how many supporting chunks to produce alongside the output
            :use_chat_engine: bool: whether or not to use the chat engine, i.e., have persistent chat contexts
            :reset_chat_engine: bool: if a chat engine was previously being used, whether or not to reset its context
            :streaming: bool: whether or not to produce a streamed chat response
        returns:
            :str | dict: containing the LLM's response and the supporting k documents and their metadata if RAG is used. Any dates in the metadata are returned as year-float (e.g. 2020-01-13 = 2020.036), prompts should take use this format too, e.g., summarize documents before June 2020 should be said summarize documents less than 2020.5
        """
        response = self.model_setup.gen_response(
            prompt=prompt,
            llm=self.llm,
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            query_mode="default",
            similarity_top_k=similarity_top_k,
            max_new_tokens=self.max_new_tokens,
            use_chat_engine=use_chat_engine,
            chat_engine=self.chat_engine,
            reset_chat_engine=reset_chat_engine,
            text_path=self.text_path,
            system_prompt=self.system_prompt,
            memory_limit=self.memory_limit,
            streaming=streaming,
        )

        self.chat_engine = response["chat_engine"]
        return response["final_response"]
