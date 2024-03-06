from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
import psycopg2
from sqlalchemy import make_url

def setup_embeddings(embedding_model_id):
    "create sentence embeddings"
    embed_model = HuggingFaceEmbedding(model_name = embedding_model_id)
    
    return embed_model

def setup_db(
    db_name = "vector_db",
    host = "localhost",
    password = "password",
    port = "5432",
    user = "user1",
    embed_dim = 384,
    table_name = "texts"
):
    "create the Postgres table for storing vectors"
    
    # create the connection
    conn = psycopg2.connect(
        dbname = "postgres",
        host = host,
        password = password,
        port = port,
        user = user,
    )
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")
        
    vector_store = PGVectorStore.from_params(
        database = db_name,
        host = host,
        password = password,
        port = port,
        user = user,
        table_name = table_name,
        embed_dim = embed_dim
    )
    
    return vector_store