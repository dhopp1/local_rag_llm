from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.core.schema import TextNode
import psycopg2
from sqlalchemy import make_url
import os
import pandas as pd
import numpy
from datetime import datetime
import re

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
    table_name = "texts",
    clear_database = True
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

    if clear_database:
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

def populate_db(
    vector_store,
    embed_model,
    text_path = None,
    metadata = None,
    chunk_size = 1024,
    separator = " "
):
    "populate the db with nodes from documents"
    
    # ingestion pipeline
    text_parser = SentenceSplitter(
        chunk_size = chunk_size,
        separator = separator
    )

    # creating document collection
    if type(text_path) != list:
        text_paths = [os.path.abspath(f"{text_path}{x}") for x in os.listdir(text_path)]
    else:
        text_paths = [os.path.abspath(x) for x in text_path]
    
    documents = []
    
    for text_i in text_paths:
        # read the file
        file = open(text_i, "r", encoding = "latin1")
        stringx = file.read().replace("\x00", "\uFFFD")
        file.close()
        
        # generate the document
        metadata_i = {}
        metadata_i["filename"] = text_i
        
        if metadata is not None:
            for col in metadata.columns[metadata.columns != "file_path"]:
                value = metadata.loc[lambda x: x.file_path == text_i, col].values[0]
                # convert to a date if possible
                try:
                    value = round((pd.Timestamp(datetime.strptime(str(value), "%Y-%m-%d")).year) + (pd.Timestamp(datetime.strptime(str(value), "%Y-%m-%d")).dayofyear / 365.25), 4) # give date in year-float
                except:
                    pass
                value = str(value) if pd.isna(value) else value
                value = int(value) if type(value) == numpy.int64 else value
                metadata_i[col] = value
                
        document = Document(
            text = stringx,
            metadata = metadata_i
        )
        documents.append(document)
    
    # chunking the documents
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))
        
    # nodes
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)
        
    # embeddings from nodes
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode = "all")
        )
        node.embedding = node_embedding
        
    # load nodes into the vectorstore db
    vector_store.add(nodes)
    
    
def convert_csv(csv_path, txt_out_path, method = "natural_language", metadata_path = "", sentence_template = "", value_col = ""):
    """Convert a CSV into a model-interpretable txt file. CSV file must be in long format
    parameters:
        :csv_path: str: path of the CSV file you want to convert
        :method: str: one of ['natural_language', 'chunks', 'metadata']. 'natural_language' = one large .txt file with data put into sentences. 'chunks' = many small csvs with one observation per file. 'metadata' = many small .txts with only the value in it, rest of the information in the metadata
        :txt_out_path: str: where to write the .txt file to. Pass a path not a .txt for 'chunks' and 'metadata' methods
        :sentence_template: str: the sentence to convert the data to, using {} as placeholders for column names. E.g., 'Indicator was {value} in {country} in {year}' 
        :value_col: str: if using the 'metadata' method, which column is the value column in the long-formatted data. The rest of the data will go to the metadata
    """
    stringx = ""
    data = pd.read_csv(csv_path)
    
    if method == "natural_language":
        # extracting columns in template
        start_indices = [x.start() for x in re.finditer("{", sentence_template)]
        end_indices = [x.start() for x in re.finditer("}", sentence_template)]
        cols = []
        for i in range(len(start_indices)):
            cols.append(sentence_template[start_indices[i]+1:end_indices[i]])
        
        for i in range(len(data)):
            tmp_stringx = sentence_template
            for col in cols:
                tmp_stringx = tmp_stringx.replace("{" + col + "}", str(data.loc[i, col]))
                
            stringx += f"{tmp_stringx}. "
            
        with open(txt_out_path, "w") as text_file:
            text_file.write(stringx)
    elif method == "chunks":
        if metadata_path != "":
            metadata = pd.read_csv(metadata_path)
        for i in range(len(data)):
            out_path = f"{txt_out_path}{csv_path.split('/')[-1].replace('.csv', '')}_{str(i+1).zfill(len(str(len(data))))}.csv"
            tmp = data.iloc[[i]].reset_index(drop = True)
            tmp.to_csv(out_path, index = False)
            
            if metadata_path != "":
                text_id = len(metadata) + 1
                metadata.loc[text_id-1, "text_id"] = text_id
                metadata.loc[text_id-1, "file_path"] = os.path.abspath(out_path)
        if metadata_path != "":
            metadata.to_csv(metadata_path, index = False)
    elif method == "metadata":
        metadata = pd.read_csv(metadata_path)
        for col in data.columns[data.columns != value_col]:
            metadata[col] = ""
        
        for i in range(len(data)):
            out_path = f"{txt_out_path}{csv_path.split('/')[-1].replace('.csv', '')}_{str(i+1).zfill(len(str(len(data))))}.txt"
            value = data.loc[i, value_col]
            
            # updating metadata
            text_id = len(metadata) + 1
            metadata.loc[text_id-1, "text_id"] = text_id
            metadata.loc[text_id-1, "file_path"] = os.path.abspath(out_path)
            for col in data.columns[data.columns != value_col]:
                metadata.loc[text_id-1, col] = data.loc[i, col]
                
            # writing out text file
            with open(out_path, "w") as text_file:
                text_file.write(str(value))
        metadata["metric"] = value_col        
        metadata.to_csv(metadata_path, index = False)

