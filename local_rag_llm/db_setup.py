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
import io
import subprocess


def pg_dump(
    host,
    port,
    user,
    password,
    db_name,
    filename,
):
    "Dump a vector database"
    command = f"""pg_dump --no-owner "postgresql://{user}:{password}@{host}:{port}/{db_name}" > {filename}"""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()


def pg_restore(
    host,
    port,
    user,
    password,
    db_name,
    filename,
):
    "Load a vector database"
    command = f"""psql "postgresql://{user}:{password}@{host}:{port}/postgres" -c 'DROP DATABASE IF EXISTS {db_name};' -c 'CREATE DATABASE {db_name} WITH OWNER {user};' && psql "postgresql://{user}:{password}@{host}:{port}/{db_name}" < {filename}"""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()


def setup_embeddings(embedding_model_id):
    "create sentence embeddings"
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_id)

    return embed_model


def setup_db(
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
    "create the Postgres table for storing vectors"

    # create the default connection
    conn = psycopg2.connect(
        dbname="postgres",
        host=host,
        password=password,
        port=port,
        user=user,
    )
    conn.autocommit = True

    with conn.cursor() as c:
        # drop and recreate the database if desired
        if clear_database:
            c.execute(f"DROP DATABASE IF EXISTS {db_name}")
            c.execute(f"CREATE DATABASE {db_name}")
        else:
            # create teh database if it doesn't exist
            try:
                c.execute(f"CREATE DATABASE {db_name}")
            except:
                pass
    conn.close()

    # connection to vector database
    conn = psycopg2.connect(
        dbname=db_name,
        host=host,
        password=password,
        port=port,
        user=user,
    )
    conn.autocommit = True

    if clear_table:
        with conn.cursor() as c:
            c.execute(f"DROP TABLE IF EXISTS data_{table_name}")

    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=host,
        password=password,
        port=port,
        user=user,
        table_name=table_name,
        embed_dim=embed_dim,
    )

    return vector_store, conn, db_name


def populate_db(
    vector_store,
    embed_model,
    text_path=None,
    metadata=None,
    chunk_overlap=200,
    chunk_size=1024,
    paragraph_separator="\n\n\n",
    separator=" ",
    quiet=True,
):
    "populate the db with nodes from documents"

    # ingestion pipeline
    text_parser = SentenceSplitter(
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size,
        paragraph_separator=paragraph_separator,
        separator=separator,
    )

    # creating document collection
    if type(text_path) != list:
        text_paths = [os.path.abspath(f"{text_path}{x}") for x in os.listdir(text_path)]
    else:
        text_paths = [os.path.abspath(x) for x in text_path]

    documents = []

    counter = 0
    for text_i in text_paths:
        counter += 1
        if not (quiet):
            print(
                f"Populating vector database (1/5), reading documents {counter}/{len(text_paths)}"
            )

        # read the file
        file = open(text_i, "r", encoding="latin1")
        stringx = file.read().replace("\x00", "\uFFFD")
        file.close()

        # generate the document
        metadata_i = {}
        metadata_i["filename"] = text_i
        metadata_i["is_csv"] = 1 if ".csv" in text_i else 0

        if metadata is not None:
            for col in metadata.columns[metadata.columns != "file_path"]:
                value = metadata.loc[lambda x: x.file_path == text_i, col].values[0]
                # convert to a date if possible
                try:
                    value = round(
                        (pd.Timestamp(datetime.strptime(str(value), "%Y-%m-%d")).year)
                        + (
                            pd.Timestamp(
                                datetime.strptime(str(value), "%Y-%m-%d")
                            ).dayofyear
                            / 365.25
                        ),
                        4,
                    )  # give date in year-float
                except:
                    pass
                value = str(value) if pd.isna(value) else value
                value = int(value) if type(value) == numpy.int64 else value
                metadata_i[col] = value

        document = Document(text=stringx, metadata=metadata_i)
        documents.append(document)

    # chunking the documents
    text_chunks = []
    doc_idxs = []
    counter = 0
    n = len(list(enumerate(documents)))
    for doc_idx, doc in enumerate(documents):
        counter += 1
        if not (quiet):
            print(f"Populating vector database (2/5), chunking documents {counter}/{n}")
        # non-CSV
        if doc.metadata["is_csv"] == 0:
            cur_text_chunks = text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
        else:
            data = pd.read_csv(io.StringIO(doc.text), sep=",")
            markdown = data.to_markdown(floatfmt="", index=False)
            header = markdown[: markdown.find("\n") * 2 + 2]
            split = SentenceSplitter(
                chunk_overlap=int(chunk_size) / 6,
                chunk_size=chunk_size,
                paragraph_separator="\n",
            ).split_text(markdown)
            cur_text_chunks = [header + x if header not in x else x for x in split]
            text_chunks.extend(cur_text_chunks)

        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    # nodes
    nodes = []
    counter = 0
    n = len(list(enumerate(text_chunks)))
    for idx, text_chunk in enumerate(text_chunks):
        counter += 1
        if not (quiet):
            print(f"Populating vector database (3/5), adding nodes {counter}/{n}")
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    # embeddings from nodes
    counter = 0
    for node in nodes:
        counter += 1
        if not (quiet):
            print(
                f"Populating vector database (4/5), adding nodes {counter}/{len(nodes)}"
            )
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    # load nodes into the vectorstore db
    if not (quiet):
        print("Populating vector database (5/5), adding nodes to vector store")
    vector_store.add(nodes)
