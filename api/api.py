import os
from local_rag_llm.model_setup import instantiate_llm
from local_rag_llm.local_llm import local_llm
from fastapi import FastAPI
import gc
import pandas as pd


# hyperparameters
db_settings = pd.read_csv("db_settings.csv")
llm_list = pd.read_csv("llm_list.csv")
corpora_list = pd.read_csv("corpora_list.csv")
try:
    hf_token = os.environ["HF_TOKEN"]
except:
    hf_token = db_settings.loc[lambda x: x.field == "hf_token", "value"].values[0]

# app
app = FastAPI()


# endpoints
@app.get("/api/v1/which_llms/")
async def which_llms():
    return list(llm_list.name.values)


@app.get("/api/v1/which_corpora/")
async def which_corpora():
    return list(corpora_list.name.values)


@app.get("/api/v1/query/")
async def gen_response(
    prompt="",
    which_llm="llama3",
    which_corpus=None,
    similarity_top_k=3,
    max_new_tokens=512,
    temperature=0,
    use_chat_engine=True,
    system_prompt="You are a chatbot.",
    context_prompt="Here is the context: {context_str}",
    chat_mode="context",
    chat_history=[],
):
    # setting up the model and LLM
    llm = instantiate_llm(
        llm_path=llm_list.loc[lambda x: x.name == which_llm, "path"].values[0],
        n_gpu_layers=100,
        context_window=llm_list.loc[
            lambda x: x.name == which_llm, "context_window"
        ].values[0],
    )

    if which_corpus is not None:
        text_path = corpora_list.loc[
            lambda x: x.name == which_corpus, "text_path"
        ].values[0]
    else:
        text_path = None
        
    model = local_llm(
        text_path=text_path,
        hf_token = hf_token,
    )

    if which_corpus is not None:
        model.setup_db(
            user=db_settings.loc[lambda x: x.field == "username", "value"].values[0],
            password=db_settings.loc[lambda x: x.field == "password", "value"].values[0],
            db_name=db_settings.loc[lambda x: x.field == "db_name", "value"].values[0],
            table_name=corpora_list.loc[
                lambda x: x.name == which_corpus, "table_name"
            ].values[0],
            clear_database=False,
            clear_table=False,
        )
        
    # generating response
    try:
        response = model.gen_response(
            prompt=prompt,
            llm=llm,
            similarity_top_k=similarity_top_k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_chat_engine=use_chat_engine,
            memory_limit=int(
                llm_list.loc[lambda x: x.name == which_llm, "context_window"].values[0] / 2
            ),
            system_prompt=system_prompt,
            context_prompt=context_prompt,
            chat_mode=chat_mode,
            chat_history=chat_history,
        )
    except:
        response = "There was an error generating the response. Try reducing your top_similarity_k parameter."

    del llm
    gc.collect()
    return {"response": response, "chat_history": model.chat_engine.chat_history}
