from llama_index.llms.llama_cpp import LlamaCPP
import urllib.request
import tempfile
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from typing import Optional
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Document
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex


def instantiate_model(
    text_path,
    llm_url,
    llm_path,
    redownload_llm,
    temperature,
    max_new_tokens,
    context_window,
    n_gpu_layers,
):
    "download/load a Hugging Face model"

    # download the hugging face model if it doesn't exist
    if llm_path is None:
        temp_file = tempfile.NamedTemporaryFile()
        llm_path = temp_file.name
        redownload_llm = True

    if redownload_llm:
        print("downloading model...")
        urllib.request.urlretrieve(llm_url, llm_path)

    # if no text path given, run vanilla LLM
    try:
        llm = LlamaCPP(
            model_url=llm_url,
            model_path=llm_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": n_gpu_layers},
            verbose=True,
        )
    except:
        print(
            "model not found, if passing an llm_path, try setting redownload_llm = True"
        )

    return llm


class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


def gen_response(
    prompt,
    llm,
    vector_store=None,
    embed_model=None,
    query_mode="default",
    similarity_top_k=4,
    max_new_tokens=512,
    use_chat_engine=False,
    chat_engine=None,
    reset_chat_engine=False,
    text_path=None,
    system_prompt="You are a chatbot.",
    memory_limit=2048,
    streaming=False,
):
    "generate a response from the LLM"
    # reset chat engine if required
    chat_query = "chat"
    if use_chat_engine and reset_chat_engine:
        if chat_engine is not None:
            chat_engine.reset()

    # use chat engine
    if use_chat_engine:
        memory = ChatMemoryBuffer.from_defaults(token_limit=memory_limit)
        if chat_engine is None:
            # non-RAG
            if text_path is None:
                index = VectorStoreIndex.from_documents(
                    [Document(text=" ", metadata={})], embed_model=embed_model
                )
                chat_engine = index.as_chat_engine(
                    verbose=True,
                    llm=llm,
                    chat_mode="context",
                    system_prompt=system_prompt,
                    memory=memory,
                    streaming=streaming,
                )
            # RAG
            else:
                index = VectorStoreIndex.from_vector_store(
                    vector_store, embed_model=embed_model
                )
                chat_engine = index.as_chat_engine(
                    verbose=True,
                    llm=llm,
                    chat_mode="context",
                    system_prompt=system_prompt,
                    similarity_top_k=similarity_top_k,
                    streaming=streaming,
                )
    else:
        # non-RAG
        if text_path is None:
            index = VectorStoreIndex.from_documents(
                [Document(text=" ", metadata={})], embed_model=embed_model
            )
            chat_engine = index.as_chat_engine(
                verbose=True, llm=llm, chat_mode="context", system_prompt=system_prompt, streaming=streaming,
            )
        # RAG
        else:
            retriever = VectorDBRetriever(
                vector_store,
                embed_model,
                query_mode=query_mode,
                similarity_top_k=similarity_top_k,
            )
            chat_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
            chat_query = "query"

    # prompt response
    if chat_query == "chat":
        if streaming:
            response = chat_engine.stream_chat(prompt)
        else:
            response = chat_engine.chat(prompt)
    elif chat_query == "query":
        response = chat_engine.query(prompt)
    final_response = {}
    
    if streaming:
        final_response["response"] = response
    else:
        final_response["response"] = response.response

    # return sources passages
    if text_path is not None:
        # getting source passages
        for j in range(len(response.source_nodes)):
            final_response[
                f"supporting_text_{str(j+1).zfill(3)}"
            ] = f"metadata: {str(response.source_nodes[j].metadata)}| source text: {response.source_nodes[j].get_text()}"
    return {"final_response": final_response, "chat_engine": chat_engine}
