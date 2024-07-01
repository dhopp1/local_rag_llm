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


def instantiate_llm(
    llm_url=None,
    llm_path=None,
    redownload_llm=False,
    temperature=None,
    max_new_tokens=None,
    context_window=3900,
    verbose=False,
    n_gpu_layers=0,
):
    """download/load a Hugging Face model. Temperature, max new tokens, and context window can be left alone and adjusted at inference time in the gen_response() function.
    parameters:
        :llm_url: str: URL for initial download of the LLM
        :llm_path: str: path of the LLM locally, if available
        :redownload_llm: bool: whether or not to force redownloading of the LLM model
        :temperature: float: number between 0 and 1, 0 = more conservative/less creative, 1 = more random/creative
        :max_new_tokens: int: limit of how many tokens to produce for an answer
        :context_window: int: 'working memory' of the LLM, varies by model
        :n_gpu_layers: str: number of GPU layers to use, set '0' for cpu
    returns:
        :LlamaCPP: LlamaCPP LLM
    """

    # download the hugging face model if it doesn't exist
    if llm_path is None:
        temp_file = tempfile.NamedTemporaryFile()
        llm_path = temp_file.name
        redownload_llm = True

    if redownload_llm:
        print("downloading model...")
        urllib.request.urlretrieve(llm_url, llm_path)

    try:
        kwargs = {
            k: v
            for k, v in [
                ("model_url", llm_url),
                ("model_path", llm_path),
                ("temperature", temperature),
                ("max_new_tokens", max_new_tokens),
                ("context_window", context_window),
                ("model_kwargs", {"n_gpu_layers": n_gpu_layers}),
                ("verbose", True),
            ]
            if v is not None
        }
        llm = LlamaCPP(**kwargs)
        llm.verbose = verbose
        llm.model_kwargs["verbose"] = verbose
        return llm
    except:
        print(
            "model not found, if passing an llm_path, try setting redownload_llm = True"
        )


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
    temperature=None,
    max_new_tokens=None,
    use_chat_engine=False,
    chat_engine=None,
    reset_chat_engine=False,
    text_path=None,
    system_prompt="",
    context_prompt=None,
    memory_limit=2048,
    streaming=False,
    chat_mode="context",
    chat_history=[],
):
    "generate a response from the LLM"
    if context_prompt == None:
        if chat_mode == "context":
            context_prompt = "Context information is below.\n--------------------\n{context_str}\n--------------------\n"
        elif chat_mode == "condense_plus_context":
            context_prompt = '\n  The following is a friendly conversation between a user and an AI assistant.\n  The assistant is talkative and provides lots of specific details from its context.\n  If the assistant does not know the answer to a question, it truthfully says it\n  does not know.\n\n  Here are the relevant documents for the context:\n\n  {context_str}\n\n  Instruction: Based on the above documents, provide a detailed answer for the user question below.\n  Answer "don\'t know" if not present in the document.\n  '
    if text_path is None:
        context_prompt = ""

    # reset chat engine if required
    if use_chat_engine and reset_chat_engine:
        if chat_engine is not None:
            chat_engine.reset()

    # adjust LLM parameters
    original_temperature = llm.temperature
    original_max_tokens = llm.max_new_tokens

    llm.generate_kwargs["temperature"] = (
        temperature if temperature is not None else original_temperature
    )
    llm.generate_kwargs["max_tokens"] = (
        max_new_tokens if max_new_tokens is not None else original_max_tokens
    )

    # use chat engine
    if chat_engine is None or not (use_chat_engine):
        memory = ChatMemoryBuffer.from_defaults(
            chat_history=[], token_limit=memory_limit
        )
    else:
        if chat_history == []:
            chat_history = chat_engine.chat_history
        memory = ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=memory_limit
        )

    kwargs = {
        "verbose": True,
        "llm": llm,
        "chat_mode": chat_mode,
        "memory": memory,
        "system_prompt": system_prompt,
        "similarity_top_k": similarity_top_k,
        "streaming": streaming,
    }
    if chat_mode == "condense_plus_context":
        kwargs["context_prompt"] = context_prompt

    # non-RAG
    if text_path is None:
        index = VectorStoreIndex.from_documents(
            [Document(text=" ", metadata={})], embed_model=embed_model
        )
    else:
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )

    # create chat engine
    chat_engine = index.as_chat_engine(**kwargs)

    if chat_mode == "context" or text_path is None:
        chat_engine._context_template = context_prompt
        chat_engine._context_prompt_template = context_prompt

    # prompt response
    if streaming:
        response = chat_engine.stream_chat(prompt)
    else:
        response = chat_engine.chat(prompt)
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

    # adjust LLM parameters back
    llm.generate_kwargs["temperature"] = original_temperature
    llm.generate_kwargs["max_tokens"] = original_max_tokens

    return {"final_response": final_response, "chat_engine": chat_engine}
