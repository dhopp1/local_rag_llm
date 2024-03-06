from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def setup_embeddings(embedding_model_id):
    "create sentence embeddings"
    embed_model = HuggingFaceEmbedding(model_name = embedding_model_id)
    
    return embed_model