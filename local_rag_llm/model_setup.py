from llama_cpp import Llama
from llama_index.llms.llama_cpp import LlamaCPP
import urllib.request
import tempfile

def instantiate_model(
    text_path,
    llm_url, 
    llm_path,
    redownload_llm,
    temperature,
    max_new_tokens,
    context_window,
    n_gpu_layers
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
        if text_path is None:
            llm = Llama(
                 model_path = llm_path, 
                 n_ctx = context_window,
                 temperature = temperature,
                 n_gpu_layers = n_gpu_layers
             )
        else:
            llm = LlamaCPP(
                model_url = llm_url,
                model_path = llm_path,
                temperature = temperature,
                max_new_tokens = max_new_tokens,
                context_window = context_window,
                generate_kwargs = {},
                model_kwargs = {"n_gpu_layers": n_gpu_layers},
                verbose = True
            )
    except:
        print("model not found, if passing an llm_path, try setting redownload_llm = True")
        
    return llm