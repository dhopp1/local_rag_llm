FROM python

WORKDIR /app

RUN pip install -r https://raw.githubusercontent.com/dhopp1/local_rag_llm/main/requirements.txt
RUN pip install local-rag-llm