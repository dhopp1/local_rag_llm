import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="local_rag_llm",
    version="0.0.3",
    author="Daniel Hopp",
    author_email="daniel.hopp@un.org",
    description="Create and run a local LLM with RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhopp1/local_rag_llm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
