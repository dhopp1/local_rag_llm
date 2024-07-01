# Change Log
### 0.0.21
### Added
* Enabled `chat_history` to be passed to `gen_response` function.
* Added FastAPI support

### 0.0.20
### Fixed
* Fixed bug with memory limit

### 0.0.19
### Fixed
* Fixed problem with context prompt for non-RAG models

### 0.0.18
### Added
* Added `condense_plus_context` option for chat mode in `gen_response`, which is better at answering follow-up questions with RAG.
* Added ability to tweak `context_prompt` in `gen_response`

### 0.0.17
### Fixed
* Fixed issue with roles in `pg_dump` and `pg_restore`

### 0.0.16
### Fixed
* Bug fix in `pg_dump` and `pg_restore` functions

### 0.0.15
### Fixed
* Bug fix in `pgdump`

### 0.0.14
### Added
* Added dockerfiles and instructions
* Added functions to dump and restore vector databases

### 0.0.13
### Fixed
* `memory_limit` in chat mode would previously overload the context window and stop working, fixed now.
* `context_window` moved to be only at LLM instantiation, removed from other locations in `gen_response` which didn't work.

### 0.0.12
### Added
* split the LLM from the vector DB/chat engine model, meaning now you can have multiple separate model objects use the same LLM. Temperature, context window, max new tokens, system prompt, etc. can also all be changed at inference time via the `model.gen_response()` function.

### 0.0.11
### Added
* added automatic handling of CSV data files by converting them to chunked markdown tables

### 0.0.10
### Added
* added `streaming` option to `.gen_response`

### 0.0.9
### Added
* `'mps'` device

### 0.0.8
### Fixed
* separated dropping database and dropping table

### 0.0.7
### Fixed
* persistent vector databases

### 0.0.6
### Added
* ability to have persistent sessions with a chat engine

### 0.0.5
### Added
* ability to change `chunk_overlap` and `paragraph_separator` parameters in `SentenceSplitter`

### 0.0.4
### Added
* ability to close vector database connection

## 0.0.3
### Added
* two additional methods for converting CSVs to LLM-readable files

## 0.0.2
### Added
* ability to convert tabular CSVs to LLM-readable text files

## 0.0.1
### Added
* initial release