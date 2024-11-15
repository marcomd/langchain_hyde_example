# HyDE-enhanced RAG

Enhanced version that applies HyDE technique via langchain

- parsing and loading of html, pdf, md, docx documents
- storing embeddings on a vector DB (pgvector)

:warning: This is a proof of concept and not intended for production use although it can be used as a starting point for a more robust implementation.

If you want to use a simpler version of the RAG, w/o DB and document parsing please check the tag v1.0 on this repository.



## Installation

Create a PostgreSQL database with user `postgres` and password `postgres` and name `langchain_hyde`

Install the pgvector extension https://github.com/pgvector/pgvector and follow instructions.

Clone this repository, go into the folder and run:

`bundle install`


## Usage

`ruby langchain_hyde.rb`

You can customize the prompt:

`ruby langchain_hyde.rb "Meditation can help sex?"`

If you change documents you need to reset the DB, use `erase` as second argument:

`ruby langchain_hyde.rb "Meditation can help sex?" erase`



## Customization

- Place your documents in the `documents` folder and edit the `documents.json` to choose which documents to load.
- You can customize the `langchain_hyde.rb` file to change the vector DB.



## Compatibility

Tested on Ruby 3.3
