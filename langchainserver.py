#!/usr/bin/env python
from fastapi import FastAPI
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

import os

os.environ["OPENAI_API_KEY"] = "API_KEY"


# 1. Create prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer. If you don't know the answer, say "Por favor, contáctese con la línea de soporte de la universidad"
Use three sentences maximum and keep the answer as concise as possible. The answer must include the articles where the user can find more information.
Always say "¿Tienes alguna otra duda?" at the end of the answer. The answer must be in Spanish.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

# 2. Create model
model = ChatOpenAI(model="gpt-4")

# 3. Create parser
parser = StrOutputParser()

# 4. Create and inyect the context
loader = PyPDFLoader(
    "Reglamento_Práctica_Profesional_-_CD_20062023.pdf",
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Create chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | model
    | StrOutputParser()
)

# 5. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 6. Adding chain route
add_routes(
    app,
    rag_chain,
    path="/rag_chain",
)

# Ejecutar la cadena con los parámetros `context` y `question`
context = "REGLAMENTO DE PRÁCTICAS PROFESIONALES EN PREGRADO."
question = "¿Cuál es su duda?"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="localhost", port=8000)
