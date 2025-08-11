import chainlit as cl
import os

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.embeddings.openai_like import OpenAILikeEmbedding


# environment host http link and key
os.environ["OPENAI_API_BASE"] = "http://localhost:8080"
os.environ["OPENAI_API_KEY"] = "token-ATRC324-IHL"

# Global setting for Chatbot model
Settings.llm = OpenAILike(
    model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    api_base="http://localhost:8080/v1",
    api_key="token-ATRC324-IHL",
    api_key_name="x-api-key",
    api_key_prefix="",
    temperature=0.1,
    max_tokens=20000,
    streaming=True,
)

# Global settings for embed model
Settings.embed_model = OpenAILikeEmbedding(
    model_name="Qwen/Qwen3-Embedding-4B",
    api_base="http://localhost:8088/v1",
    api_key="token-ATRC324-IHL",
)

reranker = LLMRerank(llm=Settings.llm, top_n=5)

Settings.context_window = 8192

# where you want your storage located
DATA_DIR = "./data"

# where you want your config files located
PERSIST_DIR = "./storage"

# if the storage is not updating remove the lines from try to except and see what
# error gets thrown
try:
    # checks storage settings and uses is nothing changes
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
except FileExistsError:
    # builds new file storage and saves settings for it
    docs = SimpleDirectoryReader(DATA_DIR).load_data(show_progress=True)
    node_parser = SentenceSplitter(chunk_size=4096, chunk_overlap=256)
    nodes = node_parser.get_nodes_from_documents(docs)
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist()


# start up of chainlit websight
@cl.on_chat_start
async def start():
    Settings.callback_manager = CallbackManager([LlamaIndexCallbackHandler()])
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=15,
        callbackmanager=Settings.callback_manager,
        node_postprocessors=[reranker],
    )
    cl.user_session.set("query_engine", query_engine)

    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()


# on every message outside of startup does this
@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")

    msg = cl.Message(content="", author="Assistant")

    res = await cl.make_async(query_engine.query)(message.content)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()
