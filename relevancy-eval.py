import sys
import logging
import pandas as pd

from typing import List
from llama_index.core import (
    Settings,
    Response,
    TreeIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
)

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.evaluation import EvaluationResult
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import RelevancyEvaluator


pd.set_option("display.max_colwidth", 0)

deepseek = OpenAILike(
    model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    api_base="http://localhost:8080/v1",
    api_key="token-ATRC324-IHL",
    api_key_name="x-api-key",
    api_key_prefix="",
    temperature=0.1,
    max_tokens=3096,
    streaming=True,
)

evaluator = RelevancyEvaluator(llm=deepseek)

documents = SimpleDirectoryReader("./data").load_data()

splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(documents, transformations=[splitter])


def display_eval_df(
    query: str, response: Response, eval_result: EvaluationResult
) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Response": str(response),
            "Source": response.source_nodes[0].node.text[:1000] + "...",
            "Evaluation Result": "Pass" if eval_result.passing else "Fail",
            "Reasoning": eval_result.feedback,
        },
        index=[0],
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"],
    )
    display(eval_df)


query_str = "What battles took place in New York City in the American Revolution?"
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query_str)
eval_result = evaluator.evaluate_response(query=query_str, response=response_vector)

display_eval_df(query_str, response_vector, eval_result)

query_str = "What are the airports in New York City?"
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query_str)
eval_result = evaluator.evaluate_response(query=query_str, response=response_vector)

display_eval_df(query_str, response_vector, eval_result)

query_str = "Who is the mayor of New York City?"
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query_str)
eval_result = evaluator.evaluate_response(query=query_str, response=response_vector)

display_eval_df(query_str, response_vector, eval_result)


def display_eval_sources(
    query: str, response: Response, eval_result: List[str]
) -> None:
    sources = [s.node.get_text() for s in response.source_nodes]
    eval_df = pd.DataFrame(
        {
            "Source": sources,
            "Eval Result": eval_result,
        },
    )
    eval_df.style.set_caption(query)
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Source"],
    )

    display(eval_df)


query_str = "What are the airports in New York City?"
query_engine = vector_index.as_query_engine(similarity_top_k=3, response_mode="no_text")
response_vector = query_engine.query(query_str)
eval_source_result_full = [
    evaluator.evaluate(
        query=query_str,
        response=response_vector.response,
        contexts=[source_node.get_content()],
    )
    for source_node in response_vector.source_nodes
]
eval_source_result = [
    "Pass" if result.passing else "Fail" for result in eval_source_result_full
]

display_eval_sources(query_str, response_vector, eval_source_result)

query_str = "Who is the mayor of New York City?"
query_engine = vector_index.as_query_engine(similarity_top_k=3, response_mode="no_text")
eval_source_result_full = [
    evaluator.evaluate(
        query=query_str,
        response=response_vector.response,
        contexts=[source_node.get_content()],
    )
    for source_node in response_vector.source_nodes
]
eval_source_result = [
    "Pass" if result.passing else "Fail" for result in eval_source_result_full
]

display_eval_sources(query_str, response_vector, eval_source_result)
