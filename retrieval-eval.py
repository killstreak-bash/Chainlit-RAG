import asyncio
import nest_asyncio
import pandas as pd

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

nest_asyncio.apply()

Settings.embed_model = OpenAILikeEmbedding(
    model_name="Qwen/Qwen3-Embedding-4B",
    api_base="http://localhost:8088/v1",
    api_key="token-ATRC324-IHL",
)

# Global setting for Chatbot model
Settings.llm = OpenAILike(
    model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    api_base="http://localhost:8080/v1",
    api_key="token-ATRC324-IHL",
    api_key_name="x-api-key",
    api_key_prefix="",
    temperature=0.1,
    max_tokens=3096,
    streaming=True,
)

documents = SimpleDirectoryReader("./data").load_data()
node_parser = SentenceSplitter(chunk_size=4096, chunk_overlap=256)
nodes = node_parser.get_nodes_from_documents(documents)

for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"

vector_index = VectorStoreIndex(nodes)
retriever = vector_index.as_retriever(similarity_top_k=2)

qa_dataset = generate_question_context_pairs(
    nodes, llm=Settings.llm, num_questions_per_chunk=2
)

queries = qa_dataset.queries.values()
print(list(queries)[2])

qa_dataset.save_json("pg_eval_dataset.json")

qa_dataset = EmbeddingQAFinetuneDataset.from_json("pg_eval_dataset.json")

include_cohere_rerank = False

metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]

if include_cohere_rerank:
    metrics.append(
        "cohere_rerank_relevancy"  # requires COHERE_API_KEY environment variable to be set
    )

retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=retriever)


# try it out on a sample query
sample_id, sample_query = list(qa_dataset.queries.items())[1]
sample_expected = qa_dataset.relevant_docs[sample_id]

eval_result = retriever_evaluator.evaluate(sample_query, sample_expected)
print(eval_result)


# try it out on an entire dataset
async def main():
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)

    def display_results(name, eval_results):
        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(metric_dicts)

        columns = {
            "retrievers": [name],
            **{k: [full_df[k].mean()] for k in metrics},
        }

        if include_cohere_rerank:
            crr_relevancy = full_df["cohere_rerank_relevancy"].mean()
            columns.update({"cohere_rerank_relevancy": [crr_relevancy]})

        metric_df = pd.DataFrame(columns)

        return metric_df

    display_results("top-2 eval", eval_results)


asyncio.run(main())
