from typing import Any, Dict, List, Union
import os
import json
import numpy as np
from dotenv import load_dotenv
from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain.memory import ChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from arxivTool.arxivTool import get_embeddings

load_dotenv(".env")


NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
SILICONFLOW_API_BASE = os.environ.get("SILICONFLOW_API_BASE")

# Connection to Neo4j
graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD, database="neo4j"
)

# Cypher validation tool for relationship directions
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in graph.structured_schema.get("relationships")
]
cypher_validation = CypherQueryCorrector(corrector_schema)

# LLMs
cypher_llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",
    openai_api_base=SILICONFLOW_API_BASE,
    openai_api_key=SILICONFLOW_API_KEY,
    temperature=0.0,
)
qa_llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",
    openai_api_base=SILICONFLOW_API_BASE,
    openai_api_key=SILICONFLOW_API_KEY,
    temperature=0.0,
)


def convert_messages(input: List[Dict[str, Any]]) -> ChatMessageHistory:
    history = ChatMessageHistory()
    for item in input:
        history.add_user_message(item["result"]["question"])
        history.add_ai_message(item["result"]["answer"])
    return history


def get_cypher_examples(input: Dict[str, Any]):
    question = input["question"]

    index_a = question.find("(A)")
    if index_a != -1:
        question_without_option = question[:index_a].strip()
    else:
        question_without_option = question.strip()
    print("CypherTemplatesSearch for question:", question_without_option)
    # 获取用户问题的嵌入向量
    user_question_embedding = get_embeddings(
        [question_without_option], token=SILICONFLOW_API_KEY
    )[0]

    # Read the cypher_templates_with_embeddings.json file
    with open(
        "../../templates/cypher_templates_with_embeddings.json", "r", encoding="utf-8"
    ) as file:
        data = json.load(file)

    cypher_templates_embeddings = []
    # Convert embeddings to numpy arrays
    for record in data["queries"]:
        if "embedding" in record:
            cypher_templates_embeddings.append(record["embedding"])
        else:
            print(
                f"Warning: record with id {record.get('id', 'unknown')} does not have an embedding."
            )
    cypher_templates_embeddings = np.array(cypher_templates_embeddings)

    # 点乘计算相似度， 可以修改为余弦相似性
    similarities = np.dot(cypher_templates_embeddings, user_question_embedding)

    # 选择相似度最高的cypher语句
    cypher = data["queries"][np.argmax(similarities)]["cypher"]
    # print(f"Most similar cypher template: {cypher}")
    return cypher


# Generate Cypher statement based on natural language input
cypher_template = """Based on the Neo4j graph schema and Examples below, write a Cypher query that would answer the user's question.
Please ensure your response consists only of the Cypher query without any additional text or explanation.

{schema}

Examples:
{cypher_examples}

Question: {question}

Cypher query:"""  # noqa: E501

cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question, convert it to a Cypher query. No pre-amble.",
        ),
        ("human", cypher_template),
    ]
)

cypher_response = (
    RunnablePassthrough.assign(
        schema=lambda _: graph.get_schema,
        cypher_examples=get_cypher_examples,
    )
    | cypher_prompt
    | cypher_llm.bind(stop=["\nCypherResult:"])
    | StrOutputParser()
)

# Generate natural language response based on database results
response_system = """
Based on the results from database queries, please answer the following question: 

please strictly follow the format below:
1. The first line should contain only the answer:
   - For choice questions, write only the option(s)
   - For other questions, write only the answer value
2. From the second line onward, provide a detailed explanation.

Note: The first line must contain only the answer, without any additional explanatory text.
"""
response_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=response_system),
        HumanMessagePromptTemplate.from_template("{question}"),
        MessagesPlaceholder(variable_name="function_response"),
    ]
)


def get_function_response(
    query: str, question: str
) -> List[Union[AIMessage, ToolMessage]]:
   
    try:
        context = graph.query(cypher_validation(query))
        print(f"Cypher: {query}")
        
    except Exception as e:
        print(f"Error: {e}")
        context = ""

    DATABASE_TOOL_ID = "call_H7fABDuzEau48T10Qn0Lsh0D"
    messages = [
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": DATABASE_TOOL_ID,
                        "function": {
                            "arguments": '{"question":"' + question + '"}',
                            "name": "GetDBInformation",
                        },
                        "type": "function",
                    }
                ]
            },
        ),
        ToolMessage(content=str(context), tool_call_id=DATABASE_TOOL_ID),
    ]
    return messages


# TODO:在指代不清的情况下，由于arxivexplorer直接使用question进行查询，会出现回答中有不相干的信息
chain = (
    RunnablePassthrough.assign(query=cypher_response)
    | RunnablePassthrough.assign(
        function_response=lambda x: get_function_response(x["query"], x["question"]),
    )
    | RunnablePassthrough.assign(
        output=response_prompt | qa_llm | StrOutputParser(),
    )
)


# Add typing for input


class Question(BaseModel):
    question: str


chain_with_kg = chain.with_types(input_type=Question)
