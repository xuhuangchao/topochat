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

# 加载.env文件
load_dotenv("D:/PycharmProjects/topochat/materialschat/matapp/packages/neo4j-cypher-memory/.env")


SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
SILICONFLOW_API_BASE = os.environ.get("SILICONFLOW_API_BASE")


qa_llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",
    openai_api_base=SILICONFLOW_API_BASE,
    openai_api_key=SILICONFLOW_API_KEY,
    temperature=0.0,  # 会消除输出的随机性，使得GPT的回答稳定不变
)


def convert_messages(input: List[Dict[str, Any]]) -> ChatMessageHistory:
    history = ChatMessageHistory()
    for item in input:
        history.add_user_message(item["result"]["question"])
        history.add_ai_message(item["result"]["answer"])
    return history


response_system = """
Based on the results from literature queries, please answer the following question: 

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
        MessagesPlaceholder(variable_name="arxiv_function_response"),
    ]
)


def get_arxiv_function_response(
    question: str, arxiv_context: str
) -> List[Union[AIMessage, ToolMessage]]:
    ARXIV_TOOL_ID = "arxiv_H7fABDuzEau48T10Qn0Lsh0E"  # 为 arXiv 工具定义一个唯一的 ID

    # 构建返回的消息列表
    messages = [
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": ARXIV_TOOL_ID,
                        "function": {
                            "arguments": '{"question":"' + question + '"}',
                            "name": "GetArxivInformation",
                        },
                        "type": "function",
                    }
                ]
            },
        ),
        ToolMessage(content=str(arxiv_context), tool_call_id=ARXIV_TOOL_ID),
    ]

    return messages


# TODO:在指代不清的情况下，由于arxivexplorer直接使用question进行查询，会出现回答中有不相干的信息
chain = (
    RunnablePassthrough.assign(
        arxiv_function_response=lambda x: get_arxiv_function_response(
            x["question"], x["arxiv_context"]
        ),
    )
    | RunnablePassthrough.assign(
        output=response_prompt | qa_llm | StrOutputParser(),
    )
)


# Add typing for input


class Question(BaseModel):
    question: str
    arxiv_context: str

chain_with_literature = chain.with_types(input_type=Question)
