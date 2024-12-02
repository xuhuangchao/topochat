import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from arxivTool.arxivTool import (
    fetch_arxiv_summaries_byexplorer,
    cluster_summaries_leiden,
    print_cluster_info_with_format,
    remove_references,
    split_into_chunks,
    get_most_relevant_chunk,
    answer_question,
)
from langchain_community.document_loaders import ArxivLoader
from neo4j_cypher_memory.chain import chain, history_graph


# 加载CSS
def load_css(css_file):
    with open(css_file, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def process_text(text):
    # 处理所有可能的特殊字符和格式
    replacements = {
        "#": "",  # 移除井号
        "\n": " ",  # 换行符替换为空格
        "\r": "",  # 移除回车符
        "\t": " ",  # 制表符替换为空格
        "  ": " ",  # 多个空格替换为单个空格
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # 清理可能残留的多余空格
    return " ".join(text.split())


def get_latest_session_id():
    # 使用Cypher查询获取最大的session id
    query = """
    MATCH (s:Session)
    RETURN COALESCE(MAX(s.id), 0) as max_id
    """
    result = history_graph.query(query)
    return result[0]["max_id"]


def create_new_session():
    # 获取最新session id并递增
    latest_id = get_latest_session_id()
    new_session_id = int(latest_id) + 1

    return new_session_id


# 获取当前用户所有会话历史
def get_session_history(user_id):
    query = """
    MATCH (u:User {id:$user_id})-[:HAS_SESSION]->(s:Session)
    MATCH (s)-[:LAST_MESSAGE]->(last_message)
    MATCH (last_message)-[:HAS_ANSWER]->(answer)
    RETURN s, last_message, answer
    ORDER BY last_message.date DESC
    """
    result = history_graph.query(query, params={"user_id": user_id})
    sessions = [
        {
            "session": record["s"],
            "last_message": record["last_message"],
            "last_answer": record["answer"],
        }
        for record in result
    ]
    return sessions


# 获取指定会话的所有问题和答案
def get_session_details(user_id, session_id, window=3):
    query = (
        """
    MATCH (u:User {id:$user_id})-[:HAS_SESSION]->(s:Session {id:$session_id}),
                       (s)-[:LAST_MESSAGE]->(last_message)
    MATCH p=(last_message)<-[:NEXT*0.."""
        + str(window)
        + """]-()
    WITH p, length(p) AS length
    ORDER BY length DESC LIMIT 1
    UNWIND reverse(nodes(p)) AS node
    MATCH (node)-[:HAS_ANSWER]->(answer)
    RETURN {question:node.text, answer:answer.text} AS result
    """
    )
    result = history_graph.query(
        query, params={"user_id": user_id, "session_id": session_id}
    )
    details = [record["result"] for record in result]

    return details


# 展示User和Assistant的聊天消息
def display_chat_messages(messages):
    # 使用 st.chat_messages 来显示所有消息
    for msg in messages:
        if msg["role"] == "user":
            st.chat_message("user", avatar="static/user.png").write(
                msg["content"]
            )  # 假设用户消息的头像是 user.png
        else:
            st.chat_message("assistant", avatar="static/chatbot.png").write(
                msg["content"]
            )  # 假设助手消息的头像是 assistant.png


# 执行文献聚类
def perform_clustering_analysis(question):
    status_placeholder = st.empty()
    try:
        with status_placeholder.container():
            # st.info("正在进行文献聚类分析...")
            start_time_1 = time.time()

            summaries, titles = fetch_arxiv_summaries_byexplorer(question)
            cluster_labels, _, embeddings, representative_docs = (
                cluster_summaries_leiden(summaries, resolution=1.0)
            )
            community_info = print_cluster_info_with_format(
                cluster_labels, summaries, titles
            )

            end_time_1 = time.time()
            elapsed_time_1 = end_time_1 - start_time_1

        return (
            community_info,
            elapsed_time_1,
            cluster_labels,
            titles,
            summaries,
            representative_docs,
        )

    except Exception as e:
        status_placeholder.error(f"处理过程中出现错误: {str(e)}")
        return None


def display_clustering_results(question, return_results=True):
    """
    第一阶段：展示聚类结果并返回中间结果
    """
    result_placeholder = st.empty()

    (
        community_info,
        elapsed_time_1,
        cluster_labels,
        titles,
        summaries,
        representative_docs,
    ) = perform_clustering_analysis(question)

    if community_info is not None:
        with result_placeholder.container():
            st.markdown(
                """
                <div class="card">
                    <h3>📊 聚类分析结果</h3>
                """,
                unsafe_allow_html=True,
            )

            st.write(community_info)
            st.markdown(
                f"""
                <p class="time-info">聚类分析耗时：{elapsed_time_1}秒</p>""",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.image("image.png", width=400, caption="聚类可视化")

            return {
                "community_info": community_info,
                "elapsed_time_1": elapsed_time_1,
                "cluster_labels": list(set(cluster_labels)),
                "titles": titles,
                "summaries": summaries,
                "representative_docs": representative_docs,
            }

    return None


def process_selected_cluster(
    clustering_results, selected_cluster, question, session_id
):
    """
    第二阶段：处理选中的聚类并生成答案
    """
    start_time_2 = time.time()

    # 获取最相关文献
    doc_index = clustering_results["representative_docs"][selected_cluster]
    most_relevant_title = clustering_results["titles"][doc_index]
    most_relevant_summary = clustering_results["summaries"][doc_index]

    st.markdown("#### 📑 最相关文献")
    st.markdown(f"**标题**: {most_relevant_title}")
    st.markdown(f"**摘要**: {most_relevant_summary}")

    # 处理PDF文本
    pdf_text = (
        ArxivLoader(query=most_relevant_title, load_max_docs=1).load()[0].page_content
    )
    pdf_text_without_references = remove_references(pdf_text)

    chunks = split_into_chunks(pdf_text_without_references, chunk_size=2000)
    top_n_chunks = get_most_relevant_chunk(question, chunks, n=3)

    # 生成最终回答
    response = answer_question(question, top_n_chunks)
    final_result = chain.invoke(
        {
            "question": question,
            "user_id": "user_123",
            "session_id": session_id,
            "arxiv_context": response,
        }
    )

    end_time_2 = time.time()
    elapsed_time_2 = end_time_2 - start_time_2
    total_time = round(clustering_results["elapsed_time_1"] + elapsed_time_2, 2)

    st.markdown("#### 💡 最终分析结果")
    st.markdown(f"**回答**: {final_result}")

    st.markdown(
        f"""
        <p class="time-info">
            深入分析耗时：{elapsed_time_2}秒<br>
            总耗时：{total_time}秒
        </p>
    """,
        unsafe_allow_html=True,
    )

    return final_result
