import streamlit as st
from datetime import datetime
from config import (
    load_css,
    process_text,
    create_new_session,
    get_session_history,
    display_chat_messages,
    get_session_details,
    process_selected_cluster,
    perform_clustering_analysis,
)


# 展示会话列表区域
def display_session_list(sessions):
    st.sidebar.header("会话列表 💭")

    # Add New Chat button
    if st.sidebar.button("➕ 新建会话", key="new_chat", help="创建新的会话"):
        new_session_id = create_new_session()
        print(f"新建session, 当前session id为{new_session_id}")
        # 更新session state
        st.session_state.selected_session_id = new_session_id
        st.session_state.current_chat = []
        # 添加新会话的基本信息到 session_state
        st.session_state.new_chat_info = {
            "id": new_session_id,
            "timestamp": datetime.now(),
            "is_new": True,
        }
        st.rerun()
        return  # 重要：防止继续执行后续代码

    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    # 创建用于显示的会话选项和映射
    session_options = []
    session_map = {}
    
    # 如果存在新会话信息，添加到列表最前面
    new_chat_info = st.session_state.get("new_chat_info")
    if new_chat_info and new_chat_info["is_new"]:
        display_text = f"""
        Session {new_chat_info["id"]}
        新会话
        {new_chat_info["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}
        """
        session_options.append(display_text)
        session_map[display_text] = new_chat_info["id"]
    
    # 添加现有会话到选项中
    for session in sessions:
        session_id = session["session"]["id"]
        last_answer_text = (
            session["last_answer"]["text"][:50] + "..."
            if len(session["last_answer"]["text"]) > 50
            else session["last_answer"]["text"]
        )
        timestamp = session["last_message"]["date"]

        # 将时间戳转换为易读格式
        readable_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # 创建显示文本
        display_text = f"""
        Session {session_id}
        {process_text(last_answer_text)}
        {readable_timestamp}
        """
        session_options.append(display_text)
        session_map[display_text] = session_id

    # 获取当前选中的索引
    current_index = 0
    if st.session_state.get("selected_session_id"):
        for i, opt in enumerate(session_options):
            if session_map[opt] == st.session_state.selected_session_id:
                current_index = i
                break

    # 使用radio组件
    selected_session = st.sidebar.radio(
        label="选择会话",
        label_visibility="collapsed",
        options=session_options,
        index=current_index,
        key="session_selector",
    )

    # 直接更新session state
    st.session_state.selected_session_id = session_map[selected_session]
    st.session_state.current_chat = []


def main():
    # 页面配置
    st.set_page_config(
        page_title="TopoChat", layout="wide", initial_sidebar_state="expanded"
    )

    # Initialize session state
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = []
    if "selected_session_id" not in st.session_state:
        st.session_state.selected_session_id = None

    load_css("static/styles.css")

    # Sidebar content
    with st.sidebar:
        sessions = get_session_history(user_id="user_123")
        display_session_list(sessions)

    # 显示历史会话或新会话
    if st.session_state.selected_session_id:
        # 检查是否是新会话
        new_chat_info = st.session_state.get("new_chat_info")
        if new_chat_info and new_chat_info["id"] == st.session_state.selected_session_id:
            # 新会话显示空白对话界面
            display_chat_messages(st.session_state.current_chat)
        else:
            # 获取并显示历史会话记录
            session_details = get_session_details(
                user_id="user_123", session_id=st.session_state.selected_session_id
            )
            messages = []
            for d in session_details:
                messages.append({"role": "user", "content": d["question"]})
                messages.append({"role": "assistant", "content": d["answer"]})

            display_chat_messages(messages)
    else:
        # 未选择任何会话时显示空白界面
        display_chat_messages([])
        

    # 创建聊天输入框
    user_input = st.chat_input("请输入您的消息:")

    if user_input:
        # 添加用户问题到聊天记录
        st.session_state.current_chat.append({"role": "user", "content": user_input})
        # 设置处理中的消息
        st.session_state.processing_message = user_input

    # 处理流程控制
    if "processing_message" in st.session_state:
        # 显示用户问题
        display_chat_messages(
            [{"role": "user", "content": st.session_state.processing_message}]
        )
        # 聚类分析阶段，如果还没有clustering_results，进行聚类分析
        if "clustering_results" not in st.session_state:
            with st.spinner("正在进行聚类分析..."):
                # 执行聚类分析
                (
                    community_info,
                    elapsed_time_1,
                    cluster_labels,
                    titles,
                    summaries,
                    representative_docs,
                ) = perform_clustering_analysis(st.session_state.processing_message)

                if community_info is not None:
                    # 显示聚类结果
                    st.markdown(
                        """
                        <div class="card">
                            <h3>📊 聚类分析结果</h3>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(community_info, unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <p class="time-info">聚类分析耗时：{elapsed_time_1}秒</p>""",
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.image("image.png", width=400, caption="聚类可视化")

                    # 保存聚类结果到session_state
                    st.session_state.clustering_results = {
                        "community_info": community_info,
                        "elapsed_time_1": elapsed_time_1,
                        "cluster_labels": list(set(cluster_labels)),
                        "titles": titles,
                        "summaries": summaries,
                        "representative_docs": representative_docs,
                    }

        # 如果有clustering_results，显示选择界面
        if "clustering_results" in st.session_state:
            # 显示社区选择
            st.markdown(
                """
                <div class="card">
                    <h3>🎯 选择感兴趣的社区</h3>
                """,
                unsafe_allow_html=True,
            )

            # 如果没有选择过cluster，创建选择框
            if "selected_cluster" not in st.session_state:
                st.session_state.selected_cluster = None

            selected_cluster = st.selectbox(
                "选择一个社区进行深入分析：",
                options=st.session_state.clustering_results["cluster_labels"],
                help="选择您感兴趣的文献社区",
                key="cluster_selector",
            )

            # 保存选择的cluster
            st.session_state.selected_cluster = selected_cluster

            # 添加确认按钮
            if st.button("确认选择并继续分析", key="confirm_cluster"):
                if st.session_state.selected_cluster is not None:
                    with st.spinner("正在深入分析选中的社区..."):
                        answer = process_selected_cluster(
                            st.session_state.clustering_results,
                            st.session_state.selected_cluster,
                            st.session_state.processing_message,
                            st.session_state.selected_session_id,
                        )

                        if answer:
                            st.session_state.current_chat.append(
                                {"role": "assistant", "content": answer}
                            )
                            # 清除新会话标记，因为已经有了对话记录
                            if st.session_state.get("new_chat_info"):
                                st.session_state.new_chat_info = None

                            # 清除处理状态
                            if "processing_message" in st.session_state:
                                del st.session_state.processing_message
                            if "clustering_results" in st.session_state:
                                del st.session_state.clustering_results
                            if "selected_cluster" in st.session_state:
                                del st.session_state.selected_cluster

                            print("cache clean!")
                            st.rerun()


if __name__ == "__main__":
    main()
