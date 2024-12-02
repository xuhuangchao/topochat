import os
import json
import pandas as pd
from dotenv import load_dotenv
import time
from arxivTool.chain_eval_kg import chain

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv("../../.env")
SILICONFLOW_API_KEY = os.environ["SILICONFLOW_API_KEY"]
SILICONFLOW_API_BASE = os.environ["SILICONFLOW_API_BASE"]


def topochat_eval(question):
    start_time = time.time()
    response = chain.invoke({"question": question})

    end_time = time.time()
    elapsed_time = round((end_time - start_time), 2)

    # print(f"final results：{final_result}")
    print("本次回答耗时：", elapsed_time, "秒")

    # 检查response对象中是否存在'output'字段
    if "output" in response and response["output"] is not None:
        print("reply response['output']")
        final_result = response["output"]
    # 如果不存在'output'字段，检查是否存在'content'字段
    elif hasattr(response, "content") and response.content is not None:
        print("reply response.content")
        final_result = response.content
    else:
        print("无法处理的回答格式：", response)

    return final_result


if __name__ == "__main__":
    df = pd.DataFrame()

    with open("topoqa-eval.json", "r") as f:
        qdata = json.load(f)
    f.close()

    print(f"共有 {len(qdata)} 个问题需要处理")

    # 确保存在输出目录
    output = f"topochat-results/"
    os.makedirs(output, exist_ok=True)

    # 初始化新列
    if "Question Info" not in df.columns:
        df["Question Info"] = ""
    if "Correct Answer" not in df.columns:
        df["Correct Answer"] = ""
    if "topochat" not in df.columns:
        df["topochat"] = ""

    counter = 0  # 计数器
    # 处理每个问题
    for i, item in enumerate(qdata):
        max_retries = 3  # 设置最大重试次数
        retries = 0  # 初始化重试计数器

        response_text = None
        question = item["question"]
        qid = item["qid"]
        correct_answer = item["answer"]
        print(f"====================正在处理问题：{qid}====================")
     
        while retries < max_retries:
            try:
                response_text = topochat_eval(question)
                break  # 如果成功，跳出重试循环
            except Exception as e:
                retries += 1
                print(f"处理问题时出现错误：{e}，正在重试...（第{retries}次尝试）")

                if retries == max_retries:
                    print(f"已达到最大重试次数，使用Qwen2.5回答：{question}")
                    chat = ChatOpenAI(
                        model="Qwen/Qwen2.5-72B-Instruct",
                        openai_api_base=SILICONFLOW_API_BASE,
                        openai_api_key=SILICONFLOW_API_KEY,
                        temperature=0.7,
                    )
                    prompt_template = """
                    Please answer: {question}

                    please strictly follow the format below:
                    1. The first line should contain only the answer:
                    - For choice questions, write only the option(s)
                    - For other questions, write only the answer value
                    2. From the second line onward, provide a detailed explanation.

                    Note: The first line must contain only the answer, without any additional explanatory text.
                    """
                    prompt = ChatPromptTemplate.from_template(prompt_template)
                    chain = prompt | chat

                    # 获取回答
                    response = chain.invoke({"question": question})
                    response_text = response.content
                    continue

        # 分离答案和解释
        lines = response_text.split("\n")
        answer = lines[0].strip()
        explanation = "\n".join(lines[1:]).strip()
        
        print(f"{qid} answer: {answer}")
        df.loc[i, "Question Info"] = qid
        df.loc[i, "Correct Answer"] = correct_answer
        df.loc[i, "topochat"] = answer

        #  为每个问题创建单独的文件保存答案
        answer_file = os.path.join(output, f"{qid}-answer.txt")
        with open(answer_file, "w", encoding="utf-8") as f:
            f.write(answer)

        # 保存解释
        details_file = os.path.join(output, f"{qid}-details.txt")
        with open(details_file, "w", encoding="utf-8") as f:
            f.write(explanation)

        counter += 1  # 每处理一个问题，计数器加1
        if counter % 50 == 0:  # 每处理50个问题
            print("已处理50个问题，休眠一分钟...")
            time.sleep(60)  # 休眠60秒

    print(f"所有问题处理完成，文件保存在 {output} 目录下")
    df.to_excel("all_questions_topoqa.xlsx", sheet_name="Sheet1", index=False)
