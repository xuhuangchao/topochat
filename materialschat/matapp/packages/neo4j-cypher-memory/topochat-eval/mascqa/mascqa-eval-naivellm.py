
import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv('../../.env')
QWEN_API_KEY = os.environ.get("QWEN_API_KEY")
QWEN_API_BASE = os.environ.get("QWEN_API_BASE")

SILICONFLOW_API_KEY = os.environ["SILICONFLOW_API_KEY"]
SILICONFLOW_API_BASE = os.environ["SILICONFLOW_API_BASE"]

df = pd.read_excel('all_questions_updated.xlsx')

with open('mascqa-eval.json','r') as f:
    questions = json.load(f)
f.close()

# 选择一个领域数据集测试回答正确率
field = 'Miscellaneous'

qids = questions[field]['qids']
qstr = questions[field]['qstr']
ques = questions[field]['questions']
nums = questions[field]['num_words']
print(f"共有 {len(ques)} 个问题需要处理")

# 初始化ChatOpenAI
chat = ChatOpenAI(
    model="qwen2.5-72b-instruct",  
    openai_api_base=QWEN_API_BASE,
    openai_api_key=QWEN_API_KEY,
    temperature=0.7
)

# 创建prompt模板
# prompt_template = """请回答:{question}

# 要求严格按照以下格式回答:
# 1. 第一行只需写出答案
# 2. 从第二行开始写出详细的解释
# """
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

# 确保存在输出目录
output = f"Qwen2.5-72b-Instruct-results/{field}"
os.makedirs(output, exist_ok=True)

# 初始化新列
if 'Qwen2.5-72b-Instruct' not in df.columns:
    df['Qwen2.5-72b-Instruct'] = ''
    
# 处理每个问题
for i, question in enumerate(ques):
    # 构建完整问题
    chain = prompt | chat
    
    # 获取回答
    response = chain.invoke({"question": question})
    response_text = response.content
    
    # 分离答案和解释
    lines = response_text.split('\n')
    answer = lines[0].strip()
    explanation = '\n'.join(lines[1:]).strip()
    
    # 为每个问题创建单独的文件
    # question_num = i + 1
    qid = qids[i]
    
    print(f"{qid} answer: {answer}")
    df.loc[df['Question Info'] == qid, 'Qwen2.5-72b-Instruct'] = answer


    # 保存答案
    answer_file = os.path.join(output, f"{qid}-answer.txt")
    with open(answer_file, 'w', encoding='utf-8') as f:
        f.write(answer)
    
    # 保存解释
    details_file = os.path.join(output, f"{qid}-details.txt")
    with open(details_file, 'w', encoding='utf-8') as f:
        f.write(explanation)
    

print(f"所有问题处理完成，文件保存在 {output} 目录下")
df.to_excel('all_questions_updated.xlsx', index=False)
