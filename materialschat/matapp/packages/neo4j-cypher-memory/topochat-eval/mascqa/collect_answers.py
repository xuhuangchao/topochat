import pandas as pd
import os
import glob

# 读取Excel文件
df = pd.read_excel('all_questions_updated.xlsx')
    
# 设置目录路径
directory = 'topochat-random-results/Fluid/'

# 构建完整的文件路径模式，匹配所有-answer.txt后缀的txt文件
file_pattern = os.path.join(directory, '*-answer.txt')

# 遍历所有匹配的文件
for file_path in glob.glob(file_pattern):
    # 提取文件名（不包括扩展名）作为qid
    filename = os.path.basename(file_path)
    qid = filename.replace('-answer.txt', '')  # 移除'-answer.txt'后缀
    print(f'Processing question {qid}...')
    # 读取txt文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        answer = file.read().strip()  # 读取字符串并去除两端空白字符
    
    # 将答案赋值给DataFrame
    df.loc[df['Question Info'] == qid, 'topochat-random'] = answer

# 保存更新后的DataFrame回Excel文件
df.to_excel('all_questions_updated.xlsx', index=False)