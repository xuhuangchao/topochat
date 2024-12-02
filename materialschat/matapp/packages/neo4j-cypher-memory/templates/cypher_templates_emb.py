import os
import sys
import json
from dotenv import load_dotenv
from arxivTool.arxivTool import get_embeddings

# 加载.env文件
load_dotenv("../.env")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

# Read the cypher_templates.json file
with open('cypher_templates.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    
templates = data['queries']

# Extract descriptions and compute embeddings
descriptions = [record['description'] for record in templates]
embeddings = get_embeddings(descriptions, token=SILICONFLOW_API_KEY)

# Add embeddings and queries to the original records
for i, (record, embedding) in enumerate(zip(templates, embeddings)):
    if isinstance(record, dict):
        record['embedding'] = embedding.tolist()
    else:
        print(f"Warning: record at index {i} is not a dictionary and will be skipped.")


# Save the updated records to a new JSON file
with open('cypher_templates_with_embeddings.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)