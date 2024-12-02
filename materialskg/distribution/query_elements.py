from py2neo import Graph
import pandas as pd
import os

# 连接Neo4j数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4jxuhc"))  # 替换为你的认证信息

# 查询语句模板
query_template = """
MATCH (f:Formula)-[:CONTAINS]->(e:Element)
WHERE e.name = $element_name
WITH f
MATCH (f)-[:HAS_TOPO_CLASS]->(t:TopoClass)
WHERE t.cate = $cate 
AND t.name <> 'trivial insulator'
RETURN f.name as Formula, t.name as Topology_Class
"""

def query_element_materials(element_name,cate):
    # 执行查询
    results = graph.run(query_template, element_name=element_name, cate=cate).data()
    return results

def save_to_csv(results, element_name, cate, len):
    # 转换结果为DataFrame
    df = pd.DataFrame(results)
    
    # 创建输出目录（如果不存在）
    output_dir = f"results/{cate}_results/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存到CSV文件
    output_file = os.path.join(output_dir, f"{element_name}_topology_materials_{len}.csv")
    df.to_csv(output_file, index=False, encoding='utf-8')
    return output_file

def batch_query_elements(elements, cate):
    results_summary = {}
    
    for element in elements:
        # 查询该元素的材料
        results = query_element_materials(element, cate)
        
        # 保存结果到CSV
        if results:
            output_file = save_to_csv(results, element, cate, len(results))
            results_summary[element] = {
                'count': len(results),
                'file': output_file
            }
        else:
            results_summary[element] = {
                'count': 0,
                'file': None
            }
    
    # 创建汇总报告
    summary_df = pd.DataFrame.from_dict(results_summary, orient='index')
    summary_df.index.name = 'Element'
    summary_df.to_csv(f'results/{cate}_summary_report.csv')
    
    return results_summary


if __name__ == '__main__':
    data = pd.read_excel('element-count.xlsx')
    element_list = []
    for i, row in data.iterrows():
        element = row['Element']
        element_list.append(element)

    batch_query_elements(element_list, 'soc_top_class')
    batch_query_elements(element_list, 'nsoc_top_class')
    