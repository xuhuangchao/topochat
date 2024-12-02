# topochat

#### 介绍
Material Science Chatbot

#### 软件架构
基于langchain提供的neo4j-cypher-memory和ArixvXexplorer开发的知识图谱+文献材料专家问答系统

![workflow](images/workflow.png)



##### 1.知识图谱MaterialsKG：

基于Materiae和Materials Project数据库构建，包括**Robocrystallographer**生成的结构描述，截止2024/11/13共有19w+节点和82w+关系

###### Nodes & Relations: 见nodes.csv和relations.csv



<img src="images/graph.png" alt="graph" style="zoom:80%;" />



##### 2.文献检索和聚类：

基于ArixvXexplorer进行初步检索，使用Leiden聚类算法进行社区分析，用户选择感兴趣的社区后自动定位中心节点文献进行回答



##### 3.最终回答基于图数据库和文献查询结果两部分综合回复



#### 效果展示
##### （旧版） 

<img src="images/Chatbot.png" alt="Chatbot" style="zoom:80%;" />

##### （新版）

![Chatbotv2](images/Chatbotv2.png)



#### 安装教程（对应的python库）
待补充



#### 使用说明

1. cd materialschat/matapp/packages/neo4j-cypher-memory

2. streamlit run app.py (old version: streamlit_app.py)

3. [浏览器访问http://localhost:8501/](http://localhost:8501/)

   

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request

