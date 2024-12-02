1. 
（1）在MaterialsKG中统计元素在非平庸拓扑材料出现次数
MATCH (f:Formula)-[:HAS_TOPO_CLASS]->(t:TopoClass)
WHERE t.name <> 'trivial insulator'
WITH f, t.cate as Category
MATCH (f)-[:CONTAINS]->(e:Element)
WITH e.name as Element, Category
RETURN 
    Element,
    COUNT(CASE WHEN Category = 'soc_top_class' THEN 1 END) as SOC_Element_Count,
    COUNT(CASE WHEN Category = 'nsoc_top_class' THEN 1 END) as NSOC_Element_Count
ORDER BY (SOC_Element_Count+NSOC_Element_Count) DESC;

（2）在MaterialsKG中统计元素在非平庸拓扑材料出现次数/元素在所有材料出现的次数
MATCH (f:Formula)-[:CONTAINS]->(e:Element)
WITH e.name as Element, count(f) as ALL_Element_Count
OPTIONAL MATCH (f2:Formula)-[:HAS_TOPO_CLASS]->(t:TopoClass)
WHERE t.name <> 'trivial insulator'
MATCH (f2)-[:CONTAINS]->(e2:Element)
WHERE e2.name = Element
WITH Element, 
     ALL_Element_Count,
     COUNT(CASE WHEN t.cate = 'soc_top_class' THEN 1 END) as SOC_Element_Count,
     COUNT(CASE WHEN t.cate = 'nsoc_top_class' THEN 1 END) as NSOC_Element_Count
RETURN 
    Element,
    ALL_Element_Count,
    SOC_Element_Count,
    NSOC_Element_Count,
    round(1.0 * SOC_Element_Count / ALL_Element_Count * 100, 2) as SOC_Percentage,
    round(1.0 * NSOC_Element_Count / ALL_Element_Count * 100, 2) as NSOC_Percentage
ORDER BY (SOC_Percentage + NSOC_Percentage) DESC


2. query_elements.py