from neo4j_cypher_memory.chain import chain
from app
if __name__ == "__main__":
    original_query = "Bi2Se3的拓扑性质有哪些?"
    print(
        chain.invoke(
            {
                "question": original_query,
                "user_id": "user_123",
                "session_id": "session_1",
            }
        )
    )
    # follow_up_query = "请推荐一些Bi2Se3的相似材料"
    # print(
    #     chain.invoke(
    #         {
    #             "question": follow_up_query,
    #             "user_id": "user_123",
    #             "session_id": "session_1",
    #         }
    #     )
    # )
