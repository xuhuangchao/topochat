import json
import random


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def select_random_items(data, count):
    return random.sample(data, count)


def main():
    file1 = "E:/凝聚态材料重大项目申请/topoqa/xhc/test_questions.json"
    file2 = "E:/凝聚态材料重大项目申请/topoqa/xhc/instruction_output_all_5.json"
    output_file = "topoqa-eval.json"

    data1 = read_json(file1)
    data2 = read_json(file2)
    count = 1
    processed_data = []
    topoclass_questions = random.sample(data1, 100)
    bandgap_questions = random.sample(data2, 100)
    for item in topoclass_questions:
        question = f"{item['Question']} (A): {item['A']}, (B): {item['B']}, (C): {item['C']}, (D): {item['D']}"
        answer = item["Answer"]
        processed_data.append(
            {"qid": f"topoqa_{count}", "question": question, "answer": answer}
        )
        count += 1

    for item in bandgap_questions:
        question = item["instruction"]
        answer = item["output"]
        processed_data.append(
            {"qid": f"topoqa_{count}", "question": question, "answer": answer}
        )
        count += 1

    write_json(processed_data, output_file)


if __name__ == "__main__":
    main()
