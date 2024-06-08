import json

# 读取原始JSON文件
with open('questions.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 初始化各题型的格式化数据列表
formatted_single_choice = []
formatted_multiple_choice = []
formatted_true_false = []
formatted_fill_in_the_blanks = []

# 处理单选题
for question, details in raw_data.get("single_choice_questions", {}).items():
    formatted_item = {
        "questionStem": question,  # 题干
        "option": [option[1:] for option in details["options"]],  # 选项
        "answer": details["answer"],  # 答案
        "id": "",  # ID 占位符
        "likeFlag": False,  # 默认值
        "markFlag": False,  # 默认值
        "abbreviationSubject": "CCPH"  # 缩写主题占位符
    }
    formatted_single_choice.append(formatted_item)

# 处理多选题
for question, details in raw_data.get("multiple_choice_questions", {}).items():
    formatted_item = {
        "questionStem": question,  # 题干
        "option": [option[1:] for option in details["options"]],  # 选项
        "answer": details["answer"],  # 答案
        "id": "",  # ID 占位符
        "likeFlag": False,  # 默认值
        "markFlag": False,  # 默认值
        "abbreviationSubject": "CCPH"  # 缩写主题占位符
    }
    formatted_multiple_choice.append(formatted_item)

# 处理判断题
for question, details in raw_data.get("true_false_questions", {}).items():
    formatted_item = {
        "questionStem": question,  # 题干
        "option": ["正确", "错误"],  # 判断题选项固定为 True 和 False
        "answer": details["answer"],  # 答案
        "id": "",  # ID 占位符
        "likeFlag": False,  # 默认值
        "markFlag": False,  # 默认值
        "abbreviationSubject": "CCPH"  # 缩写主题占位符
    }
    formatted_true_false.append(formatted_item)

# 处理填空题
for question, details in raw_data.get("fill_in_the_blanks_questions", {}).items():
    formatted_item = {
        "questionStem": question,  # 题干
        "option": "",  # 填空题没有选项
        "answer": details["answer"],  # 答案
        "id": "",  # ID 占位符
        "likeFlag": False,  # 默认值
        "markFlag": False,  # 默认值
        "abbreviationSubject": "CCPH"  # 缩写主题占位符
    }
    formatted_fill_in_the_blanks.append(formatted_item)

# 将各题型的格式化数据写入各自的JSON文件
with open('CCPH_singleChoice.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_single_choice, f, ensure_ascii=False, indent=4)

with open('CCPH_multipleChoice.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_multiple_choice, f, ensure_ascii=False, indent=4)

with open('CCPH_rightWrong.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_true_false, f, ensure_ascii=False, indent=4)

with open('CCPH_fillingBlank.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_fill_in_the_blanks, f, ensure_ascii=False, indent=4)
