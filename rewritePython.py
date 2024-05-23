import ijson
import json
import os
outputPath = "new/rewrite/"
solvedPath = "new/solved/"
result = []
mc = []
sc = []
rw =[]
fb = []

def write(thename,result,name):
    json_data = json.dumps(result, indent=4, ensure_ascii=False)

    # 写入JSON数据到文件
    with open(outputPath + thename+'_'+name+'.json', 'w', encoding='utf-8') as f:
        f.write(json_data)

def rewrite(fileName,filePath):
    thename = fileName[:len(fileName)-11]
    with open(filePath, 'r', encoding='utf-8') as fff:
        objects = ijson.items(fff, 'item')
        while True:
            try:
                a = objects.__next__()
                if a["standardAnswer"] == "正确" or a["standardAnswer"] == "错误":
                    temp = {
                        "questionStem": a["title"],
                        "option": ["正确", "错误"],
                        "answer": a["standardAnswer"]
                    }
                    rw.append(temp)
                else:
                    if ord(a["standardAnswer"][0:1]) >= 65 and ord(a["standardAnswer"][0:1]) <= 90:
                        temp = {
                            "questionStem": a["title"],
                            "option": a["options"].split("|"),
                            "answer": a["standardAnswer"]
                        }
                        if len(a["standardAnswer"])==1 and len(temp["option"])==4:
                            mc.append(temp)
                        else:
                            sc.append(temp)
                    else:
                        temp = {
                            "questionStem": a["title"],
                            "option": "",
                            "answer": a["standardAnswer"]
                        }
                        fb.append(temp)
                result.append(temp)
            except StopIteration as e:
                print("数据读取完成")
                break
    write(thename,result,"subject")
    write(thename,mc,"multipleChoice")
    write(thename,sc,"singleChoice")
    write(thename,rw,"rightWrong")
    write(thename,fb,"fillingBlank")



for dirpath, dirnames, filenames in os.walk(solvedPath):
    for filename in filenames:
        result = []
        rewrite(filename,os.path.join(dirpath, filename))

