import ijson
import json
import os
outputPath = "new3/rewrite/"
solvedPath = "new3/solved/"

def write(thename, result, name):
    json_data = json.dumps(result, indent=4, ensure_ascii=False)

    # 写入JSON数据到文件
    with open(outputPath + thename + '_'+ name +'.json', 'w', encoding='utf-8') as f:
        f.write(json_data)

def rewrite(fileName,filePath):
    thename = fileName[:len(fileName)-11]
    with open(filePath, 'r', encoding='utf-8') as fff:
        objects = ijson.items(fff, 'item')
        while True:
            try:
                a = objects.__next__()
                nn = a["standardAnswer"].replace(" ","")
                nnt = a["title"].replace(" ", "")
                nnid = a["id"].replace(" ", "")
                if nn == "正确" or nn == "错误":
                    temp = {
                        "questionStem": nnt,
                        "option": ["正确", "错误"],
                        "answer": nn,
                        "id": nnid,
                        "likeFlag": False,
                        "markFlag": False,
                        "abbreviationSubject": thename
                    }
                    rw.append(temp)
                else:
                    if ord(nn[0:1]) >= 65 and ord(nn[0:1]) <= 90:
                        temp = {
                            "questionStem": nnt,
                            "option": a["options"].replace(" ","").replace(".", "").split("|"),
                            "answer": nn,
                            "id": nnid,
                            "likeFlag": False,
                            "markFlag": False,
                            "abbreviationSubject": thename
                        }
                        if len(a["standardAnswer"]) == 1 and len(temp["option"]) == 4:
                            sc.append(temp)
                        else:
                            mc.append(temp)
                    else:
                        temp = {
                            "questionStem": nnt,
                            "option": "",
                            "answer": nn.replace("|","，"),
                            "id": nnid,
                            "likeFlag": False,
                            "markFlag": False,
                            "abbreviationSubject": thename
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
        mc = []
        sc = []
        rw =[]
        fb = []
        rewrite(filename,os.path.join(dirpath, filename))
