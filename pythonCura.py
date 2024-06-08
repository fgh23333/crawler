import json
rewritePath = "new2/rewrite/"
curaPath = "new2/cura/"

#name = ['CMH', 'MaoIntro', 'Marx', 'NCH', 'ORH', 'Political', 'XiIntro','CCPH','SDH']
name = ['NCH','ORH','CCPH','SDH']
shao = ['NCH','ORH','CCPH','SDH']
big = [20,15,15,10]
small = [15,15,15,5]

result = []
mc = []
sc = []
rw =[]
fb = []

def write(thename, result, index):
    json_data = json.dumps(result, indent=4, ensure_ascii=False)

    # 写入JSON数据到文件
    with open(curaPath + thename + '_'+ index +'.json', 'w', encoding='utf-8') as f:
        f.write(json_data)
def getmin(numlist):
    return min(numlist)
def cura(singleChoice, multipleChoice, rightWrong, fillingBlank,thename,count):
    nl = []
    fb = False
    nl.append(len(singleChoice)//count[0])
    nl.append(len(multipleChoice)//count[1])
    nl.append(len(rightWrong)//count[2])
    if len(fillingBlank) != 0:
        fb = True
        again = False
        fblift = len(fillingBlank)
        se = [0,count[3]]
    c = getmin(nl)
    print(thename, count, c)

    for i in range(0,c):
        output = []
        output = singleChoice[i*count[0]:(i+1)*count[0]]
        output.extend(multipleChoice[i*count[1]:(i+1)*count[1]])
        output.extend(rightWrong[i*count[2]:(i+1)*count[2]])
        if fb:
            if fblift >= count[3]:
                output.extend(fillingBlank[se[0]:se[1]])
                se[0] += count[3]
                se[1] += count[3]
                fblift -= count[3]
            else:
                output.extend(fillingBlank[se[0]:])
                se[0] = 0
                se[1] = count[3]-fblift
                fblift = len(fillingBlank) - se[1]
                output.extend(fillingBlank[se[0]:se[1]])
                again = True
                print(thename)
        write(thename,output,str(i+1))
    output = []
    output = singleChoice[c * count[0]:]
    output.extend(multipleChoice[c * count[1]:])
    output.extend(rightWrong[c * count[2]:])
    if fb and (not again):
        output.extend(fillingBlank[c * count[3]:])
    write(thename, output, "residual")

def read(thename):
    with open(rewritePath+thename+'_fillingBlank.json', 'r',encoding='utf-8') as file:
        fillingBlank = json.load(file)
    with open(rewritePath+thename+'_multipleChoice.json', 'r',encoding='utf-8') as file:
        multiChoice = json.load(file)
    with open(rewritePath+thename+'_rightWrong.json', 'r',encoding='utf-8') as file:
        rightWrong = json.load(file)
    with open(rewritePath+thename+'_singleChoice.json', 'r',encoding='utf-8') as file:
        singleChoice = json.load(file)
    if thename not in shao:
        cura(singleChoice, multiChoice, rightWrong, fillingBlank,thename,big)
    else:
        cura(singleChoice, multiChoice, rightWrong, fillingBlank,thename,small)

for n in name:
    read(n)
