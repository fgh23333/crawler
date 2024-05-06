const fs = require("fs")

let paramsArr = [
    {
        SubjectID: 53,
        LoreID: '',
        SubjectName: encodeURIComponent('马克思主义基本原理+全部章节'),
    },
    {
        SubjectID: 55,
        LoreID: '',
        SubjectName: encodeURIComponent('中国近现代史纲要+全部章节'),
    },
    {
        SubjectID: 56,
        LoreID: '',
        SubjectName: encodeURIComponent('思想道德与法治+全部章节'),
    },
    {
        SubjectID: 57,
        LoreID: '',
        SubjectName: encodeURIComponent('毛泽东思想和中国特色社会主义理论体系概论+全部章节'),
    },
    {
        SubjectID: 60,
        LoreID: '',
        SubjectName: encodeURIComponent('习近平新时代中国特色社会主义思想概论+全部章节'),
    },
    {
        SubjectID: 61,
        LoreID: '',
        SubjectName: encodeURIComponent('社会主义发展史+全部章节'),
    },
    {
        SubjectID: 62,
        LoreID: '',
        SubjectName: encodeURIComponent('新中国史+全部章节'),
    },
    {
        SubjectID: 63,
        LoreID: '',
        SubjectName: encodeURIComponent('党史+全部章节'),
    },
    {
        SubjectID: 65,
        LoreID: '',
        SubjectName: encodeURIComponent('改革开放史+全部章节'),
    },
];

function fetchData(params, callback) {
    let subjectName = '';
    switch (params.SubjectID) {
        case 53:
            subjectName = '马原';
            break;
        case 55:
            subjectName = '近代史';
            break;
        case 56:
            subjectName = '思政';
            break;
        case 57:
            subjectName = '毛概';
            break;
        case 60:
            subjectName = '习概';
            break;
        case 61:
            subjectName = '发展史';
            break;
        case 62:
            subjectName = '新中国史';
            break;
        case 63:
            subjectName = '党史';
            break;
        case 65:
            subjectName = '开放史';
            break;
        default:
            break;
    }

    fs.readFile(__dirname + `/${subjectName}.json`, 'utf8', async (err, data) => {
        if (err) {
            console.error(err)
        } else {
            data = JSON.parse(data)
            let content = []

            for (let i = 0; i < data.length; i++) {
                let temp = {
                    questionStem: '',
                    option: [],
                    answer: null
                }
                temp.questionStem = data[i].TestContent
                if (data[i].StandardAnswer != '正确' && data[i].StandardAnswer != '错误') {
                    let optionsArray = data[i].OptionContent.split('|')
                    temp.option = optionsArray
                    temp.answer = data[i].StandardAnswer.split("")
                } else {
                    temp.option = ['正确', '错误']
                    temp.answer = data[i].StandardAnswer
                }
                content.push(temp)
            }

            // 调用回调函数，传递处理后的数据
            callback(null, subjectName, content);
        }
    });
}

function writeFile(subjectName, jsonArray) {
    fs.writeFile(__dirname + `/${subjectName}Rewrite.json`, JSON.stringify(jsonArray), (err) => {
        if (err) {
            console.log(err);
        } else {
            console.log('成功写入文件');
        }
    });
}

function fetchAndWrite(params) {
    fetchData(params, (err, subjectName, jsonArray) => {
        if (err) {
            console.error(`请求失败`, err);
        } else {
            writeFile(subjectName, jsonArray);
        }
    });
}

paramsArr.forEach(params => {
    fetchAndWrite(params);
});
