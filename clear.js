const fs = require('fs');

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

let subject = paramsArr[i]

function fetchData(params) {
    fs.readFile(__dirname + '/习概.json', 'utf8', async (err, data) => {
        if (err) {
            console.error(err);
        } else {
            let arr = JSON.parse(data)

            // 定义一个 Set 用于存储唯一的对象
            let uniqueSet = new Set();

            // 使用 Array.filter 进行去重
            let uniqueArr = await arr.filter(obj => {
                // 将对象转换为 JSON 字符串作为唯一标识
                let objString = JSON.stringify(obj);

                // 如果 Set 中不存在相同的 JSON 字符串，则添加到 Set 中并返回 true
                if (!uniqueSet.has(objString)) {
                    uniqueSet.add(objString);
                    return true;
                }

                // 如果 Set 中已存在相同的 JSON 字符串，返回 false
                return false;
            });

            console.log(uniqueArr);

            fs.writeFile(__dirname + `/${subject}Solved.json`, JSON.stringify(uniqueArr), (err) => {
                if (err) {
                    console.log(err);
                } else {
                    console.log('success');
                }
            });
        }
    })
}



function fetchAndWrite(params) {
    fetchData(params => {
        if (error) {
            console.error(`请求失败`, error);
        } else {
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
            writeFile(subjectName, jsonArray);
        }
    });
}

paramsArr.forEach(params => {
    fetchAndWrite(params);
});