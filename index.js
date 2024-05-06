const axios = require('axios');
const { JSDOM } = require('jsdom');
const fs = require('fs');

// 目标网站的基础URL
const baseUrl = 'http://222.73.57.150:6571';

// 目标路径
const targetPath = '/Exercise/StartExerciseAll.aspx';

// 定义要发送的 Cookie
const cookies = {
    'ASP.NET_SessionId': '0loxz4xygilrhm14fxoc424k',
};

// 构造请求头，将 Cookie 添加到其中
const headers = {
    'Cookie': Object.entries(cookies).map(([name, value]) => `${name}=${value}`).join('; '),
};

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
    let jsonArray = [];

    function makeRequest(count) {
        axios.get(baseUrl + targetPath, { params, headers })
            .then(async (response) => {
                // 处理响应数据
                const html = response.data;

                // 使用 jsdom 模拟浏览器环境
                const dom = new JSDOM(html);
                const document = dom.window.document;

                // 在虚拟 DOM 中选择标签并处理
                const form = document.querySelector('body form');
                const xmlData = form.querySelectorAll('input')[5].value;

                // 提取 ds 标签数据
                const xmlDoc = new dom.window.DOMParser().parseFromString(xmlData, 'text/xml');
                const dsElements = xmlDoc.querySelectorAll('ds');

                // 处理每个 ds 元素
                Array.from(dsElements).forEach(dsElement => {
                    jsonArray.push({
                        TestContent: dsElement.querySelector('TestContent').textContent,
                        OptionContent: dsElement.querySelector('OptionContent').textContent,
                        StandardAnswer: dsElement.querySelector('StandardAnswer').textContent,
                        RubricID: dsElement.querySelector('RubricID').textContent
                    });
                });

                count++;
                if (count < 100) {
                    makeRequest(count); // 继续发起请求
                } else {
                    callback(null, jsonArray); // 请求完成，调用回调函数
                }
            })
            .catch(error => {
                callback(error); // 请求失败，调用回调函数并传递错误信息
            });
    }

    makeRequest(0); // 初始请求
}

function writeFile(subjectName, jsonArray) {
    fs.writeFile(__dirname + `/${subjectName}.json`, JSON.stringify(jsonArray), (err) => {
        if (err) {
            console.log(err);
        } else {
            console.log('success');
        }
    });
}

function fetchAndWrite(params) {
    fetchData(params, (error, jsonArray) => {
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
