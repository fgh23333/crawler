const axios = require('axios');
const { JSDOM } = require('jsdom');
const fs = require('fs');

// 目标网站的基础URL
const baseUrl = 'http://222.73.57.150:6571';

// 目标路径
const targetPath = '/Exercise/StartExerciseAll.aspx';

// 定义要发送的 Cookie
const cookies = {
    'ASP.NET_SessionId': '2hmempvrk03j2nv0n1r3krhw',
};

// 构造请求头，将 Cookie 添加到其中
const headers = {
    'Cookie': Object.entries(cookies).map(([name, value]) => `${name}=${value}`).join('; '),
};

// 构建 JSON 数组
const jsonArray = [];
let subject = '习近平新时代中国特色社会主义思想概论+全部章节'
// let subject = '思想道德与法治+全部章节'

// 循环发送多个请求，每个请求使用不同的 SubjectID
for (let i = 0; i < 100; i++) {
    // 定义要发送的参数
    const params = {
        SubjectID: '56',
        LoreID: '',
        SubjectName: encodeURIComponent(subject),

    };

    // 发送带有参数和 Cookie 的 GET 请求
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

            // 使用 Promise.all 确保所有异步操作完成后再继续
            await Promise.all(Array.from(dsElements).map(async (dsElement) => {
                const jsonResult = {
                    TestContent: dsElement.querySelector('TestContent').textContent,
                    OptionContent: dsElement.querySelector('OptionContent').textContent,
                    StandardAnswer: dsElement.querySelector('StandardAnswer').textContent,
                    RubricID: dsElement.querySelector('RubricID').textContent
                };

                jsonArray.push(jsonResult);
            }));
        })
        .catch((error) => {
            console.error(`请求失败`, error);
        });
}

setTimeout(() => {
    let subjectName = ''
    if (subject == '习近平新时代中国特色社会主义思想概论+全部章节') {
        subjectName = '习概'
    } else {
        subjectName = '思政'
    }
    fs.writeFile(__dirname + `/${subjectName}.json`, JSON.stringify(jsonArray), (err) => {
        if (err) {
            console.log(err);
        } else {
            console.log('success');
        }
    });
}, 80000);
