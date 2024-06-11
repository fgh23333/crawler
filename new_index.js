const axios = require('axios');
const fs = require('fs');

const baseUrl = 'http://222.73.57.149:6571';

const targetPath = '/examinationInfo/getPracticeInfo';

let paramsArr = [
    // {
    //     name: 'XiIntro',
    //     params: {
    //         branchId: "1705139277953761280",
    //         chapterId: "",
    //         studentId: "1795023275139531171",
    //         subjectId: "1752935841845477392",
    //     },
    // },
    // {
    //     name: 'Marx',
    //     params: {
    //         branchId: "1705139277953761280",
    //         chapterId: "",
    //         studentId: "1795023275139531171",
    //         subjectId: "1748167277460586496",
    //     },
    // },
    // {
    //     name: 'MaoIntro',
    //     params: {
    //         branchId: "1705139277953761280",
    //         chapterId: "",
    //         studentId: "1795023275139531171",
    //         subjectId: "1748168736914800651",
    //     },
    // },
    // {
    //     name: 'Political',
    //     params: {
    //         branchId: "1705139277953761280",
    //         chapterId: "",
    //         studentId: "1795023275139531171",
    //         subjectId: "1781216923707506688",
    //     },
    // },
    // {
    //     name: 'CMH',
    //     params: {
    //         branchId: "1705139277953761280",
    //         chapterId: "",
    //         studentId: "1795023275139531171",
    //         subjectId: "1748168736914800640",
    //     },
    // },
    {
        name: 'NCH',
        params: {
            branchId: "1705139277953761280",
            chapterId: "",
            studentId: "1798906253557104911",
            subjectId: "1776854236110258176",
        },
    },
    {
        name: 'SDH',
        params: {
            branchId: "1705139277953761280",
            chapterId: "",
            studentId: "1798906253557104911",
            subjectId: "1752935841845477376",
        },
    },
    {
        name: 'ORH',
        params: {
            branchId: "1705139277953761280",
            chapterId: "",
            studentId: "1798906253557104911",
            subjectId: "1752935841845477384",
        },
    },
    {
        name: 'CCPH',
        params: {
            branchId: "1705139277953761280",
            chapterId: "",
            studentId: "1798906253557104911",
            subjectId: "1798740810791911424",
        },
    }
]

const headers = {
    'Content-Type': 'application/json;charset=utf-8'
}

function fetchData(params, callback) {
    let jsonArray = [];

    function makeRequest(count) {
        axios.post(baseUrl + targetPath, JSON.stringify(params), { headers })
            .then(res => {
                let data = JSON.parse(res.data.data.paperStore.paperContent)
                let singleChoice = data.danxuan.children
                let multipleChoice = data.duoxuan.children
                let rightWrong = data.panduan.children
                let fillingBlank = data.tiankong.children
                jsonArray = jsonArray.concat(rightWrong, singleChoice, multipleChoice, fillingBlank)
                count++;
                if (count < 300) {
                    if (count % 50 == 0) {
                        console.log(count);
                    }
                    makeRequest(count);
                } else {
                    callback(null, jsonArray);
                }
            })
            .catch(error => {
                callback(error);
            });
    }

    makeRequest(0);
}

function writeFile(subjectName, jsonArray) {
    fs.writeFile(__dirname + `/new2/${subjectName}.json`, JSON.stringify(jsonArray), (err) => {
        if (err) {
            console.log(err);
        } else {
            console.log('success');
        }
    });
}

function fetchAndWrite(item) {
    fetchData(item.params, (error, jsonArray) => {
        if (error) {
            console.error(`请求失败`, error);
        } else {
            let subjectName = item.name;
            writeFile(subjectName, jsonArray);
        }
    });
}

paramsArr.forEach(item => {
    fetchAndWrite(item);
})