const fs = require('fs');

let paramsArr = [
    {
        name: 'XiIntro',
        params: {
            branchId: "1705139277953761280",
            chapterId: "",
            studentId: "1741125438396170311",
            subjectId: "1752935841845477392",
        },
    },
    {
        name: 'CMH',
        params: {
            branchId: "1705139277953761280",
            chapterId: "",
            studentId: "1741125438396170311",
            subjectId: "1748168736914800640",
        },
    },
    {
        name: 'Marx',
        params: {
            branchId: "1705139277953761280",
            chapterId: "",
            studentId: "1741125438396170311",
            subjectId: "1748167277460586496",
        },
    },
    {
        name: 'MaoIntro',
        params: {
            branchId: "1705139277953761280",
            chapterId: "",
            studentId: "1741125438396170311",
            subjectId: "1748168736914800651",
        },
    },
    {
        name: 'Political',
        params: {
            branchId: "1705139277953761280",
            chapterId: "",
            studentId: "1741125438396170311",
            subjectId: "1781216923707506688",
        },
    },
    // {
    //     name: 'NCH',
    //     params: {
    //         branchId: "1705139277953761280",
    //         chapterId: "",
    //         studentId: "1741125438396170311",
    //         subjectId: "1776854236110258176",
    //     },
    // },
    // {
    //     name: 'SDH',
    //     params: {
    //         branchId: "1705139277953761280",
    //         chapterId: "",
    //         studentId: "1741125438396170311",
    //         subjectId: "1752935841845477376",
    //     },
    // },
    // {
    //     name: 'ORH',
    //     params: {
    //         branchId: "1705139277953761280",
    //         chapterId: "",
    //         studentId: "1741125438396170311",
    //         subjectId: "1752935841845477384",
    //     },
    // }
]

function fetchData(item, callback) {
    let subjectName = item.name;

    fs.readFile(__dirname + `/new/${subjectName}.json`, 'utf8', async (err, data) => {
        if (err) {
            console.error(err);
            callback(err);
        } else {
            let arr = JSON.parse(data); 
            const seenIds = new Set();
            let uniqueArr = arr.filter(item => {
                if (seenIds.has(item.id)) {
                    return false;
                } else {
                    seenIds.add(item.id);
                    return true;
                }
            });

            callback(null, subjectName, uniqueArr);
        }
    });
}

function writeFile(subjectName, jsonArray) {
    fs.writeFile(__dirname + `/new/solved/${subjectName}Solved.json`, JSON.stringify(jsonArray), (err) => {
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

paramsArr.forEach(item => {
    fetchAndWrite(item);
})
