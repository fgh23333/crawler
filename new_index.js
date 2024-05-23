const axios = require('axios');

const baseUrl = 'http://222.73.57.153:6571';

const targetPath = '/examinationInfo/getPracticeInfo';

const params = {
    branchId: "1705139277953761280",
    chapterId: "",
    studentId: "1741125438396170311",
    subjectId: "1752935841845477392",
}

axios.post(baseUrl + targetPath, JSON.stringify(params), {
    headers: {
        'Content-Type': 'application/json;charset=utf-8'
    }
})
    .then(res => {
        let data = JSON.parse(res.data.data.paperStore.paperContent)
        let answer = JSON.parse(res.data.paperStore.paperAnswer);
        console.log(data);
        let singleChoice = data.danxuan.children
    })
