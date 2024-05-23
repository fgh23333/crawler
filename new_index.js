const axios = require('axios');

const baseUrl = 'http://222.73.57.153:6571';

const targetPath = '/examinationInfo/getPracticeInfo';

const data = {
    branchId: "1705139277953761280",
    chapterId: "",
    studentId: "1741125438396170311",
    subjectId: "1752935841845477392",
}

axios.post(baseUrl + targetPath, JSON.stringify(data), {
    headers: {
        'Content-Type': 'application/json;charset=utf-8'
    }
})
    .then(res => {
        console.log(res.data);
    })
