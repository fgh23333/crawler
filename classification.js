const fs = require('fs');
// let lesson = '思政'
let lesson = '习概'

fs.readFile(__dirname + `/${lesson}_rewrite.json`, (err, data) => {
    if (err) {
        console.error(err)
    } else {
        data = JSON.parse(data)
        let content1 = []
        let content2 = []

        for (let i = 0; i < data.length; i++) {
            let temp = {
                questionStem: '',
                answer: null
            }
            if (data[i].answer == '正确') {
                temp.questionStem = data[i].questionStem
                temp.answer = data[i].answer
                content1.push(temp)
            } else if (data[i].answer == '错误') {
                temp.questionStem = data[i].questionStem
                temp.answer = data[i].answer
                content2.push(temp)
            }
        }
        fs.writeFile(__dirname + `/${lesson}_right.json`, JSON.stringify(content1), (err) => {
            if (err) {
                console.error(err)
            } else {
                console.log('success1')
            }
        })
        fs.writeFile(__dirname + `/${lesson}_wrong.json`, JSON.stringify(content2), (err) => {
            if (err) {
                console.error(err)
            } else {
                console.log('success2')
            }
        })
    }
})