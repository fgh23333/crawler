const fs = require("fs")
// let lesson = 'politicalSolved'
let lesson = 'introductionSolved'

fs.readFile(__dirname + `/${lesson}.json`, (err, data) => {
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
        fs.writeFile(__dirname + `/${lesson}_rewrite.json`, JSON.stringify(content), (err) => {
            if (err) {
                console.error(err)
            } else {
                console.log('success')
            }
        })
    }
})