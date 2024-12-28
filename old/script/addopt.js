const fs = require('fs')
// let lesson = 'political'
let lesson = 'introduction'

fs.readFile(__dirname + `/${lesson}_right.json`, (err, data) => {
    if (err) {
        console.error(err);
    } else {
        data = JSON.parse(data)
        for (let i = 0; i < data.length; i++) {    
            data[i].option = ['正确', '错误']
        }
        fs.writeFile(__dirname + `/${lesson}_topt.json`, JSON.stringify(data), (err) => {
            if (err) {
                console.error(err)
            } else {
                console.log('success')
            }
        })
    }
})

fs.readFile(__dirname + `/${lesson}_wrong.json`, (err, data) => {
    if (err) {
        console.error(err);
    } else {
        data = JSON.parse(data)
        for (let i = 0; i < data.length; i++) {    
            data[i].option = ['正确', '错误']
        }
        fs.writeFile(__dirname + `/${lesson}_fopt.json`, JSON.stringify(data), (err) => {
            if (err) {
                console.error(err)
            } else {
                console.log('success')
            }
        })
    }
})
