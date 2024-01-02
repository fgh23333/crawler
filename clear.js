const fs = require('fs');

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

        fs.writeFile(__dirname + `/introductionSolved0.json`, JSON.stringify(uniqueArr), (err) => {
            if (err) {
                console.log(err);
            } else {
                console.log('success');
            }
        });
    }
})

