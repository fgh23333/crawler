const fs = require('fs');
const path = require('path');

// 定义分类规则的示例函数
const classifyObject = (obj) => {
    if (obj.option.length === 2) {
        obj.type = 'rightWrong';
    } else if (obj.option.length === 0) {
        obj.type = 'fillingBlank';
    } else if (obj.answer.length === 1) {
        obj.type = 'singleChoice';
    } else {
        obj.type = 'multipleChoice';
    }
    return obj;
};

// 读取指定目录中的所有 JSON 文件
const readJsonFilesFromDirectory = (directoryPath) => {
    return fs.promises.readdir(directoryPath)
        .then(files => {
            return files.filter(file => file.endsWith('.json'));
        });
};

// 读取并处理 JSON 文件
const processJsonFiles = async (directoryPath) => {
    const jsonFiles = await readJsonFilesFromDirectory(directoryPath);

    for (const file of jsonFiles) {
        const filePath = path.join(directoryPath, file);
        try {
            const fileContent = await fs.promises.readFile(filePath, 'utf-8');
            const jsonObjects = JSON.parse(fileContent);

            // 遍历每个对象并根据分类规则添加属性
            const updatedObjects = jsonObjects.map(classifyObject);

            // 输出处理后的对象（或保存到文件等）
            // console.log(`Processed ${file}:`, updatedObjects);

            // 如果需要保存修改后的文件，可以使用下面的代码
            await fs.promises.writeFile(filePath, JSON.stringify(updatedObjects, null, 2), 'utf-8');
        } catch (err) {
            console.error(`Error reading or processing file ${file}:`, err);
        }
    }
};

// 调用函数，传入目标目录路径
const directoryPath = (__dirname + "/merge"); // 替换为你的目录路径
processJsonFiles(directoryPath);
