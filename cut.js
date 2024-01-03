const fs = require('fs').promises;

async function readQuestionsFromFile() {
  try {
    // 使用async/await读取JSON文件
    const jsonString = await fs.readFile(__dirname + '/politicalSolved_rewrite.json', 'utf8');
    return JSON.parse(jsonString);
  } catch (error) {
    console.error('读取文件出错:', error);
    throw error;
  }
}

async function writeQuestionsToJsonFile(questions, fileName) {
  try {
    // 使用async/await写入JSON文件
    const jsonContent = JSON.stringify(questions, null, 2);
    await fs.writeFile(fileName, jsonContent, 'utf8');
    console.log(`写入文件: ${fileName}`);
  } catch (error) {
    console.error('写入文件出错:', error);
    throw error;
  }
}

async function splitAndWriteQuestions() {
  const allQuestions = await readQuestionsFromFile();
  const batchSize = 50;

  for (let i = 0; i < allQuestions.length; i += batchSize) {
    const batch = allQuestions.slice(i, i + batchSize);
    const fileName = `political_${i / batchSize}.json`;

    await writeQuestionsToJsonFile(batch, fileName);
  }
}

// 调用函数
splitAndWriteQuestions();
