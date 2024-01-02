const fs = require('fs');

// 定义一个数组用于存储所有 JSON 文件的数据
let combinedData = [];

// 循环读取七个 JSON 文件
for (let i = 0; i < 7; i++) {
  const fileName = `introductionSolved${i}.json`;
  const fileContent = fs.readFileSync(fileName, 'utf8');
  const jsonData = JSON.parse(fileContent);
  
  // 将当前文件的数据合并到数组中
  combinedData = combinedData.concat(jsonData); // 或者使用 [...combinedData, ...jsonData];
}

// 将合并后的数据写入新的 JSON 文件
const combinedJson = JSON.stringify(combinedData, null, 2);
fs.writeFileSync('introductionSolved.json', combinedJson, 'utf8');

console.log('success');
