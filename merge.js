const fs = require('fs');
const path = require('path');

// 读取 JSON 文件
const readJsonFile = (filePath) => {
  if (!fs.existsSync(filePath)) return [];
  const data = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(data);
};

// 写入 JSON 文件
const writeJsonFile = (filePath, data) => {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
};

// 去重、合并和删除不在新的中的旧数据
const mergeJsonData = (oldData, newData) => {
  // 删除旧数据中不在新数据中的项
  const updatedOldData = oldData.filter(item =>
    newData.some(newItem => JSON.stringify(newItem) === JSON.stringify(item))
  );

  // 将新数据中不在旧数据中的添加
  const nonDuplicateNewData = newData.filter(item =>
    !updatedOldData.some(oldItem => JSON.stringify(oldItem) === JSON.stringify(item))
  );

  // 合并结果
  return [...updatedOldData, ...nonDuplicateNewData];
};

// 批量处理文件夹中的文件
const processFolders = (oldDir, newDir, outputDir) => {
  if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

  const oldFiles = fs.readdirSync(oldDir);
  const newFiles = fs.readdirSync(newDir);

  oldFiles.forEach((file) => {
    const oldFilePath = path.join(oldDir, file);
    const newFilePath = path.join(newDir, file);
    const outputFilePath = path.join(outputDir, file);

    // 确保新文件夹中也有对应文件
    if (newFiles.includes(file)) {
      const oldData = readJsonFile(oldFilePath);
      const newData = readJsonFile(newFilePath);

      // 合并数据
      const mergedData = mergeJsonData(oldData, newData);

      // 写入合并后的数据
      writeJsonFile(outputFilePath, mergedData);
      console.log(`已处理文件：${file}`);
    } else {
      console.log(`跳过文件：${file}（新文件夹中不存在）`);
    }
  });

  console.log('所有文件处理完成！');
};

// 使用示例
const oldFolderPath = './new2/rewrite'; // 旧文件所在文件夹路径
const newFolderPath = './new/rewrite'; // 新文件所在文件夹路径
const outputFolderPath = './merge'; // 输出文件夹路径

processFolders(oldFolderPath, newFolderPath, outputFolderPath);
