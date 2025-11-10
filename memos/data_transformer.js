const fs = require('fs');
const path = require('path');

// 学科名称映射
const subjectMapping = {
    'XiIntro': '习近平新时代中国特色社会主义思想概论',
    'Marx': '马克思主义基本原理',
    'MaoIntro': '毛泽东思想和中国特色社会主义理论体系概论',
    'Political': '思想道德与法治',
    'CMH': '中国近现代史纲要',
    'NCH': '新中国史',
    'SDH': '社会主义发展史',
    'ORH': '改革开放史',
    'CCPH': '中共党史'
};

// 转换函数：将rewrite目录下的分类数据转换为目标格式
function transformCategorizedQuestion(question, subject, questionType, index) {
    const subjectName = subjectMapping[subject] || subjectMapping['Marx'];

    // 处理问题文本
    const questionText = question.questionStem || '';

    // 处理选项
    let options = [];
    if (question.option && Array.isArray(question.option)) {
        question.option.forEach((opt, i) => {
            if (opt && opt.trim()) {
                options.push({
                    letter: String.fromCharCode(65 + i), // A, B, C, D...
                    content: opt.trim()
                });
            }
        });
    } else {
        // 如果没有选项，创建默认选项
        options = [
            { letter: "A", content: "选项A" },
            { letter: "B", content: "选项B" },
            { letter: "C", content: "选项C" },
            { letter: "D", content: "选项D" }
        ];
    }

    // 处理答案标签
    let label = question.answer || 'A';

    // 处理判断题答案
    if (label === '正确') label = 'A';
    if (label === '错误') label = 'B';

    // 处理多选题答案（如"ABCD"）
    if (typeof label === 'string' && label.length > 1) {
        // 对于多选题，保持原格式但确保是有效的字母组合
        label = label.toUpperCase().replace(/[^A-J]/g, '');
    }

    // 确保答案是有效的
    if (!label || !/^[A-J]+$/.test(label)) {
        label = 'A';
    }

    // 构建完整的问题文本（包含选项）
    let fullQuestion = questionText;
    if (options.length > 0) {
        const optionsText = options.map(opt => `(${opt.letter}) ${opt.content}`).join(' ');
        fullQuestion = `${questionText}\nAnswer Choices: ${optionsText}`;
    }

    return {
        id: `Text-${index}`,
        question: fullQuestion,
        options: options,
        label: [label],
        subject_name: subjectName,
        question_type: questionType
    };
}

// 主处理函数
function processData() {
    const rewriteDir = path.join(__dirname, '2025-05-27', 'rewrite');
    const outputFile = path.join(__dirname, 'transformed_political_data.json');
    const results = [];
    let questionIndex = 0;

    // 要处理的科目列表（按照您提供的顺序）
    const subjects = ['XiIntro', 'Marx', 'MaoIntro', 'Political', 'CMH', 'NCH', 'SDH', 'ORH', 'CCPH'];

    // 题目类型列表（subject是学科总题目，其他是具体题型）
    const questionTypes = ['singleChoice', 'multipleChoice', 'rightWrong', 'fillingBlank', 'subject'];

    console.log('开始转换分类数据...\n');

    subjects.forEach(subject => {
        questionTypes.forEach(questionType => {
            const filePath = path.join(rewriteDir, `${subject}_${questionType}.json`);

            if (fs.existsSync(filePath)) {
                try {
                    const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
                    console.log(`正在处理 ${subject}_${questionType}: ${data.length} 道题目`);

                    data.forEach(question => {
                        if (question.questionStem && question.questionStem.trim()) {
                            const transformed = transformCategorizedQuestion(question, subject, questionType, questionIndex++);
                            results.push(transformed);
                        }
                    });

                    console.log(`✓ ${subject}_${questionType} 处理完成`);
                } catch (error) {
                    console.error(`处理 ${subject}_${questionType} 时出错:`, error.message);
                }
            } else {
                // console.log(`文件未找到: ${filePath}`);
            }
        });
        console.log('');
    });

    // 写入转换结果
    fs.writeFileSync(outputFile, JSON.stringify(results, null, 2));

    console.log('\n=== 转换完成! ===');
    console.log(`总题目数量: ${results.length}`);
    console.log(`输出文件: ${outputFile}`);

    // 显示统计信息
    const stats = {};
    results.forEach(q => {
        const key = q.subject_name;
        stats[key] = (stats[key] || 0) + 1;
    });

    console.log('\n按学科分类统计:');
    Object.entries(stats).forEach(([subject, count]) => {
        console.log(`${subject}: ${count} 道题目`);
    });

    // 显示样例
    if (results.length > 0) {
        console.log('\n样例数据:');
        console.log(JSON.stringify(results[0], null, 2));
    }
}

// 如果直接运行此脚本
if (require.main === module) {
    processData();
}

module.exports = { transformCategorizedQuestion, processData };