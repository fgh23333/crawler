const axios = require('axios');
const fs = require('fs');
const path = require('path');

// ======================== 配置 ========================

const BASE_URL = 'http://222.73.57.153:6571';
const TARGET_PATH = '/examinationInfo/getPracticeInfo';
const FETCH_COUNT = 1000; // 每个科目请求次数

const SUBJECTS = [
    { name: 'XiIntro',  branchId: '1705139277953761280', subjectId: '1752935841845477392' },
    { name: 'Marx',     branchId: '1705139277953761280', subjectId: '1748167277460586496' },
    { name: 'MaoIntro', branchId: '1705139277953761280', subjectId: '1748168736914800651' },
    { name: 'Political', branchId: '1705139277953761280', subjectId: '1781216923707506688' },
    { name: 'CMH',      branchId: '1705139277953761280', subjectId: '1748168736914800640' },
    { name: 'NCH',      branchId: '1705139277953761280', subjectId: '1776854236110258176' },
    { name: 'SDH',      branchId: '1705139277953761280', subjectId: '1752935841845477376' },
    { name: 'ORH',      branchId: '1705139277953761280', subjectId: '1752935841845477384' },
    { name: 'CCPH',     branchId: '1705139277953761280', subjectId: '1798740810791911424' },
];

const STUDENT_ID = '1798906253557104911';

// ======================== 路径管理 ========================

// 从命令行参数获取日期，默认使用今天
const dateArg = process.argv.find(a => a.startsWith('--date='));
const DATE = dateArg ? dateArg.split('=')[1] : new Date().toISOString().split('T')[0];

// 项目根目录
const ROOT = path.resolve(__dirname, '..');

const DIR = {
    raw:     path.join(ROOT, DATE),
    solved:  path.join(ROOT, DATE, 'solved'),
    rewrite: path.join(ROOT, DATE, 'rewrite'),
    oldRaw:  path.join(ROOT, 'new'),
    oldRewrite: path.join(ROOT, 'new', 'rewrite'),
    merge:   path.join(ROOT, 'merge'),
    cura:    path.join(ROOT, 'cura'),
};

// 小科目（每卷题量少）
const SMALL_SUBJECTS = new Set(['NCH', 'ORH', 'CCPH', 'SDH']);
// 大科目每卷题量: [单选, 多选, 判断, 填空]
const BIG_COUNT  = [20, 15, 15, 10];
// 小科目每卷题量
const SMALL_COUNT = [15, 15, 15, 5];

// ======================== 工具函数 ========================

function ensureDir(dir) {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
}

function writeJson(filePath, data) {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
}

function readJson(filePath) {
    if (!fs.existsSync(filePath)) return [];
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
}

function log(step, msg) {
    const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    console.log(`[${ts}] [${step}] ${msg}`);
}

// ======================== Step 1: 抓取（高并发版） ========================

const CONCURRENCY = 16; // 最大并发请求数
const RETRY_LIMIT = 3;

async function singleRequest(params, headers) {
    const res = await axios.post(BASE_URL + TARGET_PATH, JSON.stringify(params), { headers, timeout: 15000 });
    const data = JSON.parse(res.data.data.paperStore.paperContent);
    return [].concat(data.panduan.children, data.danxuan.children, data.duoxuan.children, data.tiankong.children);
}

async function fetchSubject(subject) {
    const params = {
        branchId: subject.branchId,
        chapterId: '',
        studentId: STUDENT_ID,
        subjectId: subject.subjectId,
    };
    const headers = { 'Content-Type': 'application/json;charset=utf-8' };
    let jsonArray = [];
    let completed = 0;

    // 生成任务队列
    const tasks = [];
    for (let i = 0; i < FETCH_COUNT; i++) {
        tasks.push(i);
    }

    // 并发池
    let index = 0;
    async function runNext() {
        while (index < tasks.length) {
            const i = index++;
            let retryCount = 0;
            let success = false;

            while (retryCount <= RETRY_LIMIT && !success) {
                try {
                    const result = await singleRequest(params, headers);
                    jsonArray = jsonArray.concat(result);
                    success = true;
                } catch (err) {
                    retryCount++;
                    if (retryCount > RETRY_LIMIT) {
                        // 放弃这个请求
                    }
                }
            }

            completed++;
            if (completed % 100 === 0 || completed === FETCH_COUNT) {
                log('Fetch', `${subject.name}: ${completed}/${FETCH_COUNT} (已收集 ${jsonArray.length} 题)`);
            }
        }
    }

    // 启动并发池
    const workers = [];
    for (let w = 0; w < Math.min(CONCURRENCY, FETCH_COUNT); w++) {
        workers.push(runNext());
    }
    await Promise.all(workers);

    writeJson(path.join(DIR.raw, `${subject.name}.json`), jsonArray);
    log('Fetch', `${subject.name} 完成，共 ${jsonArray.length} 题`);
    return { name: subject.name, count: jsonArray.length };
}

async function fetchAll() {
    ensureDir(DIR.raw);
    log('Fetch', `开始抓取所有科目（并发: ${CONCURRENCY}）...`);

    // 9 个科目同时跑
    const results = await Promise.all(SUBJECTS.map(s => fetchSubject(s)));

    const total = results.reduce((s, r) => s + r.count, 0);
    log('Fetch', `所有科目抓取完成，共 ${total} 题`);
    return results;
}

// ======================== Step 2: 去重 ========================

function dedup() {
    ensureDir(DIR.solved);
    log('Dedup', '开始去重...');

    let totalBefore = 0, totalAfter = 0;

    for (const subject of SUBJECTS) {
        const rawPath = path.join(DIR.raw, `${subject.name}.json`);
        if (!fs.existsSync(rawPath)) {
            log('Dedup', `跳过 ${subject.name}（原始文件不存在）`);
            continue;
        }

        const arr = readJson(rawPath);
        totalBefore += arr.length;

        const seenIds = new Set();
        const unique = arr.filter(item => {
            if (seenIds.has(item.id)) return false;
            seenIds.add(item.id);
            return true;
        });

        totalAfter += unique.length;
        writeJson(path.join(DIR.solved, `${subject.name}Solved.json`), unique);
        log('Dedup', `${subject.name}: ${arr.length} → ${unique.length} (去除 ${arr.length - unique.length} 个重复)`);
    }

    log('Dedup', `去重完成: ${totalBefore} → ${totalAfter}`);
}

// ======================== Step 3: 改写分类（Node.js 版 rewritePython.py） ========================

function rewrite() {
    ensureDir(DIR.rewrite);
    log('Rewrite', '开始改写分类...');

    const files = fs.readdirSync(DIR.solved).filter(f => f.endsWith('Solved.json'));
    let totalQuestions = 0;

    for (const file of files) {
        const subjectName = file.replace('Solved.json', '');
        const data = readJson(path.join(DIR.solved, file));

        const sc = []; // singleChoice
        const mc = []; // multipleChoice
        const rw = []; // rightWrong
        const fb = []; // fillingBlank
        const all = []; // subject (全部)

        for (const item of data) {
            const answer = (item.standardAnswer || '').replace(/ /g, '');
            const title = (item.title || '').replace(/ /g, '');
            const id = (item.id || '').replace(/ /g, '');

            let classified;

            if (answer === '正确' || answer === '错误') {
                // 判断题
                classified = {
                    questionStem: title,
                    option: ['正确', '错误'],
                    answer: answer,
                    id: id,
                    likeFlag: false,
                    markFlag: false,
                    abbreviationSubject: subjectName,
                };
                rw.push(classified);
            } else if (answer.length > 0 && answer.charCodeAt(0) >= 65 && answer.charCodeAt(0) <= 90) {
                // 选择题（A-D 开头）
                const options = (item.options || '').replace(/ /g, '').replace(/\./g, '').split('|');
                classified = {
                    questionStem: title,
                    option: options,
                    answer: answer,
                    id: id,
                    likeFlag: false,
                    markFlag: false,
                    abbreviationSubject: subjectName,
                };
                if (answer.length === 1 && options.length === 4) {
                    sc.push(classified);
                } else {
                    mc.push(classified);
                }
            } else {
                // 填空题
                classified = {
                    questionStem: title,
                    option: '',
                    answer: answer.replace(/\|/g, '，'),
                    id: id,
                    likeFlag: false,
                    markFlag: false,
                    abbreviationSubject: subjectName,
                };
                fb.push(classified);
            }

            all.push(classified);
        }

        // 写出 5 个分类文件
        writeJson(path.join(DIR.rewrite, `${subjectName}_subject.json`), all);
        writeJson(path.join(DIR.rewrite, `${subjectName}_singleChoice.json`), sc);
        writeJson(path.join(DIR.rewrite, `${subjectName}_multipleChoice.json`), mc);
        writeJson(path.join(DIR.rewrite, `${subjectName}_rightWrong.json`), rw);
        writeJson(path.join(DIR.rewrite, `${subjectName}_fillingBlank.json`), fb);

        totalQuestions += all.length;
        log('Rewrite', `${subjectName}: 共 ${all.length} 题 (单选${sc.length} 多选${mc.length} 判断${rw.length} 填空${fb.length})`);
    }

    log('Rewrite', `改写分类完成，共处理 ${totalQuestions} 题`);
}

// ======================== Step 4: 合并新旧数据 ========================

function merge() {
    ensureDir(DIR.merge);
    log('Merge', '开始合并新旧数据...');

    const types = ['singleChoice', 'multipleChoice', 'rightWrong', 'fillingBlank', 'subject'];
    let totalMerged = 0;

    for (const subject of SUBJECTS) {
        for (const type of types) {
            const fileName = `${subject.name}_${type}.json`;
            const newFilePath = path.join(DIR.rewrite, fileName);
            const oldFilePath = path.join(DIR.oldRewrite, fileName);
            const outputFilePath = path.join(DIR.merge, fileName);

            if (!fs.existsSync(newFilePath)) {
                // 新数据中不存在该文件，如果旧数据有则直接复制
                if (fs.existsSync(oldFilePath)) {
                    writeJson(outputFilePath, readJson(oldFilePath));
                }
                continue;
            }

            const newData = readJson(newFilePath);

            if (!fs.existsSync(oldFilePath)) {
                // 没有旧数据，直接使用新数据
                writeJson(outputFilePath, newData);
                totalMerged += newData.length;
                continue;
            }

            const oldData = readJson(oldFilePath);

            // 保留旧数据中仍然存在于新数据中的条目
            const newDataStrSet = new Set(newData.map(item => JSON.stringify(item)));
            const retainedOld = oldData.filter(item => newDataStrSet.has(JSON.stringify(item)));

            // 添加新数据中旧数据没有的条目
            const retainedOldStrSet = new Set(retainedOld.map(item => JSON.stringify(item)));
            const freshNew = newData.filter(item => !retainedOldStrSet.has(JSON.stringify(item)));

            const merged = [...retainedOld, ...freshNew];
            writeJson(outputFilePath, merged);
            totalMerged += merged.length;
        }
        log('Merge', `${subject.name} 合并完成`);
    }

    log('Merge', `所有科目合并完成，共 ${totalMerged} 条数据`);
}

// ======================== Step 5: 分类标注 ========================

function classify() {
    ensureDir(DIR.merge);
    log('Classify', '开始分类标注...');

    const files = fs.readdirSync(DIR.merge).filter(f => f.endsWith('.json'));
    let total = 0;

    for (const file of files) {
        const filePath = path.join(DIR.merge, file);
        const data = readJson(filePath);

        for (const item of data) {
            if (item.option && item.option.length === 2) {
                item.type = 'rightWrong';
            } else if (!item.option || item.option.length === 0 || item.option === '') {
                item.type = 'fillingBlank';
            } else if (item.answer && item.answer.length === 1) {
                item.type = 'singleChoice';
            } else {
                item.type = 'multipleChoice';
            }
        }

        writeJson(filePath, data);
        total += data.length;
    }

    log('Classify', `分类标注完成，共 ${files.length} 个文件，${total} 条数据`);
}

// ======================== Step 6: 组卷（Node.js 版 pythonCura.py） ========================

function cura() {
    ensureDir(DIR.cura);
    log('Cura', '开始组卷...');

    let totalPapers = 0;

    for (const subject of SUBJECTS) {
        const name = subject.name;
        const isSmall = SMALL_SUBJECTS.has(name);
        const count = isSmall ? SMALL_COUNT : BIG_COUNT;

        // 读取分类后的数据（从 merge 目录，因为已经合并了新旧数据）
        const scPath = path.join(DIR.merge, `${name}_singleChoice.json`);
        const mcPath = path.join(DIR.merge, `${name}_multipleChoice.json`);
        const rwPath = path.join(DIR.merge, `${name}_rightWrong.json`);
        const fbPath = path.join(DIR.merge, `${name}_fillingBlank.json`);

        const singleChoice   = readJson(scPath);
        const multipleChoice = readJson(mcPath);
        const rightWrong     = readJson(rwPath);
        const fillingBlank   = readJson(fbPath);

        if (singleChoice.length === 0 && multipleChoice.length === 0 && rightWrong.length === 0) {
            log('Cura', `${name}: 无题目数据，跳过`);
            continue;
        }

        // 计算能组多少套卷（取各题型能分的套数最小值）
        const splits = [
            Math.floor(singleChoice.length / count[0]),
            Math.floor(multipleChoice.length / count[1]),
            Math.floor(rightWrong.length / count[2]),
        ];
        const numPapers = Math.min(...splits);

        if (numPapers === 0) {
            log('Cura', `${name}: 题目不足一套卷，跳过`);
            continue;
        }

        let fbOffset = 0;
        const hasFb = fillingBlank.length > 0;

        for (let i = 0; i < numPapers; i++) {
            const paper = [];
            paper.push(...singleChoice.slice(i * count[0], (i + 1) * count[0]));
            paper.push(...multipleChoice.slice(i * count[1], (i + 1) * count[1]));
            paper.push(...rightWrong.slice(i * count[2], (i + 1) * count[2]));

            if (hasFb) {
                const fbEnd = fbOffset + count[3];
                if (fbEnd <= fillingBlank.length) {
                    paper.push(...fillingBlank.slice(fbOffset, fbEnd));
                    fbOffset = fbEnd;
                } else {
                    // 填空题不够，用剩余的
                    paper.push(...fillingBlank.slice(fbOffset));
                    fbOffset = fillingBlank.length;
                }
            }

            writeJson(path.join(DIR.cura, `${name}_${i + 1}.json`), paper);
            totalPapers++;
        }

        // 写剩余题目
        const residual = [];
        residual.push(...singleChoice.slice(numPapers * count[0]));
        residual.push(...multipleChoice.slice(numPapers * count[1]));
        residual.push(...rightWrong.slice(numPapers * count[2]));
        if (hasFb && fbOffset < fillingBlank.length) {
            residual.push(...fillingBlank.slice(fbOffset));
        }
        writeJson(path.join(DIR.cura, `${name}_residual.json`), residual);

        log('Cura', `${name}: 生成 ${numPapers} 套卷 (每套: 单选${count[0]} 多选${count[1]} 判断${count[2]} 填空${hasFb ? count[3] : 0})`);
    }

    log('Cura', `组卷完成，共 ${totalPapers} 套试卷`);
}

// ======================== Step 7: 同步到 new/ 目录 ========================

function syncToNew() {
    log('Sync', '开始同步到 new/ 目录...');

    const types = ['singleChoice', 'multipleChoice', 'rightWrong', 'fillingBlank', 'subject'];

    for (const subject of SUBJECTS) {
        // 同步合并后的 rewrite 数据
        for (const type of types) {
            const fileName = `${subject.name}_${type}.json`;
            const srcPath = path.join(DIR.merge, fileName);
            const dstPath = path.join(DIR.oldRewrite, fileName);

            if (fs.existsSync(srcPath)) {
                writeJson(dstPath, readJson(srcPath));
            }
        }

        // 同步原始数据（合并后 rewrite 中的 subject 文件作为原始数据源）
        const subjectPath = path.join(DIR.merge, `${subject.name}_subject.json`);
        const rawPath = path.join(DIR.oldRaw, `${subject.name}.json`);
        if (fs.existsSync(subjectPath)) {
            writeJson(rawPath, readJson(subjectPath));
        }
    }

    log('Sync', '同步完成，new/ 目录已更新为最新数据');
}

// ======================== 主流程 ========================

async function main() {
    console.log('='.repeat(60));
    log('Pipeline', `题库自动更新管道启动 | 日期: ${DATE}`);
    console.log('='.repeat(60));

    try {
        // Step 1: 抓取
        console.log('\n--- Step 1/7: 数据抓取 ---');
        await fetchAll();

        // Step 2: 去重
        console.log('\n--- Step 2/7: 数据去重 ---');
        dedup();

        // Step 3: 改写分类
        console.log('\n--- Step 3/7: 改写分类 ---');
        rewrite();

        // Step 4: 合并
        console.log('\n--- Step 4/7: 合并新旧数据 ---');
        merge();

        // Step 5: 分类标注
        console.log('\n--- Step 5/7: 分类标注 ---');
        classify();

        // Step 6: 组卷
        console.log('\n--- Step 6/7: 自动组卷 ---');
        cura();

        // Step 7: 同步
        console.log('\n--- Step 7/7: 同步到 new/ ---');
        syncToNew();

        console.log('\n' + '='.repeat(60));
        log('Pipeline', '全部流程执行完成！');
        console.log('='.repeat(60));
    } catch (err) {
        log('Pipeline', `执行失败: ${err.message}`);
        console.error(err);
        process.exit(1);
    }
}

main();
