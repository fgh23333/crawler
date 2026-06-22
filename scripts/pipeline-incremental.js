/**
 * 增量抓取脚本：逐科目抓取，完成一个就上传 R2
 * 用法: node scripts/pipeline-incremental.js [科目名]
 * 示例: node scripts/pipeline-incremental.js MaoIntro    # 只跑 MaoIntro
 *       node scripts/pipeline-incremental.js              # 跑全部
 */

const axios = require('axios');
const fs = require('fs');
const path = require('path');
const https = require('https');

// ======================== 配置 ========================

const BASE_URL = 'http://222.73.57.153:6571';
const TARGET_PATH = '/examinationInfo/getPracticeInfo';
const FETCH_COUNT = 1000;
const CONCURRENCY = 5; // 保守并发，防服务器限流
const RETRY_LIMIT = 3;

// ======================== studentId 轮换池 ========================
// 从环境变量 STUDENT_IDS（JSON 数组字符串）加载多个 studentId，按 round-robin 轮换使用，
// 避免单一 studentId 被上游服务器限流。未设置或格式非法时回退到单一内置 studentId。
const FALLBACK_STUDENT_ID = '1798906253557104911';

function loadStudentIdPool() {
    const raw = process.env.STUDENT_IDS;
    if (raw && raw.trim()) {
        try {
            const arr = JSON.parse(raw);
            if (Array.isArray(arr) && arr.length > 0 &&
                arr.every(x => typeof x === 'string' && x.trim() !== '')) {
                return arr.map(x => x.trim());
            }
            console.warn('[StudentPool] STUDENT_IDS 非非空字符串数组，回退到单一 studentId');
        } catch (e) {
            console.warn(`[StudentPool] STUDENT_IDS JSON 解析失败: ${e.message}，回退到单一 studentId`);
        }
    } else {
        console.warn('[StudentPool] 未设置 STUDENT_IDS 环境变量，回退到单一 studentId');
    }
    return [FALLBACK_STUDENT_ID];
}

const STUDENT_ID_POOL = loadStudentIdPool();
let rrIndex = 0;
function nextStudentId() {
    const id = STUDENT_ID_POOL[rrIndex % STUDENT_ID_POOL.length];
    rrIndex++;
    return id;
}
console.log(`[StudentPool] 已加载 ${STUDENT_ID_POOL.length} 个 studentId 进入轮换池`);

const ALL_SUBJECTS = [
    { name: 'MaoIntro', branchId: '1705139277953761280', subjectId: '1748168736914800651' },
    { name: 'XiIntro',  branchId: '1705139277953761280', subjectId: '1752935841845477392' },
    { name: 'Marx',     branchId: '1705139277953761280', subjectId: '1748167277460586496' },
    { name: 'Political', branchId: '1705139277953761280', subjectId: '1781216923707506688' },
    { name: 'CMH',      branchId: '1705139277953761280', subjectId: '1748168736914800640' },
    { name: 'NCH',      branchId: '1705139277953761280', subjectId: '1776854236110258176' },
    { name: 'SDH',      branchId: '1705139277953761280', subjectId: '1752935841845477376' },
    { name: 'ORH',      branchId: '1705139277953761280', subjectId: '1752935841845477384' },
    { name: 'CCPH',     branchId: '1705139277953761280', subjectId: '1798740810791911424' },
];

// R2 上传配置
const ACCOUNT_ID = 'bbd869342ef49cfea41170378427db5d';
const APIToken = process.env.CF_API_TOKEN;
const BUCKET = 'question-bank';

const ROOT = path.resolve(__dirname, '..');

// ======================== 命令行参数 ========================

const filterArg = process.argv[2]; // 可选: 指定科目名
const SUBJECTS = filterArg
    ? ALL_SUBJECTS.filter(s => s.name === filterArg)
    : ALL_SUBJECTS;

if (SUBJECTS.length === 0) {
    console.error(`未知科目: ${filterArg}`);
    console.error(`可选: ${ALL_SUBJECTS.map(s => s.name).join(', ')}`);
    process.exit(1);
}

// ======================== 工具函数 ========================

function log(msg) {
    const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    console.log(`[${ts}] ${msg}`);
}

function ensureDir(dir) {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

// ======================== 并发抓取单个科目 ========================

async function singleRequest(subject, headers) {
    const params = {
        branchId: subject.branchId,
        chapterId: '',
        studentId: nextStudentId(),
        subjectId: subject.subjectId,
    };
    const res = await axios.post(BASE_URL + TARGET_PATH, JSON.stringify(params), { headers, timeout: 15000 });
    const data = JSON.parse(res.data.data.paperStore.paperContent);
    return [].concat(data.panduan.children, data.danxuan.children, data.duoxuan.children, data.tiankong.children);
}

async function fetchSubject(subject) {
    const headers = { 'Content-Type': 'application/json;charset=utf-8' };
    let jsonArray = [];
    let completed = 0;
    let errors = 0;

    let index = 0;

    async function runNext() {
        while (index < FETCH_COUNT) {
            const i = index++;
            let retryCount = 0;

            while (retryCount <= RETRY_LIMIT) {
                try {
                    const result = await singleRequest(subject, headers);
                    jsonArray = jsonArray.concat(result);
                    break;
                } catch (err) {
                    retryCount++;
                    if (retryCount > RETRY_LIMIT) errors++;
                }
            }

            completed++;
            if (completed % 200 === 0 || completed === FETCH_COUNT) {
                log(`${subject.name}: ${completed}/${FETCH_COUNT} (已收集 ${jsonArray.length} 题, 失败 ${errors})`);
            }
        }
    }

    const workers = [];
    for (let w = 0; w < Math.min(CONCURRENCY, FETCH_COUNT); w++) {
        workers.push(runNext());
    }
    await Promise.all(workers);

    return { jsonArray, errors, count: jsonArray.length };
}

// ======================== 改写分类 ========================

function rewrite(subjectName, data) {
    const sc = [], mc = [], rw = [], fb = [], all = [];

    for (const item of data) {
        const answer = (item.standardAnswer || '').replace(/ /g, '');
        const title = (item.title || '').replace(/ /g, '');
        const id = (item.id || '').replace(/ /g, '');

        let classified;
        if (answer === '正确' || answer === '错误') {
            classified = { questionStem: title, option: ['正确', '错误'], answer, id, likeFlag: false, markFlag: false, abbreviationSubject: subjectName };
            rw.push(classified);
        } else if (answer.length > 0 && answer.charCodeAt(0) >= 65 && answer.charCodeAt(0) <= 90) {
            const options = (item.options || '').replace(/ /g, '').replace(/\./g, '').split('|');
            classified = { questionStem: title, option: options, answer, id, likeFlag: false, markFlag: false, abbreviationSubject: subjectName };
            if (answer.length === 1 && options.length === 4) sc.push(classified);
            else mc.push(classified);
        } else {
            classified = { questionStem: title, option: '', answer: answer.replace(/\|/g, '，'), id, likeFlag: false, markFlag: false, abbreviationSubject: subjectName };
            fb.push(classified);
        }
        all.push(classified);
    }

    return {
        subject: all,
        singleChoice: sc,
        multipleChoice: mc,
        rightWrong: rw,
        fillingBlank: fb,
    };
}

// ======================== 合并去重 ========================

/**
 * 合并新旧分类数据（按 questionStem + answer 去重）
 * 保留旧数据中仍然有效的题目 + 添加新发现的题目
 */
function mergeClassified(oldClassified, newClassified) {
    const types = ['singleChoice', 'multipleChoice', 'rightWrong', 'fillingBlank', 'subject'];
    const merged = {};

    for (const type of types) {
        const oldData = oldClassified[type] || [];
        const newData = newClassified[type] || [];

        // 用 id 去重（id 是题目的唯一标识）
        const seenIds = new Set();
        const result = [];

        // 先放旧数据
        for (const item of oldData) {
            if (!seenIds.has(item.id)) {
                seenIds.add(item.id);
                result.push(item);
            }
        }

        // 再放新数据中旧数据没有的
        let added = 0;
        for (const item of newData) {
            if (!seenIds.has(item.id)) {
                seenIds.add(item.id);
                result.push(item);
                added++;
            }
        }

        merged[type] = result;
        if (added > 0) {
            log(`  ${type}: ${oldData.length} 旧 + ${added} 新 = ${result.length} 合计`);
        }
    }

    return merged;
}

// ======================== R2 上传 ========================

function uploadToR2(key, content) {
    return new Promise((resolve, reject) => {
        const body = Buffer.isBuffer(content) ? content : Buffer.from(content, 'utf-8');
        const req = https.request({
            hostname: 'api.cloudflare.com',
            path: `/client/v4/accounts/${ACCOUNT_ID}/r2/buckets/${BUCKET}/objects/${key}`,
            method: 'PUT',
            headers: { 'Authorization': `Bearer ${APIToken}`, 'Content-Type': 'application/json', 'Content-Length': body.length },
        }, (res) => {
            let d = '';
            res.on('data', c => d += c);
            res.on('end', () => res.statusCode === 200 ? resolve(key) : reject(new Error(`HTTP ${res.statusCode}: ${d.slice(0, 150)}`)));
        });
        req.on('error', reject);
        req.write(body);
        req.end();
    });
}

async function uploadSubjectToR2(subjectName, classified) {
    const types = ['singleChoice', 'multipleChoice', 'rightWrong', 'fillingBlank', 'subject'];
    for (const type of types) {
        const key = `rewrite/${subjectName}_${type}.json`;
        await uploadToR2(key, JSON.stringify(classified[type]));
    }
    log(`  ✓ R2 上传完成: ${types.length} 个文件`);
}

// ======================== 主流程 ========================

async function main() {
    if (!APIToken) {
        console.error('请设置环境变量 CF_API_TOKEN');
        process.exit(1);
    }

    log(`开始增量抓取，共 ${SUBJECTS.length} 个科目，并发 ${CONCURRENCY}`);

    for (const subject of SUBJECTS) {
        log(`\n${'='.repeat(50)}`);
        log(`开始抓取: ${subject.name}`);

        // Step 1: 抓取
        const { jsonArray, errors, count } = await fetchSubject(subject);
        log(`${subject.name}: 抓取完成，共 ${count} 题（失败 ${errors} 次）`);

        if (count === 0) {
            log(`${subject.name}: 无数据，跳过`);
            continue;
        }

        // Step 2: 去重
        const seenIds = new Set();
        const unique = jsonArray.filter(item => {
            if (seenIds.has(item.id)) return false;
            seenIds.add(item.id);
            return true;
        });
        log(`${subject.name}: 去重 ${jsonArray.length} → ${unique.length}`);

        // Step 3: 改写分类
        const classified = rewrite(subject.name, unique);
        log(`${subject.name}: 分类完成 (单选${classified.singleChoice.length} 多选${classified.multipleChoice.length} 判断${classified.rightWrong.length} 填空${classified.fillingBlank.length})`);

        // Step 4: 读取旧数据并合并去重
        const rewriteDir = path.join(ROOT, 'new', 'rewrite');
        const oldClassified = {};
        const types = ['singleChoice', 'multipleChoice', 'rightWrong', 'fillingBlank', 'subject'];
        for (const type of types) {
            const oldFile = path.join(rewriteDir, `${subject.name}_${type}.json`);
            if (fs.existsSync(oldFile)) {
                try { oldClassified[type] = JSON.parse(fs.readFileSync(oldFile, 'utf-8')); } catch {}
            }
        }

        const hasOld = Object.keys(oldClassified).length > 0;
        const finalClassified = hasOld ? mergeClassified(oldClassified, classified) : classified;

        if (hasOld) {
            const newTotal = finalClassified.subject.length;
            const oldTotal = (oldClassified.subject || []).length;
            log(`${subject.name}: 合并完成 ${oldTotal} 旧 + 新 → ${newTotal} 总计`);
        }

        // Step 5: 上传到 R2
        log(`${subject.name}: 上传到 R2...`);
        await uploadSubjectToR2(subject.name, finalClassified);

        // Step 6: 保存到本地
        ensureDir(rewriteDir);
        for (const type of types) {
            fs.writeFileSync(
                path.join(rewriteDir, `${subject.name}_${type}.json`),
                JSON.stringify(finalClassified[type], null, 2), 'utf-8'
            );
        }
        log(`${subject.name}: ✓ 本地文件已保存 (${finalClassified.subject.length} 题)`);
    }

    // 更新 version.json
    const version = { version: new Date().toISOString(), updated: new Date().toISOString() };
    await uploadToR2('version.json', JSON.stringify(version));
    log(`\nversion.json 已更新: ${version.version}`);
    log('\n全部完成！');
}

main().catch(err => { console.error('执行失败:', err); process.exit(1); });
