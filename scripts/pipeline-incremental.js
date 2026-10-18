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
const { mergeById } = require('./lib/merge');
const { createAccountPool } = require('./lib/accountPool');

// ======================== 配置 ========================

const BASE_URL = 'http://222.73.57.153:6571';
const TARGET_PATH = '/examinationInfo/getPracticeInfo';
const FETCH_COUNT = 400; // 与主流水线一致：降低请求压力，样本量仍足够统计抽题权重
const CONCURRENCY = 5; // 保守并发，防服务器限流
const RETRY_LIMIT = 3;

// ======================== 学号轮换池 + 登录 ========================
// 池子加载（数组/网格规格）与并发预热在 scripts/lib/accountPool.js，两条 pipeline 共用。
// 密码=学号本身；抓取时按 round-robin 轮换学号，避免单一身份被服务器限流。
const BRANCH_ID = '1705139277953761280';
const LOGIN_PATH = '/login/practiceLogin';

async function login(studentNum) {
    const body = JSON.stringify({
        branchId: BRANCH_ID,
        studentNum,
        practiceType: '1',
        password: studentNum, // 密码=学号本身
    });
    const res = await axios.post(BASE_URL + LOGIN_PATH, body, {
        headers: { 'Content-Type': 'application/json;charset=utf-8' },
        timeout: 15000,
    });
    const bodyJson = res.data;
    if (!bodyJson || String(bodyJson.code) !== '200' || !bodyJson.data || !bodyJson.data.studentInfo) {
        throw new Error(`登录失败: ${bodyJson && bodyJson.msg}`);
    }
    return bodyJson.data.studentInfo.id;
}

const accountPool = createAccountPool({ login });
const { nextStudentNum, getSid, refreshSid } = accountPool;

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
    const studentNum = nextStudentNum();
    let sid = await getSid(studentNum);
    const params = {
        branchId: subject.branchId,
        chapterId: '',
        studentId: sid,
        subjectId: subject.subjectId,
    };

    let res = await axios.post(BASE_URL + TARGET_PATH, JSON.stringify(params), { headers, timeout: 15000 });

    // 检测 sid 过期
    const body = res.data;
    const isExpired = !body || String(body.code) !== '200' || !body.data || !body.data.paperStore;
    if (isExpired) {
        const msg = body && body.msg ? String(body.msg) : '';
        const code = body && body.code;
        if (String(code) === '401' || /登录|过期|未登录|失效|超时/.test(msg)) {
            sid = await refreshSid(studentNum);
            params.studentId = sid;
            res = await axios.post(BASE_URL + TARGET_PATH, JSON.stringify(params), { headers, timeout: 15000 });
        } else {
            throw new Error(`请求异常: code=${code}, msg=${msg}`);
        }
    }

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
 * 合并新旧分类数据（按 id 安全合并，逻辑统一在 scripts/lib/merge.js）
 * 保留全部旧条目 + 追加新发现的 id；opts 透传元数据打标参数
 */
function mergeClassified(oldClassified, newClassified, opts) {
    const types = ['singleChoice', 'multipleChoice', 'rightWrong', 'fillingBlank', 'subject'];
    const merged = {};

    for (const type of types) {
        const oldData = oldClassified[type] || [];
        const newData = newClassified[type] || [];

        const { merged: result, added } = mergeById(oldData, newData, opts);

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

    // 先登录所有学号预热 sid 缓存
    await accountPool.warmup((m) => log(`[Login] ${m}`));

    for (const subject of SUBJECTS) {
        log(`\n${'='.repeat(50)}`);
        log(`开始抓取: ${subject.name}`);

        // Step 1: 抓取
        const { jsonArray, errors, count } = await fetchSubject(subject);
        log(`${subject.name}: 抓取完成，共 ${count} 题（失败 ${errors} 次）`);

        if (count === 0) {
            // 空爬：保留旧数据、不上传（安全），但 loudly 提醒，避免静默失败被忽视
            const rewriteDir0 = path.join(ROOT, 'new', 'rewrite');
            const oldExists = fs.existsSync(path.join(rewriteDir0, `${subject.name}_subject.json`));
            log(`${subject.name}: ⚠ 本次爬取 0 题${oldExists ? '（旧库有数据，疑似服务器/登录异常，已保留旧数据不上传）' : ''}，跳过`);
            continue;
        }

        // Step 2: 去重（去重前统计每题出现次数——官方抽题权重数据）
        const seenIds = new Set();
        const countMap = new Map();
        const unique = jsonArray.filter(item => {
            if (item && item.id) {
                const k = String(item.id);
                countMap.set(k, (countMap.get(k) || 0) + 1);
            }
            if (seenIds.has(item.id)) return false;
            seenIds.add(item.id);
            return true;
        });
        log(`${subject.name}: 去重 ${jsonArray.length} → ${unique.length}`);

        // Step 3: 改写分类
        const classified = rewrite(subject.name, unique);
        log(`${subject.name}: 分类完成 (单选${classified.singleChoice.length} 多选${classified.multipleChoice.length} 判断${classified.rightWrong.length} 填空${classified.fillingBlank.length})`);

        // Step 4: 读取旧数据并合并去重（带元数据打标）
        const rewriteDir = path.join(ROOT, 'new', 'rewrite');
        const oldClassified = {};
        const types = ['singleChoice', 'multipleChoice', 'rightWrong', 'fillingBlank', 'subject'];
        for (const type of types) {
            const oldFile = path.join(rewriteDir, `${subject.name}_${type}.json`);
            if (fs.existsSync(oldFile)) {
                try { oldClassified[type] = JSON.parse(fs.readFileSync(oldFile, 'utf-8')); } catch {}
            }
        }

        const stampOpts = { countMap, today: new Date().toISOString().split('T')[0] };
        const hasOld = Object.keys(oldClassified).length > 0;
        const finalClassified = hasOld ? mergeClassified(oldClassified, classified, stampOpts) : mergeClassified({}, classified, stampOpts);

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
