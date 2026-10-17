/**
 * 灾难恢复脚本：从线上 R2 数据 API 拉回 canonical 题库到 new/rewrite/
 *
 * 背景：2026-06-08 空爬 + 旧版破坏式 merge 把仓库 new/rewrite 45 个文件
 * 全部清成 []（commit f138d59）；R2 上 2026-06-01 CI 上传的快照是唯一完好副本
 * （前端数据 API https://bankapi.635262140.xyz 即代理该 bucket）。
 *
 * 用法：
 *   node scripts/restore-from-r2.mjs             # 下载全部 45 个文件并覆盖本地
 *   node scripts/restore-from-r2.mjs --verify    # 只校验线上可达性与各文件题数，不写盘
 *
 * 安全性：先全部拉取并校验（HTTP 200 + 非空数组），任一文件异常即整体中止，
 * 校验全部通过后才写盘，避免半截恢复。
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DATA_API = process.env.DATA_API || 'https://bankapi.635262140.xyz';

const SUBJECTS = [
    'XiIntro', 'Marx', 'MaoIntro', 'Political', 'CMH',
    'NCH', 'SDH', 'ORH', 'CCPH',
];
const TYPES = ['singleChoice', 'multipleChoice', 'rightWrong', 'fillingBlank', 'subject'];

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TARGET_DIR = path.join(ROOT, 'new', 'rewrite');

const verifyOnly = process.argv.includes('--verify');

async function fetchOne(subject, type) {
    const key = `${subject}_${type}.json`;
    const res = await fetch(`${DATA_API}/rewrite/${key}`, {
        headers: { 'Cache-Control': 'no-cache' },
        signal: AbortSignal.timeout(30000),
    });
    if (!res.ok) throw new Error(`${key}: HTTP ${res.status}`);
    const data = await res.json();
    if (!Array.isArray(data) || data.length === 0) {
        throw new Error(`${key}: 内容为空或不是数组（线上数据可能已损坏，停止恢复！）`);
    }
    // 基本结构抽检：首条必须有题干和 id
    const first = data[0];
    if (!first || !first.questionStem || !first.id) {
        throw new Error(`${key}: 首条数据缺少 questionStem/id，疑似非题库文件`);
    }
    return { key, data };
}

async function main() {
    console.log(`数据源: ${DATA_API}${verifyOnly ? '（仅校验模式）' : ''}`);

    // Pass 1: 全部拉取并校验
    const results = [];
    for (const subject of SUBJECTS) {
        for (const type of TYPES) {
            const { key, data } = await fetchOne(subject, type);
            results.push({ key, data });
            console.log(`  ✓ ${key}: ${data.length} 题`);
        }
    }

    const total = results.reduce((s, r) => s + r.data.length, 0);
    console.log(`\n校验通过: ${results.length} 个文件，共 ${total} 题`);

    if (verifyOnly) {
        console.log('仅校验模式，未写盘');
        return;
    }

    // Pass 2: 校验全部通过后统一写盘（仓库惯例：2 空格缩进）
    fs.mkdirSync(TARGET_DIR, { recursive: true });
    for (const { key, data } of results) {
        fs.writeFileSync(path.join(TARGET_DIR, key), JSON.stringify(data, null, 2), 'utf-8');
    }
    console.log(`已恢复 ${results.length} 个文件 → new/rewrite/`);
}

main().catch(err => {
    console.error(`恢复失败: ${err.message}`);
    console.error('未写入任何文件（本地数据未动）');
    process.exit(1);
});
