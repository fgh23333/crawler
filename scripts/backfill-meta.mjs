/**
 * 一次性回填：为存量题库补 collectedAt = R2 恢复快照日期
 *
 * 背景：canonical 数据于 2026-06-08 被空爬清空，2026-06-01 的 R2 快照是唯一副本。
 * 快照内的题目全部在 2026-06-01（含）之前收录，无更精确的首收日期可考，
 * 统一回填为快照日期。此后 pipeline 新增的题目会记录真实首收日期。
 *
 * 不回填 appearCount（历史出现次数已不可考，缺省视为未加权=1，
 * 首次成功爬取后会自然开始累计）。
 *
 * 用法：node scripts/backfill-meta.mjs            # 执行回填
 *       node scripts/backfill-meta.mjs --dry      # 只统计不写盘
 * 幂等：重复运行零变更。
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const BACKFILL_DATE = process.env.BACKFILL_DATE || '2026-06-01';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIR = path.join(ROOT, 'new', 'rewrite');

const dry = process.argv.includes('--dry');

const files = fs.readdirSync(DIR).filter(f => f.endsWith('.json'));
let totalQ = 0, stamped = 0, already = 0;
const touched = [];

for (const file of files) {
	const p = path.join(DIR, file);
	const data = JSON.parse(fs.readFileSync(p, 'utf-8'));
	if (!Array.isArray(data)) throw new Error(`${file}: 非数组`);
	let changed = 0;
	for (const q of data) {
		totalQ++;
		if (!q.collectedAt) {
			q.collectedAt = BACKFILL_DATE;
			stamped++;
			changed++;
		} else {
			already++;
		}
	}
	if (changed > 0) {
		touched.push(`${file}: +${changed}`);
		if (!dry) fs.writeFileSync(p, JSON.stringify(data, null, 2), 'utf-8');
	}
}

console.log(`共 ${files.length} 个文件 / ${totalQ} 题：本次回填 ${stamped}，已有日期 ${already}`);
for (const t of touched) console.log('  ' + t);
if (dry) console.log('（--dry 模式，未写盘）');
else if (touched.length) console.log(`已写回 ${touched.length} 个文件`);
else console.log('无变更（幂等）');
