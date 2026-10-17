/**
 * 新旧题库安全合并（单一来源，pipeline.js / pipeline-incremental.js 共用）
 *
 * 规则（非破坏式，按 id）：
 * - 旧条目全部保留，顺序不变
 * - 新数据中旧库没有的 id 追加到末尾
 * - 缺 id 的异常条目直接丢弃并计数，不进入去重集合
 *
 * 背景：pipeline.js 旧版 merge 用「全对象 JSON 字符串交集」过滤旧数据，
 * 2026-06-08 一次空爬（服务器不可用）把 canonical 数据 45 个文件全部清成 []。
 * id 为源站 snowflake，天然稳定，按 id 合并即可。
 */

/**
 * 按 id 安全合并，可选打标元数据。
 * @param {Array} oldData 旧库条目
 * @param {Array} newData 本次爬取（去重后）条目
 * @param {object} [opts]
 * @param {Map<string,number>} [opts.countMap] 本次运行每个 id 的出现次数（去重前统计，跨题型全局）
 * @param {string} [opts.today] 'YYYY-MM-DD'，新题首收日期
 * 打标语义：
 * - collectedAt = 首次收录日期，一旦写入永不覆盖（旧条目缺字段时补写 today，仅为迁移兜底）
 * - appearCount = 历史累计出现次数；旧条目累加本次次数，新条目取本次次数（至少 1）
 */
function mergeById(oldData, newData, opts = {}) {
    const oldArr = Array.isArray(oldData) ? oldData : [];
    const newArr = Array.isArray(newData) ? newData : [];
    const { countMap, today } = opts;
    const runCount = (id) => (countMap ? (countMap.get(String(id)) || 0) : 0);
    const stamp = opts.stamp !== false && (today || countMap); // 未传打标参数则完全不改对象

    const seenIds = new Set();
    const merged = [];
    let droppedNoId = 0;

    // 先放旧数据（保持原顺序，全部保留）
    for (const item of oldArr) {
        if (!item || !item.id) { droppedNoId++; continue; }
        if (seenIds.has(item.id)) continue; // 旧数据内部去重
        seenIds.add(item.id);
        if (stamp) {
            if (!item.collectedAt && today) item.collectedAt = today;
            if (countMap) {
                const add = runCount(item.id);
                // 只在确有增量或已有累计值时写入，避免给从未出现的题造 appearCount:0
                if (add > 0 || item.appearCount !== undefined) {
                    item.appearCount = (Number(item.appearCount) || 0) + add;
                }
            }
        }
        merged.push(item);
    }

    // 再追加新数据中旧库没有的
    let added = 0;
    for (const item of newArr) {
        if (!item || !item.id) { droppedNoId++; continue; }
        if (seenIds.has(item.id)) continue;
        seenIds.add(item.id);
        if (stamp && today) {
            item.collectedAt = today;            // 首次收录
            if (countMap) item.appearCount = runCount(item.id) || 1;
        }
        merged.push(item);
        added++;
    }

    return { merged, added, droppedNoId };
}

/**
 * 空爬守卫：旧库有数据而本次爬取一题未得 → 判定为空爬，返回违规科目列表。
 * 调用方据此 process.exit(1)，让 CI 跳过 auto-commit 与 R2 上传。
 * 注意：不能用「新题量 < 旧库一半」做硬门槛——题库随时间累积，
 * 单次爬取的去重题量占比天然下降，会误杀正常运行的管道。
 */
function checkEmptyCrawlGuard(subjectTotals) {
    return subjectTotals.filter(t => t.oldTotal > 0 && t.newTotal === 0);
}

module.exports = { mergeById, checkEmptyCrawlGuard };
