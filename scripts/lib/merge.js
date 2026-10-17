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

function mergeById(oldData, newData) {
    const oldArr = Array.isArray(oldData) ? oldData : [];
    const newArr = Array.isArray(newData) ? newData : [];

    const seenIds = new Set();
    const merged = [];
    let droppedNoId = 0;

    // 先放旧数据（保持原顺序，全部保留）
    for (const item of oldArr) {
        if (!item || !item.id) { droppedNoId++; continue; }
        if (seenIds.has(item.id)) continue; // 旧数据内部去重
        seenIds.add(item.id);
        merged.push(item);
    }

    // 再追加新数据中旧库没有的
    let added = 0;
    for (const item of newArr) {
        if (!item || !item.id) { droppedNoId++; continue; }
        if (seenIds.has(item.id)) continue;
        seenIds.add(item.id);
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
