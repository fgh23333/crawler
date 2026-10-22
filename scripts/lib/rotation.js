/**
 * 科目轮转（单一来源，pipeline.js 使用）
 *
 * 背景：马院服务器对每 IP 设有内容预算（2026-10-22 实测约 100 份卷子 / 300 题，
 * 用尽后返回空卷），且预算几乎总被抓取队列最前的科目独吞——10-19/10-22 三次
 * 运行全部落在 XiIntro（固定队首）。按运行日轮转起始科目，让 9 科轮流吃到预算，
 * 避免增量永远堆在一科。
 *
 * 无状态、确定性：同一日期多次运行结果一致，跨天自动轮换，无需持久化。
 */

/**
 * 按日期把科目列表循环左移。
 * @param {Array} subjects 科目列表（原数组不被修改）
 * @param {string} dateStr 'YYYY-MM-DD'（运行日）
 * @returns {Array} 轮转后的新数组；长度 < 2 时原样返回
 */
function rotateSubjects(subjects, dateStr) {
    const list = [...subjects];
    if (list.length < 2) return list;
    const d = new Date(`${dateStr}T00:00:00Z`);
    if (Number.isNaN(d.getTime())) return list; // 日期非法时不轮转，保持行为可预期
    // 一年中的第几天（1 起）：1 月 1 日 = 1
    const dayOfYear = Math.floor((d - Date.UTC(d.getUTCFullYear(), 0, 0)) / 86400000);
    const start = dayOfYear % list.length;
    return start === 0 ? list : list.slice(start).concat(list.slice(0, start));
}

module.exports = { rotateSubjects };
