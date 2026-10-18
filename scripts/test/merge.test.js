// merge.js 回归测试——本模块是 2026-06-08 空爬清库事故的修复，
// 「旧库非空 + 新数据空 → 绝不返回 []」是最高优先级不变量。
// 纯 node 运行：node scripts/test/merge.test.js
const { mergeById, checkEmptyCrawlGuard } = require('../lib/merge.js');

let failures = 0;
function check(name, cond, detail) {
    if (cond) { console.log(`  ✓ ${name}`); }
    else { failures++; console.error(`  ✗ ${name} ${detail || ''}`); }
}
const q = (id, extra = {}) => ({ id, questionStem: `题${id}`, ...extra });

// ---- 1. 事故回归：空爬不毁数据 ----
console.log('[1] 空爬守卫（事故回归）');
{
    const oldData = [q('a'), q('b'), q('c')];
    const { merged, added } = mergeById(oldData, []);
    check('旧库非空 + 新数据空 → 原样返回，绝不 []', merged.length === 3 && merged.map(x => x.id).join(',') === 'a,b,c', `实际 ${JSON.stringify(merged.map(x => x.id))}`);
    check('added = 0', added === 0);

    const violations = checkEmptyCrawlGuard([
        { subject: 'Marx', oldTotal: 100, newTotal: 0 },
        { subject: 'CMH', oldTotal: 80, newTotal: 12 },
        { subject: 'New', oldTotal: 0, newTotal: 0 },
    ]);
    check('空爬守卫只抓「旧>0 且 新=0」的科目', violations.length === 1 && violations[0].subject === 'Marx', JSON.stringify(violations));
}

// ---- 2. 增量合并语义 ----
console.log('[2] 按 id 增量合并');
{
    const oldData = [q('a'), q('b')];
    const newData = [q('b'), q('c'), q('d')];
    const { merged, added, droppedNoId } = mergeById(oldData, newData);
    check('旧条目保序在前，新 id 追加在后', merged.map(x => x.id).join(',') === 'a,b,c,d');
    check('added = 2（c、d）', added === 2);
    check('无丢弃', droppedNoId === 0);
    check('旧对象引用不换（元数据原地保留）', merged[0] === oldData[0]);
}

// ---- 3. 元数据打标语义 ----
console.log('[3] collectedAt / appearCount 打标');
{
    const today = '2026-10-19';
    const oldData = [
        q('a', { collectedAt: '2026-06-01', appearCount: 5 }),   // 完整旧条目
        q('b', { collectedAt: '2026-06-01' }),                    // 旧条目无累计
        q('c'),                                                   // 迁移兜底：缺 collectedAt
    ];
    const newData = [q('d')];
    const countMap = new Map([['a', 3], ['d', 2]]); // 本次 a 出现 3 次，d 出现 2 次
    const { merged } = mergeById(oldData, newData, { countMap, today });

    check('旧条目 collectedAt 永不覆盖', merged[0].collectedAt === '2026-06-01' && merged[1].collectedAt === '2026-06-01');
    check('旧条目缺 collectedAt 时补 today', merged[2].collectedAt === today);
    check('旧条目 appearCount 累加本次出现数', merged[0].appearCount === 8, `实际 ${merged[0].appearCount}`);
    check('旧条目无累计且本次未出现 → 不造 appearCount', merged[1].appearCount === undefined && merged[2].appearCount === undefined);
    check('新条目 collectedAt = today', merged[3].collectedAt === today);
    check('新条目 appearCount = 本次出现数', merged[3].appearCount === 2, `实际 ${merged[3].appearCount}`);
}
{
    // countMap 缺席时不虚构出现次数（与存量回填「不造 appearCount」一致），|| 1 仅兜 countMap 缺 id
    const { merged } = mergeById([], [q('x')], { today: '2026-10-19' });
    check('无 countMap 时不虚构 appearCount（collectedAt 仍打标）', merged[0].appearCount === undefined && merged[0].collectedAt === '2026-10-19');
    const { merged: m1 } = mergeById([], [q('y')], { countMap: new Map(), today: '2026-10-19' });
    check('countMap 缺 id 时新条目 appearCount 兜底为 1', m1[0].appearCount === 1);
}
{
    // 不传打标参数 → 对象完全不改（纯净合并）
    const oldData = [q('a')];
    mergeById(oldData, [q('b')]);
    check('未传打标参数时旧对象零改动', Object.keys(oldData[0]).join(',') === 'id,questionStem');
}

// ---- 4. 异常输入防御 ----
console.log('[4] 异常输入');
{
    const { merged, droppedNoId } = mergeById([q('a'), null, { questionStem: '无id' }], [q('b'), undefined, {}]);
    check('缺 id / null 条目丢弃并计数', droppedNoId === 4 && merged.map(x => x.id).join(',') === 'a,b', `dropped=${droppedNoId}`);
    const { merged: m2 } = mergeById(null, undefined);
    check('非数组输入安全（视作空）', Array.isArray(m2) && m2.length === 0);
    const dup = mergeById([q('a'), q('a', { collectedAt: '2026-06-01' })], [q('a')]);
    check('旧数据内部重复 id 去重', dup.merged.length === 1);
}

console.log(failures === 0 ? '\n全部通过' : `\n${failures} 项失败`);
process.exit(failures === 0 ? 0 : 1);
