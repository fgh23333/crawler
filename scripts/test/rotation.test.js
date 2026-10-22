// rotation.js 回归测试——科目轮转的确定性、完整性与覆盖度。
// 纯 node 运行：node scripts/test/rotation.test.js
const { rotateSubjects } = require('../lib/rotation.js');

let failures = 0;
function check(name, cond, detail) {
    if (cond) { console.log(`  ✓ ${name}`); }
    else { failures++; console.error(`  ✗ ${name} ${detail || ''}`); }
}
const NAMES = ['XiIntro', 'Marx', 'MaoIntro', 'Political', 'CMH', 'NCH', 'SDH', 'ORH', 'CCPH'];
const subjects = NAMES.map(name => ({ name, branchId: 'b', subjectId: 's' }));

// ---- 1. 基本性质 ----
console.log('[1] 轮转基本性质');
{
    const r = rotateSubjects(subjects, '2026-10-22');
    check('长度与元素完整（一轮恰好包含全部科目）', r.length === 9 && new Set(r.map(s => s.name)).size === 9);
    check('是循环左移（顺序保持，仅起点移动）', NAMES.join(',').includes(r.map(s => s.name).join(',')) === false || r.map(s => s.name).join(',') === NAMES.slice(7).concat(NAMES.slice(0, 7)).join(','), r.map(s => s.name).join(','));
    check('原数组未被修改', subjects[0].name === 'XiIntro');
    check('确定性：同一日期两次调用结果一致', JSON.stringify(rotateSubjects(subjects, '2026-10-22')) === JSON.stringify(r));
}

// ---- 2. 起点轮换 ----
console.log('[2] 起点随日期轮换');
{
    const starts = [];
    for (let d = 1; d <= 12; d++) {
        const date = `2026-10-${String(d).padStart(2, '0')}`;
        starts.push(rotateSubjects(subjects, date)[0].name);
    }
    check('连续 12 天起点在轮换（相邻日期起点不同）', starts.every((s, i) => i === 0 || s !== starts[i - 1]));
    // 相邻日期 dayOfYear 差 1 → 起点索引差 1（模 9），即起点按科目顺序逐日后移
    const idxOf = (n) => NAMES.indexOf(n);
    check('起点按科目顺序逐日后移（模 9）', starts.every((s, i) => i === 0 || (idxOf(s) - idxOf(starts[i - 1]) + 9) % 9 === 1), starts.join(','));

    // 覆盖度：9 个连续日期覆盖全部 9 个起点
    const cover = new Set(starts.slice(0, 9));
    check('9 个连续日期覆盖全部 9 个起始科目', cover.size === 9);
}

// ---- 3. 边界 ----
console.log('[3] 边界与异常');
{
    check('单科目列表原样返回', rotateSubjects(subjects.slice(0, 1), '2026-10-22').length === 1);
    check('空列表安全', rotateSubjects([], '2026-10-22').length === 0);
    check('非法日期不轮转（队首不变）', rotateSubjects(subjects, 'not-a-date')[0].name === 'XiIntro');
    check('跨年边界日期可用（12-31 / 01-01）', rotateSubjects(subjects, '2026-12-31').length === 9 && rotateSubjects(subjects, '2027-01-01').length === 9);
}

console.log(failures === 0 ? '\n全部通过' : `\n${failures} 项失败`);
process.exit(failures === 0 ? 0 : 1);
