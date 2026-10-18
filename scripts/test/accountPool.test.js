// accountPool.js 行为测试（随机洗牌 / 早停 / 剔除无效 / 快速失败 / 随机起点）
// 纯 node 运行，无第三方依赖：node scripts/test/accountPool.test.js
// 依赖 STUDENT_NUMS 环境变量，但测试内全部显式设置，不读真实学号池（敏感数据不进测试）。
const { createAccountPool, expandGrid, shuffle } = require('../lib/accountPool.js');

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
// FNV-1a 字符串哈希 → 均匀伪随机，用于模拟确定性"有效/无效"分布
const hash = (s) => { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; } return h; };

const FULL_SPEC = {
    cohorts: ['23', '24', '25', '26'],
    majors: ['11', '13', '15', '21', '22', '23', '25', '27', '29', '31', '33', '34', '35', '36', '40', '41', '42', '43', '44', '46', '50', '52', '60', '61', '62', '63', '64', '69'],
};

let failures = 0;
function check(name, cond, detail) {
    if (cond) { console.log(`  ✓ ${name}`); }
    else { failures++; console.error(`  ✗ ${name} ${detail || ''}`); }
}

(async () => {
    // ---- 1. shuffle：保持元素集合、且确实打乱 ----
    console.log('[1] shuffle');
    {
        const base = expandGrid({ cohorts: ['23', '24'], majors: ['11', '52'], classes: ['1', '2'], nums: ['01', '02'] });
        const a = shuffle(base.slice());
        const b = shuffle(base.slice());
        check('元素集合不变', a.sort().join(',') === base.slice().sort().join(','));
        check('两次洗牌顺序不同', a.join(',') !== b.join(',') || base.length < 2);
        // 大数组：前 50% 位置的重合度应接近 50%（完全随机），而非 100%（未洗牌）
        const big = expandGrid(FULL_SPEC);
        const s1 = shuffle(big.slice()).slice(0, big.length / 2);
        const s2 = shuffle(big.slice()).slice(0, big.length / 2);
        const set2 = new Set(s2);
        const inter = s1.filter(n => set2.has(n)).length;
        check('大数组前缀重合度接近 50%（洗牌充分）', Math.abs(inter / s1.length - 0.5) < 0.02, `实际 ${(inter / s1.length * 100).toFixed(1)}%`);
    }

    // ---- 2. 大网格 + 17.4% 有效率 → 早停 + 轮换均摊 ----
    console.log('[2] 大网格早停（16,800 账号，模拟 17.4% 有效）');
    {
        const grid = expandGrid(FULL_SPEC);
        check('网格展开 16,800', grid.length === 16800, `实际 ${grid.length}`);
        const validSet = new Set(grid.filter(n => hash(n) % 1000 < 174));
        const rate = validSet.size / grid.length;
        check('模拟有效率 ≈17.4%', Math.abs(rate - 0.174) < 0.01, `实际 ${(rate * 100).toFixed(1)}%`);

        process.env.STUDENT_NUMS = JSON.stringify(FULL_SPEC);
        let attempted = 0;
        const t0 = Date.now();
        const p = createAccountPool({
            login: async (num) => {
                await sleep(1); attempted++;
                if (validSet.has(num)) return 'sid-' + num;
                throw new Error('登录失败: 账号不存在');
            },
        });
        await p.warmup(() => {});
        check(`收集 ${p.pool.length} 个有效（目标 300，允许并发超采 ≤+8）`, p.pool.length >= 300 && p.pool.length <= 308, `实际 ${p.pool.length}`);
        check('全部有效', p.pool.every(n => validSet.has(n)));
        check('早停生效（尝试 ≈300/0.174 ≈ 1,724，远小于 16,800）', Math.abs(attempted - 300 / rate) / (300 / rate) < 0.15 && attempted < grid.length / 5, `尝试 ${attempted}`);

        // 轮换：3000 次调用均摊到每个有效账号（round-robin，随机起点）
        const counts = new Map();
        for (let i = 0; i < 3000; i++) {
            const n = p.nextStudentNum();
            counts.set(n, (counts.get(n) || 0) + 1);
        }
        const vals = [...counts.values()];
        check('轮换覆盖全部有效账号', counts.size === p.pool.length);
        check('轮换均摊（max-min ≤ 1）', Math.max(...vals) - Math.min(...vals) <= 1);
        check('总数守恒', vals.reduce((a, b) => a + b, 0) === 3000);
        console.log(`    尝试 ${attempted} 次 / ${grid.length}，耗时 ${Date.now() - t0}ms`);
    }

    // ---- 3. 随机噪声：两次运行收集到不同批次（大 universe 下交集应小） ----
    console.log('[3] 随机噪声：多次运行批次不同');
    {
        const validSet = new Set(expandGrid(FULL_SPEC).filter(n => hash(n) % 10 < 3)); // 30% 有效 → universe ~5,000
        const run = async () => {
            const p = createAccountPool({
                login: async (n) => { await sleep(0); if (validSet.has(n)) return 's'; throw new Error('登录失败: x'); },
            });
            await p.warmup(() => {});
            return new Set(p.pool);
        };
        const s1 = await run(), s2 = await run();
        const inter = [...s1].filter(n => s2.has(n)).length;
        // E[交集/批大小] = 300/universe ≈ 6%；宽放到 <25% 判定"批次确实轮换"
        check(`两次批次交集比例 < 25%（期望 ~6%）`, inter / Math.min(s1.size, s2.size) < 0.25, `交集 ${inter}/${s1.size} = ${(inter / s1.size * 100).toFixed(1)}%）`);
        console.log(`    批次1=${s1.size} 批次2=${s2.size} 交集=${inter}（${(inter / s1.size * 100).toFixed(1)}%）`);
    }

    // ---- 4. 全超时 → 快速失败 ----
    console.log('[4] 服务器整体不可用快速失败');
    {
        process.env.STUDENT_NUMS = JSON.stringify({ cohorts: ['23', '24'], majors: ['11', '52'] });
        let attempts4 = 0;
        const t0 = Date.now();
        const p = createAccountPool({ login: async () => { attempts4++; await sleep(5); throw new Error('timeout of 15000ms exceeded'); } });
        let threw = null;
        try { await p.warmup(() => {}); } catch (e) { threw = e; }
        check('抛出整体不可用错误', threw && /整体不可用/.test(threw.message));
        check('尝试次数 ≈60+并发余量', attempts4 >= 60 && attempts4 <= 75, `实际 ${attempts4}`);
        check('耗时 < 1s', Date.now() - t0 < 1000);
    }

    // ---- 5. 小数组池（目标未达但队列耗尽）→ 正常返回 ----
    console.log('[5] 小池子');
    {
        process.env.STUDENT_NUMS = JSON.stringify(['A1', 'A2', 'A3', 'A4', 'A5']);
        const p = createAccountPool({ login: async (n) => { await sleep(0); if (n === 'A2' || n === 'A4') throw new Error('登录失败: 无效'); return 'sid-' + n; } });
        await p.warmup(() => {});
        check('仅保留有效账号（顺序随机）', p.pool.slice().sort().join(',') === 'A1,A3,A5', `实际 ${p.pool.slice().sort().join(',')}`);
        const first = p.nextStudentNum();
        check('轮换起点在池内', p.pool.includes(first));
    }

    // ---- 6. #28 式混合场景（20% 超时 + 56% 无效 + 24% 有效）→ 不误杀 ----
    console.log('[6] 混合场景不误杀');
    {
        const grid = expandGrid({ cohorts: ['23', '24'], majors: ['11', '52'] });
        process.env.STUDENT_NUMS = JSON.stringify(grid);
        const p = createAccountPool({
            login: async (num) => {
                await sleep(1);
                const d = Number(num.slice(-1));
                if (d === 0 || d === 1) throw new Error('timeout of 15000ms exceeded'); // 超时
                if (d <= 6) throw new Error('登录失败: 账号不存在');                      // 无效
                return 'sid-' + num;                                                     // 有效
            },
        });
        await p.warmup(() => {});
        check('未误杀，收集到有效账号', p.pool.length > 0, `池 ${p.pool.length}`);
    }

    console.log(failures === 0 ? '\n全部通过' : `\n${failures} 项失败`);
    process.exit(failures === 0 ? 0 : 1);
})().catch(e => { console.error(e); process.exit(1); });
