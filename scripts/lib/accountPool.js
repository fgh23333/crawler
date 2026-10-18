// 学号轮换池：加载（数组 / 网格规格两种格式）+ 并发预热登录。
// pipeline.js 与 pipeline-incremental.js 共用，单一实现。

const FALLBACK_STUDENT_NUM = '2352613';
const WARMUP_CONCURRENCY = 8; // 预热登录并发数（与抓取并发同量级，降低对服务器的瞬时压力）
const WARMUP_TARGET_VALID = 300; // 预热早停：收集到该数量有效学号即停止扫描，不再扫完整网格

// 快速失败阈值：尝试数达到该值、成功数为 0、且过半是超时 → 判定服务器整体不可用。
// 避免大池子（上万账号）× 15s 超时把整个 job 拖死在 warmup 阶段。
const EARLY_BAIL_ATTEMPTS = 60;

// Fisher-Yates 原地洗牌。
// 每次运行随机打乱扫描顺序，让早停收集到的有效学号在多次运行间均匀轮换，
// 避免总是集中使用网格顺序靠前的同一批账号。
function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

// 网格规格展开：届 × 专业 × 班 × 号 → 完整学号列表（去重、保序）。
// 大池子明文放不进 GitHub secret（48KB 上限，16,800 个账号约 185KB），
// 改存紧凑规格由代码展开。
function expandGrid(spec) {
    const classes = Array.isArray(spec.classes) && spec.classes.length ? spec.classes : ['1', '2', '3', '4', '5', '6'];
    const nums = Array.isArray(spec.nums) && spec.nums.length
        ? spec.nums
        : Array.from({ length: 25 }, (_, i) => String(i + 1).padStart(2, '0'));
    const list = [];
    for (const yy of spec.cohorts) {
        for (const major of spec.majors) {
            for (const cls of classes) {
                for (const num of nums) {
                    list.push(`${yy}${major}${cls}${num}`);
                }
            }
        }
    }
    return [...new Set(list)];
}

// 解析 STUDENT_NUMS 环境变量，支持两种格式：
// 1. JSON 字符串数组（旧格式）：'["2352613","2352614"]'
// 2. 网格规格对象（新格式）：'{"cohorts":["23","24"],"majors":["11","52"],"classes":["1","2"],"nums":["01","02"]}'
//    其中 classes/nums 可省略，默认 1-6 班 / 01-25 号。
// 未设置或格式非法时回退到单一内置学号。
function loadStudentNumPool() {
    const raw = process.env.STUDENT_NUMS;
    if (raw && raw.trim()) {
        try {
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed) && parsed.length > 0 &&
                parsed.every(x => typeof x === 'string' && x.trim() !== '')) {
                return parsed.map(x => x.trim());
            }
            if (parsed && typeof parsed === 'object' && !Array.isArray(parsed) &&
                Array.isArray(parsed.cohorts) && parsed.cohorts.length > 0 &&
                Array.isArray(parsed.majors) && parsed.majors.length > 0) {
                const grid = expandGrid(parsed);
                if (grid.length > 0) {
                    const clsCount = (Array.isArray(parsed.classes) && parsed.classes.length
                        ? parsed.classes : ['1', '2', '3', '4', '5', '6']).length;
                    const numCount = (Array.isArray(parsed.nums) && parsed.nums.length
                        ? parsed.nums : Array.from({ length: 25 })).length;
                    console.log(`[AccountPool] 网格规格展开为 ${grid.length} 个学号（` +
                        `${parsed.cohorts.length} 届 × ${parsed.majors.length} 专业 × ${clsCount} 班 × ${numCount} 号）`);
                    return grid;
                }
                console.warn('[AccountPool] 网格规格展开结果为空，回退到单一学号');
            } else {
                console.warn('[AccountPool] STUDENT_NUMS 既非字符串数组也非网格规格，回退到单一学号');
            }
        } catch (e) {
            console.warn(`[AccountPool] STUDENT_NUMS JSON 解析失败: ${e.message}，回退到单一学号`);
        }
    } else {
        console.warn('[AccountPool] 未设置 STUDENT_NUMS 环境变量，回退到单一学号');
    }
    return [FALLBACK_STUDENT_NUM];
}

// 创建账号池实例。login(num) 由调用方注入（两条 pipeline 的登录实现各自独立）。
// 返回 { pool, nextStudentNum, getSid, refreshSid, warmup }。
function createAccountPool({ login }) {
    const pool = loadStudentNumPool();
    const sidCache = new Map(); // 学号 → studentId
    let numIndex = 0;

    // Node 单线程，读 + 自增同步执行，高并发下也安全。
    // 池子内容经预热随机筛选 + 轮换起点随机化，请求不会总落在固定账号上。
    function nextStudentNum() {
        const num = pool[numIndex % pool.length];
        numIndex++;
        return num;
    }

    async function getSid(studentNum) {
        if (sidCache.has(studentNum)) return sidCache.get(studentNum);
        const sid = await login(studentNum);
        sidCache.set(studentNum, sid);
        return sid;
    }

    async function refreshSid(studentNum) {
        const sid = await login(studentNum);
        sidCache.set(studentNum, sid);
        return sid;
    }

    // 并发预热：按随机顺序逐个登录验证可用性，不可用账号各消耗一次登录即被剔除，
    // 收集满 WARMUP_TARGET_VALID 个有效学号即早停（不再扫完整个网格）。
    // logFn(msg) 由调用方传入（两条 pipeline 的 log 签名不同）。
    async function warmup(logFn) {
        const queue = shuffle(pool.slice()); // 不改变 pool 原始顺序，仅在副本上洗牌
        logFn(`开始登录验证 ${queue.length} 个学号（随机顺序，并发: ${WARMUP_CONCURRENCY}，收集 ${WARMUP_TARGET_VALID} 个有效即停）...`);
        const ok = [];
        let attempted = 0;
        let timeouts = 0;
        let aborted = false;
        let index = 0;

        async function runNext() {
            // 停止条件：队列扫完 / 已收集足够有效学号（早停）/ 服务器整体不可用
            while (index < queue.length && !aborted && ok.length < WARMUP_TARGET_VALID) {
                const num = queue[index++];
                attempted++;
                try {
                    const sid = await login(num);
                    sidCache.set(num, sid);
                    ok.push(num);
                    logFn(`${num} ✓ (studentId=${sid})`);
                } catch (e) {
                    if (/timeout/i.test(e.message)) timeouts++;
                    logFn(`${num} ✗ ${e.message}`);
                }
                // 服务器疑似整体不可用（零成功 + 过半超时）→ 快速失败
                if (ok.length === 0 && attempted >= EARLY_BAIL_ATTEMPTS &&
                    timeouts * 2 >= attempted) {
                    aborted = true;
                }
            }
        }

        const workers = [];
        for (let w = 0; w < Math.min(WARMUP_CONCURRENCY, queue.length); w++) {
            workers.push(runNext());
        }
        await Promise.all(workers);

        if (aborted) {
            throw new Error(`服务器疑似整体不可用（前 ${attempted} 个登录 ${timeouts} 个超时且零成功），已中止预热`);
        }
        const earlyStopped = index < queue.length;
        logFn(`登录完成: ${ok.length} 个有效（尝试 ${attempted}/${queue.length}${earlyStopped ? `，收集满 ${WARMUP_TARGET_VALID} 个已早停` : ''}）`);
        if (ok.length === 0) throw new Error('所有学号登录失败，无法继续');
        // 仅保留登录成功的学号；轮换起点随机化，避免每次都从同一账号开始
        pool.length = 0;
        pool.push(...ok);
        numIndex = Math.floor(Math.random() * pool.length);
        console.log(`[AccountPool] ${ok.length} 个学号进入轮换池`);
    }

    return { pool, nextStudentNum, getSid, refreshSid, warmup };
}

module.exports = { createAccountPool, expandGrid, shuffle, loadStudentNumPool, FALLBACK_STUDENT_NUM };
