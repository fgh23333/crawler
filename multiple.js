const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

const baseUrl = 'http://222.73.57.153:6571';
const targetPath = '/examinationInfo/getPracticeInfo';
const headers = { 'Content-Type': 'application/json;charset=utf-8' };

const outputDir = path.join(__dirname, 'test');
if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

const MAX_WORKERS = 1024; // 限制最大并发数，避免服务器过载
const RETRY_LIMIT = 3; // 失败时最大重试次数

const paramsArr = [
    { name: 'XiIntro', params: { branchId: "1705139277953761280", chapterId: "", studentId: "1798906253557104911", subjectId: "1752935841845477392" } },
    { name: 'Marx', params: { branchId: "1705139277953761280", chapterId: "", studentId: "1798906253557104911", subjectId: "1748167277460586496" } },
    { name: 'MaoIntro', params: { branchId: "1705139277953761280", chapterId: "", studentId: "1798906253557104911", subjectId: "1748168736914800651" } },
    { name: 'Political', params: { branchId: "1705139277953761280", chapterId: "", studentId: "1798906253557104911", subjectId: "1781216923707506688" } },
    { name: 'CMH', params: { branchId: "1705139277953761280", chapterId: "", studentId: "1798906253557104911", subjectId: "1748168736914800640" } },
    { name: 'NCH', params: { branchId: "1705139277953761280", chapterId: "", studentId: "1798906253557104911", subjectId: "1776854236110258176" } },
    { name: 'SDH', params: { branchId: "1705139277953761280", chapterId: "", studentId: "1798906253557104911", subjectId: "1752935841845477376" } },
    { name: 'ORH', params: { branchId: "1705139277953761280", chapterId: "", studentId: "1798906253557104911", subjectId: "1752935841845477384" } },
    { name: 'CCPH', params: { branchId: "1705139277953761280", chapterId: "", studentId: "1798906253557104911", subjectId: "1798740810791911424" } }
];

if (isMainThread) {
    let results = {};
    let pendingRequests = paramsArr.length * 1000;
    let activeWorkers = 0;
    let queue = [];

    console.log(`Total Requests: ${pendingRequests}`);

    // 预先分配存储空间
    paramsArr.forEach(({ name }) => { results[name] = []; });

    // 生成任务队列
    paramsArr.forEach(({ name, params }) => {
        for (let i = 0; i < 1000; i++) {
            queue.push({ name, params, attempt: i, retryCount: 0 });
        }
    });

    function startWorker(task) {
        activeWorkers++;
        const worker = new Worker(__filename, { workerData: task });

        worker.on('message', ({ name, attempt, data }) => {
            results[name] = results[name].concat(data);
            console.log(`[${name}] Received data from attempt ${attempt}. Remaining: ${--pendingRequests}`);

            activeWorkers--;
            if (queue.length > 0) {
                startWorker(queue.shift()); // 继续执行下一个任务
            } else if (activeWorkers === 0 && pendingRequests === 0) {
                writeResults(); // 所有任务完成，写入文件
            }
        });

        worker.on('error', err => console.error(`Worker Error:`, err));

        worker.on('exit', code => {
            if (code !== 0) console.error(`Worker stopped with exit code ${code}`);
        });
    }

    function processQueue() {
        while (activeWorkers < MAX_WORKERS && queue.length > 0) {
            startWorker(queue.shift());
        }
    }

    function writeResults() {
        Object.entries(results).forEach(([name, data]) => {
            fs.writeFileSync(path.join(outputDir, `${name}.json`), JSON.stringify(data, null, 2));
        });
        console.log("All files written successfully.");
    }

    processQueue(); // 启动任务队列
} else {
    (async function fetchData() {
        const { name, params, attempt, retryCount } = workerData;

        try {
            const res = await axios.post(baseUrl + targetPath, JSON.stringify(params), { headers });
            const responseData = JSON.parse(res.data.data.paperStore.paperContent);
            const extractedData = [].concat(
                responseData.panduan.children,
                responseData.danxuan.children,
                responseData.duoxuan.children,
                responseData.tiankong.children
            );

            parentPort.postMessage({ name, attempt, data: extractedData });
        } catch (error) {
            console.error(`[${name}] Attempt ${attempt} failed (Retry: ${retryCount}): ${error.message}`);

            if (retryCount < RETRY_LIMIT) {
                parentPort.postMessage({ name, attempt, data: [], retryCount: retryCount + 1 });
            } else {
                console.error(`[${name}] Attempt ${attempt} failed after ${RETRY_LIMIT} retries.`);
                parentPort.postMessage({ name, attempt, data: [] }); // 避免阻塞，返回空数据
            }
        }
    })();
}
