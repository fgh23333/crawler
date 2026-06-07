/**
 * 将本地 JSON 文件批量上传到 Cloudflare R2
 *
 * 用法:
 *   node scripts/upload-r2.js
 *
 * 需要环境变量:
 *   CLOUDFLARE_ACCOUNT_ID
 *   CLOUDFLARE_API_TOKEN
 *
 * 或者直接用 wrangler:
 *   wrangler r2 object put question-bank/rewrite/Marx_singleChoice.json --file new/rewrite/Marx_singleChoice.json
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

const ACCOUNT_ID = process.env.CLOUDFLARE_ACCOUNT_ID;
const API_TOKEN = process.env.CLOUDFLARE_API_TOKEN;

if (!ACCOUNT_ID || !API_TOKEN) {
    console.error('请设置环境变量 CLOUDFLARE_ACCOUNT_ID 和 CLOUDFLARE_API_TOKEN');
    process.exit(1);
}

const ROOT = path.resolve(__dirname, '..');
const BUCKET = 'question-bank';

function uploadToR2(key, filePath) {
    return new Promise((resolve, reject) => {
        const content = fs.readFileSync(filePath);
        const urlPath = `/client/v4/accounts/${ACCOUNT_ID}/r2/buckets/${BUCKET}/objects/${key}`;

        const options = {
            hostname: 'api.cloudflare.com',
            path: urlPath,
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${API_TOKEN}`,
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(content),
            },
        };

        const req = https.request(options, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                if (res.statusCode === 200) {
                    resolve({ key, size: content.length });
                } else {
                    reject(new Error(`HTTP ${res.statusCode}: ${data}`));
                }
            });
        });

        req.on('error', reject);
        req.write(content);
        req.end();
    });
}

async function uploadDirectory(localDir, r2Prefix) {
    if (!fs.existsSync(localDir)) {
        console.log(`目录不存在，跳过: ${localDir}`);
        return 0;
    }

    const files = fs.readdirSync(localDir).filter(f => f.endsWith('.json'));
    let count = 0;

    for (const file of files) {
        const key = `${r2Prefix}/${file}`;
        const filePath = path.join(localDir, file);

        try {
            const result = await uploadToR2(key, filePath);
            count++;
            console.log(`  ✓ ${key} (${(result.size / 1024).toFixed(1)} KB)`);
        } catch (err) {
            console.error(`  ✗ ${key}: ${err.message}`);
        }
    }

    return count;
}

async function main() {
    console.log('开始上传数据到 R2...\n');

    // 上传 rewrite 数据
    console.log('--- Rewrite 数据 ---');
    const rewriteCount = await uploadDirectory(
        path.join(ROOT, 'new', 'rewrite'),
        'rewrite'
    );

    // 上传 cura 数据（从 Vue 项目复制过来或直接上传）
    console.log('\n--- Cura 试卷数据 ---');
    const curaLocal = path.join(ROOT, 'cura');
    const curaVue = path.resolve(ROOT, '..', 'Vue', 'crawlerVisualization', 'src', 'assets', 'cura');
    const curaDir = fs.existsSync(curaLocal) ? curaLocal : (fs.existsSync(curaVue) ? curaVue : null);
    const curaCount = curaDir ? await uploadDirectory(curaDir, 'cura') : 0;

    console.log(`\n完成！共上传 ${rewriteCount} 个 rewrite 文件, ${curaCount} 个 cura 文件`);
}

main();
