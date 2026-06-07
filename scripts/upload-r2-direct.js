/**
 * 一次性脚本：将现有题库数据上传到 Cloudflare R2
 * 通过 Cloudflare REST API 直接上传，无需 wrangler
 *
 * 用法: node scripts/upload-r2-direct.js
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

const ACCOUNT_ID = 'bbd869342ef49cfea41170378427db5d';
const BUCKET = 'question-bank';

// 从环境变量获取 API Token，或留空由用户填写
const APIToken = process.env.CF_API_TOKEN;

if (!APIToken) {
    console.error('请设置环境变量 CF_API_TOKEN (Cloudflare API Token)');
    console.error('  或直接编辑此文件填入 token');
    process.exit(1);
}

const ROOT = path.resolve(__dirname, '..');

function upload(key, content) {
    return new Promise((resolve, reject) => {
        const urlPath = `/client/v4/accounts/${ACCOUNT_ID}/r2/buckets/${BUCKET}/objects/${key}`;
        const body = Buffer.isBuffer(content) ? content : Buffer.from(content, 'utf-8');

        const options = {
            hostname: 'api.cloudflare.com',
            path: urlPath,
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${APIToken}`,
                'Content-Type': 'application/json',
                'Content-Length': body.length,
            },
        };

        const req = https.request(options, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                if (res.statusCode === 200) {
                    resolve({ key, size: body.length });
                } else {
                    reject(new Error(`HTTP ${res.statusCode} for ${key}: ${data.slice(0, 200)}`));
                }
            });
        });
        req.on('error', reject);
        req.write(body);
        req.end();
    });
}

async function uploadDir(dir, prefix) {
    if (!fs.existsSync(dir)) {
        console.log(`  目录不存在，跳过: ${dir}`);
        return 0;
    }
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.json'));
    let ok = 0;
    for (const f of files) {
        try {
            await upload(`${prefix}/${f}`, fs.readFileSync(path.join(dir, f)));
            ok++;
            process.stdout.write(`\r  ${ok}/${files.length} 已上传`);
        } catch (err) {
            console.error(`\n  ✗ ${f}: ${err.message}`);
        }
    }
    console.log('');
    return ok;
}

async function main() {
    console.log('上传 rewrite 数据...');
    const r1 = await uploadDir(path.join(ROOT, 'new', 'rewrite'), 'rewrite');

    // 尝试上传 cura 数据（先从本项目找，再从 Vue 项目找）
    console.log('上传 cura 试卷...');
    const curaLocal = path.join(ROOT, 'cura');
    const curaVue = path.resolve(ROOT, '..', 'Vue', 'crawlerVisualization', 'src', 'assets', 'cura');
    const curaDir = fs.existsSync(curaLocal) ? curaLocal : curaVue;
    const r2 = await uploadDir(curaDir, 'cura');

    // 上传 version.json
    console.log('上传 version.json...');
    const version = { version: new Date().toISOString(), updated: new Date().toISOString() };
    await upload('version.json', JSON.stringify(version));

    console.log(`\n完成！rewrite: ${r1} 文件, cura: ${r2} 文件`);
}

main().catch(console.error);
