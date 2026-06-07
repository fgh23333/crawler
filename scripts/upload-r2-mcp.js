/**
 * 通过 Cloudflare REST API 批量上传 JSON 文件到 R2
 *
 * 用法 (二选一):
 *
 * 方式 1 - 环境变量传入 token:
 *   set CF_API_TOKEN=你的token
 *   node scripts/upload-r2-mcp.js
 *
 * 方式 2 - 直接用 wrangler (需先 wrangler login):
 *   wrangler r2 object put question-bank/rewrite/XiIntro_singleChoice.json --file new/rewrite/XiIntro_singleChoice.json
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

const ACCOUNT_ID = 'bbd869342ef49cfea41170378427db5d';
const BUCKET = 'question-bank';
const APIToken = process.env.CF_API_TOKEN;

if (!APIToken) {
    console.log('用法:');
    console.log('  set CF_API_TOKEN=你的Cloudflare_API_Token');
    console.log('  node scripts/upload-r2-mcp.js');
    console.log('');
    console.log('在 Cloudflare 控制台获取 API Token:');
    console.log('  https://dash.cloudflare.com/profile/api-tokens');
    console.log('  权限: Account > Cloudflare R2 > Edit');
    process.exit(1);
}

const ROOT = path.resolve(__dirname, '..');

function upload(key, filePath) {
    return new Promise((resolve, reject) => {
        const content = fs.readFileSync(filePath);
        const urlPath = `/client/v4/accounts/${ACCOUNT_ID}/r2/buckets/${BUCKET}/objects/${encodeURIComponent(key).replace(/%2F/g, '/')}`;

        const req = https.request({
            hostname: 'api.cloudflare.com',
            path: urlPath,
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${APIToken}`,
                'Content-Type': 'application/json',
                'Content-Length': content.length,
            },
        }, (res) => {
            let body = '';
            res.on('data', c => body += c);
            res.on('end', () => {
                res.statusCode === 200 ? resolve(key) : reject(new Error(`${res.statusCode} ${key}: ${body.slice(0, 150)}`));
            });
        });
        req.on('error', reject);
        req.write(content);
        req.end();
    });
}

async function main() {
    console.log('R2 批量上传工具\n');

    // 1. Rewrite 数据
    const rewriteDir = path.join(ROOT, 'new', 'rewrite');
    const rewriteFiles = fs.readdirSync(rewriteDir).filter(f => f.endsWith('.json') && !f.includes('subject'));
    console.log(`[1/3] 上传 rewrite 数据 (${rewriteFiles.length} 文件)...`);
    for (const f of rewriteFiles) {
        try {
            await upload(`rewrite/${f}`, path.join(rewriteDir, f));
            console.log(`  ✓ rewrite/${f}`);
        } catch (e) { console.error(`  ✗ ${e.message}`); }
    }

    // 2. Subject 数据（合集，较大）
    const subjectFiles = fs.readdirSync(rewriteDir).filter(f => f.endsWith('_subject.json'));
    console.log(`\n[2/3] 上传 subject 数据 (${subjectFiles.length} 文件)...`);
    for (const f of subjectFiles) {
        try {
            await upload(`rewrite/${f}`, path.join(rewriteDir, f));
            console.log(`  ✓ rewrite/${f}`);
        } catch (e) { console.error(`  ✗ ${e.message}`); }
    }

    // 3. Cura 试卷数据
    const curaVue = path.resolve(ROOT, '..', 'Vue', 'crawlerVisualization', 'src', 'assets', 'cura');
    if (fs.existsSync(curaVue)) {
        const curaFiles = fs.readdirSync(curaVue).filter(f => f.endsWith('.json'));
        console.log(`\n[3/3] 上传 cura 试卷 (${curaFiles.length} 文件)...`);
        for (const f of curaFiles) {
            try {
                await upload(`cura/${f}`, path.join(curaVue, f));
                console.log(`  ✓ cura/${f}`);
            } catch (e) { console.error(`  ✗ ${e.message}`); }
        }
    } else {
        console.log('\n[3/3] cura 目录不存在，跳过');
    }

    // 4. version.json
    const version = { version: new Date().toISOString(), updated: new Date().toISOString() };
    const tmpVer = path.join(ROOT, '_version_tmp.json');
    fs.writeFileSync(tmpVer, JSON.stringify(version));
    await upload('version.json', tmpVer);
    fs.unlinkSync(tmpVer);
    console.log(`\n✓ version.json 已更新`);

    console.log('\n上传完成！');
}

main().catch(console.error);
