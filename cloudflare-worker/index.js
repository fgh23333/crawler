/**
 * Cloudflare Worker — 题库数据 API 网关
 *
 * 功能：
 *   1. 来源校验（Referer / Origin）
 *   2. IP 频率限制（基于 KV 计数器）
 *   3. 从 R2 读取数据并返回（带 CDN 缓存头）
 *   4. CORS 支持
 */

// ======================== 配置 ========================

// RATE_LIMIT 在 fetch handler 中从 env.RATE_LIMIT_PER_MINUTE 读取
const DEFAULT_RATE_LIMIT = 60; // 每分钟每 IP 最大请求数（默认）
const RATE_WINDOW = 60; // 限流窗口（秒）
const CACHE_MAX_AGE = 86400; // 数据缓存 1 天（数据一周才更新一次）

// ======================== CORS ========================

function corsHeaders(origin, allowedOrigins) {
  const allow = allowedOrigins.includes('*')
    ? '*'
    : (allowedOrigins.includes(origin) ? origin : '');
  return {
    'Access-Control-Allow-Origin': allow,
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Max-Age': '86400',
  };
}

// ======================== 来源校验 ========================

function checkOrigin(request, allowedOrigins) {
  if (allowedOrigins.includes('*')) return true;

  const origin = request.headers.get('Origin') || '';
  const referer = request.headers.get('Referer') || '';

  return allowedOrigins.some(allowed => {
    return origin.includes(allowed) || referer.includes(allowed);
  });
}

// ======================== 频率限制 ========================

/**
 * 基于 KV 的滑动窗口计数器
 * key 格式: rate:<ip>:<window_index>
 * 同一窗口内计数超过阈值则拒绝
 */
async function checkRateLimit(kv, ip, limit) {
  const windowIndex = Math.floor(Date.now() / 1000 / RATE_WINDOW);
  const key = `rate:${ip}:${windowIndex}`;

  const count = parseInt(await kv.get(key)) || 0;
  if (count >= limit) {
    return { allowed: false, remaining: 0 };
  }

  // 写入计数
  await kv.put(key, String(count + 1), { expirationTtl: RATE_WINDOW * 2 });

  return { allowed: true, remaining: limit - count - 1 };
}

// ======================== R2 数据读取 ========================

async function serveFromR2(bucket, key) {
  if (!key) return null;

  const object = await bucket.get(key);
  if (!object) return null;

  return object;
}

// ======================== 主处理逻辑 ========================

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const allowedOrigins = (env.ALLOWED_ORIGINS || '*').split(',').map(s => s.trim());
    const origin = request.headers.get('Origin') || '';

    // ---------- OPTIONS 预检 ----------
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: corsHeaders(origin, allowedOrigins),
      });
    }

    // ---------- 只允许 GET ----------
    if (request.method !== 'GET') {
      return new Response('Method Not Allowed', { status: 405 });
    }

    // ---------- 来源校验 ----------
    if (!checkOrigin(request, allowedOrigins)) {
      return new Response('Forbidden', {
        status: 403,
        headers: { 'Content-Type': 'text/plain' },
      });
    }

    // ---------- 频率限制 ----------
    const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
    const limit = parseInt(env.RATE_LIMIT_PER_MINUTE) || DEFAULT_RATE_LIMIT;
    const rateResult = await checkRateLimit(env.RATE_LIMIT, ip, limit);

    if (!rateResult.allowed) {
      return new Response('Too Many Requests', {
        status: 429,
        headers: {
          'Content-Type': 'text/plain',
          'Retry-After': String(RATE_WINDOW),
          ...corsHeaders(origin, allowedOrigins),
        },
      });
    }

    // ---------- 从 R2 读取数据 ----------
    // URL 路径格式: /rewrite/Marx_singleChoice.json 或 /version.json
    const key = url.pathname.slice(1); // 去掉开头的 /

    if (!key || key.includes('..')) {
      return new Response('Bad Request', { status: 400 });
    }

    const object = await serveFromR2(env.BUCKET, key);

    if (!object) {
      return new Response('Not Found', {
        status: 404,
        headers: corsHeaders(origin, allowedOrigins),
      });
    }

    // ---------- 返回数据 ----------
    const headers = {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': `public, max-age=${CACHE_MAX_AGE}`,
      'CDN-Cache-Control': `public, max-age=${CACHE_MAX_AGE}`,
      'X-RateLimit-Remaining': String(rateResult.remaining),
      ...corsHeaders(origin, allowedOrigins),
    };

    // 如果 R2 对象有 httpMetadata，使用它
    if (object.httpMetadata?.contentType) {
      headers['Content-Type'] = object.httpMetadata.contentType;
    }

    return new Response(object.body, { headers });
  },
};
