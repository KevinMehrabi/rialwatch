/* Regression: a returning browser must fetch a new release's chart and ledger.
 * Use a real HTTP cache: Playwright route interception disables browser caching.
 * node verify/dashboard-cache.cjs /path/to/built/site
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {chromium} = require('playwright');

(async () => {
  const site = path.resolve(process.argv[2]);
  const html = fs.readFileSync(path.join(site, 'index.html'), 'utf8');
  const originalRelease = html.match(/data-release="([^"]+)"/)?.[1];
  assert.ok(originalRelease && !originalRelease.includes('${'), 'homepage must render a release identifier');
  let revision = 1;
  const requests = [];
  const rows = [
    {date:'2026-11-23',fix:2738000,p25:2738000,p75:2741500,withheld:false},
    {date:'2026-11-24',fix:2842000,p25:2841225,p75:2843750,withheld:false}
  ];
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/') {
      res.writeHead(200, {'Content-Type':'text/html', 'Cache-Control':'no-store'});
      res.end(html.replaceAll(originalRelease, `release-${revision}`));
      return;
    }
    res.setHeader('Cache-Control', 'public, max-age=3600');
    if (url.pathname.startsWith('/api/')) {
      requests.push({revision, pathname:url.pathname, release:url.searchParams.get('release')});
      const payload = url.pathname === '/api/series.json' ? {rows:rows.slice(0, revision)} :
        url.pathname === '/api/chart_history.json' ? {rows:[]} : {cards:[]};
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify(payload));
      return;
    }
    const file = path.resolve(site, '.' + url.pathname);
    if (!file.startsWith(site + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      res.writeHead(404); res.end(); return;
    }
    if (file.endsWith('.js')) res.setHeader('Content-Type','text/javascript');
    if (file.endsWith('.css')) res.setHeader('Content-Type','text/css');
    res.end(fs.readFileSync(file));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch(process.env.PW_CHROMIUM ? {executablePath:process.env.PW_CHROMIUM} : {});
    const page = await browser.newPage({viewport:{width:1280,height:900}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.waitForFunction(() => document.querySelector('#recent-days-summary').textContent.includes('1 of 1'));
    assert.match(await page.locator('#readout-date').textContent(), /Nov 23/);
    revision = 2;
    await page.reload();
    await page.waitForFunction(() => document.querySelector('#recent-days-summary').textContent.includes('2 of 2'));
    assert.match(await page.locator('#readout-date').textContent(), /Nov 24/);
    assert.match(await page.locator('#historyChart').getAttribute('aria-label'), /2,842,000/);
    for (const endpoint of ['/api/series.json','/api/chart_history.json','/api/regional_market_signals_card.json']) {
      assert.ok(requests.some(r => r.pathname === endpoint && r.revision === 2 && r.release === 'release-2'), `${endpoint} must use the new release`);
    }
    assert.deepEqual(errors, []);
    console.log('PASS: cached first visit → new release reload updates chart, ledger and regional requests.');
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
