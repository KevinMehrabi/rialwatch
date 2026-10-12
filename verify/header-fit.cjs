/* Layout acceptance check for the RialWatch redesign.
 *   npm i -D playwright            (once)
 *   node verify/header-fit.cjs http://localhost:8000/        # your dev server
 *   node verify/header-fit.cjs file:///abs/path/design/preview/mobile.html   # the reference
 * Optional: PW_CHROMIUM=/path/to/chromium  to use an already-installed browser.
 * Exits non-zero on any failure. Selectors are text/role based so they work on any markup. */
const { chromium } = require('playwright');
const url = process.argv[2];
if (!url) { console.error('usage: node header-fit.cjs <url>'); process.exit(2); }
const WIDTHS = [320, 360, 375, 390, 430, 768, 1024, 1280];
(async () => {
  const browser = await chromium.launch(process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM, args: ['--no-sandbox'] } : {});
  let failures = 0;
  const fail = (w, msg) => { failures++; console.log(`  FAIL @${w}: ${msg}`); };
  for (const w of WIDTHS) {
    const ctx = await browser.newContext({ viewport: { width: w, height: 900 } });
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(url, { waitUntil: 'load' });
    await page.waitForTimeout(600);
    console.log(`width ${w}`);
    // 1. no horizontal scroll, nothing sticking out of the viewport
    const o = await page.evaluate(() => {
      const vw = document.documentElement.clientWidth, bad = [];
      document.querySelectorAll('body *').forEach((e) => {
        if (e.closest('svg') && e.tagName.toLowerCase() !== 'svg') return;
        const r = e.getBoundingClientRect();
        if (r.width && (r.right > vw + 0.5 || r.left < -0.5)) bad.push(e.tagName.toLowerCase() + ':' + (e.innerText || '').trim().slice(0, 20));
      });
      return { vw, sw: document.documentElement.scrollWidth, bad: bad.slice(0, 5) };
    });
    if (o.sw > o.vw) fail(w, `horizontal scroll (scrollWidth ${o.sw} > ${o.vw})`);
    if (o.bad.length) fail(w, `elements outside viewport: ${o.bad.join(', ')}`);
    // 2. the seven header elements are present, visible and inside the viewport
    const need = {
      logo: page.getByRole('img', { name: /rialwatch/i }).first(),
      title: page.getByText(/^RialWatch$/).first(),
      subtitle: page.getByText('Independent Iranian Rial (IRR) Benchmark').first(),
      methodology: page.getByRole('link', { name: /methodology/i }).first(),
      status: page.getByRole('link', { name: /status/i }).first(),
      updated: page.getByText(/^(site )?updated$/i).first(),
      themeToggle: page.getByRole('button', { name: /switch to (light|dark) mode/i }).first(),
    };
    for (const [k, loc] of Object.entries(need)) {
      const box = await loc.boundingBox().catch(() => null);
      if (!box || box.width === 0) { fail(w, `header element missing/hidden: ${k}`); continue; }
      if (box.x < -0.5 || box.x + box.width > w + 0.5) fail(w, `header element outside viewport: ${k}`);
      if (['methodology', 'status', 'themeToggle'].includes(k) && box.height < 43.5) fail(w, `tap target < 44px: ${k} (${Math.round(box.height)}px)`);
    }
    if (errors.length) fail(w, `page errors: ${errors.join(' | ')}`);
    await ctx.close();
  }
  await browser.close();
  console.log(failures ? `\n${failures} failure(s)` : '\nall checks passed');
  process.exit(failures ? 1 : 0);
})();
