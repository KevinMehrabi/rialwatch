/* Browser behavior checks for the data-bound redesign.
 * node verify/redesign.cjs http://localhost:8000/
 * Set PW_CHROMIUM to reuse an installed Chromium binary.
 */
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const url = process.argv[2] || 'http://127.0.0.1:8765/';
const launch = process.env.PW_CHROMIUM ? {executablePath:process.env.PW_CHROMIUM} : {};
const fixtures = {
 rows: Array.from({length:40},(_,i)=>({ date:new Date(Date.UTC(2026,0,i+1+(i>15 ? 2 : 0))).toISOString().slice(0,10),fix:1000000+i*1000,p25:1000000+i*1000,p75:1000000+i*1000,status:'Green',withheld:i===38 })),
 cards: [
  {basket_name:'First',signal_label:'Network',weighted_rate:1200000,median_rate:1199000,spread_vs_benchmark_pct:40,usable_record_count:25,contributing_source_count:4,fresh_contributing_source_count:3,remembered_source_count:0,basket_confidence:82.5,freshness_status:'fresh',dispersion_level:'high',display_state:'publish',alignment_label:'Divergent'},
  {basket_name:'Second',signal_label:'Transfer',weighted_rate:900000,median_rate:901000,spread_vs_benchmark_pct:-32,usable_record_count:10,contributing_source_count:2,fresh_contributing_source_count:1,remembered_source_count:5,basket_confidence:65,freshness_status:'fresh',dispersion_level:'medium',display_state:'monitor',suppression_reason:'limited_coverage',alignment_label:'Mild divergence'},
  {basket_name:'Hidden',display_state:'hide'},
  {basket_name:'Excluded',render_on_homepage:false}
 ]
};
(async()=> {
 const browser=await chromium.launch(launch);
 try {
  const page=await browser.newPage({viewport:{width:390,height:900}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto(url);await page.waitForSelector('.regional-row');await page.waitForSelector('#historyRangePanel:not([hidden])');await page.evaluate(()=>document.fonts.ready);
  assert.equal(await page.locator('.indicator-card').count(),6);
  assert.equal(await page.locator('.indicator-sparkline-svg').count(),6);
  assert.equal(await page.locator('.reading-card').count(),4);
  assert.equal(await page.locator('[data-history-window]').count(),7);
  const calculations=await page.locator('.indicator-card').evaluateAll(cards=>cards.map(card=>{
   const number=text=>Number(text.match(/[+−-]?[\d,]+(?:\.\d+)?/)?.[0]?.replace(/,/g,'').replace('−','-'));
   return {value:number(card.querySelector('.indicator-value-main').textContent),values:[...card.querySelectorAll('.indicator-compare strong')].map(el=>number(el.textContent))};
  }));
  calculations.forEach(({value,values:[street,peer]},index)=>{
   if(![value,street,peer].every(Number.isFinite)) return;
   const expected=index<2 ? (street/peer-1)*100 : index<4 ? (peer/street-1)*100 : street-peer;
   assert.ok(Math.abs(value-expected)<(index<4 ? .051 : .5),`indicator ${index} must agree with its displayed comparison rates`);
  });
  assert.match(await page.locator('#regional-context').textContent(),/Diagnostics updated/);
  assert.equal(await page.locator('.ledger-table tbody tr:visible').count(),10);
  await page.getByRole('button',{name:'Show all 30'}).click();
  assert.equal(await page.locator('.ledger-table tbody tr:visible').count(),30);
  await page.getByRole('button',{name:'Show fewer'}).click();
  const fullEnd=Number(await page.locator('#historyRangeRightHandle').getAttribute('aria-valuenow'));
  await page.getByRole('button',{name:'7D',exact:true}).click();
  assert.equal(await page.getByRole('button',{name:'7D',exact:true}).getAttribute('aria-pressed'),'true');
  const start=Number(await page.locator('#historyRangeLeftHandle').getAttribute('aria-valuenow'));
  await page.locator('#historyRangeLeftHandle').focus();await page.keyboard.press('ArrowLeft');
  const manualStart=Number(await page.locator('#historyRangeLeftHandle').getAttribute('aria-valuenow'));
  assert.ok(manualStart<start,'ArrowLeft expands the selected history to the left');
  assert.ok(fullEnd-manualStart+1>=7,'manual ranges include at least seven observations, even after a sparse calendar preset');
  assert.equal(await page.locator('[data-history-window][aria-pressed="true"]').count(),0);
  await page.getByRole('button',{name:'Reset',exact:true}).click();
  const handle=await page.locator('#historyRangeLeftHandle').boundingBox();
  await page.mouse.move(handle.x+handle.width/2,handle.y+handle.height/2);await page.mouse.down();await page.mouse.move(handle.x+handle.width/2+80,handle.y+handle.height/2,{steps:8});await page.mouse.up();
  assert.ok(Number(await page.locator('#historyRangeLeftHandle').getAttribute('aria-valuenow'))>0,'left grip must drag');
  const right=await page.locator('#historyRangeRightHandle').boundingBox();
  await page.mouse.move(right.x+right.width/2,right.y+right.height/2);await page.mouse.down();await page.mouse.move(right.x+right.width/2-40,right.y+right.height/2,{steps:8});await page.mouse.up();
  assert.ok(Number(await page.locator('#historyRangeRightHandle').getAttribute('aria-valuenow'))<fullEnd,'right grip must drag');
  const beforePan=[Number(await page.locator('#historyRangeLeftHandle').getAttribute('aria-valuenow')),Number(await page.locator('#historyRangeRightHandle').getAttribute('aria-valuenow'))];
  const windowBox=await page.locator('#historyRangeWindow').boundingBox();
  await page.mouse.move(windowBox.x+windowBox.width/2,windowBox.y+20);await page.mouse.down();await page.mouse.move(windowBox.x+windowBox.width/2-20,windowBox.y+20,{steps:8});await page.mouse.up();
  const afterPan=[Number(await page.locator('#historyRangeLeftHandle').getAttribute('aria-valuenow')),Number(await page.locator('#historyRangeRightHandle').getAttribute('aria-valuenow'))];
  assert.ok(afterPan[0]<beforePan[0]);assert.equal(afterPan[1]-afterPan[0],beforePan[1]-beforePan[0]);
  await page.locator('#historyChart').focus();await page.keyboard.press('Home');
  assert.match(await page.locator('#readout-date').textContent(),/Selected/);
  assert.ok((await page.locator('#chart-announcement').textContent()).length>20);
  // Scrubbing only patches the readout/selection: the plot path must retain its identity.
  await page.evaluate(()=>{window.__plotPath=document.querySelector('#historyChart .chart-line')});
  const chart=await page.locator('#historyChart').boundingBox();await page.mouse.move(chart.x+100,chart.y+50);
  assert.equal(await page.evaluate(()=>window.__plotPath===document.querySelector('#historyChart .chart-line')),true);
  await page.locator('.regional-row').nth(1).click();
  assert.equal(await page.locator('.regional-mobile-detail:visible').count(),1);
  assert.equal(await page.locator('.regional-row').nth(1).getAttribute('aria-expanded'),'true');
  await page.setViewportSize({width:1280,height:900});
  assert.equal(await page.locator('.regional-row').nth(1).getAttribute('aria-pressed'),'true');
  await page.locator('.regional-row').nth(1).focus();await page.keyboard.press('ArrowDown');
  assert.equal(await page.locator('.regional-row').nth(2).getAttribute('aria-pressed'),'true');
  assert.equal(await page.locator('.ledger-table tbody tr:visible').count(),30);
  await page.getByRole('button',{name:'Switch to light mode'}).click();await page.reload();
  assert.equal(await page.locator('html').getAttribute('data-theme'),'light');
  await page.emulateMedia({reducedMotion:'reduce'});
  assert.equal(await page.locator('.chart-pulse').evaluate(el=>getComputedStyle(el).display),'none');
  // Check all requested widths on all shared pages, in both themes.
  for(const path of ['', 'methodology/', 'status/']) {
   for(const theme of ['dark','light']) {
    await page.evaluate(theme=>localStorage.setItem('rw-theme',theme),theme);
    for(const width of [320,360,375,390,430,768,900,1024,1280]) {
     await page.setViewportSize({width,height:900});await page.goto(new URL(path,url).href);
     await page.evaluate(()=>document.fonts.ready);
     assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth),true,`${path} ${theme} ${width} overflow`);
     assert.equal(await page.locator('.site-logo:visible').count(),1);
     const brand=await page.locator('.site-brand').boundingBox(),toggle=await page.locator('#theme-toggle').boundingBox();
     assert.ok(brand.x+brand.width<=toggle.x+1 || brand.y!==toggle.y,'header brand overlaps toggle');
     if(path==='') {
      const rate=await page.locator('.primary-rate-value').boundingBox();
      const chart=await page.locator('.history-chart-wrap').boundingBox();
      const band=await page.locator('.band-block').boundingBox();
      const preset=await page.locator('[data-history-window="7"]').boundingBox();
      assert.ok(chart.y>=rate.y+rate.height && chart.y<rate.y+rate.height+80,'chart belongs directly below the main rate');
      assert.ok(band.y>chart.y+chart.height,'band belongs below the chart');
      assert.ok(preset.height<=30 && preset.width<50,'range presets retain their compact dimensions');
     }
    }
   }
  }
  // Deterministic fixtures cover calendar gaps, withheld rows, filtering, diagnostics, and short/empty history.
  const fixturePage=await browser.newPage({viewport:{width:390,height:900}});fixturePage.on('pageerror',error=>errors.push(error.message));
  let rows=fixtures.rows, regional={generated_at:new Date().toISOString(),cards:fixtures.cards};
  let failFetch=false;
  await fixturePage.route('**/api/series.json',route=>failFetch ? route.fulfill({status:503,body:'unavailable'}) : route.fulfill({json:{rows}}));
  await fixturePage.route('**/api/regional_market_signals_card.json',route=>failFetch ? route.fulfill({status:503,body:'unavailable'}) : route.fulfill({json:regional}));
  await fixturePage.goto(url);await fixturePage.waitForSelector('.regional-row');
  assert.equal(await fixturePage.locator('.regional-row').count(),2);
  assert.match(await fixturePage.locator('.regional-row').first().textContent(),/First/);
  assert.match(await fixturePage.locator('.regional-mobile-detail').first().textContent(),/3 fresh \/ 4 used/);
  assert.doesNotMatch(await fixturePage.locator('.regional-mobile-detail').first().textContent(),/remembered/);
  assert.equal(await fixturePage.locator('.ledger-publication.withheld').count(),1);
  await fixturePage.getByRole('button',{name:'30D',exact:true}).click();
  assert.equal(Number(await fixturePage.locator('#historyRangeLeftHandle').getAttribute('aria-valuenow')),11,'presets use calendar days across gaps');
  await fixturePage.getByRole('button',{name:'1D',exact:true}).click();
  assert.equal(Number(await fixturePage.locator('#historyRangeRightHandle').getAttribute('aria-valuenow'))-Number(await fixturePage.locator('#historyRangeLeftHandle').getAttribute('aria-valuenow')),0,'1D must not include a withheld or older reference merely to draw a line');
  assert.match(await fixturePage.locator('#timeline-message').textContent(),/1 day has no published rate/);
  await fixturePage.locator('#historyRangeLeftHandle').press('ArrowRight');
  assert.ok(Number(await fixturePage.locator('#historyRangeRightHandle').getAttribute('aria-valuenow'))-Number(await fixturePage.locator('#historyRangeLeftHandle').getAttribute('aria-valuenow'))>=6,'manual range minimum is seven points');
  regional={...regional,generated_at:'2000-01-01T00:00:00Z'};await fixturePage.reload();await fixturePage.waitForSelector('.regional-row');
  assert.match(await fixturePage.locator('.regional-mobile-detail').first().textContent(),/0 fresh \/ 4 used/);
  assert.match(await fixturePage.locator('.regional-mobile-detail').first().textContent(),/stale diagnostics artifact/);
  for(const count of [1,2,6]) {
   rows=fixtures.rows.slice(0,count);await fixturePage.reload();await fixturePage.waitForSelector('#historyRangePanel:not([hidden])');
   await fixturePage.getByRole('button',{name:'7D',exact:true}).click();
   await fixturePage.locator('#historyRangeRightHandle').press('ArrowLeft');
   assert.equal(await fixturePage.locator('#historyChart').evaluate(el=>/NaN|Infinity/.test(el.innerHTML)),false);
  }
  rows=['2026-10-05','2026-10-06','2026-10-11'].map((date,i)=>({date,fix:2600000+i*10000,withheld:false}));
  await fixturePage.reload();await fixturePage.waitForSelector('#historyRangePanel:not([hidden])');
  const gapPath=await fixturePage.locator('#historyChart .chart-line').getAttribute('d');
  const coords=[...gapPath.matchAll(/[ML]([\d.]+),([\d.]+)/g)].map(m=>Number(m[1]));
  assert.equal((gapPath.match(/M/g)||[]).length,2,'missing days break the line');
  assert.ok(Math.abs((coords[1]-coords[0])/(coords[2]-coords[0])-1/6)<.001,'horizontal spacing follows calendar time');
  assert.match(await fixturePage.locator('#timeline-message').textContent(),/4 days have no published rate/);
  rows=[];regional={cards:[]};await fixturePage.reload();await fixturePage.getByText('Timeline unavailable',{exact:true}).waitFor();
  await fixturePage.getByText('Regional signals unavailable',{exact:true}).waitFor();
  failFetch=true;await fixturePage.reload();await fixturePage.getByText('Timeline unavailable',{exact:true}).waitFor();
  await fixturePage.getByText('Regional signals unavailable',{exact:true}).waitFor();
  // Site refresh time must not conceal the age of the benchmark snapshot.
  const asOf=Date.parse(await fixturePage.locator('#primary-benchmark').getAttribute('data-as-of'));
  for(const [ageHours,visible] of [[47,false],[48,true],[-1,false]]) {
   await fixturePage.clock.setFixedTime(new Date(asOf+ageHours*3600000));
   await fixturePage.reload();
   assert.equal(await fixturePage.locator('#benchmark-age').isVisible(),visible);
   if(visible) assert.match(await fixturePage.locator('#benchmark-age').textContent(),/2 days old/);
  }
  assert.equal(await fixturePage.locator('.updated-label').textContent(),'Site updated');
  assert.deepEqual(errors,[]);
  console.log('PASS: content, presets, drag/pan, keyboard, scrub, themes, reduced motion, shared pages, stale/withheld/missing data, and short series');
 } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1});
