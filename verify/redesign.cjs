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
  assert.match(await page.locator('#recent-days-summary').textContent(),/^Showing 10 of \d+ daily references/);
  await page.getByRole('button',{name:'View 30 more'}).click();
  assert.equal(await page.locator('.ledger-table tbody tr:visible').count(),40);
  assert.match(await page.locator('#recent-days-summary').textContent(),/^Showing 40 of \d+ daily references/);
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
  assert.equal(await page.locator('.ledger-table tbody tr:visible').count(),40,'expanded history survives the desktop breakpoint');
  await page.getByRole('button',{name:'Switch to light mode'}).click();await page.reload();
  assert.equal(await page.locator('html').getAttribute('data-theme'),'light');
  await page.emulateMedia({reducedMotion:'reduce'});
  // The resize observer replaces SVG children; read the current pulse from its stable root.
  assert.equal(await page.locator('#historyChart').evaluate(el=>getComputedStyle(el.querySelector('.chart-pulse')).display),'none');
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
  let archiveRows=[], archiveUnavailable=false;
  let failFetch=false;
  await fixturePage.route('**/api/series.json',route=>failFetch ? route.fulfill({status:503,body:'unavailable'}) : route.fulfill({json:{rows}}));
  await fixturePage.route('**/api/chart_history.json',route=>archiveUnavailable ? route.fulfill({status:503,body:'unavailable'}) : route.fulfill({json:{rows:archiveRows}}));
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
  assert.equal(await fixturePage.locator('#timeline-message').isVisible(),false,'no permanent missing-days banner');
  await fixturePage.locator('#historyRangeLeftHandle').press('ArrowRight');
  assert.ok(Number(await fixturePage.locator('#historyRangeRightHandle').getAttribute('aria-valuenow'))-Number(await fixturePage.locator('#historyRangeLeftHandle').getAttribute('aria-valuenow'))>=6,'manual range minimum is seven points');
  regional={...regional,generated_at:'2000-01-01T00:00:00Z'};await fixturePage.reload();await fixturePage.waitForSelector('.regional-row');
  assert.match(await fixturePage.locator('.regional-mobile-detail').first().textContent(),/0 fresh \/ 4 used/);
  assert.match(await fixturePage.locator('.regional-mobile-detail').first().textContent(),/stale diagnostics artifact/);
  for(const count of [1,2,6]) {
   rows=fixtures.rows.slice(0,count);await fixturePage.reload();await fixturePage.waitForSelector('#historyRangePanel:not([hidden])');
   assert.equal(await fixturePage.locator('.ledger-table tbody tr:visible').count(),count);
   assert.equal(await fixturePage.locator('#recent-days-summary').textContent(),`Showing ${count} of ${count} daily references · Full history`);
   assert.equal(await fixturePage.locator('#ledger-more').isVisible(),false);
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
  assert.equal(await fixturePage.locator('#timeline-message').isVisible(),false);
  archiveRows=['2026-10-07','2026-10-08','2026-10-09','2026-10-10'].map((date,i)=>({date,fix:2600000+i*1000,kind:'historical_archive',p25:null,p75:null}));
  // An archive cannot override a published reference, even if its endpoint disagrees.
  archiveRows.push({date:'2026-10-11',fix:9,kind:'historical_archive'});
  await fixturePage.reload();await fixturePage.waitForSelector('#historyRangePanel:not([hidden])');
  assert.equal(await fixturePage.locator('.chart-archive-point').count(),0,'archive observations use the same line without separate markers');
  assert.equal(((await fixturePage.locator('#historyChart .chart-line').getAttribute('d')).match(/M/g)||[]).length,1,'verified archive quotes fill the chart gaps');
  assert.equal((await fixturePage.locator('#readout-value').textContent()).replace(/,/g,''),'2620000');
  assert.equal(await fixturePage.locator('.ledger-table tbody tr').count(),3,'archive points never appear as published fixes');
  await fixturePage.locator('#historyChart').press('End');await fixturePage.locator('#historyChart').press('ArrowLeft');
  assert.match(await fixturePage.locator('#readout-date').textContent(),/Oct 10, 2026 · Recovered/);
  assert.match(await fixturePage.locator('#readout-source').textContent(),/not a RialWatch daily fix/);
  assert.match(await fixturePage.locator('#readout-band').textContent(),/Bonbast archive/);
  assert.equal(await fixturePage.getByRole('link',{name:'Source record',exact:true}).getAttribute('href'),'/history/backfills/2026-10-10.json');
  await fixturePage.locator('#historyChart').press('End');
  assert.equal(await fixturePage.locator('#readout-source').isVisible(),false);
  // Hover details must be visible beside the point, not only in the distant readout.
  // Check both ends and an archive point at narrow/mobile/desktop sizes in both themes.
  for (const width of [320,390,1280]) {
   await fixturePage.setViewportSize({width,height:900});
   for (const theme of ['dark','light']) {
    await fixturePage.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);
    await fixturePage.getByRole('button',{name:'All',exact:true}).click();
    await fixturePage.locator('#historyChart').scrollIntoViewIfNeeded();
    const plot=await fixturePage.locator('#historyChart').boundingBox();
    const points=[...(await fixturePage.locator('#historyChart .chart-line').getAttribute('d')).matchAll(/[ML]([\d.]+),([\d.]+)/g)].map(m=>({x:Number(m[1]),y:Number(m[2])}));
    for (const [index,date,value,recovered] of [[0,'Oct 5, 2026','2,600,000',false],[2,'Oct 7, 2026','2,600,000',true],[6,'Oct 11, 2026','2,620,000',false]]) {
     await fixturePage.mouse.move(0,0);
     await fixturePage.mouse.move(plot.x+points[index].x,plot.y+points[index].y);
     const tooltip=fixturePage.getByRole('tooltip');
     assert.equal(await tooltip.isVisible(),true,`${width} ${theme}: hover tooltip must appear`);
     assert.match(await fixturePage.locator('#tooltip-date').textContent(),new RegExp(date));
     assert.equal(await fixturePage.locator('#tooltip-value').textContent(),value);
     assert.match(await tooltip.textContent(),recovered ? /Recovered quote[\s\S]*Bonbast archive/ : /Published reference/);
     assert.equal(await fixturePage.locator('#tooltip-value').textContent(),await fixturePage.locator('#readout-value').textContent());
     const box=await tooltip.boundingBox();
     assert.ok(box.x>=plot.x && box.x+box.width<=plot.x+plot.width+1 && box.y>=plot.y && box.y+box.height<=plot.y+plot.height+1,'tooltip stays inside the chart');
     await tooltip.hover();
     assert.equal(await tooltip.isVisible(),true,'tooltip remains available when hovered');
     assert.equal(await fixturePage.locator('#tooltip-value').textContent(),value,'hovering the tooltip preserves the selected point');
     await fixturePage.keyboard.press('Escape');
     assert.equal(await tooltip.isVisible(),false,'Escape dismisses pointer tooltip without focusing the plot');
    }
    await fixturePage.locator('#historyChart').press('Home');
    assert.equal(await fixturePage.getByRole('tooltip').isVisible(),true,'keyboard inspection shows the same tooltip');
    await fixturePage.keyboard.press('Tab');
    assert.equal(await fixturePage.getByRole('tooltip').isVisible(),false,'keyboard blur dismisses tooltip');
    await fixturePage.mouse.move(0,0);
    await fixturePage.mouse.move(plot.x+points[2].x,plot.y+points[2].y);
    await fixturePage.mouse.move(0,0);
    assert.equal(await fixturePage.getByRole('tooltip').isVisible(),false,'leaving the chart dismisses pointer tooltip');
   }
  }
  const touchPage=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
  touchPage.on('pageerror',error=>errors.push(error.message));
  await touchPage.goto(url);await touchPage.waitForSelector('#historyRangePanel:not([hidden])');
  await touchPage.locator('#historyChart').scrollIntoViewIfNeeded();
  const touchPlot=await touchPage.locator('#historyChart').boundingBox();
  await touchPage.touchscreen.tap(touchPlot.x+touchPlot.width/2,touchPlot.y+touchPlot.height/2);
  assert.equal(await touchPage.getByRole('tooltip').isVisible(),true,'touch tap opens point details');
  await touchPage.getByRole('button',{name:'7D',exact:true}).tap();
  assert.equal(await touchPage.getByRole('tooltip').isVisible(),false,'changing the range clears the tooltip');
  await touchPage.close();
  // Progressively reach every reference, including a partial final batch and a year boundary.
  const ledgerPage=await browser.newPage();ledgerPage.on('pageerror',error=>errors.push(error.message));
  const ledgerRows=Array.from({length:97},(_,i)=>({date:new Date(Date.UTC(2025,10,i+1)).toISOString().slice(0,10),fix:1000000+i*2000+(i>=70 ? 8000 : 0),withheld:i===90}));
  await ledgerPage.route('**/api/series.json',route=>route.fulfill({json:{rows:ledgerRows}}));
  await ledgerPage.route('**/api/chart_history.json',route=>route.fulfill({json:{rows:[]}}));
  const dates=ledgerRows.map(row=>row.date).reverse();
  for (const width of [390,1280]) {
   await ledgerPage.setViewportSize({width,height:900});await ledgerPage.goto(url);
   await ledgerPage.waitForSelector('.ledger-date');
   let shown=width<900 ? 10 : 30;
   assert.equal(await ledgerPage.locator('#recent-days-summary').textContent(),`Showing ${shown} of 97 daily references`);
   while(shown<97) {
    const previous=shown;
    await ledgerPage.getByRole('button',{name:`View ${Math.min(30,97-shown)} more`,exact:true}).click();
    shown=Math.min(97,shown+30);
    assert.equal(await ledgerPage.locator('.ledger-table tbody tr:visible').count(),shown);
    assert.equal(await ledgerPage.locator('#recent-days-summary').textContent(),`Showing ${shown} of 97 daily references${shown===97 ? ' · Full history' : ''}`);
    assert.deepEqual(await ledgerPage.locator('.ledger-date time').evaluateAll(nodes=>nodes.map(node=>node.dateTime)),dates.slice(0,shown),'no duplicates, missing rows, or reordering between batches');
    assert.equal(await ledgerPage.evaluate(()=>document.activeElement.getAttribute('href')),`/fix/${dates[previous]}/`,'focus begins at the newly revealed references');
    if(previous===(width<900 ? 10 : 30)) {
     await ledgerPage.setViewportSize({width:width<900 ? 1280 : 390,height:900});
     assert.equal(await ledgerPage.locator('.ledger-table tbody tr:visible').count(),shown,'resize keeps the revealed count');
     await ledgerPage.setViewportSize({width,height:900});
    }
   }
   assert.equal(await ledgerPage.locator('#ledger-more').isVisible(),false);
   assert.equal(await ledgerPage.locator('.ledger-publication.withheld').count(),1,'withheld entries retain their label and are counted as references');
   assert.match(await ledgerPage.locator('.ledger-date').last().textContent(),/2025/,'older years are explicit');
   assert.match(await ledgerPage.locator('.ledger-table tbody tr').last().textContent(),/1,000,000/);
   assert.match(await ledgerPage.locator('.ledger-table tbody tr').nth(30).textContent(),/\+2,000/,'change at a batch boundary uses the true preceding date');
   assert.equal(await ledgerPage.locator('.ledger-change-track span').first().getAttribute('style'),await ledgerPage.locator('.ledger-change-track span').nth(30).getAttribute('style'),'equal changes use the same bar scale across batches');
  }
  await ledgerPage.close();
  archiveUnavailable=true;await fixturePage.reload();await fixturePage.waitForSelector('#historyRangePanel:not([hidden])');
  assert.equal(((await fixturePage.locator('#historyChart .chart-line').getAttribute('d')).match(/M/g)||[]).length,2,'archive outages retain the official timeline and its real gaps');
  archiveUnavailable=false;archiveRows=[];
  rows=[];regional={cards:[]};await fixturePage.reload();await fixturePage.getByText('Timeline unavailable',{exact:true}).waitFor();
  await fixturePage.getByText('Regional signals unavailable',{exact:true}).waitFor();
  assert.equal(await fixturePage.locator('#recent-days-summary').textContent(),'History is temporarily unavailable');
  assert.equal(await fixturePage.locator('#ledger-more').isVisible(),false);
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
  console.log('PASS: full-history pagination and counts, content, presets, drag/pan, keyboard, scrub, hover/touch tooltips, themes, reduced motion, shared pages, stale/withheld/missing data, and short series');
 } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1});
