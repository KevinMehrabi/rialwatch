/* Lighthouse accessibility audit in both persisted themes. */
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');
(async()=> {
 const {default:lighthouse}=await import('lighthouse');
 const url=process.argv[2] || 'http://localhost:8000/';
 const output=process.env.RW_AUDIT_OUTPUT || path.join(os.tmpdir(),'rialwatch-audits');
 fs.mkdirSync(output,{recursive:true});
 const data=fs.mkdtempSync(path.join(os.tmpdir(),'rialwatch-browser-'));
 const context=await chromium.launchPersistentContext(data,{...(process.env.PW_CHROMIUM ? {executablePath:process.env.PW_CHROMIUM} : {}),args:['--remote-debugging-port=9234'],headless:true});
 try {
  const page=await context.newPage();
  for(const theme of ['dark','light']) {
   await page.goto(url);await page.evaluate(theme=>localStorage.setItem('rw-theme',theme),theme);await page.reload();
   const {lhr}=await lighthouse(url,{port:9234,onlyCategories:['accessibility'],disableStorageReset:true,output:'json',logLevel:'error'});
   fs.writeFileSync(path.join(output,`${theme}.json`),JSON.stringify(lhr,null,2));
   const failures=Object.values(lhr.audits).filter(a=>a.score!==null && a.score<1);
   const score=Math.round(lhr.categories.accessibility.score*100);
   console.log(`${theme}: ${score}/100`);
   failures.forEach(a=>console.log(JSON.stringify({id:a.id,title:a.title,items:a.details?.items?.map(i=>({node:i.node?.snippet,explanation:i.node?.explanation}))})));
   if(score<95) process.exitCode=1;
  }
 } finally { await context.close();fs.rmSync(data,{recursive:true,force:true}); }
})().catch(error=>{console.error(error);process.exitCode=1});
