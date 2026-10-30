(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const desktop = matchMedia('(min-width: 900px)');
  const svgNS = 'http://www.w3.org/2000/svg';
  const numeric = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value)) ? Number(value) : null;
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const fmt = value => numeric(value) === null ? 'Unavailable' : Number(value).toLocaleString('en-US', { maximumFractionDigits:0 });
  const signed = (value, digits = 0) => numeric(value) === null ? 'Unavailable' : `${value < 0 ? '−' : value > 0 ? '+' : ''}${Math.abs(value).toLocaleString('en-US', { minimumFractionDigits:digits, maximumFractionDigits:digits })}`;
  const direction = value => value > 0 ? 'up' : value < 0 ? 'down' : 'flat';
  const arrow = value => value > 0 ? '▲' : value < 0 ? '▼' : '–';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, character => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' })[character]);
  const dayTime = value => Date.parse(`${value}T00:00:00Z`);
  const date = (value, options = {}) => new Date(`${value}T00:00:00Z`).toLocaleDateString('en-US', { month:'short', day:'numeric', timeZone:'UTC', ...options });
  const timestamp = value => {
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return 'Unavailable';
    const day = parsed.toLocaleDateString('en-US', { month:'short', day:'numeric', year:'numeric', timeZone:'UTC' });
    const time = parsed.toLocaleTimeString('en-GB', { hour:'2-digit', minute:'2-digit', timeZone:'UTC' });
    return `${day} · ${time} UTC`;
  };
  const humanize = (value, fallback = 'Unavailable') => String(value || fallback).replace(/_/g, ' ');
  const textNumber = value => numeric(String(value).trim().replace(/,/g, '').replace(/−/g, '-'));
  const rateNumber = element => element ? textNumber(element.firstChild?.textContent || '') : null;
  const fetchJSON = async url => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  };

  function enhancePrimary() {
    const card = $('primary-benchmark');
    document.querySelectorAll('[data-format-timestamp]').forEach(element => {
      element.textContent = timestamp(element.dateTime);
    });
    const asOf = Date.parse(card.dataset.asOf);
    const wholeDays = Math.floor((Date.now() - asOf) / 86400000);
    if (Number.isFinite(wholeDays) && wholeDays >= 2) {
      const note = $('benchmark-age');
      note.textContent = `Benchmark data is ${wholeDays} days old. The site update time does not indicate a newer benchmark.`;
      note.hidden = false;
    }
    const fix = textNumber(card.dataset.fix), p25 = textNumber(card.dataset.p25), p75 = textNumber(card.dataset.p75);
    if (fix !== null && p25 !== null && p75 !== null) {
      $('band-gauge').hidden = false;
      $('band-gauge').querySelector('.band-marker').style.left = `${6 + 88 * (p75 === p25 ? .5 : clamp((fix - p25) / (p75 - p25), 0, 1))}%`;
    }
    const observed = $('primary-observed');
    if (/^\d{4}-\d{2}-\d{2}$/.test(observed.dataset.date) && /\d{2}:\d{2} UTC/.test(observed.textContent)) {
      observed.textContent = `${date(observed.dataset.date)} · ${observed.textContent}`;
    }
    // Keep the server's daily publication state, including withheld/candidate explanations.
    const change = $('primary-change');
    const match = change.textContent.match(/([+−-]?[\d,]+)\s*IRR per USD\s*\(([+−-]?[\d.]+)%\)/);
    if (match) renderPrimaryChange(textNumber(match[1]), Number(match[2].replace('−', '-')));
    document.querySelectorAll('.indicator-value-main').forEach(element => {
      element.textContent = element.textContent.replace(/^-/, '−');
      element.classList.toggle('unavailable', !/[0-9]/.test(element.textContent));
    });
    document.querySelectorAll('.indicator-sparkline-svg').forEach(svg => {
      const line = svg.querySelector('polyline');
      if (line && !svg.querySelector('polygon')) {
        const area = document.createElementNS(svgNS, 'polygon');
        area.setAttribute('points', `3,41 ${line.getAttribute('points')} 217,41`);
        svg.insertBefore(area, line);
      }
    });
  }
  function renderPrimaryChange(delta, percent) {
    $('primary-change').innerHTML = `<span class="change-pill direction-${direction(delta)}">${arrow(delta)} ${signed(percent, 1)}%</span><span><strong>${signed(delta)}</strong> IRR per USD <span class="text-secondary">· previous reference</span></span>`;
  }

  function renderLadder() {
    const values = {
      official: rateNumber(document.querySelector('[data-rate-layer="official"]')),
      crypto: rateNumber(document.querySelector('[data-rate-layer="crypto"]')),
      street: rateNumber(document.querySelector('.comparison-street')),
      transfer: rateNumber(document.querySelector('[data-rate-layer="transfer"]')),
      gold: rateNumber(document.querySelector('[data-rate-layer="gold"]'))
    };
    const svg = $('rate-ladder-svg'), available = Object.values(values).filter(v => v !== null && v > 0);
    if (!available.length) { svg.hidden = true; $('rate-ladder-caption').textContent = 'Rate ladder unavailable'; return; }
    const mobile = !desktop.matches, w = svg.clientWidth || 328, h = mobile ? 176 : 265;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    const pad = mobile ? 12 : 36, axis = mobile ? 78 : 134;
    const lo = Math.min(1600000, Math.max(1, Math.floor(Math.min(...available) / 500000) * 500000));
    const hi = Math.max(4000000, Math.ceil(Math.max(...available) / 500000) * 500000);
    const x = v => pad + Math.log(v / lo) / Math.log(hi / lo) * (w - 2 * pad);
    const short = (v, digits = 2) => `${(v / 1e6).toFixed(digits).replace(/0+$/, '').replace(/\.$/, '')}M`;
    const tx = (px, y, text, anchor = 'middle', bold = false) => `<text x="${px}" y="${y}" text-anchor="${anchor}" font-size="${mobile ? 11 : 20}"${bold ? ' font-weight="700"' : ''}>${esc(text)}</text>`;
    let html = `<line x1="${pad}" x2="${w-pad}" y1="${axis}" y2="${axis}" stroke="var(--bd)" stroke-width="2"/>`;
    const tickStep = Math.max(1, Math.ceil((hi - lo) / 5000000)) * 1000000;
    for (let v = Math.ceil(lo / tickStep) * tickStep; v <= hi; v += tickStep) {
      html += `<line x1="${x(v)}" x2="${x(v)}" y1="${axis-5}" y2="${axis+5}" stroke="var(--bd)"/>${tx(x(v), axis+22, short(v, 1))}`;
    }
    const marker = (key, label, color, radius, labelX, labelY, anchor = 'middle') => {
      const v = values[key]; if (!(v > 0)) return;
      const px = x(v);
      if (key === 'crypto' || key === 'transfer') html += `<line x1="${px}" y1="${axis-5}" x2="${labelX}" y2="${labelY+6}" stroke="var(--mut)"/>`;
      html += `<circle cx="${px}" cy="${axis}" r="${radius}" fill="${color}" stroke="${key==='street' ? 'var(--strong)' : 'var(--mut)'}" stroke-width="${key==='street' ? 2.5 : 1.5}"/>`;
      html += tx(labelX, labelY, `${label} ${short(v, key==='street' ? 3 : 2)}`, anchor, key==='street');
    };
    if (values.official > 0) marker('official','Official','var(--s3)',6, mobile ? pad : clamp(x(values.official), 80, w-80), axis+54, mobile ? 'start' : 'middle');
    if (values.gold > 0) marker('gold','Gold-implied','var(--gold)',6, mobile ? w-pad : clamp(x(values.gold), 110, w-110), axis+(mobile ? 85 : 54), mobile ? 'end' : 'middle');
    if (values.crypto > 0) marker('crypto','Crypto','var(--s1)',5, clamp(x(values.crypto)-28, mobile ? 95 : 130, w*.55), axis-48,'end');
    if (values.transfer > 0) marker('transfer','Transfer','var(--s1)',5, clamp(x(values.transfer)+28, w*.45, w-(mobile ? 100 : 145)),axis-48,'start');
    if (values.street > 0) marker('street','Street','var(--acc)',8, clamp(x(values.street), mobile ? 165 : 100, w-(mobile ? 55 : 100)),axis+54);
    for (const [a,b,pct] of [[values.official,values.street, values.street/values.official-1],[values.street,values.gold, values.gold/values.street-1]]) {
      if (!(a > 0 && b > 0)) continue;
      const chipWidth = mobile ? 58 : 84;
      const middle = clamp((x(a)+x(b))/2, chipWidth/2+2, w-chipWidth/2-2), y = axis-(mobile ? 30 : 42);
      html += `<rect x="${middle-chipWidth/2}" y="${y}" width="${chipWidth}" height="${mobile ? 22 : 30}" rx="15" fill="var(--gold-t)" stroke="var(--gold)"/><text x="${middle}" y="${y+(mobile ? 15 : 21)}" text-anchor="middle" style="fill:var(--gold)" font-size="${mobile ? 11 : 16}" font-weight="700">${signed(pct*100,1)}%</text>`;
    }
    svg.innerHTML = html;
    svg.setAttribute('aria-label', `Rate ladder on a log scale, IRR per USD: ${Object.entries(values).map(([key,v]) => `${key === 'gold' ? 'gold-implied' : key} ${fmt(v)}`).join(', ')}`);
    let caption = 'IRR per USD, log scale.';
    if ([values.street, values.crypto, values.transfer].every(v => v > 0)) {
      const trio = [values.street, values.crypto, values.transfer];
      const difference = (Math.max(...trio) / Math.min(...trio) - 1) * 100;
      if (difference <= 2) caption += ` The street, crypto and transfer rates sit within about ${Math.ceil(difference)}% of each other.`;
    }
    if (available.length < 5) caption += ' Unavailable rate layers are omitted.';
    $('rate-ladder-caption').textContent = caption;
  }

  class Timeline {
    constructor(rows) {
      this.rows = rows; this.times = rows.map(row => dayTime(row.date)); this.start = 0; this.end = rows.length-1; this.preset = 'all'; this.selected = null;
      this.svg = $('historyChart'); this.tooltip = $('history-chart-tooltip'); this.track = $('historyRangeBrushTrack'); this.minPoints = Math.min(7, rows.length);
      $('timeline-message').hidden = true; $('historyRangePanel').hidden = false;
      document.querySelectorAll('[data-history-window]').forEach(button => {
        button.disabled = false;
        button.addEventListener('click', () => this.setPreset(button.dataset.historyWindow));
      });
      $('historyRangeReset').addEventListener('click', () => this.setPreset('all'));
      this.svg.addEventListener('pointermove', event => this.scrub(event));
      this.svg.addEventListener('pointerdown', event => this.scrub(event));
      // Keep the tooltip open when the pointer moves from the plot onto its text.
      this.svg.parentElement.addEventListener('pointerleave', event => {
        if (event.pointerType === 'mouse' && this.selectionMode !== 'keyboard') { this.selected = null; this.updateSelection(); }
      });
      this.svg.addEventListener('pointerup', () => this.announce());
      document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && this.selected !== null) {
          this.selected = null; this.updateSelection();
          if (document.activeElement === this.svg) this.announce();
        }
      });
      this.svg.addEventListener('blur', () => {
        if (this.selectionMode === 'keyboard') { this.selected = null; this.updateSelection(); }
      });
      this.svg.addEventListener('keydown', event => {
        if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
        event.preventDefault();
        this.selectionMode = 'keyboard';
        if (event.key === 'Home') this.selected = this.start;
        else if (event.key === 'End') this.selected = this.end;
        else this.selected = clamp((this.selected ?? this.end) + (event.key === 'ArrowRight' ? 1 : -1), this.start, this.end);
        this.updateSelection(); this.announce();
      });
      for (const [id,mode] of [['historyRangeLeftHandle','start'],['historyRangeRightHandle','end'],['historyRangeWindow','pan']]) {
        const target = $(id);
        target.addEventListener('pointerdown', event => this.beginDrag(event, mode));
        if (mode !== 'pan') target.addEventListener('keydown', event => this.keyRange(event, mode));
      }
      const all = rows.map(row => row.fix), min = Math.min(...all), span = Math.max(1, Math.max(...all)-min);
      const points = all.map((v,i) => `${i && this.times[i]-this.times[i-1]===86400000 ? 'L' : 'M'}${this.fraction(i)*100},${36-(v-min)/span*32}`).join(' ');
      $('historyRangeOverviewLine').setAttribute('d', points);
      $('historyRangeOverviewArea').setAttribute('d', '');
      this.observer = new ResizeObserver(() => this.render()); this.observer.observe(this.svg);
      this.render();
    }
    fraction(index) {
      const span = this.times.at(-1)-this.times[0];
      return span ? (this.times[index]-this.times[0])/span : .5;
    }
    nearest(time, start = 0, end = this.rows.length-1) {
      let best = start;
      for (let i=start+1; i<=end; i++) if (Math.abs(this.times[i]-time)<Math.abs(this.times[best]-time)) best=i;
      return best;
    }
    setPreset(preset) {
      this.preset = preset; this.end = this.rows.length-1; this.start = 0; this.selected = null;
      if (preset !== 'all') {
        const cutoff = Date.parse(`${this.rows[this.end].date}T00:00:00Z`) - Number(preset)*86400000;
        this.start = this.rows.findIndex(row => Date.parse(`${row.date}T00:00:00Z`) >= cutoff);
        this.start = Math.max(0, this.start);
      }
      this.render(); this.announce();
    }
    render() {
      const w = this.svg.clientWidth || 320, h = this.svg.clientHeight || 224;
      const left = desktop.matches ? 64 : 54, right = w-10, top = 14, bottom = h-30;
      this.geometry = {w,h,left,right,top,bottom};
      const visible = this.rows.slice(this.start, this.end+1), values = visible.map(row => row.fix);
      const min = Math.min(...values), max = Math.max(...values), pad = Math.max((max-min)*.1,8000);
      const firstTime = this.preset && this.preset !== 'all' ? Math.max(this.times[0],this.times[this.end]-Number(this.preset)*86400000) : this.times[this.start];
      const lastTime = this.times[this.end];
      this.domain = {firstTime,lastTime};
      const xTime = time => left + (lastTime===firstTime ? .5 : (time-firstTime)/(lastTime-firstTime))*(right-left);
      this.x = i => xTime(this.times[i]);
      this.y = v => bottom - (v-(min-pad))/(max-min+2*pad)*(bottom-top);
      this.svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      const ticks = desktop.matches ? 5 : 4, decimals = max-min < 120000 ? 3 : max-min < 1500000 ? 2 : 1;
      let html = '';
      for (let i=0; i<ticks; i++) {
        const value = max+pad-i/(ticks-1)*(max-min+2*pad), py = this.y(value);
        html += `<line class="chart-grid" x1="${left}" x2="${right}" y1="${py}" y2="${py}"/><text x="${left-8}" y="${py+4}" text-anchor="end">${(value/1e6).toFixed(decimals)}M</text>`;
      }
      const segments = [];
      visible.forEach((row,i) => {
        const index = i+this.start;
        if (!i || this.times[index]-this.times[index-1]>86400000) segments.push([]);
        segments.at(-1).push(index);
      });
      const path = segment => segment.map((index,i) => `${i ? 'L':'M'}${this.x(index).toFixed(2)},${this.y(this.rows[index].fix).toFixed(2)}`).join(' ');
      html += `<path class="chart-area" d="${segments.filter(part=>part.length>1).map(part=>`${path(part)} L${this.x(part.at(-1))},${bottom} L${this.x(part[0])},${bottom} Z`).join(' ')}"/><path class="chart-line" d="${segments.map(path).join(' ')}"/>`;
      visible.forEach((row,i) => {
        if (row.kind === 'historical_archive') html += `<circle class="chart-archive-point" cx="${this.x(i+this.start)}" cy="${this.y(row.fix)}" r="3.5"/>`;
      });
      segments.filter(part=>part.length===1 && part[0]!==this.end).forEach(part=>{ html += `<circle class="chart-end" cx="${this.x(part[0])}" cy="${this.y(this.rows[part[0]].fix)}" r="3"/>`; });
      const labelTimes = [...new Set(Array.from({length:ticks}, (_,i) => firstTime+Math.round(i/(ticks-1)*(lastTime-firstTime)/86400000)*86400000))];
      for (const [i,time] of labelTimes.entries()) html += `<text x="${xTime(time)}" y="${h-6}" text-anchor="${i===0 ? 'start' : i===labelTimes.length-1 ? 'end' : 'middle'}">${date(new Date(time).toISOString().slice(0,10))}</text>`;
      $('timeline-message').hidden = true;
      $('timeline-message').textContent = '';
      if (this.end === this.rows.length-1) html += `<circle class="chart-pulse" cx="${this.x(this.end)}" cy="${this.y(this.rows[this.end].fix)}" r="5"/>`;
      html += `<circle class="chart-end" cx="${this.x(this.end)}" cy="${this.y(this.rows[this.end].fix)}" r="4.5"/><g id="chart-crosshair" visibility="hidden"><line class="chart-crosshair" y1="${top}" y2="${bottom}"/><circle class="chart-selection" r="5"/></g>`;
      this.svg.innerHTML = html;
      this.svg.setAttribute('aria-label', `Street rate from ${date(this.rows[this.start].date, {year:'numeric'})} to ${date(this.rows[this.end].date, {year:'numeric'})}. Latest ${fmt(this.rows.at(-1).fix)} IRR per USD. Use left and right arrow keys to explore.`);
      this.updateNavigator(); this.updateSelection();
    }
    updateSelection() {
      const index = this.selected ?? this.end, row = this.rows[index], prior = this.rows[index-1];
      const crosshair = $('chart-crosshair'), show = this.selected !== null;
      crosshair.setAttribute('visibility', show ? 'visible' : 'hidden');
      if (show) {
        const x = this.x(index), y = this.y(row.fix);
        crosshair.querySelector('line').setAttribute('x1',x); crosshair.querySelector('line').setAttribute('x2',x);
        crosshair.querySelector('circle').setAttribute('cx',x); crosshair.querySelector('circle').setAttribute('cy',y);
      }
      const recovered = row.kind === 'historical_archive';
      $('readout-date').textContent = `${show ? 'Selected' : index===this.rows.length-1 ? 'Latest' : 'Range end'} · ${date(row.date, {year:'numeric'})}${recovered ? ' · Recovered' : ''}`;
      $('readout-value').textContent = fmt(row.fix);
      const delta = prior ? row.fix-prior.fix : null;
      $('readout-change').className = `direction-${direction(delta)}`;
      const comparison = recovered || prior?.kind === 'historical_archive' ? 'previous chart point' : 'previous reference';
      $('readout-change').textContent = delta === null ? 'First reference' : `${arrow(delta)} ${signed(delta)}${desktop.matches ? ` vs ${comparison}` : ''}`;
      $('readout-band').textContent = recovered ? 'Bonbast archive · buy/sell midpoint' : numeric(row.p25) !== null && numeric(row.p75) !== null ? `Band ${fmt(row.p25)} – ${fmt(row.p75)}` : 'Band unavailable';
      const source = $('readout-source');
      source.hidden = !recovered;
      source.innerHTML = recovered ? `Historical market quote recovered after the fact; not a RialWatch daily fix. <a href="/history/backfills/${row.date}.json">Source record</a> · <a href="/methodology/#historical-recovery">Method</a>` : '';
      this.tooltip.hidden = !show;
      if (show) {
        $('tooltip-date').textContent = `${date(row.date, {year:'numeric'})} · ${recovered ? 'Recovered quote' : 'Published reference'}`;
        $('tooltip-value').textContent = fmt(row.fix);
        $('tooltip-change').className = `direction-${direction(delta)}`;
        $('tooltip-change').textContent = delta === null ? 'First reference' : `${arrow(delta)} ${signed(delta)} vs ${comparison}`;
        $('tooltip-detail').textContent = $('readout-band').textContent;
        // Anchor to the selected point and flip at the plot edges, including after resize.
        const x = this.x(index), y = this.y(row.fix), {w,h} = this.geometry;
        const width = this.tooltip.offsetWidth, height = this.tooltip.offsetHeight;
        this.tooltip.style.left = `${clamp(x+width+12 <= w-8 ? x+12 : x-width-12,8,Math.max(8,w-width-8))}px`;
        this.tooltip.style.top = `${clamp(y-height-12 >= 8 ? y-height-12 : y+12,8,Math.max(8,h-height-8))}px`;
      }
    }
    scrub(event) {
      const rect = this.svg.getBoundingClientRect(), x = event.clientX-rect.left;
      if (x < this.geometry.left || x > this.geometry.right) return;
      const time = this.domain.firstTime+(x-this.geometry.left)/(this.geometry.right-this.geometry.left)*(this.domain.lastTime-this.domain.firstTime);
      const index = this.nearest(time,this.start,this.end);
      this.selectionMode = event.pointerType;
      if (this.selected !== index) { this.selected = index; this.updateSelection(); }
    }
    announce() {
      $('chart-announcement').textContent = `${$('readout-date').textContent}: ${$('readout-value').textContent} IRR per USD. ${$('readout-change').textContent}. ${$('readout-band').textContent}. ${$('historyRangeSummary').textContent}.`;
    }
    updateNavigator() {
      const max = this.rows.length-1, left = max ? this.fraction(this.start)*100 : 0, right = max ? this.fraction(this.end)*100 : 100;
      $('historyRangeDimLeft').style.width = `${left}%`; $('historyRangeDimRight').style.width = `${100-right}%`;
      $('historyRangeWindow').style.left = `${left}%`; $('historyRangeWindow').style.width = `${right-left}%`;
      for (const [id,index,pos] of [['historyRangeLeftHandle',this.start,left],['historyRangeRightHandle',this.end,right]]) {
        const handle = $(id); handle.style.left = `${pos}%`;
        handle.setAttribute('aria-valuemin','0'); handle.setAttribute('aria-valuemax',max); handle.setAttribute('aria-valuenow',index);
        handle.setAttribute('aria-valuetext',date(this.rows[index].date, {year:'numeric'}));
      }
      document.querySelectorAll('[data-history-window]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.historyWindow===this.preset)));
      const label = this.preset === 'all' ? 'Full history' : this.preset === null ? 'Custom range' : this.preset === '365' ? '1Y' : `${this.preset}D`;
      $('historyRangeSummary').textContent = `${label}: ${date(new Date(this.domain.firstTime).toISOString().slice(0,10), this.preset==='all' ? {year:'numeric'} : {})} – ${date(this.rows[this.end].date, this.preset==='all' ? {year:'numeric'} : {})}`;
    }
    setManual(start, end) {
      this.start = start; this.end = end; this.preset = null; this.selected = null; this.render();
    }
    keyRange(event, mode) {
      if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(event.key)) return;
      event.preventDefault();
      const max = this.rows.length-1, step = (event.shiftKey ? 7 : 1)*(['ArrowLeft','ArrowDown'].includes(event.key) ? -1 : 1);
      let start = this.start, end = this.end;
      if (mode==='start') {
        end = Math.max(end,this.minPoints-1);
        start = clamp(event.key==='Home' ? 0 : event.key==='End' ? end : start+step, 0, end-this.minPoints+1);
      } else {
        start = Math.min(start,max-this.minPoints+1);
        end = clamp(event.key==='Home' ? start : event.key==='End' ? max : end+step, start+this.minPoints-1, max);
      }
      this.setManual(start,end); this.announce();
    }
    beginDrag(event, mode) {
      if (event.button !== 0) return;
      event.preventDefault();
      const target = event.currentTarget, rect = this.track.getBoundingClientRect(), max = this.rows.length-1;
      let start = this.start, end = this.end;
      if (end-start+1<this.minPoints) { start = clamp(start,0,max-this.minPoints+1); end = start+this.minPoints-1; }
      const origin = event.clientX;
      target.setPointerCapture(event.pointerId);
      let frame = null, pending = null;
      const apply = () => {
        frame = null;
        const shiftTime = (pending-origin)/rect.width*(this.times.at(-1)-this.times[0]);
        const originIndex = mode==='end' ? end : start;
        const shift = this.nearest(this.times[originIndex]+shiftTime)-originIndex;
        let a = start, b = end;
        if (mode==='start') a = clamp(start+shift,0,end-this.minPoints+1);
        else if (mode==='end') b = clamp(end+shift,start+this.minPoints-1,max);
        else { a = clamp(start+shift,0,max-(end-start)); b = a+(end-start); }
        this.setManual(a,b);
      };
      const move = e => { pending = e.clientX; if (frame===null) frame = requestAnimationFrame(apply); };
      const finish = () => {
        if (frame!==null) { cancelAnimationFrame(frame); apply(); }
        target.removeEventListener('pointermove',move); target.removeEventListener('pointerup',finish); target.removeEventListener('pointercancel',finish);
        this.announce();
      };
      target.addEventListener('pointermove',move); target.addEventListener('pointerup',finish); target.addEventListener('pointercancel',finish);
    }
  }

  function renderLedger(rows) {
    const history = rows.map((row,index) => ({...row, delta: index && numeric(row.fix)!==null && numeric(rows[index-1].fix)!==null ? row.fix-rows[index-1].fix : null})).reverse();
    const holder = $('recent-days'), summary = $('recent-days-summary'), more = $('ledger-more');
    const batchSize = 30;
    const max = Math.max(1,...history.map(row=>Math.abs(row.delta || 0)));
    let shown = Math.min(history.length, desktop.matches ? batchSize : 10), expanded = false;
    const header = '<thead><tr><th scope="col">DATE</th><th scope="col">FIX</th><th scope="col">CHANGE</th></tr></thead>';
    const renderRow = (row,max) => `<tr><td><a class="ledger-date" href="/fix/${esc(row.date)}/"><time datetime="${esc(row.date)}">${date(row.date,{year:'numeric'})}</time><span class="ledger-publication${row.withheld ? ' withheld' : ''}">${row.withheld ? 'Withheld' : '✓ Published'}</span></a></td><td>${fmt(row.fix)}</td><td><div class="ledger-change direction-${direction(row.delta)}">${row.delta===null ? '–' : `${arrow(row.delta)} ${signed(row.delta)}`}<div class="ledger-change-track"><span style="width:${Math.abs(row.delta || 0)/max*100}%"></span></div></div></td></tr>`;
    const render = () => {
      const batches = [];
      // Fixed batches keep earlier desktop columns in place as older rows are revealed.
      for (let start=0; start<shown; start+=batchSize) {
        const batch = history.slice(start,Math.min(start+batchSize,shown));
        const columns = desktop.matches ? [batch.slice(0,15),batch.slice(15)] : [batch];
        batches.push(`<div class="ledger-batch">${columns.filter(group=>group.length).map((group,i)=>`<table class="ledger-table" aria-label="Daily references ${start+i*15+1}–${start+i*15+group.length}">${header}<tbody>${group.map(row=>renderRow(row,max)).join('')}</tbody></table>`).join('')}</div>`);
      }
      holder.innerHTML = batches.join('') || '<p class="empty-message">No daily references yet</p>';
      summary.textContent = `Showing ${fmt(shown)} of ${fmt(history.length)} daily references${shown===history.length && shown ? ' · Full history' : ''}`;
      more.hidden = shown>=history.length;
      more.textContent = `View ${Math.min(batchSize,history.length-shown)} more`;
    };
    more.addEventListener('click',()=> {
      const firstNew = shown;
      shown = Math.min(history.length,shown+batchSize); expanded = true;
      render();
      // Start reading at the new rows instead of following the button to the new bottom.
      const next = holder.querySelectorAll('.ledger-date')[firstNew];
      if (next) { next.focus({preventScroll:true}); next.scrollIntoView({block:'nearest'}); }
    });
    desktop.addEventListener('change',()=> {
      if (!expanded) shown = Math.min(history.length,desktop.matches ? batchSize : 10);
      render();
    });
    render();
  }

  const alignmentClass = label => ({'aligned':'alignment-aligned','mild divergence':'alignment-mild','divergent':'alignment-divergent'})[String(label).toLowerCase()] || '';
  function badge(card) {
    const cls = alignmentClass(card.alignment_label);
    const shape = cls==='alignment-divergent' ? '<path d="M6 1 12 11H0Z" fill="currentColor"/>' : cls==='alignment-mild' ? '<circle cx="6" cy="6" r="5" fill="none" stroke="currentColor"/><path d="M6 1a5 5 0 0 0 0 10Z" fill="currentColor"/>' : '<circle cx="6" cy="6" r="5" fill="currentColor"/>';
    return `<span class="alignment-badge ${cls}"><svg viewBox="0 0 12 12" aria-hidden="true">${shape}</svg>${esc(card.alignment_label || 'Unknown')}</span>`;
  }
  function detail(card) {
    const fresh = Math.max(0,numeric(card.fresh_contributing_source_count)||0), used = Math.max(fresh,numeric(card.contributing_source_count)||0), remembered = Math.max(0,numeric(card.remembered_source_count)||0);
    const total = Math.max(1,used+remembered), confidence = numeric(card.basket_confidence);
    const coverage = `${fresh} fresh / ${used} used${remembered ? ` / ${remembered} remembered` : ''}`;
    const dispersion = ({high:3,medium:2,low:1})[card.dispersion_level] || 0;
    return `<p class="diagnostic-intro">${card.display_state==='monitor' ? 'Monitoring' : 'Diagnostics signal available'} • Signal type: ${esc(card.signal_label || humanize(card.signal_type_used))}</p>
      <div class="diagnostic-grid ${alignmentClass(card.alignment_label)}">
        <div><span class="diagnostic-label">Weighted rate</span><strong class="diagnostic-value">${fmt(card.weighted_rate)}</strong><div class="unit-label">IRR per USD</div></div>
        <div><span class="diagnostic-label">Median</span><strong class="diagnostic-value">${fmt(card.median_rate)}</strong><div class="unit-label">IRR per USD</div></div>
        <div><span class="diagnostic-label">Spread vs benchmark</span><strong class="diagnostic-value signal-color">${numeric(card.spread_vs_benchmark_pct)===null ? 'Unavailable' : `${signed(card.spread_vs_benchmark_pct,2)}%`}</strong></div>
        <div><span class="diagnostic-label">Usable records</span><strong class="diagnostic-value">${fmt(card.usable_record_count)}</strong></div>
        <div class="diagnostic-wide"><div class="diagnostic-sources-heading"><span class="diagnostic-label">Sources</span><span>${coverage}</span></div><div class="meter sources-meter" aria-hidden="true"><span style="width:${fresh/total*100}%"></span><span class="used" style="width:${(used-fresh)/total*100}%"></span><span class="remembered" style="width:${remembered/total*100}%"></span></div></div>
        <div class="diagnostic-wide"><div class="diagnostic-sources-heading"><span class="diagnostic-label">Confidence</span><strong>${confidence===null ? 'Unavailable' : confidence.toFixed(1)}</strong></div><div class="meter" aria-hidden="true"><span style="width:${clamp(confidence||0,0,100)}%"></span></div></div>
        <div><span class="diagnostic-label">Freshness</span><strong>${esc(humanize(card.freshness_status))}</strong></div>
        <div><span class="diagnostic-label">Dispersion</span><strong>${esc(humanize(card.dispersion_level))}</strong><span class="dispersion-meter" aria-hidden="true">${[1,2,3].map(i=>`<i${i<=dispersion ? ' class="filled"' : ''}></i>`).join('')}</span></div>
      </div>
      ${card.dispersion_level==='high' ? '<p class="diagnostic-warning">High dispersion — indicative signal only</p>' : ''}
      ${card.display_state==='monitor' ? `<p class="diagnostic-warning">Monitoring only. Reason: ${esc(humanize(card.suppression_reason,'limited coverage'))}</p>` : ''}`;
  }
  function renderRegional(payload) {
    const generated = Date.parse(payload?.generated_at);
    const stale = Number.isFinite(generated) && Date.now()-generated > 30*60*60*1000;
    // Preserve the production filter: visible monitoring cards carry useful diagnostics too.
    const cards = (Array.isArray(payload?.cards) ? payload.cards : []).filter(card=>card && card.display_state!=='hide' && card.render_on_homepage!==false).map(card=>stale ? {...card,display_state:'monitor',freshness_status:'stale',fresh_contributing_source_count:0,suppression_reason:'stale_diagnostics_artifact'} : card);
    const holder = $('regional-market-cards');
    const context = $('regional-context');
    context.hidden = !cards.length;
    if (!cards.length) { holder.innerHTML='<div class="card empty-message">Regional signals unavailable</div>'; return; }
    const primary = $('primary-benchmark');
    const benchmark = textNumber(primary.dataset.fix);
    const comparisons = cards.filter(card => numeric(card.weighted_rate)!==null && numeric(card.spread_vs_benchmark_pct)!==null);
    // The regional export has no benchmark date. Name this page's benchmark only
    // when every published spread demonstrably uses that same rate.
    const matchesBenchmark = benchmark>0 && comparisons.length>0 && comparisons.every(card =>
      Math.abs((Number(card.weighted_rate)/benchmark-1)*100-Number(card.spread_vs_benchmark_pct))<0.0001
    );
    context.textContent = `Diagnostics updated ${timestamp(payload.generated_at)}. ` + (
      matchesBenchmark && /^\d{4}-\d{2}-\d{2}$/.test(primary.dataset.date)
        ? `Compared with the ${date(primary.dataset.date, {year:'numeric'})} benchmark: ${fmt(benchmark)} IRR per USD.`
        : 'The comparison benchmark differs from the displayed daily reference, or its date is unavailable.'
    );
    const spreads = cards.map(card=>numeric(card.spread_vs_benchmark_pct)).filter(value=>value!==null);
    const lo = -Math.max(10,Math.ceil(Math.abs(Math.min(0,...spreads))/5)*5+1), hi = Math.max(10,Math.ceil(Math.max(0,...spreads)/5)*5+1);
    const pos = value=>clamp((value-lo)/(hi-lo)*100,0,100), zero = pos(0);
    const spreadBar = value => {
      const end = pos(value||0), width = value===null || value===0 ? 0 : Math.max(.8,Math.abs(end-zero));
      return `<span class="spread-bar" aria-hidden="true"><span class="spread-tick" style="left:${pos(-10)}%"></span><span class="spread-tick" style="left:${pos(10)}%"></span><span class="spread-fill" style="left:${Math.min(zero,end)}%;width:${width}%"></span><span class="spread-zero" style="left:${zero}%"></span></span>`;
    };
    const published = cards.filter(card=>card.display_state==='publish').length;
    holder.innerHTML = `<div class="card regional-list"><div class="regional-list-heading"><span>${published} signals published${published<cards.length ? ` · ${cards.length-published} monitoring` : ''}</span><span>Spread vs benchmark</span></div>${cards.map((card,i)=>`<div class="regional-item"><button type="button" class="regional-row ${alignmentClass(card.alignment_label)}" id="regional-row-${i}" data-regional-index="${i}"><span class="regional-name">${esc(card.basket_name)}${card.display_state==='monitor' ? ' (monitoring)' : ''}<span class="regional-signal">${esc(card.signal_label || humanize(card.signal_type_used))}</span></span>${badge(card)}<span class="regional-rate">${fmt(card.weighted_rate)}<svg class="regional-chevron" viewBox="0 0 12 12" aria-hidden="true"><path d="m2 4 4 4 4-4"/></svg></span>${spreadBar(numeric(card.spread_vs_benchmark_pct))}<span class="regional-spread">${numeric(card.spread_vs_benchmark_pct)===null ? 'N/A' : `${signed(card.spread_vs_benchmark_pct,2)}%`}</span></button><div id="regional-panel-${i}" class="regional-mobile-detail" role="region" aria-labelledby="regional-row-${i}" hidden>${detail(card)}</div></div>`).join('')}</div><div class="card regional-detail" id="regional-detail" role="region" aria-label="Selected regional signal"></div>`;
    let selected = 0;
    const buttons = [...holder.querySelectorAll('.regional-row')];
    const select = index => {
      selected = index;
      buttons.forEach((button,i)=> {
        button.removeAttribute(desktop.matches ? 'aria-expanded' : 'aria-pressed');
        button.setAttribute(desktop.matches ? 'aria-pressed' : 'aria-expanded',String(i===selected));
        button.setAttribute('aria-controls',desktop.matches ? 'regional-detail' : `regional-panel-${i}`);
        $(`regional-panel-${i}`).hidden = desktop.matches || i!==selected;
      });
      if (selected>=0) {
        const card = cards[selected];
        $('regional-detail').innerHTML = `<div class="regional-detail-heading"><h3>${esc(card.basket_name)}</h3>${badge(card)}</div>${detail(card)}`;
      }
    };
    buttons.forEach((button,i)=> {
      button.addEventListener('click',()=>select(!desktop.matches && selected===i ? -1 : i));
      button.addEventListener('keydown',event=> {
        if (!['ArrowUp','ArrowDown','Home','End'].includes(event.key)) return;
        event.preventDefault();
        const next = event.key==='Home' ? 0 : event.key==='End' ? cards.length-1 : clamp(i+(event.key==='ArrowDown' ? 1 : -1),0,cards.length-1);
        buttons[next].focus(); select(next);
      });
    });
    desktop.addEventListener('change',()=>select(selected<0 && desktop.matches ? 0 : selected));
    select(0);
  }

  enhancePrimary();
  const ladderObserver = new ResizeObserver(renderLadder); ladderObserver.observe($('rate-ladder'));
  fetchJSON('/api/series.json').then(payload=> {
    const rows = (Array.isArray(payload) ? payload : Array.isArray(payload?.rows) ? payload.rows : []).filter(row=>row && /^\d{4}-\d{2}-\d{2}$/.test(row.date) && Number.isFinite(Date.parse(`${row.date}T00:00:00Z`))).sort((a,b)=>a.date.localeCompare(b.date));
    if (!rows.length) throw new Error('No daily references');
    const valid = rows.filter(row=>numeric(row.fix)>0 && !row.withheld).map(row=>({...row,fix:Number(row.fix)}));
    // Archive recovery is optional: a missing endpoint must not hide daily data.
    fetchJSON('/api/chart_history.json').then(history => {
      const byDay = new Map(valid.map(row=>[row.date,row]));
      const first = valid[0]?.date, last = valid.at(-1)?.date;
      for (const row of history?.rows || []) {
        if (row?.kind === 'historical_archive' && /^\d{4}-\d{2}-\d{2}$/.test(row.date) && Number.isFinite(dayTime(row.date)) && row.date>first && row.date<last && numeric(row.fix)>0 && !byDay.has(row.date)) byDay.set(row.date,{...row,fix:Number(row.fix)});
      }
      return [...byDay.values()].sort((a,b)=>a.date.localeCompare(b.date));
    }).catch(()=>valid).then(chartRows=> {
      if (chartRows.length) new Timeline(chartRows);
      else { $('timeline-message').textContent='Timeline unavailable'; $('historyChart').hidden=true; }
    });
    renderLedger(rows);
    const primary = $('primary-benchmark');
    const index = rows.findIndex(row=>row.date===primary.dataset.date);
    if (index>0 && primary.dataset.publication==='Published' && numeric(rows[index-1].fix)>0 && textNumber(primary.dataset.fix)!==null) {
      const delta = textNumber(primary.dataset.fix)-rows[index-1].fix;
      renderPrimaryChange(delta,delta/rows[index-1].fix*100);
    }
  }).catch(()=> {
    $('timeline-message').textContent='Timeline unavailable'; $('historyChart').hidden=true;
    $('recent-days').innerHTML='<p class="empty-message">Daily references unavailable</p>';
    $('recent-days-summary').textContent='History is temporarily unavailable';
  });
  fetchJSON('/api/regional_market_signals_card.json').then(renderRegional).catch(()=>renderRegional({cards:[]}));
})();
