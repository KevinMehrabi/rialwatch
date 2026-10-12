(() => {
  'use strict';
  const root = document.documentElement;
  const toggle = document.getElementById('theme-toggle');
  const updateLabel = () => {
    const next = root.dataset.theme === 'light' ? 'dark' : 'light';
    toggle.setAttribute('aria-label', `Switch to ${next} mode`);
    toggle.title = `Switch to ${next} mode`;
  };
  updateLabel();
  toggle.addEventListener('click', () => {
    const theme = root.dataset.theme === 'light' ? 'dark' : 'light';
    root.dataset.theme = theme;
    try { localStorage.setItem('rw-theme', theme); } catch (_) { /* Storage may be disabled. */ }
    updateLabel();
    document.dispatchEvent(new CustomEvent('rw-theme-change', { detail: { theme } }));
  });
  const updated = document.getElementById('site-last-update');
  const timestamp = new Date(updated.dataset.ts);
  const desktop = matchMedia('(min-width: 900px)');
  const formatUpdate = () => {
    if (Number.isNaN(timestamp.getTime())) return;
    const date = timestamp.toLocaleDateString('en-US', { month:'short', day:'numeric', ...(desktop.matches ? { year:'numeric' } : {}), timeZone:'UTC' });
    const time = timestamp.toLocaleTimeString('en-GB', { hour:'2-digit', minute:'2-digit', timeZone:'UTC' });
    updated.querySelector('time').textContent = `${date} · ${time} UTC`;
  };
  formatUpdate();
  desktop.addEventListener('change', formatUpdate);
})();
