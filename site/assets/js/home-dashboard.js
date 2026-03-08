(function () {
  function fmt(value) {
    if (value === null || value === undefined) return 'N/A';
    return Number(value).toLocaleString('en-US', { maximumFractionDigits: 0 });
  }

  function parseRows(rows) {
    return (rows || [])
      .filter(function (row) {
        return row && row.date;
      })
      .map(function (row) {
        return {
          date: String(row.date),
          fix: typeof row.fix === 'number' ? row.fix : null,
          p25: typeof row.p25 === 'number' ? row.p25 : null,
          p75: typeof row.p75 === 'number' ? row.p75 : null,
          status: row.status || 'N/A',
          withheld: Boolean(row.withheld)
        };
      })
      .sort(function (a, b) {
        return a.date.localeCompare(b.date);
      });
  }

  function renderSparkline(rows) {
    var svg = document.getElementById('sparkline');
    if (!svg) return;

    var points = rows.filter(function (r) { return r.fix !== null; });
    if (points.length < 2) {
      svg.innerHTML = '<text x="4" y="18" fill="#99a9c4" font-size="12">No sparkline data</text>';
      return;
    }

    var width = svg.clientWidth || 360;
    var height = svg.clientHeight || 56;
    var values = points.map(function (p) { return p.fix; });
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    var span = Math.max(1, max - min);

    var path = points.map(function (p, i) {
      var x = (i / (points.length - 1)) * (width - 4) + 2;
      var y = height - (((p.fix - min) / span) * (height - 8) + 4);
      return (i === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + y.toFixed(1);
    }).join(' ');

    svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    svg.innerHTML = '<path d="' + path + '" fill="none" stroke="#66a3ff" stroke-width="2"/>';
  }

  function renderHistoryChart(rows) {
    var svg = document.getElementById('history-chart-svg');
    var empty = document.getElementById('history-chart-empty');
    if (!svg || !empty) return;

    var points = rows.filter(function (r) { return r.fix !== null; });
    if (points.length < 2) {
      svg.innerHTML = '';
      empty.style.display = 'block';
      return;
    }

    empty.style.display = 'none';

    var width = svg.clientWidth || 820;
    var height = svg.clientHeight || 250;
    var padX = 32;
    var padY = 18;
    var values = points.map(function (p) { return p.fix; });
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    var span = Math.max(1, max - min);

    var grid = [];
    for (var i = 0; i <= 4; i += 1) {
      var gy = padY + ((height - padY * 2) * i / 4);
      grid.push('<line x1="' + padX + '" y1="' + gy.toFixed(1) + '" x2="' + (width - padX) + '" y2="' + gy.toFixed(1) + '" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>');
    }

    var path = points.map(function (p, i) {
      var x = padX + ((width - padX * 2) * i / (points.length - 1));
      var y = height - padY - (((p.fix - min) / span) * (height - padY * 2));
      return (i === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + y.toFixed(1);
    }).join(' ');

    svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    svg.innerHTML = grid.join('') +
      '<path d="' + path + '" fill="none" stroke="#66a3ff" stroke-width="2.3"/>' +
      '<text x="8" y="14" fill="#99a9c4" font-size="11">' + fmt(max) + '</text>' +
      '<text x="8" y="' + (height - 6) + '" fill="#99a9c4" font-size="11">' + fmt(min) + '</text>';
  }

  function renderTape(rows) {
    var body = document.getElementById('history-table-body');
    if (!body) return;

    var recent = rows.slice(-30).reverse();
    if (!recent.length) {
      body.innerHTML = '<tr><td colspan="6" class="text-secondary">No historical daily rows yet.</td></tr>';
      return;
    }

    body.innerHTML = recent.map(function (r) {
      return '<tr>' +
        '<td>' + r.date + '</td>' +
        '<td>' + fmt(r.fix) + '</td>' +
        '<td>' + fmt(r.p25) + '</td>' +
        '<td>' + fmt(r.p75) + '</td>' +
        '<td>' + (r.status || 'N/A') + '</td>' +
        '<td>' + (r.withheld ? 'Yes' : 'No') + '</td>' +
        '</tr>';
    }).join('');
  }

  function loadSeries() {
    fetch('api/series.json', { cache: 'no-store' })
      .then(function (res) { return res.ok ? res.json() : []; })
      .then(function (rows) {
        var parsed = parseRows(rows);
        renderSparkline(parsed);
        renderHistoryChart(parsed);
        renderTape(parsed);
      })
      .catch(function () {
        renderSparkline([]);
        renderHistoryChart([]);
        renderTape([]);
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadSeries);
  } else {
    loadSeries();
  }
})();
