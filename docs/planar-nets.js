'use strict';
const math=x=>`\\(${x}\\)`;
const sci=x=>{if(x===0)return '0';const [a,b]=x.toExponential(2).split('e');return `${a}\\times10^{${Number(b)}}`;};
fetch('data/planar-nets/results.json').then(r=>{if(!r.ok)throw Error('Results unavailable');return r.json();}).then(async d=>{
  document.querySelector('#net-table').innerHTML=d.fixed_time.map(r=>`<tr><td>${math(r.beta)}</td><td>${math(r.net_sum.toFixed(14))}</td><td>${math(r.spectral.toFixed(14))}</td><td>${math(sci(r.net_absolute_error))}</td></tr>`).join('');
  document.querySelector('#matrix-table').innerHTML=d.matrix_checks.filter(r=>r.beta===.1).map(r=>`<tr><td>${math(r.mode_cutoff)}</td><td>${math(r.dimension)}</td><td>${math(r.fixed_time_matrix_trace.toFixed(12))}</td><td>${math(sci(r.fixed_time_cutoff_error))}</td></tr>`).join('');
  if(window.MathJax?.startup?.promise)await window.MathJax.startup.promise;
  if(window.MathJax?.typesetPromise)await window.MathJax.typesetPromise([document.querySelector('#net-table'),document.querySelector('#matrix-table')]);
}).catch(error=>{document.querySelector('#net-table').innerHTML='<tr><td colspan="4">Results could not be loaded. Use the JSON linked below.</td></tr>';console.error(error);});
