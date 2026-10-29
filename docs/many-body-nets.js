"use strict";
const math = s => `\\(${s}\\)`;
function sci(x) {
  if (x === 0) return "0";
  const [m,e] = x.toExponential(2).split("e");
  return `${m}\\times10^{${Number(e)}}`;
}
let mathQueue = Promise.resolve();
function typeset(elements) {
  mathQueue = mathQueue.then(async () => {
    // The deferred MathJax loader initializes asynchronously.
    if (window.MathJax?.startup?.promise) await window.MathJax.startup.promise;
    if (window.MathJax?.typesetPromise) await window.MathJax.typesetPromise(elements);
  }).catch(console.error);
}
function ground() {
  const c = 10 ** Number(document.querySelector("#coupling").value);
  let lo = 0,hi = Math.PI;
  for (let i=0;i<65;i++) {
    const q = (lo+hi)/2;
    if(q*Math.tan(q/2) < c/2) lo=q; else hi=q;
  }
  const q = (lo+hi)/2;
  const element = document.querySelector("#ground-readout");
  window.MathJax?.typesetClear?.([element]);
  element.innerHTML = `${math(`c=${c.toFixed(4)}`)}<br>${math(`q_0=${q.toFixed(9)}`)}<br>${math(`E_0=${(2*q*q).toFixed(9)}`)}`;
  typeset([element]);
}
document.querySelector("#coupling").addEventListener("input",ground);
ground();
fetch("data/many-body/results.json").then(r=>{
  if(!r.ok) throw Error(`Results request failed: ${r.status}`);
  return r.json();
}).then(d=>{
  const max = (rows,key) => Math.max(...rows.map(r=>r[key]));
  const net = d.resolvent.filter(r=>r.winding_cutoff===32);
  const rows = [
    ["Free cycles / occupation sums",`${math("1\\leq N\\leq4")}; bosons and fermions`,`${math(sci(max(d.free,"relative_error")))} relative`],
    ["First contact / differentiated spectrum",`${math("\\beta=0.02,0.05,0.1")}`,`${math(sci(max(d.contact_derivative,"absolute_error")))} absolute`],
    ["Bethe equation residual",`Three particles, ${math("c=0.1,1,10")}`,`${math(sci(max(d.bethe,"residual")))}`],
    ["Winding resolvent / closed formula",`${math("W=32,\\ s=2,8,32")}; both twists`,`${math(sci(max(net,"net_error")))} absolute`],
    ["Spectral resolvent / winding formula",`${math("J=2048")}; both twists`,`${math(sci(max(d.resolvent,"spectral_error")))} absolute`],
    ["Three-body contact factorization",`${math("M=6,12,24,48")}`,`${math(sci(max(d.collision_operator,"factor_error")))} max entry`]
  ];
  document.querySelector("#summary-table").innerHTML=rows.map(r=>`<tr>${r.map(x=>`<td>${x}</td>`).join("")}</tr>`).join("");
  document.querySelector("#collision-table").innerHTML=d.collision_operator.map(r=>`<tr><td>${math(r.cutoff)}</td><td>${math(r.hilbert_dimension)}</td><td>${math(r.collision_dimension)}</td><td>${math(sci(r.resolvent_error))}</td></tr>`).join("");
  document.querySelector("#energy-table").innerHTML=d.diagonalization.filter(r=>r.n===3&&r.c===1).map(r=>`<tr><td>${math(r.cutoff)}</td><td>${math(r.energies[0].toFixed(12))}</td><td>${math(r.absolute_errors[0].toFixed(9))}</td></tr>`).join("");
  document.querySelector("#versions").textContent=`Recorded environment: Python ${d.versions.python}, NumPy ${d.versions.numpy}, SciPy ${d.versions.scipy}, Matplotlib ${d.versions.matplotlib}. Timings in JSON describe this run only.`;
  typeset([document.querySelector("#evidence"),document.querySelector("#three")]);
}).catch(error=>{
  document.querySelector("#summary-table").innerHTML=`<tr><td colspan="3">Results could not be loaded. Use the complete results JSON linked below.</td></tr>`;
  console.error(error);
});
