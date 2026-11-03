const byId = id => document.getElementById(id);
const colors = ["#bd6335", "#007c83", "#6876a4"];
let report, current, mathTimer;
const norm = v => Math.hypot(v[0], v[1]);
const sub = (a,b) => [a[0]-b[0], a[1]-b[1]];
const cross = (a,b) => a[0]*b[1]-a[1]*b[0];
const lerp = (a,b,t) => a.map((x,i) => x+(b[i]-x)*t);
const number = x => Number(x).toFixed(9);
const sci = x => { if (!x) return "0"; const p=x.toExponential(1).split("e"); return p[0]+"\\times10^{"+Number(p[1])+"}"; };

async function math(element, source) {
  await MathJax.startup.promise;
  MathJax.typesetClear([element]);
  element.textContent=source;
  await MathJax.typesetPromise([element]);
}

function intersections(paths,u,v) {
  const segments=[], result=[];
  paths.forEach((p,e) => p.slice(1).forEach((b,k) => segments.push({a:p[k], b, e, k})));
  segments.forEach((s,i) => {
    segments.slice(0,i).forEach(q => {
      if(s.e===q.e && Math.abs(s.k-q.k)===1) return;
      const a=sub(s.b,s.a), b=sub(q.b,q.a), det=cross(a,b);
      if(Math.abs(det)<1e-12) return;
      const delta=sub(q.a,s.a), t=cross(delta,b)/det, r=cross(delta,a)/det;
      if(t< -1e-8 || t>1+1e-8 || r< -1e-8 || r>1+1e-8) return;
      const p=[s.a[0]+t*a[0],s.a[1]+t*a[1]];
      if(Math.min(norm(sub(p,u)),norm(sub(p,v)))<1e-7) return;
      if(!result.some(x=>norm(sub(p,x))<1e-7)) result.push(p);
    });
  });
  return result;
}

function draw(vertices, frame) {
  const xs=vertices.map(p=>p[0]), ys=vertices.map(p=>p[1]);
  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const scale=Math.max(maxX-minX,maxY-minY),pad=.13*scale;
  const svg=byId("netView");
  svg.setAttribute("viewBox",[minX-pad,-maxY-pad,maxX-minX+2*pad,maxY-minY+2*pad].join(" "));
  const points=p=>p.map(q=>q[0]+","+(-q[1])).join(" ");
  let content='<polygon points="'+points(vertices)+'" fill="#fffdf8" stroke="#18333a" stroke-width="'+(.008*scale)+'"/>';
  if(current.witness?.admissible_region && byId("mode").value==="immersed") {
    content+='<polygon points="'+points(current.witness.admissible_region.first_junction_region)+'" fill="#007c8314" stroke="#007c8350" stroke-width="'+(.003*scale)+'"/>';
  }
  vertices.forEach((a,i)=>{
    const b=vertices[(i+1)%vertices.length], edge=sub(b,a), length=norm(edge);
    const p=[(a[0]+b[0])/2+.055*scale*edge[1]/length,(a[1]+b[1])/2-.055*scale*edge[0]/length];
    content+='<text x="'+p[0]+'" y="'+(-p[1])+'" text-anchor="middle" dominant-baseline="middle" fill="#50656b" font-size="'+(.065*scale)+'">'+(i+1)+'</text>';
  });
  if(frame) {
    frame.paths.forEach((path,i)=>{
      content+='<polyline class="net-path" points="'+points(path)+'" fill="none" stroke="'+colors[i]+'" stroke-width="'+(.009*scale)+'" stroke-linejoin="round"/>';
      path.slice(1,-1).forEach(p=>content+='<circle cx="'+p[0]+'" cy="'+(-p[1])+'" r="'+(.009*scale)+'" fill="'+colors[i]+'"/>');
    });
    const crosses=intersections(frame.paths,frame.u,frame.v);
    if(byId("showCrossings").checked) crosses.forEach(p=>content+='<circle cx="'+p[0]+'" cy="'+(-p[1])+'" r="'+(.016*scale)+'" fill="#f7f5ef" stroke="#62757a" stroke-width="'+(.003*scale)+'"/>');
    [frame.u,frame.v].forEach((p,i)=>content+='<circle class="junction" cx="'+p[0]+'" cy="'+(-p[1])+'" r="'+(.022*scale)+'" fill="'+(i===0?"#18333a":"#bd6335")+'" stroke="#fffdf8" stroke-width="'+(.004*scale)+'"/>');
    byId("netType").textContent=crosses.length ? "Immersed net: "+crosses.length+" extra crossing points." : "Crossing-free net.";
  }
  svg.innerHTML=content;
}

function render() {
  clearTimeout(mathTimer);
  current=report.cases.find(c=>c.id===byId("shape").value);
  const embedded=byId("mode").value==="embedded";
  const witness=embedded ? current.embedded_witness : current.witness;
  byId("position").disabled=!witness;
  if(!witness) {
    draw(current.vertices,null);
    const depth=embedded ? current.embedded_depth : current.depth;
    byId("netType").textContent="No witness found in this bounded search.";
    byId("caseInfo").textContent="Search: one direct edge, with at most "+depth+" reflections on each other edge. This does not establish nonexistence.";
    byId("angleCheck").textContent="";
    math(byId("netNumbers"),"");
    return;
  }
  const first=witness.frames[0],last=witness.frames[witness.frames.length-1],t=Number(byId("position").value)/1000;
  const frame={u:lerp(first.u,last.u,t),v:lerp(first.v,last.v,t),
    paths:first.paths.map((p,e)=>p.map((q,k)=>lerp(q,last.paths[e][k],t)))};
  const lengths=frame.paths.map(p=>p.slice(1).reduce((sum,q,k)=>sum+norm(sub(q,p[k])),0));
  let angleError=0;
  [frame.paths.map(p=>sub(p[1],p[0])),frame.paths.map(p=>sub(p[p.length-2],p[p.length-1]))].forEach(directions=>{
    const unit=directions.map(d=>d.map(x=>x/norm(d)));
    [[0,1],[1,2],[2,0]].forEach(([i,j])=>{
      const cosine=Math.max(-1,Math.min(1,unit[i][0]*unit[j][0]+unit[i][1]*unit[j][1]));
      angleError=Math.max(angleError,Math.abs(Math.acos(cosine)*180/Math.PI-120));
    });
  });
  draw(current.vertices,frame);
  byId("caseInfo").textContent="Two-parameter affine family; the slider follows one line through it. Nine positions on this line were independently ray-traced."+(embedded ? "" : " The shaded area shows the computed admissible region for the first junction.");
  mathTimer=setTimeout(()=>{
    const total=lengths.reduce((a,b)=>a+b,0);
    math(byId("netNumbers"),"\\[\\begin{aligned}\\ell_1&="+number(lengths[0])+"\\\\\\ell_2&="+number(lengths[1])+"\\\\\\ell_3&="+number(lengths[2])+"\\\\L&="+number(total)+"\\end{aligned}\\]");
    math(byId("angleCheck"),"\\(\\varepsilon_{\\mathrm{angle}}="+sci(angleError)+"{}^\\circ\\)");
  },50);
}

async function start() {
  report=await fetch("data/polygon-families/report.json").then(r=>r.json());
  byId("shape").innerHTML=report.cases.map(c=>'<option value="'+c.id+'">'+c.label+'</option>').join("");
  byId("shape").value="rhombus_60";
  ["shape","mode"].forEach(id=>byId(id).addEventListener("change",()=>{byId("position").value=500;render();}));
  byId("position").addEventListener("input",render);
  byId("showCrossings").addEventListener("change",render);
  await MathJax.startup.promise;
  render();
}
start().catch(error=>{byId("caseInfo").textContent="Could not load the experiment data: "+error.message;});
