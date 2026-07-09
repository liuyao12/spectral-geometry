const state = {
  data: null,
  selectedIndex: 0,
  model: "disk",
  showTiles: true,
  showLabels: true,
  showDomainSegment: true,
  vertices: [],
  sideSamples: [],
  sideCircles: [],
  projection: null
};

const el = {
  loadStatus: document.querySelector("#loadStatus"),
  surfaceSelect: document.querySelector("#surfaceSelect"),
  modelSelect: document.querySelector("#modelSelect"),
  geodesicSelect: document.querySelector("#geodesicSelect"),
  tileToggle: document.querySelector("#tileToggle"),
  labelToggle: document.querySelector("#labelToggle"),
  domainToggle: document.querySelector("#domainToggle"),
  surfaceDetails: document.querySelector("#surfaceDetails"),
  surfaceTitle: document.querySelector("#surfaceTitle"),
  surfaceSubtitle: document.querySelector("#surfaceSubtitle"),
  canvas: document.querySelector("#surfaceCanvas"),
  errorBox: document.querySelector("#errorBox"),
  geodesicRows: document.querySelector("#geodesicRows"),
  recordCount: document.querySelector("#recordCount"),
  geodesicDetails: document.querySelector("#geodesicDetails"),
  selectedJson: document.querySelector("#selectedJson")
};

const ctx = el.canvas.getContext("2d");
const pairColors = ["#0b7285", "#b05c00", "#6d5bd0", "#247a38"];
const identity = {
  a: { x: 1, y: 0 },
  b: { x: 0, y: 0 }
};
const cayleyAngle = Math.PI / 5;

function fmt(value, digits = 5) {
  if (value === null || value === undefined) return "";
  if (typeof value !== "number") return String(value);
  return value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function c(x, y = 0) {
  return { x, y };
}

function add(a, b) {
  return c(a.x + b.x, a.y + b.y);
}

function sub(a, b) {
  return c(a.x - b.x, a.y - b.y);
}

function mul(a, b) {
  return c(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

function div(a, b) {
  const den = b.x * b.x + b.y * b.y;
  return c((a.x * b.x + a.y * b.y) / den, (a.y * b.x - a.x * b.y) / den);
}

function conj(a) {
  return c(a.x, -a.y);
}

function scale(a, k) {
  return c(a.x * k, a.y * k);
}

function norm2(a) {
  return a.x * a.x + a.y * a.y;
}

function norm(a) {
  return Math.hypot(a.x, a.y);
}

function dot(a, b) {
  return a.x * b.x + a.y * b.y;
}

function cross(a, b) {
  return a.x * b.y - a.y * b.x;
}

function fromPair(pair) {
  return c(pair[0], pair[1]);
}

function fromComplexObject(value) {
  return c(value.re, value.im);
}

function matrixFromRecord(record) {
  return {
    a: fromComplexObject(record.matrix.a),
    b: fromComplexObject(record.matrix.b)
  };
}

function inverseMatrix(matrix) {
  return {
    a: conj(matrix.a),
    b: scale(matrix.b, -1)
  };
}

function applyMatrix(matrix, z) {
  const numerator = add(mul(matrix.a, z), matrix.b);
  const denominator = add(mul(conj(matrix.b), z), conj(matrix.a));
  return div(numerator, denominator);
}

function normalizeDelta(delta) {
  let value = delta;
  while (value <= -Math.PI) value += Math.PI * 2;
  while (value > Math.PI) value -= Math.PI * 2;
  return value;
}

function geodesicCircle(p, q) {
  const rhs1 = (norm2(p) + 1) / 2;
  const rhs2 = (norm2(q) + 1) / 2;
  const det = cross(p, q);
  if (Math.abs(det) < 1e-10) return null;
  const center = c(
    (rhs1 * q.y - rhs2 * p.y) / det,
    (p.x * rhs2 - q.x * rhs1) / det
  );
  const radius = Math.sqrt(Math.max(0, norm2(center) - 1));
  return { center, radius };
}

function geodesicSamples(p, q, count = 80, trim = 0) {
  const circle = geodesicCircle(p, q);
  const start = Math.max(0, trim);
  const end = Math.min(1, 1 - trim);
  if (!circle || circle.radius < 1e-10) {
    const points = [];
    const p0 = norm2(p) > 0.999 ? scale(p, 1 - trim) : p;
    const q0 = norm2(q) > 0.999 ? scale(q, 1 - trim) : q;
    for (let i = 0; i <= count; i += 1) {
      const t = i / count;
      points.push(add(scale(p0, 1 - t), scale(q0, t)));
    }
    return points;
  }

  const a0 = Math.atan2(p.y - circle.center.y, p.x - circle.center.x);
  const a1 = Math.atan2(q.y - circle.center.y, q.x - circle.center.x);
  const d0 = normalizeDelta(a1 - a0);
  const d1 = d0 > 0 ? d0 - Math.PI * 2 : d0 + Math.PI * 2;
  const candidates = [d0, d1].map((delta) => {
    const mid = c(
      circle.center.x + circle.radius * Math.cos(a0 + delta / 2),
      circle.center.y + circle.radius * Math.sin(a0 + delta / 2)
    );
    return { delta, score: norm2(mid) < 1.00001 ? Math.abs(delta) : 100 + Math.abs(delta) };
  });
  const delta = candidates.sort((a, b) => a.score - b.score)[0].delta;
  const points = [];
  for (let i = 0; i <= count; i += 1) {
    const t = start + ((end - start) * i) / count;
    const angle = a0 + delta * t;
    points.push(c(
      circle.center.x + circle.radius * Math.cos(angle),
      circle.center.y + circle.radius * Math.sin(angle)
    ));
  }
  return points;
}

function toHalfPlane(z) {
  const rot = c(Math.cos(-cayleyAngle), Math.sin(-cayleyAngle));
  const zr = mul(z, rot);
  return mul(c(0, 1), div(add(c(1, 0), zr), sub(c(1, 0), zr)));
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * q)));
  return sorted[index];
}

function canvasSize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = el.canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (el.canvas.width !== width || el.canvas.height !== height) {
    el.canvas.width = width;
    el.canvas.height = height;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width: rect.width, height: rect.height };
}

function selectedGeodesic() {
  return state.data?.geodesics[state.selectedIndex] || null;
}

function axisSamples() {
  const geo = selectedGeodesic();
  if (!geo) return [];
  return geodesicSamples(fromPair(geo.fixed_points[0]), fromPair(geo.fixed_points[1]), 420, 0.008);
}

function transformedBoundary(matrix = identity) {
  const points = [];
  state.sideSamples.forEach((samples, index) => {
    const mapped = samples.map((point) => applyMatrix(matrix, point));
    if (index === 0) points.push(...mapped);
    else points.push(...mapped.slice(1));
  });
  return points;
}

function neighborTransforms() {
  if (!state.showTiles || !state.data) return [];
  const transforms = [];
  for (const generator of state.data.generators) {
    const matrix = matrixFromRecord(generator);
    transforms.push(matrix, inverseMatrix(matrix));
  }
  return transforms;
}

function prepareProjection(size) {
  if (state.model === "disk") {
    const scaleValue = Math.min(size.width, size.height) * 0.44;
    state.projection = {
      project(point) {
        return c(size.width / 2 + point.x * scaleValue, size.height / 2 - point.y * scaleValue);
      }
    };
    return;
  }

  const fitPoints = [
    ...transformedBoundary(identity),
    ...axisSamples()
  ].map(toHalfPlane).filter((point) => (
    Number.isFinite(point.x) &&
    Number.isFinite(point.y) &&
    Math.abs(point.x) < 1e5 &&
    Math.abs(point.y) < 1e5
  ));

  const xs = fitPoints.map((point) => point.x);
  const ys = fitPoints.map((point) => Math.max(0, point.y));
  let minX = quantile(xs, 0.02);
  let maxX = quantile(xs, 0.98);
  let maxY = Math.max(quantile(ys, 0.98), 1);
  if (Math.abs(maxX - minX) < 0.5) {
    minX -= 0.5;
    maxX += 0.5;
  }
  const padX = (maxX - minX) * 0.1;
  const padY = maxY * 0.14;
  minX -= padX;
  maxX += padX;
  maxY += padY;
  const padding = 34;
  const scaleX = (size.width - padding * 2) / (maxX - minX);
  const scaleY = (size.height - padding * 2) / maxY;
  const scaleValue = Math.max(1, Math.min(scaleX, scaleY));
  const plotWidth = (maxX - minX) * scaleValue;
  const originX = (size.width - plotWidth) / 2 - minX * scaleValue;
  const originY = size.height - padding;
  state.projection = {
    project(point) {
      const w = toHalfPlane(point);
      return c(originX + w.x * scaleValue, originY - w.y * scaleValue);
    },
    halfplane: { minX, maxX, maxY, originX, originY, scale: scaleValue }
  };
}

function project(point) {
  return state.projection.project(point);
}

function drawPolyline(points, options = {}) {
  if (!points.length) return;
  const screen = points.map(project).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (screen.length < 2) return;
  ctx.save();
  if (options.dash) ctx.setLineDash(options.dash);
  ctx.beginPath();
  ctx.moveTo(screen[0].x, screen[0].y);
  for (let i = 1; i < screen.length; i += 1) {
    ctx.lineTo(screen[i].x, screen[i].y);
  }
  if (options.close) ctx.closePath();
  if (options.fill) {
    ctx.fillStyle = options.fill;
    ctx.fill();
  }
  if (options.stroke) {
    ctx.strokeStyle = options.stroke;
    ctx.lineWidth = options.width || 1;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.stroke();
  }
  ctx.restore();
}

function roundedRect(x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function drawLabel(text, point, color) {
  const screen = project(point);
  if (!Number.isFinite(screen.x) || !Number.isFinite(screen.y)) return;
  ctx.save();
  ctx.font = "700 12px Inter, system-ui, sans-serif";
  const metrics = ctx.measureText(text);
  const width = Math.max(28, metrics.width + 14);
  const height = 22;
  roundedRect(screen.x - width / 2, screen.y - height / 2, width, height, 6);
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4;
  ctx.stroke();
  ctx.fillStyle = "#1c2026";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, screen.x, screen.y + 0.5);
  ctx.restore();
}

function insidePolygon(point) {
  if (norm2(point) > 1.0001) return false;
  for (const side of state.sideCircles) {
    if (norm(sub(point, side.center)) < side.radius - 1e-5) return false;
  }
  return true;
}

function clippedAxisSegments(samples) {
  const segments = [];
  let current = [];
  for (const point of samples) {
    if (insidePolygon(point)) {
      current.push(point);
    } else if (current.length > 1) {
      segments.push(current);
      current = [];
    } else {
      current = [];
    }
  }
  if (current.length > 1) segments.push(current);
  return segments;
}

function drawDiskBoundary(size) {
  const radius = Math.min(size.width, size.height) * 0.44;
  ctx.save();
  ctx.beginPath();
  ctx.arc(size.width / 2, size.height / 2, radius, 0, Math.PI * 2);
  ctx.fillStyle = "#fcfefe";
  ctx.fill();
  ctx.strokeStyle = "#b9c5ca";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();
}

function drawHalfPlaneBoundary(size) {
  const hp = state.projection.halfplane;
  if (!hp) return;
  const left = project(c(-0.999, 0)).x;
  const right = project(c(0.999, 0)).x;
  ctx.save();
  ctx.strokeStyle = "#aebbc3";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(Math.max(22, Math.min(left, right) - 80), hp.originY);
  ctx.lineTo(Math.min(size.width - 22, Math.max(left, right) + 80), hp.originY);
  ctx.stroke();
  ctx.fillStyle = "#64717d";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("real axis", size.width - 26, hp.originY - 8);
  ctx.restore();
}

function drawTiles() {
  for (const matrix of neighborTransforms()) {
    const boundary = transformedBoundary(matrix);
    drawPolyline(boundary, {
      close: true,
      stroke: "rgba(77, 91, 105, 0.24)",
      width: 1,
      fill: "rgba(11, 111, 112, 0.025)"
    });
  }
}

function drawCentralPolygon() {
  const boundary = transformedBoundary(identity);
  drawPolyline(boundary, {
    close: true,
    fill: "rgba(255,255,255,0.86)",
    stroke: "rgba(32,36,42,0.30)",
    width: 1
  });

  state.data.polygon.sides.forEach((side) => {
    drawPolyline(state.sideSamples[side.index], {
      stroke: pairColors[side.pair_color_index],
      width: 3
    });
  });

  state.vertices.forEach((vertex) => {
    const screen = project(vertex);
    ctx.save();
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = "#20242a";
    ctx.fill();
    ctx.restore();
  });
}

function drawAxis() {
  const samples = axisSamples();
  if (!samples.length) return;
  drawPolyline(samples, {
    stroke: "rgba(214,63,47,0.42)",
    width: 2.2
  });

  if (state.showDomainSegment) {
    for (const segment of clippedAxisSegments(samples)) {
      drawPolyline(segment, {
        stroke: "#d63f2f",
        width: 6
      });
      drawPolyline(segment, {
        stroke: "#fff8f6",
        width: 2
      });
    }
  }

  if (state.model === "disk") {
    const geo = selectedGeodesic();
    for (const endpoint of geo.fixed_points) {
      const screen = project(scale(fromPair(endpoint), 0.995));
      ctx.save();
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#d63f2f";
      ctx.fill();
      ctx.restore();
    }
  }
}

function drawLabels() {
  if (!state.showLabels) return;
  for (const side of state.data.polygon.sides) {
    const samples = state.sideSamples[side.index];
    const midpoint = samples[Math.floor(samples.length / 2)];
    const inward = scale(midpoint, 0.92);
    drawLabel(side.label, inward, pairColors[side.pair_color_index]);
  }
}

function drawScene() {
  if (!state.data) return;
  const size = canvasSize();
  prepareProjection(size);
  ctx.clearRect(0, 0, size.width, size.height);
  ctx.fillStyle = "#f7f9fa";
  ctx.fillRect(0, 0, size.width, size.height);

  if (state.model === "disk") drawDiskBoundary(size);
  else drawHalfPlaneBoundary(size);

  drawTiles();
  drawCentralPolygon();
  drawAxis();
  drawLabels();
}

function setDetails(node, rows) {
  node.innerHTML = "";
  for (const [key, value] of rows) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = value;
    node.append(dt, dd);
  }
}

function populateGeodesicSelect() {
  el.geodesicSelect.innerHTML = "";
  state.data.geodesics.forEach((geo, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${geo.word}  |  L=${fmt(geo.length, 4)}`;
    el.geodesicSelect.append(option);
  });
}

function renderGeodesicRows() {
  el.geodesicRows.innerHTML = "";
  state.data.geodesics.forEach((geo, index) => {
    const row = document.createElement("tr");
    if (index === state.selectedIndex) row.classList.add("is-selected");
    const word = document.createElement("td");
    const strong = document.createElement("strong");
    strong.textContent = geo.word;
    word.append(strong);
    const letters = document.createElement("td");
    letters.textContent = String(geo.word_length);
    const length = document.createElement("td");
    length.textContent = fmt(geo.length, 5);
    const trace = document.createElement("td");
    trace.textContent = fmt(geo.trace_abs, 5);
    row.append(word, letters, length, trace);
    row.addEventListener("click", () => {
      state.selectedIndex = index;
      el.geodesicSelect.value = String(index);
      render();
    });
    el.geodesicRows.append(row);
  });
}

function renderDetails() {
  const polygon = state.data.polygon;
  const geo = selectedGeodesic();
  setDetails(el.surfaceDetails, [
    ["quotient", `genus ${state.data.surface.genus}, compact`],
    ["area", `${fmt(polygon.area, 5)} = 4*pi`],
    ["polygon", "regular hyperbolic octagon"],
    ["edge rule", "opposite sides"],
    ["records", String(state.data.geodesics.length)],
    ["search", `words up to ${state.data.enumeration.max_word_length} letters`]
  ]);

  el.surfaceTitle.textContent = state.data.surface.name;
  el.surfaceSubtitle.textContent = `${geo.word}: length ${fmt(geo.length, 6)}, |trace| ${fmt(geo.trace_abs, 6)}`;
  el.recordCount.textContent = `${state.data.geodesics.length} records`;

  setDetails(el.geodesicDetails, [
    ["word", geo.word],
    ["inverse", geo.inverse_word],
    ["letters", String(geo.word_length)],
    ["length", fmt(geo.length, 9)],
    ["|trace|", fmt(geo.trace_abs, 9)],
    ["fixed point 1", `(${fmt(geo.fixed_points[0][0], 5)}, ${fmt(geo.fixed_points[0][1], 5)})`],
    ["fixed point 2", `(${fmt(geo.fixed_points[1][0], 5)}, ${fmt(geo.fixed_points[1][1], 5)})`]
  ]);
  el.selectedJson.textContent = JSON.stringify(geo, null, 2);
}

function render() {
  if (!state.data) return;
  state.model = el.modelSelect.value;
  state.showTiles = el.tileToggle.checked;
  state.showLabels = el.labelToggle.checked;
  state.showDomainSegment = el.domainToggle.checked;
  renderDetails();
  renderGeodesicRows();
  drawScene();
}

function prepareGeometry() {
  const polygon = state.data.polygon;
  state.vertices = polygon.vertices.map((vertex) => fromPair(vertex.point));
  state.sideSamples = polygon.sides.map((side) => {
    const start = state.vertices[side.vertices[0]];
    const end = state.vertices[side.vertices[1]];
    return geodesicSamples(start, end, 64, 0);
  });
  const foot = polygon.side_foot_radius_disk;
  state.sideCircles = polygon.sides.map((side) => {
    const centerDistance = 1 / foot;
    const angle = side.normal_angle;
    return {
      center: c(centerDistance * Math.cos(angle), centerDistance * Math.sin(angle)),
      radius: Math.sqrt(centerDistance * centerDistance - 1)
    };
  });
}

async function init() {
  const res = await fetch("data/hyperbolic/bolza_octagon_geodesics.json");
  if (!res.ok) throw new Error(`Could not load geodesic dataset (${res.status})`);
  state.data = await res.json();
  prepareGeometry();
  populateGeodesicSelect();
  el.loadStatus.textContent = `${state.data.geodesics.length} geodesic records generated locally`;
  render();
}

el.modelSelect.addEventListener("change", render);
el.geodesicSelect.addEventListener("change", () => {
  state.selectedIndex = Number(el.geodesicSelect.value);
  render();
});
el.tileToggle.addEventListener("change", render);
el.labelToggle.addEventListener("change", render);
el.domainToggle.addEventListener("change", render);
window.addEventListener("resize", drawScene);

if ("ResizeObserver" in window) {
  const resizeObserver = new ResizeObserver(drawScene);
  resizeObserver.observe(el.canvas);
}

init().catch((error) => {
  el.loadStatus.textContent = "Dataset load failed";
  el.errorBox.textContent = error.message;
  el.errorBox.classList.remove("is-hidden");
});
