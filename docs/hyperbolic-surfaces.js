const state = {
  data: null,
  selectedIndex: 0,
  model: "disk",
  showTiles: true,
  showDomainSegment: true,
  vertices: [],
  sideSamples: [],
  sideCircles: [],
  generatorMatrices: {},
  sideReentryMatrices: [],
  trace: null,
  projection: null,
  sceneSize: null,
  viewZoom: 1,
  viewPan: { x: 0, y: 0 },
  viewDragging: null,
  wrapYaw: 0.74,
  wrapPitch: 0.18
};

const el = {
  loadStatus: document.querySelector("#loadStatus"),
  surfaceSelect: document.querySelector("#surfaceSelect"),
  modelSelect: document.querySelector("#modelSelect"),
  geodesicSelect: document.querySelector("#geodesicSelect"),
  tileToggle: document.querySelector("#tileToggle"),
  domainToggle: document.querySelector("#domainToggle"),
  edgeLegend: document.querySelector("#edgeLegend"),
  sourceLink: document.querySelector("#sourceLink"),
  surfaceDetails: document.querySelector("#surfaceDetails"),
  surfaceTitle: document.querySelector("#surfaceTitle"),
  surfaceSubtitle: document.querySelector("#surfaceSubtitle"),
  canvas: document.querySelector("#surfaceCanvas"),
  zoomIn: document.querySelector("#zoomIn"),
  zoomOut: document.querySelector("#zoomOut"),
  resetView: document.querySelector("#resetView"),
  errorBox: document.querySelector("#errorBox"),
  geodesicRows: document.querySelector("#geodesicRows"),
  recordCount: document.querySelector("#recordCount"),
  geodesicDetails: document.querySelector("#geodesicDetails"),
  selectedJson: document.querySelector("#selectedJson")
};

const ctx = el.canvas.getContext("2d");
const surfaceCatalog = [
  { id: "regular-8gon-genus-2", name: "Regular octagon genus-2 surface", json: "data/hyperbolic/bolza_octagon_geodesics.json" },
  { id: "regular-12gon-genus-3", name: "Regular dodecagon genus-3 surface", json: "data/hyperbolic/regular_genus3_dodecagon_geodesics.json" },
  { id: "regular-16gon-genus-4", name: "Regular 16-gon genus-4 surface", json: "data/hyperbolic/regular_genus4_16gon_geodesics.json" },
  { id: "regular-20gon-genus-5", name: "Regular 20-gon genus-5 surface", json: "data/hyperbolic/regular_genus5_20gon_geodesics.json" }
];
const pairColors = [
  "#0b7285",
  "#b05c00",
  "#6d5bd0",
  "#247a38",
  "#c23b6a",
  "#7b6f13",
  "#3457b1",
  "#7a3c9f",
  "#9a5a2e",
  "#2b7a78",
  "#a13f3f",
  "#506a2f"
];
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

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpPoint(a, b, t) {
  return c(lerp(a.x, b.x, t), lerp(a.y, b.y, t));
}

function smoothstep(t) {
  const clamped = Math.max(0, Math.min(1, t));
  return clamped * clamped * (3 - 2 * clamped);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
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

function pairColor(index) {
  return pairColors[index % pairColors.length];
}

function rgba(hex, alpha) {
  const value = hex.replace("#", "");
  const red = parseInt(value.slice(0, 2), 16);
  const green = parseInt(value.slice(2, 4), 16);
  const blue = parseInt(value.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
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

function composeMatrices(left, right) {
  return {
    a: add(mul(left.a, right.a), mul(left.b, conj(right.b))),
    b: add(mul(left.a, right.b), mul(left.b, conj(right.a)))
  };
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

function geodesicArcData(p, q) {
  const circle = geodesicCircle(p, q);
  if (!circle || circle.radius < 1e-10) {
    return null;
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
  return { circle, startAngle: a0, delta };
}

function geodesicPointBetween(p, q, t) {
  const data = geodesicArcData(p, q);
  if (!data) {
    return add(scale(p, 1 - t), scale(q, t));
  }
  const angle = data.startAngle + data.delta * t;
  return c(
    data.circle.center.x + data.circle.radius * Math.cos(angle),
    data.circle.center.y + data.circle.radius * Math.sin(angle)
  );
}

function geodesicSamples(p, q, count = 80, trim = 0) {
  const start = Math.max(0, trim);
  const end = Math.min(1, 1 - trim);
  const p0 = norm2(p) > 0.999 ? scale(p, 1 - trim) : p;
  const q0 = norm2(q) > 0.999 ? scale(q, 1 - trim) : q;
  const points = [];
  for (let i = 0; i <= count; i += 1) {
    const t = start + ((end - start) * i) / count;
    points.push(geodesicPointBetween(p0, q0, t));
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
  state.sceneSize = { width: rect.width, height: rect.height };
  return state.sceneSize;
}

function viewCenter() {
  const size = state.sceneSize || { width: el.canvas.clientWidth || 1, height: el.canvas.clientHeight || 1 };
  return c(size.width / 2, size.height / 2);
}

function applyViewTransform(point) {
  const center = viewCenter();
  return c(
    center.x + (point.x - center.x) * state.viewZoom + state.viewPan.x,
    center.y + (point.y - center.y) * state.viewZoom + state.viewPan.y
  );
}

function resetViewState() {
  state.viewZoom = 1;
  state.viewPan = c(0, 0);
  state.viewDragging = null;
}

function canvasPoint(event) {
  const rect = el.canvas.getBoundingClientRect();
  return c(event.clientX - rect.left, event.clientY - rect.top);
}

function zoomView(factor, anchor = viewCenter()) {
  const previousZoom = state.viewZoom;
  const nextZoom = clamp(previousZoom * factor, 0.45, 4.5);
  const ratio = nextZoom / previousZoom;
  const center = viewCenter();
  state.viewPan = c(
    anchor.x - center.x - ratio * (anchor.x - center.x - state.viewPan.x),
    anchor.y - center.y - ratio * (anchor.y - center.y - state.viewPan.y)
  );
  state.viewZoom = nextZoom;
  drawScene();
}

function selectedGeodesic() {
  return state.data?.geodesics[state.selectedIndex] || null;
}

function axisSamples() {
  const geo = selectedGeodesic();
  if (!geo) return [];
  return geodesicSamples(fromPair(geo.fixed_points[0]), fromPair(geo.fixed_points[1]), 420, 0.008);
}

function allTracePoints() {
  const trace = state.trace;
  if (!trace) return [];
  return trace.segments.flatMap((segment) => segment);
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
    ...allTracePoints()
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
  return applyViewTransform(state.projection.project(point));
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
    if (sideValue(point, side) < -1e-5) return false;
  }
  return true;
}

function sideValue(point, side) {
  return norm(sub(point, side.center)) - side.radius;
}

function mostViolatedSide(point, epsilon = 1e-7) {
  let index = -1;
  let value = Infinity;
  state.sideCircles.forEach((side, sideIndex) => {
    const current = sideValue(point, side);
    if (current < value) {
      value = current;
      index = sideIndex;
    }
  });
  return value < -epsilon ? index : -1;
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

function sideLabel(index) {
  return state.data.polygon.sides[index]?.label || "";
}

function sideColor(index) {
  const side = state.data.polygon.sides[index];
  return pairColor(side?.pair_color_index || 0);
}

function pairedSideIndex(index) {
  const halfSides = state.data.polygon.p / 2;
  return (index + halfSides) % state.data.polygon.p;
}

function reduceToDomain(rawPoint) {
  let point = rawPoint;
  let map = identity;
  const path = [];
  for (let step = 0; step < 24; step += 1) {
    const sideIndex = mostViolatedSide(point);
    if (sideIndex < 0) break;
    const reentry = state.sideReentryMatrices[sideIndex];
    point = applyMatrix(reentry, point);
    map = composeMatrices(reentry, map);
    path.push(sideIndex);
  }
  return {
    point,
    map,
    key: path.join("/")
  };
}

function longestSegment(segments) {
  let best = null;
  let bestLength = -Infinity;
  for (const segment of segments) {
    let length = 0;
    for (let i = 1; i < segment.length; i += 1) {
      length += norm(sub(segment[i], segment[i - 1]));
    }
    if (length > bestLength) {
      best = segment;
      bestLength = length;
    }
  }
  return best;
}

function startDataForQuotientTrace() {
  const segment = longestSegment(clippedAxisSegments(axisSamples()));
  if (!segment?.length) return null;
  return {
    point: segment[Math.floor(segment.length / 2)],
    conjugator: identity
  };
}

function reducedAxisStartData() {
  const samples = axisSamples();
  if (!samples.length) return null;
  const rawPoint = samples[Math.floor(samples.length / 2)];
  const reduction = reduceToDomain(rawPoint);
  if (!insidePolygon(reduction.point)) return null;
  return {
    point: reduction.point,
    conjugator: reduction.map
  };
}

function exitTransitionBetween(previousRaw, nextRaw, reductionMap) {
  const nextInPreviousChart = applyMatrix(reductionMap, nextRaw);
  let sideIndex = mostViolatedSide(nextInPreviousChart, 1e-10);
  if (sideIndex < 0) {
    let minValue = Infinity;
    state.sideCircles.forEach((side, index) => {
      const value = sideValue(nextInPreviousChart, side);
      if (value < minValue) {
        minValue = value;
        sideIndex = index;
      }
    });
  }

  let lo = 0;
  let hi = 1;
  for (let step = 0; step < 36; step += 1) {
    const mid = (lo + hi) / 2;
    const raw = geodesicPointBetween(previousRaw, nextRaw, mid);
    const point = applyMatrix(reductionMap, raw);
    if (sideValue(point, state.sideCircles[sideIndex]) >= 0) lo = mid;
    else hi = mid;
  }

  const rawBoundary = geodesicPointBetween(previousRaw, nextRaw, hi);
  const exitPoint = applyMatrix(reductionMap, rawBoundary);
  return { sideIndex, exitPoint };
}

function buildQuotientTrace() {
  const geo = selectedGeodesic();
  const startData = startDataForQuotientTrace() || reducedAxisStartData();
  if (!geo || !startData) {
    return {
      segments: clippedAxisSegments(axisSamples()),
      transitions: [],
      fallback: true
    };
  }

  const start = startData.point;
  const matrix = composeMatrices(
    startData.conjugator,
    composeMatrices(matrixFromRecord(geo), inverseMatrix(startData.conjugator))
  );
  const end = applyMatrix(matrix, start);
  const rawSamples = geodesicSamples(start, end, 960, 0);
  if (!rawSamples.length) return { segments: [], transitions: [], fallback: true };

  const first = reduceToDomain(rawSamples[0]);
  const segments = [[first.point]];
  const transitions = [];
  let currentSegment = segments[0];
  let previousRaw = rawSamples[0];
  let previousReduction = first;

  for (let index = 1; index < rawSamples.length; index += 1) {
    const raw = rawSamples[index];
    const reduction = reduceToDomain(raw);
    const lastPoint = currentSegment[currentSegment.length - 1];
    if (reduction.key !== previousReduction.key) {
      const transition = exitTransitionBetween(previousRaw, raw, previousReduction.map);
      const reentryMatrix = state.sideReentryMatrices[transition.sideIndex];
      const reentryPoint = applyMatrix(reentryMatrix, transition.exitPoint);
      currentSegment.push(transition.exitPoint);
      transitions.push({
        index: transitions.length + 1,
        exitSide: transition.sideIndex,
        exitLabel: sideLabel(transition.sideIndex),
        reentrySide: pairedSideIndex(transition.sideIndex),
        reentryLabel: sideLabel(pairedSideIndex(transition.sideIndex)),
        exitPoint: transition.exitPoint,
        reentryPoint
      });
      currentSegment = [reentryPoint, reduction.point];
      segments.push(currentSegment);
    } else if (lastPoint && norm(sub(reduction.point, lastPoint)) > 0.42) {
      currentSegment = [reduction.point];
      segments.push(currentSegment);
    } else {
      currentSegment.push(reduction.point);
    }
    previousRaw = raw;
    previousReduction = reduction;
  }

  return { segments, transitions, fallback: false };
}

function drawDiskBoundary(size) {
  const radius = Math.min(size.width, size.height) * 0.44;
  const center = applyViewTransform(c(size.width / 2, size.height / 2));
  ctx.save();
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius * state.viewZoom, 0, Math.PI * 2);
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
  const axisY = applyViewTransform(c(0, hp.originY)).y;
  ctx.save();
  ctx.strokeStyle = "#aebbc3";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(Math.max(22, Math.min(left, right) - 80 * state.viewZoom), axisY);
  ctx.lineTo(Math.min(size.width - 22, Math.max(left, right) + 80 * state.viewZoom), axisY);
  ctx.stroke();
  ctx.fillStyle = "#64717d";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("real axis", size.width - 26, axisY - 8);
  ctx.restore();
}

function surfaceMeshSpec(neighbor = false) {
  const polygon = state.data.polygon;
  const genus = state.data.surface.genus;
  if (neighbor) {
    return {
      sectors: Math.min(28, Math.max(polygon.p, polygon.p * 2)),
      rings: Math.min(6, Math.max(4, genus + 3))
    };
  }
  return {
    sectors: Math.min(56, Math.max(32, polygon.p * 3)),
    rings: Math.min(10, Math.max(7, genus + 6))
  };
}

function domainMeshPoints(spec) {
  const points = [];
  for (let ring = 0; ring <= spec.rings; ring += 1) {
    const radial = ring / spec.rings;
    const row = [];
    for (let sector = 0; sector <= spec.sectors; sector += 1) {
      const angle = (sector + 0.5) * Math.PI * 2 / spec.sectors;
      const radius = boundaryRadiusAtAngle(angle) * radial;
      row.push(c(radius * Math.cos(angle), radius * Math.sin(angle)));
    }
    points.push(row);
  }
  return points;
}

function projectedGeodesicEdge(a, b, matrix, sampleCount = 7) {
  const start = applyMatrix(matrix, a);
  const end = applyMatrix(matrix, b);
  return geodesicSamples(start, end, sampleCount, 0)
    .map(project)
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
}

function drawProjectedTriangle(vertices, matrix, fill, stroke, width = 0.55) {
  const edges = [
    projectedGeodesicEdge(vertices[0], vertices[1], matrix),
    projectedGeodesicEdge(vertices[1], vertices[2], matrix),
    projectedGeodesicEdge(vertices[2], vertices[0], matrix)
  ];
  if (edges.some((edge) => edge.length < 2)) return;
  ctx.beginPath();
  edges.forEach((edge, edgeIndex) => {
    edge.forEach((point, pointIndex) => {
      if (edgeIndex === 0 && pointIndex === 0) ctx.moveTo(point.x, point.y);
      else if (pointIndex > 0) ctx.lineTo(point.x, point.y);
    });
  });
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.stroke();
}

function drawDomainMesh(matrix = identity, neighbor = false) {
  const polygon = state.data.polygon;
  const spec = surfaceMeshSpec(neighbor);
  const points = domainMeshPoints(spec);
  const stroke = neighbor ? "rgba(38, 47, 55, 0.11)" : "rgba(31, 41, 48, 0.20)";
  const lineWidth = neighbor ? 0.42 : 0.58;
  ctx.save();
  ctx.lineJoin = "round";
  for (let ring = 0; ring < spec.rings; ring += 1) {
    for (let sector = 0; sector < spec.sectors; sector += 1) {
      const sideIndex = Math.min(polygon.p - 1, Math.floor((sector + 0.5) * polygon.p / spec.sectors));
      const color = pairColor(polygon.sides[sideIndex].pair_color_index);
      const alpha = neighbor ? 0.035 : 0.10 + 0.07 * (ring / spec.rings);
      drawProjectedTriangle(
        [points[ring][sector], points[ring + 1][sector], points[ring + 1][sector + 1]],
        matrix,
        rgba(color, alpha),
        stroke,
        lineWidth
      );
      drawProjectedTriangle(
        [points[ring][sector], points[ring + 1][sector + 1], points[ring][sector + 1]],
        matrix,
        rgba(color, alpha * 0.82),
        stroke,
        lineWidth
      );
    }
  }
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
    drawDomainMesh(matrix, true);
  }
}

function drawCentralPolygon() {
  const boundary = transformedBoundary(identity);
  drawPolyline(boundary, {
    close: true,
    fill: "rgba(255,255,255,0.58)"
  });
  drawDomainMesh(identity, false);
  drawPolyline(boundary, {
    close: true,
    stroke: "rgba(32,36,42,0.32)",
    width: 1.1
  });

  state.data.polygon.sides.forEach((side) => {
    drawPolyline(state.sideSamples[side.index], {
      stroke: pairColor(side.pair_color_index),
      width: 3
    });
  });
}

function drawCrossingMarker(point, label, color, filled) {
  const screen = project(point);
  if (!Number.isFinite(screen.x) || !Number.isFinite(screen.y)) return;
  ctx.save();
  ctx.beginPath();
  ctx.arc(screen.x, screen.y, 8, 0, Math.PI * 2);
  ctx.fillStyle = filled ? color : "#ffffff";
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = filled ? "#ffffff" : "#1c2026";
  ctx.font = "700 10px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, screen.x, screen.y + 0.5);
  ctx.restore();
}

function drawQuotientTrace() {
  if (!state.showDomainSegment || !state.trace) return;
  for (const segment of state.trace.segments) {
    drawPolyline(segment, {
      stroke: "#d63f2f",
      width: 3.4
    });
  }

  for (const transition of state.trace.transitions) {
    const color = sideColor(transition.exitSide);
    drawCrossingMarker(transition.exitPoint, String(transition.index), color, true);
    drawCrossingMarker(transition.reentryPoint, String(transition.index), color, false);
  }
}

function wrapLayout(size) {
  const genus = state.data.surface.genus;
  const width = Math.min(size.width - 64, Math.max(380, 145 * genus + 230));
  const height = Math.min(size.height * 0.34, 210);
  const depth = Math.min(width * 0.34, 280);
  const center = c(size.width / 2, size.height / 2);
  const holes = Array.from({ length: genus }, (_, index) => ({
    u: (index + 0.5) / genus,
    rx: Math.min(48, width / (genus * 4.2)),
    ry: Math.min(44, height * 0.24)
  }));
  return { genus, width, height, depth, center, holes };
}

function wrapAngularDistance(a, b) {
  return Math.atan2(Math.sin(a - b), Math.cos(a - b));
}

function projectWrapPoint(layout, point) {
  const yaw = state.wrapYaw;
  const pitch = state.wrapPitch;
  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);
  const cosPitch = Math.cos(pitch);
  const sinPitch = Math.sin(pitch);
  const x1 = point.x * cosYaw - point.z * sinYaw;
  const z1 = point.x * sinYaw + point.z * cosYaw;
  const y2 = point.y * cosPitch - z1 * sinPitch;
  const z2 = point.y * sinPitch + z1 * cosPitch;
  const scaleFactor = 1 / (1 + z2 / 1200);
  const screen = applyViewTransform(c(
    layout.center.x + x1 * scaleFactor,
    layout.center.y + y2 * scaleFactor
  ));
  return {
    x: screen.x,
    y: screen.y,
    depth: z2,
    scale: scaleFactor * state.viewZoom
  };
}

function drawWrapSurfaceBase(size, opacity) {
  if (opacity <= 0.01) return;
  const layout = wrapLayout(size);
  const center = applyViewTransform(c(layout.center.x, layout.center.y + layout.height * 0.42));
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.beginPath();
  ctx.ellipse(center.x, center.y, layout.width * 0.43 * state.viewZoom, layout.height * 0.38 * state.viewZoom, 0, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(56, 76, 82, 0.10)";
  ctx.fill();
  ctx.restore();
}

function drawWrapSurfaceHoles(size, opacity) {
  if (opacity <= 0.01) return;
  const layout = wrapLayout(size);
  const holes = layout.holes.map((hole) => {
    const center = wrapSurfaceScreenPoint(size, layout, hole.u, 0.5);
    return { ...hole, center };
  }).sort((a, b) => b.center.depth - a.center.depth);
  ctx.save();
  for (const hole of holes) {
    const frontness = clamp(0.72 - hole.center.depth / (layout.depth * 1.55), 0.18, 1);
    ctx.globalAlpha = opacity * frontness;
    const rx = hole.rx * hole.center.scale * (0.86 + 0.10 * Math.cos(state.wrapYaw));
    const ry = hole.ry * hole.center.scale;
    ctx.beginPath();
    ctx.ellipse(hole.center.x, hole.center.y, rx, ry, 0, 0, Math.PI * 2);
    ctx.fillStyle = "#f7f9fa";
    ctx.fill();
    ctx.strokeStyle = "rgba(46, 58, 66, 0.50)";
    ctx.lineWidth = 1.8;
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(hole.center.x, hole.center.y + ry * 0.16, rx * 0.72, ry * 0.43, 0, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(46, 58, 66, 0.16)";
    ctx.lineWidth = 1.1;
    ctx.stroke();
  }
  ctx.restore();
}

function boundaryRadiusAtAngle(angle) {
  let lo = 0;
  let hi = 0.999;
  for (let step = 0; step < 34; step += 1) {
    const mid = (lo + hi) / 2;
    const point = c(mid * Math.cos(angle), mid * Math.sin(angle));
    if (insidePolygon(point)) lo = mid;
    else hi = mid;
  }
  return lo;
}

function wrapSurfaceScreenPoint(size, layout, u, v) {
  const theta = u * Math.PI * 2;
  const phi = v * Math.PI * 2;
  const radial = Math.cos(phi);
  const lobe = 1 + 0.08 * Math.cos(theta * layout.genus);
  const majorX = layout.width * 0.30 * lobe;
  const majorZ = layout.depth * 0.76;
  const tubeX = layout.height * (0.35 + 0.035 * Math.sin(theta * layout.genus));
  const tubeY = layout.height * 0.34;
  const x = Math.sin(theta) * (majorX + tubeX * radial);
  const z = Math.cos(theta) * (majorZ + tubeX * radial);
  let y = Math.sin(phi) * tubeY;
  for (const hole of layout.holes) {
    const holeTheta = hole.u * Math.PI * 2;
    const local = wrapAngularDistance(theta, holeTheta);
    const influence = Math.exp(-(local * local) / 0.20);
    y += Math.sin(phi) * influence * hole.ry * 0.18;
  }
  y += Math.sin(theta * layout.genus) * layout.height * 0.045 * (1 + radial * 0.2);
  return projectWrapPoint(layout, { x, y, z });
}

function wrapMeshPoint(size, layout, sector, ring, sectors, rings) {
  const u = sector / sectors;
  const v = ring / rings;
  return wrapSurfaceScreenPoint(size, layout, u, v);
}

function drawScreenTriangle(points, fill, stroke) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  ctx.lineTo(points[1].x, points[1].y);
  ctx.lineTo(points[2].x, points[2].y);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 0.7;
  ctx.stroke();
}

function drawWrapMesh(size) {
  const polygon = state.data.polygon;
  const layout = wrapLayout(size);
  const sectors = Math.max(32, polygon.p * 4);
  const rings = 8;
  const points = [];
  for (let ring = 0; ring <= rings; ring += 1) {
    const row = [];
    for (let sector = 0; sector <= sectors; sector += 1) {
      row.push(wrapMeshPoint(size, layout, sector, ring, sectors, rings));
    }
    points.push(row);
  }

  ctx.save();
  ctx.lineJoin = "round";
  const triangles = [];
  for (let ring = 0; ring < rings; ring += 1) {
    for (let sector = 0; sector < sectors; sector += 1) {
      const sideIndex = Math.min(polygon.p - 1, Math.floor((sector + 0.5) * polygon.p / sectors));
      const color = pairColor(polygon.sides[sideIndex].pair_color_index);
      const alpha = 0.45 + 0.12 * (ring / rings);
      const stroke = "rgba(38, 47, 55, 0.30)";
      const first = [points[ring][sector], points[ring + 1][sector], points[ring + 1][sector + 1]];
      const second = [points[ring][sector], points[ring + 1][sector + 1], points[ring][sector + 1]];
      triangles.push({
        points: first,
        fill: rgba(color, alpha),
        stroke,
        depth: first.reduce((total, point) => total + (point.depth || 0), 0) / first.length
      });
      triangles.push({
        points: second,
        fill: rgba(color, alpha * 0.86),
        stroke,
        depth: second.reduce((total, point) => total + (point.depth || 0), 0) / second.length
      });
    }
  }
  triangles.sort((a, b) => b.depth - a.depth);
  for (const triangle of triangles) {
    drawScreenTriangle(triangle.points, triangle.fill, triangle.stroke);
  }

  ctx.strokeStyle = "rgba(21, 31, 38, 0.46)";
  ctx.lineWidth = 1.15;
  for (let ring = 1; ring < rings; ring += 1) {
    ctx.beginPath();
    for (let sector = 0; sector <= sectors; sector += 1) {
      const point = points[ring][sector];
      if (sector === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    }
    ctx.stroke();
  }

  const stride = Math.max(1, Math.floor(sectors / polygon.p));
  for (let sector = 0; sector < sectors; sector += stride) {
    ctx.beginPath();
    for (let ring = 0; ring <= rings; ring += 1) {
      const point = points[ring][sector];
      if (ring === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    }
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(21, 31, 38, 0.30)";
  ctx.lineWidth = 0.85;
  for (let ring = 0; ring < rings; ring += 1) {
    ctx.beginPath();
    for (let sector = 0; sector < sectors; sector += 2) {
      const a = points[ring][sector];
      const b = points[ring + 1][sector + 1];
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();
  }

  for (let sideIndex = 0; sideIndex < polygon.p; sideIndex += 1) {
    const sector = Math.floor(sideIndex * sectors / polygon.p);
    ctx.beginPath();
    for (let ring = 0; ring <= rings; ring += 1) {
      const point = points[ring][sector];
      if (ring === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    }
    ctx.strokeStyle = pairColor(polygon.sides[sideIndex].pair_color_index);
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.stroke();
  }
  ctx.restore();
}

function drawWrapScene(size) {
  drawWrapSurfaceBase(size, 1);
  drawWrapMesh(size);
  drawWrapSurfaceHoles(size, 1);
}

function drawScene() {
  if (!state.data) return;
  const size = canvasSize();
  ctx.clearRect(0, 0, size.width, size.height);
  ctx.fillStyle = "#f7f9fa";
  ctx.fillRect(0, 0, size.width, size.height);

  if (state.model === "wrap") {
    drawWrapScene(size);
    return;
  }

  prepareProjection(size);
  if (state.model === "disk") drawDiskBoundary(size);
  else drawHalfPlaneBoundary(size);

  drawTiles();
  drawCentralPolygon();
  drawQuotientTrace();
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
    const multiplicity = geo.symmetry_multiplicity || 1;
    const orbitText = multiplicity > 1 ? `  |  mult=${multiplicity}` : "";
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${geo.word}${orbitText}  |  L=${fmt(geo.length, 4)}`;
    el.geodesicSelect.append(option);
  });
}

function populateSurfaces() {
  el.surfaceSelect.innerHTML = "";
  for (const surface of surfaceCatalog) {
    const option = document.createElement("option");
    option.value = surface.id;
    option.textContent = surface.name;
    el.surfaceSelect.append(option);
  }
}

function renderLegend() {
  el.edgeLegend.innerHTML = "";
  for (const generator of state.data.generators) {
    const swatch = document.createElement("i");
    swatch.className = "swatch";
    swatch.style.background = pairColor(generator.side_pair[0]);
    swatch.title = `${generator.letter}/${generator.inverse}`;
    el.edgeLegend.append(swatch);
  }
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
    const multiplicity = document.createElement("td");
    multiplicity.textContent = String(geo.symmetry_multiplicity || 1);
    const letters = document.createElement("td");
    letters.textContent = String(geo.word_length);
    const length = document.createElement("td");
    length.textContent = fmt(geo.length, 5);
    const trace = document.createElement("td");
    trace.textContent = fmt(geo.trace_abs, 5);
    row.append(word, multiplicity, letters, length, trace);
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
  const multiplicity = geo.symmetry_multiplicity || 1;
  const equivalentWords = geo.equivalent_words || [geo.word];
  const visibleEquivalentWords = equivalentWords.slice(0, 12).join(", ");
  const equivalentSummary = equivalentWords.length > 12 ? `${visibleEquivalentWords}, ...` : visibleEquivalentWords;
  const transitions = state.trace?.transitions || [];
  const itinerary = transitions.map((transition) => `${transition.exitLabel}->${transition.reentryLabel}`).join(", ");
  const visiblePath = state.model === "wrap"
    ? "topological wrap sketch"
    : state.trace?.fallback ? "axis clip" : "folded quotient trace";
  setDetails(el.surfaceDetails, [
    ["quotient", `genus ${state.data.surface.genus}, compact`],
    ["area", `${fmt(polygon.area, 5)} = 4*pi*${state.data.surface.genus - 1}`],
    ["polygon", `regular hyperbolic ${polygon.p}-gon`],
    ["edge rule", "opposite sides"],
    ["orbits", String(state.data.geodesics.length)],
    ["search", `words up to ${state.data.enumeration.max_word_length} letters`],
    ["visible path", visiblePath]
  ]);

  el.surfaceTitle.textContent = state.data.surface.name;
  el.surfaceSubtitle.textContent = `${geo.word}: length ${fmt(geo.length, 6)}, |trace| ${fmt(geo.trace_abs, 6)}, multiplicity ${multiplicity}`;
  el.recordCount.textContent = `${state.data.geodesics.length} symmetry orbits`;

  setDetails(el.geodesicDetails, [
    ["word", geo.word],
    ["inverse", geo.inverse_word],
    ["multiplicity", String(multiplicity)],
    ["equivalent words", equivalentSummary],
    ["letters", String(geo.word_length)],
    ["length", fmt(geo.length, 9)],
    ["|trace|", fmt(geo.trace_abs, 9)],
    ["crossings", String(transitions.length)],
    ["itinerary", itinerary || ""],
    ["fixed point 1", `(${fmt(geo.fixed_points[0][0], 5)}, ${fmt(geo.fixed_points[0][1], 5)})`],
    ["fixed point 2", `(${fmt(geo.fixed_points[1][0], 5)}, ${fmt(geo.fixed_points[1][1], 5)})`]
  ]);
  el.selectedJson.textContent = JSON.stringify({
    ...geo,
    quotient_trace: {
      crossings: transitions.length,
      itinerary: transitions.map((transition) => ({
        exit_side: transition.exitLabel,
        reentry_side: transition.reentryLabel
      }))
    }
  }, null, 2);
}

function render() {
  if (!state.data) return;
  state.model = el.modelSelect.value;
  state.showTiles = el.tileToggle.checked;
  state.showDomainSegment = el.domainToggle.checked;
  state.trace = state.model === "wrap" ? null : buildQuotientTrace();
  renderDetails();
  renderGeodesicRows();
  drawScene();
}

function requestStaticRedraw() {
  drawScene();
}

function handleCanvasPointerDown(event) {
  state.viewDragging = {
    pointerId: event.pointerId,
    x: event.clientX,
    y: event.clientY,
    pan: c(state.viewPan.x, state.viewPan.y)
  };
  el.canvas.setPointerCapture?.(event.pointerId);
  el.canvas.classList.add("is-panning");
  event.preventDefault();
}

function handleCanvasPointerMove(event) {
  const drag = state.viewDragging;
  if (!drag || drag.pointerId !== event.pointerId) return;
  state.viewPan = c(
    drag.pan.x + event.clientX - drag.x,
    drag.pan.y + event.clientY - drag.y
  );
  event.preventDefault();
  drawScene();
}

function handleCanvasPointerUp(event) {
  const drag = state.viewDragging;
  if (!drag || drag.pointerId !== event.pointerId) return;
  state.viewDragging = null;
  el.canvas.releasePointerCapture?.(event.pointerId);
  el.canvas.classList.remove("is-panning");
  event.preventDefault();
}

function handleCanvasWheel(event) {
  event.preventDefault();
  const factor = Math.exp(-event.deltaY * 0.0012);
  zoomView(factor, canvasPoint(event));
}

function prepareGeometry() {
  const polygon = state.data.polygon;
  state.generatorMatrices = {};
  for (const generator of state.data.generators) {
    const matrix = matrixFromRecord(generator);
    state.generatorMatrices[generator.letter] = matrix;
    state.generatorMatrices[generator.inverse] = inverseMatrix(matrix);
  }
  const foot = polygon.side_foot_radius_disk;
  const centerDistance = (foot + 1 / foot) / 2;
  const sideRadius = (1 / foot - foot) / 2;
  state.sideCircles = polygon.sides.map((side) => {
    const angle = side.normal_angle;
    return {
      center: c(centerDistance * Math.cos(angle), centerDistance * Math.sin(angle)),
      radius: sideRadius
    };
  });
  const vertexRadius = polygon.vertex_radius_disk;
  const halfAngle = Math.PI / polygon.p;
  state.sideSamples = polygon.sides.map((side) => {
    const angle = side.normal_angle;
    const start = c(vertexRadius * Math.cos(angle - halfAngle), vertexRadius * Math.sin(angle - halfAngle));
    const end = c(vertexRadius * Math.cos(angle + halfAngle), vertexRadius * Math.sin(angle + halfAngle));
    return geodesicSamples(start, end, 96, 0);
  });
  state.sideReentryMatrices = polygon.sides.map((side) => {
    const base = side.pairing_generator;
    return side.index < polygon.p / 2 ? state.generatorMatrices[base.toUpperCase()] : state.generatorMatrices[base];
  });
}

async function init() {
  populateSurfaces();
  await loadSurface(surfaceCatalog[0].id);
}

async function loadSurface(surfaceId) {
  const surface = surfaceCatalog.find((item) => item.id === surfaceId) || surfaceCatalog[0];
  el.surfaceSelect.value = surface.id;
  el.sourceLink.href = surface.json;
  el.loadStatus.textContent = `Loading ${surface.name}...`;
  const res = await fetch(surface.json);
  if (!res.ok) throw new Error(`Could not load geodesic dataset (${res.status})`);
  state.data = await res.json();
  state.selectedIndex = 0;
  state.trace = null;
  resetViewState();
  prepareGeometry();
  populateGeodesicSelect();
  renderLegend();
  el.loadStatus.textContent = `${state.data.geodesics.length} symmetry orbits generated locally`;
  render();
}

el.surfaceSelect.addEventListener("change", () => {
  loadSurface(el.surfaceSelect.value).catch((error) => {
    el.loadStatus.textContent = "Dataset load failed";
    el.errorBox.textContent = error.message;
    el.errorBox.classList.remove("is-hidden");
  });
});
el.modelSelect.addEventListener("change", () => {
  resetViewState();
  render();
});
el.geodesicSelect.addEventListener("change", () => {
  state.selectedIndex = Number(el.geodesicSelect.value);
  render();
});
el.tileToggle.addEventListener("change", render);
el.domainToggle.addEventListener("change", render);
window.addEventListener("resize", requestStaticRedraw);
el.zoomIn.addEventListener("click", () => zoomView(1.22));
el.zoomOut.addEventListener("click", () => zoomView(1 / 1.22));
el.resetView.addEventListener("click", () => {
  resetViewState();
  drawScene();
});
el.canvas.addEventListener("pointerdown", handleCanvasPointerDown);
el.canvas.addEventListener("pointermove", handleCanvasPointerMove);
el.canvas.addEventListener("pointerup", handleCanvasPointerUp);
el.canvas.addEventListener("pointercancel", handleCanvasPointerUp);
el.canvas.addEventListener("wheel", handleCanvasWheel, { passive: false });

if ("ResizeObserver" in window) {
  const resizeObserver = new ResizeObserver(requestStaticRedraw);
  resizeObserver.observe(el.canvas);
}

init().catch((error) => {
  el.loadStatus.textContent = "Dataset load failed";
  el.errorBox.textContent = error.message;
  el.errorBox.classList.remove("is-hidden");
});
