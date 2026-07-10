import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DATA_URL = "data/tetra/billiards_inventory.json";
const SINGULAR_DATA_URL = "data/tetra/singular_normal_cones.json";
const VERTEX_NAMES = ["A", "B", "C", "D"];
const FACES = {
  A: ["B", "C", "D"],
  B: ["A", "C", "D"],
  C: ["A", "B", "D"],
  D: ["A", "B", "C"]
};
const EDGES = [
  ["A", "B"], ["A", "C"], ["A", "D"],
  ["B", "C"], ["B", "D"], ["C", "D"]
];
const CUBE_EDGES = [
  [[0, 0, 0], [1, 0, 0]],
  [[0, 0, 0], [0, 1, 0]],
  [[0, 0, 0], [0, 0, 1]],
  [[1, 1, 1], [0, 1, 1]],
  [[1, 1, 1], [1, 0, 1]],
  [[1, 1, 1], [1, 1, 0]],
  [[1, 1, 0], [1, 0, 0]],
  [[1, 1, 0], [0, 1, 0]],
  [[1, 0, 1], [1, 0, 0]],
  [[1, 0, 1], [0, 0, 1]],
  [[0, 1, 1], [0, 1, 0]],
  [[0, 1, 1], [0, 0, 1]]
];
const FACE_COLORS = {
  A: 0x4e79a7,
  B: 0xf28e2b,
  C: 0x59a14f,
  D: 0xe15759,
  "?": 0x777777
};
const FACE_CSS = {
  A: "#4e79a7",
  B: "#f28e2b",
  C: "#59a14f",
  D: "#e15759",
  "?": "#777777"
};
const STRATUM_COLORS = {
  edge: 0xff7a00,
  vertex: 0x8b5cf6,
  singular: 0xff7a00
};
const STRATUM_CSS = {
  edge: "#ff7a00",
  vertex: "#8b5cf6",
  singular: "#ff7a00"
};
const FILTER_LABELS = {
  all: "All paths",
  ordinary: "Ordinary face hits",
  edge: "Singular edge-only hits",
  vertex: "Singular vertex-only hits",
  edge_vertex: "Singular edge + vertex hits"
};

const els = {
  status: document.getElementById("inventoryStatus"),
  period: document.getElementById("periodSelect"),
  kind: document.getElementById("kindSelect"),
  search: document.getElementById("wordSearch"),
  sort: document.getElementById("sortSelect"),
  summary: document.getElementById("summaryDetails"),
  title: document.getElementById("orbitTitle"),
  subtitle: document.getElementById("orbitSubtitle"),
  reset: document.getElementById("resetCamera"),
  play: document.getElementById("playAnimation"),
  stage: document.getElementById("stage"),
  foldedCanvas: document.getElementById("foldedCanvas"),
  unfoldedCanvas: document.getElementById("unfoldedCanvas"),
  empty: document.getElementById("stageEmpty"),
  note: document.getElementById("stageNote"),
  rows: document.getElementById("orbitRows"),
  sortHeaders: [...document.querySelectorAll(".sort-header")],
  count: document.getElementById("recordCount"),
  details: document.getElementById("orbitDetails"),
  points: document.getElementById("pointRows")
};

let inventory = null;
let selectedId = null;
let sortKey = "period";
let sortDirection = "asc";
let selectedPointIndex = 0;
let animationRunning = false;
let animationProgress = 0;
let lastAnimationTime = null;
let currentPathPoints = { folded: [], unfolded: [] };
const ANIMATION_SECONDS = 7.5;
const FLICK_RADIANS_PER_PIXEL = 0.004;
const FLICK_DECAY_PER_SECOND = 2.6;
const FLICK_STOP_SPEED = 0.018;

const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x15181d, transparent: true, opacity: 0.82 });
const unfoldedCopyEdgeMaterial = new THREE.LineBasicMaterial({ color: 0x515963, transparent: true, opacity: 0.5 });
const cubeMaterial = new THREE.LineBasicMaterial({ color: 0x3f4c5a, transparent: true, opacity: 0.28 });
const pathMaterial = new THREE.LineBasicMaterial({ color: 0xd0342c });
const singularPathMaterial = new THREE.LineBasicMaterial({ color: 0xff7a00, linewidth: 2 });
const unfoldedPathMaterial = new THREE.LineBasicMaterial({ color: 0xd0342c, linewidth: 2 });

function createViewer(canvas, mode) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0xffffff, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 2000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;

  const root = new THREE.Group();
  const overlay = new THREE.Group();
  const movingMarker = new THREE.Mesh(
    new THREE.SphereGeometry(1, 24, 16),
    new THREE.MeshBasicMaterial({ color: 0x111111 })
  );
  const wrapMarker = new THREE.Mesh(
    new THREE.SphereGeometry(1, 24, 16),
    new THREE.MeshBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.45 })
  );
  movingMarker.visible = false;
  wrapMarker.visible = false;
  overlay.add(movingMarker, wrapMarker);
  scene.add(root, overlay);

  return {
    mode,
    canvas,
    renderer,
    scene,
    camera,
    controls,
    root,
    overlay,
    movingMarker,
    wrapMarker,
    bounds: null,
    markerRadius: 0.035,
    spin: {
      dragging: false,
      pointerId: null,
      lastX: 0,
      lastY: 0,
      lastTime: 0,
      lastFrameTime: null,
      velocity: new THREE.Vector2()
    }
  };
}

const views = {
  folded: createViewer(els.foldedCanvas, "folded"),
  unfolded: createViewer(els.unfoldedCanvas, "unfolded")
};

function applyCameraSpin(view, yaw, pitch) {
  const target = view.controls.target;
  const offset = view.camera.position.clone().sub(target);
  const up = view.camera.up.clone().normalize();
  offset.applyAxisAngle(up, -yaw);

  const right = new THREE.Vector3().crossVectors(up, offset).normalize();
  if (right.lengthSq() > 0.0001) {
    const nextOffset = offset.clone().applyAxisAngle(right, -pitch);
    const tilt = Math.abs(nextOffset.clone().normalize().dot(up));
    if (tilt < 0.96) offset.copy(nextOffset);
  }

  view.camera.position.copy(target).add(offset);
  view.camera.lookAt(target);
}

function updateFlickSpin(view, time) {
  if (view.mode !== "folded") return;
  const spin = view.spin;
  if (spin.lastFrameTime == null) {
    spin.lastFrameTime = time;
    return;
  }
  const delta = Math.min(Math.max((time - spin.lastFrameTime) / 1000, 0), 0.05);
  spin.lastFrameTime = time;
  if (spin.dragging || spin.velocity.length() < FLICK_STOP_SPEED) return;
  applyCameraSpin(view, spin.velocity.x * delta, spin.velocity.y * delta);
  spin.velocity.multiplyScalar(Math.exp(-FLICK_DECAY_PER_SECOND * delta));
}

function installFlickSpin(view) {
  const spin = view.spin;
  view.canvas.addEventListener("pointerdown", event => {
    if (event.button != null && event.button !== 0) return;
    spin.dragging = true;
    spin.pointerId = event.pointerId;
    spin.lastX = event.clientX;
    spin.lastY = event.clientY;
    spin.lastTime = event.timeStamp;
    spin.velocity.set(0, 0);
  }, { passive: true });

  window.addEventListener("pointermove", event => {
    if (!spin.dragging || event.pointerId !== spin.pointerId) return;
    const elapsed = Math.max((event.timeStamp - spin.lastTime) / 1000, 1 / 120);
    const dx = event.clientX - spin.lastX;
    const dy = event.clientY - spin.lastY;
    spin.velocity.set(
      THREE.MathUtils.clamp((dx / elapsed) * FLICK_RADIANS_PER_PIXEL, -4.2, 4.2),
      THREE.MathUtils.clamp((dy / elapsed) * FLICK_RADIANS_PER_PIXEL, -4.2, 4.2)
    );
    spin.lastX = event.clientX;
    spin.lastY = event.clientY;
    spin.lastTime = event.timeStamp;
  }, { passive: true });

  const endDrag = event => {
    if (!spin.dragging || event.pointerId !== spin.pointerId) return;
    spin.dragging = false;
    spin.pointerId = null;
  };
  window.addEventListener("pointerup", endDrag, { passive: true });
  window.addEventListener("pointercancel", endDrag, { passive: true });
}

installFlickSpin(views.folded);

function animate(time = 0) {
  updateAnimation(time);
  for (const view of Object.values(views)) {
    updateFlickSpin(view, time);
    view.controls.update();
    view.renderer.render(view.scene, view.camera);
  }
  requestAnimationFrame(animate);
}
animate();

function resizeViewer(view) {
  const rect = view.canvas.getBoundingClientRect();
  const width = Math.max(260, Math.round(rect.width));
  const height = Math.max(260, Math.round(rect.height || rect.width));
  view.renderer.setSize(width, height, false);
  view.camera.aspect = width / height;
  view.camera.updateProjectionMatrix();
  resetCamera(view);
}

for (const view of Object.values(views)) {
  new ResizeObserver(() => resizeViewer(view)).observe(view.canvas);
  resizeViewer(view);
}
window.addEventListener("resize", () => {
  for (const view of Object.values(views)) resizeViewer(view);
});

function bigIntFrom(value) {
  return BigInt(String(value));
}

function integerText(value) {
  return /^[+-]?\d+$/.test(String(value));
}

function zeroValue(value) {
  const text = String(value);
  if (integerText(text)) return BigInt(text) === 0n;
  return Math.abs(parseFractionApprox(text)) < 1e-12;
}

function scalarApprox(value) {
  return parseFractionApprox(String(value));
}

function formatApprox(value) {
  if (Math.abs(value) < 5e-13) return "0";
  return Number(value).toPrecision(12).replace(/\.?0+$/, "");
}

function ratio(value, denominator) {
  if (!integerText(value) || !integerText(denominator)) {
    return scalarApprox(value) / scalarApprox(denominator);
  }
  let a = bigIntFrom(value);
  let b = bigIntFrom(denominator);
  if (a === 0n) return 0;
  const neg = (a < 0n) !== (b < 0n);
  if (a < 0n) a = -a;
  if (b < 0n) b = -b;
  const sa = a.toString();
  const sb = b.toString();
  const k = 16;
  const ma = Number(sa.slice(0, k)) / 10 ** Math.min(k, sa.length);
  const mb = Number(sb.slice(0, k)) / 10 ** Math.min(k, sb.length);
  const estimate = (ma / mb) * 10 ** (sa.length - sb.length);
  return neg ? -estimate : estimate;
}

function compareIntegerStrings(a, b) {
  if (!integerText(a) || !integerText(b)) {
    const aa = scalarApprox(a);
    const bb = scalarApprox(b);
    if (Number.isFinite(aa) && Number.isFinite(bb)) return aa - bb;
    return String(a).localeCompare(String(b));
  }
  const aa = String(a).replace(/^-/, "");
  const bb = String(b).replace(/^-/, "");
  if (aa.length !== bb.length) return aa.length - bb.length;
  return aa.localeCompare(bb);
}

function parseFractionApprox(text) {
  if (text == null || text === "") return Number.POSITIVE_INFINITY;
  const parts = String(text).split("/");
  if (parts.length === 1) return Number(parts[0]);
  return Number(parts[0]) / Number(parts[1]);
}

function gcdBigInt(a, b) {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b) {
    const next = a % b;
    a = b;
    b = next;
  }
  return a;
}

function fractionText(numerator, denominator) {
  if (!integerText(numerator) || !integerText(denominator)) {
    const den = scalarApprox(denominator);
    return den === 1 ? String(numerator) : formatApprox(scalarApprox(numerator) / den);
  }
  let a = bigIntFrom(numerator);
  let b = bigIntFrom(denominator);
  if (b < 0n) {
    a = -a;
    b = -b;
  }
  const g = gcdBigInt(a, b);
  a /= g;
  b /= g;
  return b === 1n ? a.toString() : `${a}/${b}`;
}

function barycentricExact(row, denominator) {
  return row.map(value => fractionText(value, denominator));
}

function pointExactFromBarycentric(row, denominator) {
  if (!integerText(denominator) || row.some(value => !integerText(value))) {
    const point = pointFloatFromBarycentric(row, denominator);
    return point.map(formatApprox);
  }
  const nums = row.map(bigIntFrom);
  const x = nums[0] + nums[3];
  const y = nums[1] + nums[3];
  const z = nums[2] + nums[3];
  return [x, y, z].map(value => fractionText(value, denominator));
}

function pointFloatFromBarycentric(row, denominator) {
  const nums = row.map(value => ratio(value, denominator));
  return [
    nums[0] + nums[3],
    nums[1] + nums[3],
    nums[2] + nums[3]
  ];
}

function faceFromBarycentric(row) {
  const zeroIndex = row.findIndex(value => zeroValue(value));
  return zeroIndex >= 0 ? VERTEX_NAMES[zeroIndex] : "?";
}

function inferStratum(row) {
  const zero = [];
  row.forEach((value, index) => {
    if (zeroValue(value)) zero.push(VERTEX_NAMES[index]);
  });
  const label = zero.join("");
  if (zero.length === 1) return { label, type: "face", face: zero[0] };
  if (zero.length === 2) return { label, type: "edge", face: "edge" };
  if (zero.length === 3) return { label, type: "vertex", face: "vertex" };
  return { label: "?", type: "singular", face: "?" };
}

function stratumFromLabel(label, row) {
  if (row) return inferStratum(row);
  const text = String(label ?? "");
  if (/^\([A-D]{2}\)$/.test(text)) return { label: text, type: "edge", face: "edge" };
  if (/^\[[A-D]\]$/.test(text)) return { label: text, type: "vertex", face: "vertex" };
  if (/^[A-D]$/.test(text)) return { label: text, type: "face", face: text };
  return inferStratum(row);
}

function isSingular(orbit) {
  return orbit.kind === "singular_normal_cone" || orbit.singular === true;
}

function pointStyle(orbit, index) {
  if (isSingular(orbit)) {
    const stratum = stratumFromLabel(orbit.stratum_word?.[index], orbit.barycentric_points[index]);
    if (stratum.type === "face") {
      return {
        ...stratum,
        color: FACE_COLORS[stratum.face] ?? FACE_COLORS["?"],
        css: FACE_CSS[stratum.face] ?? FACE_CSS["?"]
      };
    }
    return {
      ...stratum,
      color: STRATUM_COLORS[stratum.type] ?? STRATUM_COLORS.singular,
      css: STRATUM_CSS[stratum.type] ?? STRATUM_CSS.singular
    };
  }
  const face = orbit.faces?.[index] ?? faceFromBarycentric(orbit.barycentric_points[index]);
  return {
    label: face,
    type: "face",
    face,
    color: FACE_COLORS[face] ?? FACE_COLORS["?"],
    css: FACE_CSS[face] ?? FACE_CSS["?"]
  };
}

function displayWord(orbit) {
  if (isSingular(orbit) && Array.isArray(orbit.barycentric_points)) {
    return orbit.barycentric_points.map(row => inferStratum(row).label).join(" ");
  }
  return orbit.word;
}

function stratumProfile(orbit) {
  if (!isSingular(orbit)) return { ordinary: true, hasEdge: false, hasVertex: false };
  const strata = Array.isArray(orbit.barycentric_points)
    ? orbit.barycentric_points.map(inferStratum)
    : [];
  return {
    ordinary: false,
    hasEdge: strata.some(stratum => stratum.type === "edge"),
    hasVertex: strata.some(stratum => stratum.type === "vertex")
  };
}

function pathClassLabel(orbit) {
  const profile = stratumProfile(orbit);
  if (profile.ordinary) return "ordinary face hits";
  if (profile.hasEdge && profile.hasVertex) return "singular edge + vertex hits";
  if (profile.hasVertex) return "singular vertex-only hits";
  if (profile.hasEdge) return "singular edge-only hits";
  return "singular boundary hits";
}

function stratumFilterKey(orbit) {
  const profile = stratumProfile(orbit);
  if (profile.ordinary) return "ordinary";
  if (profile.hasEdge && profile.hasVertex) return "edge_vertex";
  if (profile.hasVertex) return "vertex";
  if (profile.hasEdge) return "edge";
  return "edge_vertex";
}

const PERMUTATIONS_4 = (() => {
  const out = [];
  const rec = (pool, row) => {
    if (row.length === 4) {
      out.push(row);
      return;
    }
    pool.forEach((value, index) => rec(pool.filter((_, j) => j !== index), [...row, value]));
  };
  rec([0, 1, 2, 3], []);
  return out;
})();
const copyCountCache = new Map();

function rowKey(row) {
  return row.map(value => String(value)).join(",");
}

function cyclicKeys(matrix) {
  const rows = matrix.map(rowKey);
  const keys = [];
  for (let shift = 0; shift < rows.length; shift++) {
    keys.push(rows.map((_, i) => rows[(i + shift) % rows.length]).join(";"));
  }
  const reversed = [...rows].reverse();
  for (let shift = 0; shift < reversed.length; shift++) {
    keys.push(reversed.map((_, i) => reversed[(i + shift) % reversed.length]).join(";"));
  }
  return keys;
}

function canonicalPathKey(matrix) {
  return cyclicKeys(matrix).sort()[0];
}

function copyCountOf(orbit) {
  if (copyCountCache.has(orbit.id)) return copyCountCache.get(orbit.id);
  const seen = new Set();
  for (const permutation of PERMUTATIONS_4) {
    const matrix = orbit.barycentric_points.map(row => permutation.map(index => row[index]));
    seen.add(canonicalPathKey(matrix));
  }
  const count = seen.size;
  copyCountCache.set(orbit.id, count);
  return count;
}

function degeneracyLabel(orbit) {
  const familyDimension = Number(orbit.family_dimension ?? orbit.family?.dimension ?? 0);
  if (familyDimension > 0) {
    const familyCopyCount = Number(orbit.family_copy_count ?? orbit.family?.copy_count ?? copyCountOf(orbit));
    return familyCopyCount === 1 ? `inf (${familyDimension}D)` : `${familyCopyCount} inf (${familyDimension}D)`;
  }
  return String(orbit.copyCounts ?? orbit.copy_count ?? copyCountOf(orbit));
}

function normalizeOrbit(raw, fallbackKind) {
  const orbit = { ...raw };
  orbit.kind = orbit.kind ?? fallbackKind;
  orbit.word = String(orbit.word ?? orbit.id ?? "");
  orbit.period = Number(orbit.period ?? orbit.word.length);
  orbit.length_numeric = Number(orbit.length_numeric ?? orbit.length ?? 0);
  orbit.barycentric_denominator = String(orbit.barycentric_denominator);
  orbit.barycentric_points = orbit.barycentric_points.map(row => row.map(value => String(value)));
  orbit.faces = orbit.faces ?? orbit.barycentric_points.map(faceFromBarycentric);
  orbit.barycentric_exact = orbit.barycentric_exact ?? orbit.barycentric_points.map(row => barycentricExact(row, orbit.barycentric_denominator));
  orbit.points_xyz_exact = orbit.points_xyz_exact ?? orbit.barycentric_points.map(row => pointExactFromBarycentric(row, orbit.barycentric_denominator));
  orbit.points_xyz = orbit.points_xyz ?? orbit.barycentric_points.map(row => pointFloatFromBarycentric(row, orbit.barycentric_denominator));
  orbit.axis_direction = orbit.axis_direction ?? [];
  orbit.initial_direction = orbit.initial_direction ?? [];
  orbit.height = String(orbit.height ?? orbit.barycentric_denominator);
  if (!isSingular(orbit) && orbit.boundary_margin == null) {
    const positive = orbit.barycentric_points.flat().filter(value => scalarApprox(value) > 0);
    orbit.boundary_margin = positive.length
      ? fractionText(positive.reduce((a, b) => scalarApprox(a) < scalarApprox(b) ? a : b), orbit.barycentric_denominator)
      : "";
  }
  orbit.boundary_margin = orbit.boundary_margin ?? "";
  if (!Number.isFinite(orbit.centroid_distance_numeric)) {
    const center = [0, 1, 2].map(axis => orbit.points_xyz.reduce((sum, point) => sum + point[axis], 0) / orbit.points_xyz.length);
    orbit.path_center_xyz = orbit.path_center_xyz ?? center;
    orbit.centroid_distance_numeric = Math.hypot(center[0] - 0.5, center[1] - 0.5, center[2] - 0.5);
  }
  orbit.search_text = `${orbit.id} ${displayWord(orbit)}`.toUpperCase();
  return orbit;
}

function combinedPeriodCounts(orbits) {
  const counts = {};
  for (const orbit of orbits) counts[orbit.period] = (counts[orbit.period] ?? 0) + 1;
  return Object.fromEntries(Object.entries(counts).sort((a, b) => Number(a[0]) - Number(b[0])));
}

function vertexMapFromInventory() {
  const vertices = inventory.tetrahedron.vertices;
  return Object.fromEntries(
    VERTEX_NAMES.map(name => [name, new THREE.Vector3(...vertices[name])])
  );
}

function baryPoint(row, denominator, vertices) {
  const out = new THREE.Vector3();
  for (let i = 0; i < VERTEX_NAMES.length; i++) {
    out.addScaledVector(vertices[VERTEX_NAMES[i]], ratio(row[i], denominator));
  }
  return out;
}

function clearRoot(view) {
  while (view.root.children.length) {
    const child = view.root.children.pop();
    child.traverse?.(node => {
      if (node.geometry) node.geometry.dispose();
    });
  }
}

function triangleGeometry(points) {
  const geometry = new THREE.BufferGeometry();
  geometry.setFromPoints(points);
  geometry.setIndex([0, 1, 2]);
  geometry.computeVertexNormals();
  return geometry;
}

function addFace(group, vertices, face, opacity = 0.16) {
  const material = new THREE.MeshBasicMaterial({
    color: FACE_COLORS[face],
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
    depthWrite: false
  });
  const mesh = new THREE.Mesh(triangleGeometry(FACES[face].map(name => vertices[name])), material);
  group.add(mesh);
}

function addEdges(group, vertices, material = edgeMaterial) {
  const points = [];
  for (const [a, b] of EDGES) {
    points.push(vertices[a], vertices[b]);
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  group.add(new THREE.LineSegments(geometry, material));
}

function addUnitCubeWireframe(group) {
  const points = [];
  for (const [a, b] of CUBE_EDGES) {
    points.push(new THREE.Vector3(...a), new THREE.Vector3(...b));
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  group.add(new THREE.LineSegments(geometry, cubeMaterial));
}

function addTetrahedron(group, vertices, opacity = 0.15, material = edgeMaterial, coloredFace = undefined) {
  const faces = coloredFace === undefined ? VERTEX_NAMES : [coloredFace].filter(Boolean);
  for (const face of faces) addFace(group, vertices, face, opacity);
  addEdges(group, vertices, material);
}

function markerGeometry(style, radius) {
  if (style.type === "edge") return new THREE.BoxGeometry(radius * 1.55, radius * 1.55, radius * 1.55);
  if (style.type === "vertex") return new THREE.OctahedronGeometry(radius * 1.4, 0);
  return new THREE.SphereGeometry(radius, 18, 12);
}

function addPointMarker(group, point, style, selected) {
  const radius = 0.026;
  const material = new THREE.MeshBasicMaterial({ color: style.color });
  const mesh = new THREE.Mesh(markerGeometry(style, radius), material);
  mesh.position.copy(point);
  group.add(mesh);
}

function addPath(group, points, material = pathMaterial) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  group.add(new THREE.Line(geometry, material));
}

function reflectedVerticesAcrossPlane(vertices, point, normal) {
  const unit = normal.clone();
  if (unit.lengthSq() < 1e-18) return Object.fromEntries(VERTEX_NAMES.map(name => [name, vertices[name].clone()]));
  unit.normalize();
  return Object.fromEntries(
    VERTEX_NAMES.map(name => {
      const vertex = vertices[name];
      return [name, vertex.clone().sub(unit.clone().multiplyScalar(2 * vertex.clone().sub(point).dot(unit)))];
    })
  );
}

function foldedPoints(orbit) {
  return orbit.points_xyz.map(point => new THREE.Vector3(point[0], point[1], point[2]));
}

function repeatedInitialSegmentPath(points) {
  if (points.length < 2) return points.map(point => point.clone());
  return [
    points[points.length - 1].clone(),
    ...points.map(point => point.clone()),
    points[0].clone()
  ];
}

function copyPoint(orbit, vertices, index) {
  const n = orbit.barycentric_points.length;
  return baryPoint(orbit.barycentric_points[((index % n) + n) % n], orbit.barycentric_denominator, vertices);
}

function reflectionNormal(orbit, vertices, index) {
  const previous = copyPoint(orbit, vertices, index - 1);
  const current = copyPoint(orbit, vertices, index);
  const next = copyPoint(orbit, vertices, index + 1);
  const incoming = current.clone().sub(previous);
  const outgoing = next.clone().sub(current);
  if (incoming.lengthSq() > 1e-18) incoming.normalize();
  if (outgoing.lengthSq() > 1e-18) outgoing.normalize();
  return incoming.sub(outgoing);
}

function reflectionFaceForHit(orbit, index) {
  const style = pointStyle(orbit, index);
  return style.type === "face" && FACE_COLORS[style.face] ? style.face : null;
}

function unfoldedChain(orbit) {
  const n = orbit.barycentric_points.length;
  let vertices = vertexMapFromInventory();
  const copies = [];
  const points = [];
  const markerIndices = [];
  const pushCopy = reflectionFace => {
    copies.push({
      vertices: Object.fromEntries(VERTEX_NAMES.map(name => [name, vertices[name].clone()])),
      reflectionFace
    });
  };

  pushCopy(n > 0 ? reflectionFaceForHit(orbit, 0) : null);
  if (n === 0) return { copies, points, markerIndices };

  points.push(copyPoint(orbit, vertices, n - 1));
  markerIndices.push(n - 1);
  points.push(copyPoint(orbit, vertices, 0));
  markerIndices.push(0);

  for (let i = 0; i < n; i++) {
    const point = copyPoint(orbit, vertices, i);
    const normal = reflectionNormal(orbit, vertices, i);
    vertices = reflectedVerticesAcrossPlane(vertices, point, normal);
    pushCopy(i + 1 < n ? reflectionFaceForHit(orbit, i + 1) : null);
    const nextIndex = (i + 1) % n;
    points.push(copyPoint(orbit, vertices, nextIndex));
    markerIndices.push(nextIndex);
  }
  return { copies, points, markerIndices };
}

function updateBoundsAndCamera(view) {
  const group = view.root;
  const box = new THREE.Box3().setFromObject(group);
  if (box.isEmpty()) {
    view.bounds = null;
    return;
  }
  view.bounds = box;
  const size = new THREE.Vector3();
  box.getSize(size);
  view.markerRadius = Math.max(size.x, size.y, size.z, 1) * 0.018;
  resetCamera(view);
}

function fitDistanceForBox(camera, box, direction, padding) {
  const center = new THREE.Vector3();
  box.getCenter(center);
  const forward = direction.clone().multiplyScalar(-1).normalize();
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  if (right.lengthSq() < 0.0001) right.set(1, 0, 0);
  const up = new THREE.Vector3().crossVectors(right, forward).normalize();
  const corners = [
    [box.min.x, box.min.y, box.min.z],
    [box.min.x, box.min.y, box.max.z],
    [box.min.x, box.max.y, box.min.z],
    [box.min.x, box.max.y, box.max.z],
    [box.max.x, box.min.y, box.min.z],
    [box.max.x, box.min.y, box.max.z],
    [box.max.x, box.max.y, box.min.z],
    [box.max.x, box.max.y, box.max.z]
  ].map(point => new THREE.Vector3(...point).sub(center));
  const halfWidth = Math.max(...corners.map(point => Math.abs(point.dot(right))));
  const halfHeight = Math.max(...corners.map(point => Math.abs(point.dot(up))));
  const halfDepth = Math.max(...corners.map(point => Math.abs(point.dot(forward))));
  const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
  const widthDistance = halfWidth / (tanHalfFov * camera.aspect);
  const heightDistance = halfHeight / tanHalfFov;
  return (Math.max(widthDistance, heightDistance) + halfDepth) * padding;
}

function resetCamera(view = null) {
  if (view == null) {
    for (const item of Object.values(views)) resetCamera(item);
    return;
  }
  const { camera, controls } = view;
  view.spin.dragging = false;
  view.spin.pointerId = null;
  view.spin.velocity.set(0, 0);
  view.spin.lastFrameTime = null;
  if (!view.bounds) {
    camera.position.set(2.2, 2.4, 2.0);
    controls.target.set(0.5, 0.5, 0.5);
    controls.update();
    return;
  }
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  view.bounds.getCenter(center);
  view.bounds.getSize(size);
  const maxSize = Math.max(size.x, size.y, size.z, 1);
  const direction = view.mode === "unfolded"
    ? new THREE.Vector3(0.35, -0.9, 0.5)
    : new THREE.Vector3(1.35, 1.45, 1.2);
  direction.normalize();
  const padding = view.mode === "unfolded" ? 0.92 : 0.88;
  const minDistanceScale = view.mode === "unfolded" ? 0.7 : 0.62;
  const fitDistance = Math.max(fitDistanceForBox(camera, view.bounds, direction, padding), maxSize * minDistanceScale);
  camera.position.copy(center).addScaledVector(direction, fitDistance);
  camera.near = Math.max(0.01, maxSize / 500);
  camera.far = Math.max(1000, maxSize * 10);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function orientedChain(orbit) {
  const chain = unfoldedChain(orbit);
  if (chain.points.length < 2) return chain;
  const directionEnd = chain.points.length > 2 ? chain.points[chain.points.length - 2] : chain.points[chain.points.length - 1];
  const direction = directionEnd.clone().sub(chain.points[0]).normalize();
  if (direction.lengthSq() < 0.0001) return chain;
  const q = new THREE.Quaternion().setFromUnitVectors(direction, new THREE.Vector3(1, 0, 0));
  return {
    copies: chain.copies.map(copy => ({
      ...copy,
      vertices: Object.fromEntries(VERTEX_NAMES.map(name => [name, copy.vertices[name].clone().applyQuaternion(q)]))
    })),
    points: chain.points.map(point => point.clone().applyQuaternion(q)),
    markerIndices: chain.markerIndices
  };
}

function drawFolded(view, orbit) {
  const vertices = vertexMapFromInventory();
  addUnitCubeWireframe(view.root);
  addTetrahedron(view.root, vertices, 0.16, edgeMaterial);
  const points = foldedPoints(orbit);
  const pathPoints = [...points, points[0]];
  addPath(view.root, pathPoints, isSingular(orbit) ? singularPathMaterial : pathMaterial);
  points.forEach((point, i) => addPointMarker(view.root, point, pointStyle(orbit, i), i === selectedPointIndex));
  return repeatedInitialSegmentPath(points);
}

function drawUnfolded(view, orbit) {
  const { copies, points, markerIndices } = orientedChain(orbit);
  copies.forEach(copy => {
    addTetrahedron(view.root, copy.vertices, 0.18, unfoldedCopyEdgeMaterial, copy.reflectionFace);
  });
  if (points.length > 1) addPath(view.root, points, unfoldedPathMaterial);
  points.forEach((point, i) => {
    addPointMarker(view.root, point, pointStyle(orbit, markerIndices[i]), markerIndices[i] === selectedPointIndex);
  });
  return points;
}

function polylineLengths(points) {
  const lengths = [];
  let total = 0;
  for (let i = 0; i < points.length - 1; i++) {
    total += points[i].distanceTo(points[i + 1]);
    lengths.push(total);
  }
  return { lengths, total };
}

function pointAtProgress(points, progress) {
  if (!points.length) return null;
  if (points.length === 1) return points[0].clone();
  const { lengths, total } = polylineLengths(points);
  if (total <= 0) return points[0].clone();
  const target = ((progress % 1) + 1) % 1 * total;
  let previous = 0;
  for (let i = 0; i < lengths.length; i++) {
    const next = lengths[i];
    if (target <= next || i === lengths.length - 1) {
      const span = Math.max(next - previous, 1e-9);
      const t = (target - previous) / span;
      return points[i].clone().lerp(points[i + 1], t);
    }
    previous = next;
  }
  return points[points.length - 1].clone();
}

function progressAtPoint(points, pointIndex) {
  if (!points.length || pointIndex <= 0) return 0;
  const { lengths, total } = polylineLengths(points);
  if (total <= 0) return 0;
  return (lengths[Math.min(pointIndex - 1, lengths.length - 1)] ?? 0) / total;
}

function progressAtHitIndex(points, hitIndex) {
  return progressAtPoint(points, hitIndex + 1);
}

function setAnimationMarker(view, points) {
  const point = pointAtProgress(points, animationProgress);
  view.movingMarker.visible = animationRunning && point != null;
  view.wrapMarker.visible = false;
  if (!point) return;
  view.movingMarker.position.copy(point);
  view.movingMarker.scale.setScalar(view.markerRadius);
}

function updateAnimation(time) {
  if (animationRunning) {
    if (lastAnimationTime == null) lastAnimationTime = time;
    const delta = Math.max(0, time - lastAnimationTime) / 1000;
    animationProgress = (animationProgress + delta / ANIMATION_SECONDS) % 1;
    lastAnimationTime = time;
  }
  setAnimationMarker(views.folded, currentPathPoints.folded);
  setAnimationMarker(views.unfolded, currentPathPoints.unfolded);
}

function stopAnimation() {
  animationRunning = false;
  lastAnimationTime = null;
  els.play.textContent = "Animate point";
  els.play.classList.remove("is-running");
  for (const view of Object.values(views)) {
    view.movingMarker.visible = false;
    view.wrapMarker.visible = false;
  }
}

function toggleAnimation() {
  const orbit = selectedOrbit();
  if (!orbit) return;
  if (animationRunning) {
    stopAnimation();
    return;
  }
  animationProgress = progressAtHitIndex(currentPathPoints.folded, selectedPointIndex);
  animationRunning = true;
  lastAnimationTime = performance.now();
  els.play.textContent = "Pause point";
  els.play.classList.add("is-running");
  updateAnimation(lastAnimationTime);
}

function selectedOrbit() {
  if (!inventory) return null;
  const rows = visibleOrbits();
  return rows.find(orbit => orbit.id === selectedId) ?? defaultOrbitFromRows(rows);
}

function defaultOrbitFromRows(rows) {
  if (els.kind.value === "singular") return rows[0] ?? null;
  return rows.find(orbit => !isSingular(orbit)) ?? rows[0] ?? null;
}

function selectDefaultVisibleOrbit() {
  selectedId = defaultOrbitFromRows(visibleOrbits())?.id ?? null;
  selectedPointIndex = 0;
}

function drawSelectedOrbit() {
  clearRoot(views.folded);
  clearRoot(views.unfolded);
  currentPathPoints = { folded: [], unfolded: [] };
  const orbit = selectedOrbit();
  if (!orbit) {
    els.empty.textContent = "No path matches the current filters.";
    els.empty.classList.remove("is-hidden");
    els.note.classList.add("is-hidden");
    els.title.textContent = "Inventory";
    els.subtitle.textContent = "";
    els.details.innerHTML = "";
    els.points.innerHTML = "";
    for (const view of Object.values(views)) {
      view.bounds = null;
      view.movingMarker.visible = false;
      view.wrapMarker.visible = false;
    }
    resetCamera();
    return;
  }
  els.empty.classList.add("is-hidden");
  els.note.classList.add("is-hidden");
  selectedId = orbit.id;
  selectedPointIndex = Math.min(selectedPointIndex, orbit.barycentric_points.length - 1);
  currentPathPoints.folded = drawFolded(views.folded, orbit);
  updateBoundsAndCamera(views.folded);
  currentPathPoints.unfolded = drawUnfolded(views.unfolded, orbit);
  updateBoundsAndCamera(views.unfolded);
  updateAnimation(performance.now());
  renderDetails(orbit);
  renderRows();
}

function orbitLengthLabel(orbit) {
  return orbit.length_exact || String(orbit.length_numeric.toFixed(8));
}

function provenanceLabel(orbit) {
  if (isSingular(orbit)) {
    return orbit.coordinate_kind === "numeric" ? "singular numeric" : "singular exact";
  }
  if (orbit.provenance === "exploratory_exactified") return "explore";
  if (orbit.period > (inventory?.summary?.exhaustive_checked_through ?? 40)) return "extra";
  return "exhaustive";
}

function centroidDistanceValue(orbit) {
  if (Number.isFinite(orbit.centroid_distance_numeric)) return orbit.centroid_distance_numeric;
  if (orbit.centroid_distance_squared_exact) return Math.sqrt(parseFractionApprox(orbit.centroid_distance_squared_exact));
  return Number.POSITIVE_INFINITY;
}

function centroidDistanceSquaredLabel(orbit) {
  return orbit.centroid_distance_squared_exact ?? "";
}

function compareNumbers(a, b) {
  if (a === b) return 0;
  if (!Number.isFinite(a) && !Number.isFinite(b)) return 0;
  if (!Number.isFinite(a)) return 1;
  if (!Number.isFinite(b)) return -1;
  return a - b;
}

function compareBySortKey(a, b) {
  if (sortKey === "length") return a.length_numeric - b.length_numeric || a.period - b.period;
  if (sortKey === "center") return compareNumbers(centroidDistanceValue(a), centroidDistanceValue(b)) || a.period - b.period;
  if (sortKey === "height") return compareIntegerStrings(a.height, b.height) || a.period - b.period;
  if (sortKey === "margin") return compareNumbers(parseFractionApprox(a.boundary_margin), parseFractionApprox(b.boundary_margin));
  if (sortKey === "word") return displayWord(a).localeCompare(displayWord(b)) || a.period - b.period;
  return a.period - b.period || a.length_numeric - b.length_numeric || displayWord(a).localeCompare(displayWord(b));
}

function visibleOrbits() {
  if (!inventory) return [];
  const period = els.period.value;
  const kind = els.kind.value;
  const query = els.search.value.trim().toUpperCase();
  let rows = inventory.orbits.filter(orbit => {
    if (period !== "all" && String(orbit.period) !== period) return false;
    if (kind !== "all" && stratumFilterKey(orbit) !== kind) return false;
    if (query && !orbit.search_text.includes(query)) return false;
    return true;
  });
  rows = [...rows].sort((a, b) => {
    const value = compareBySortKey(a, b);
    return sortDirection === "asc" ? value : -value;
  });
  return rows;
}

function renderSortHeaders() {
  for (const button of els.sortHeaders) {
    const active = button.dataset.sortKey === sortKey;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-sort", active ? (sortDirection === "asc" ? "ascending" : "descending") : "none");
    button.textContent = button.textContent.replace(/\s+\^(asc|desc)$/, "");
    if (active) button.textContent += sortDirection === "asc" ? " ^asc" : " ^desc";
  }
  if (els.sort.value !== sortKey && [...els.sort.options].some(option => option.value === sortKey)) {
    els.sort.value = sortKey;
  }
}

function renderColoredWord(cell, orbit) {
  cell.replaceChildren();
  for (const char of displayWord(orbit)) {
    const token = document.createElement("span");
    if (FACE_CSS[char]) {
      token.className = "word-letter";
      token.style.setProperty("--letter-color", FACE_CSS[char]);
    } else {
      token.className = /\s/.test(char) ? "word-space" : "word-separator";
    }
    token.textContent = char;
    cell.appendChild(token);
  }
}

function renderRows() {
  const rows = visibleOrbits();
  els.count.textContent = `${rows.length} shown`;
  els.rows.innerHTML = "";
  renderSortHeaders();
  for (const orbit of rows) {
    const row = document.createElement("tr");
    row.className = `orbit-row${orbit.id === selectedId ? " is-selected" : ""}${isSingular(orbit) ? " is-singular" : ""}`;
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-label", `Select period ${orbit.period} path ${displayWord(orbit)}`);
    row.innerHTML = `
      <td class="period-cell"></td>
      <td class="word-cell mono"></td>
      <td class="numeric-cell"></td>
      <td class="exact-cell"></td>
      <td class="exact-cell"></td>
      <td class="deg-cell"></td>
      <td class="numeric-cell"></td>
      <td class="numeric-cell"></td>
      <td class="source-cell"></td>
    `;
    row.children[0].textContent = `p${String(orbit.period).padStart(2, "0")}`;
    renderColoredWord(row.children[1], orbit);
    row.children[2].textContent = orbit.length_numeric.toFixed(10);
    row.children[3].textContent = centroidDistanceSquaredLabel(orbit);
    row.children[4].textContent = orbitLengthLabel(orbit);
    row.children[5].textContent = degeneracyLabel(orbit);
    row.children[6].textContent = orbit.height;
    row.children[7].textContent = orbit.boundary_margin;
    row.children[8].textContent = provenanceLabel(orbit);
    const selectRow = () => {
      stopAnimation();
      selectedId = orbit.id;
      selectedPointIndex = 0;
      drawSelectedOrbit();
    };
    row.addEventListener("click", selectRow);
    row.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectRow();
      }
    });
    els.rows.appendChild(row);
  }
}

function detailRow(term, value, className = "") {
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  if (className) dd.className = className;
  dd.textContent = value;
  els.details.append(dt, dd);
}

function renderPointRows(orbit) {
  els.points.innerHTML = "";
  orbit.barycentric_points.forEach((row, index) => {
    const style = pointStyle(orbit, index);
    const button = document.createElement("button");
    button.type = "button";
    button.className = `point-row${index === selectedPointIndex ? " is-selected" : ""}`;
    button.style.setProperty("--point-color", style.css);
    button.setAttribute("aria-label", `Select point ${index} on ${style.label}`);

    const number = document.createElement("span");
    number.className = "point-index";
    number.textContent = String(index).padStart(2, "0");

    const label = document.createElement("span");
    label.className = "point-label";
    label.textContent = style.label;

    const coords = document.createElement("span");
    coords.className = "point-coords mono";
    coords.textContent = `[${row.join(", ")}]`;

    const xyz = document.createElement("span");
    xyz.className = "point-xyz mono";
    xyz.textContent = `(${orbit.points_xyz_exact[index].join(", ")})`;

    button.append(number, label, coords, xyz);
    button.addEventListener("click", () => {
      stopAnimation();
      selectedPointIndex = index;
      drawSelectedOrbit();
    });
    els.points.appendChild(button);
  });
}

function equivalenceLabel(orbit) {
  if (isSingular(orbit)) return "normal-cone representative";
  if (orbit.equivalence === "full") return "canonical full";
  return orbit.equivalence ? `canonical ${orbit.equivalence}` : "canonical";
}

function renderDetails(orbit) {
  els.title.textContent = displayWord(orbit);
  els.subtitle.textContent = `period ${orbit.period} | ${pathClassLabel(orbit)} | ${equivalenceLabel(orbit)} | ${orbit.id}`;
  els.details.innerHTML = "";
  detailRow("Class", pathClassLabel(orbit));
  detailRow("Equivalence", equivalenceLabel(orbit));
  detailRow("Length", `${orbit.length_exact} (${orbit.length_numeric.toFixed(10)})`);
  if (orbit.path_center_xyz_exact) detailRow("Center", orbit.path_center_xyz_exact.join(", "), "mono");
  if (orbit.centroid_distance_squared_exact) detailRow("Center d^2", orbit.centroid_distance_squared_exact, "mono");
  if (Number.isFinite(orbit.centroid_distance_numeric)) detailRow("Center dist", orbit.centroid_distance_numeric.toFixed(10));
  detailRow("Degeneracy", degeneracyLabel(orbit));
  detailRow("Height", orbit.height);
  if (orbit.boundary_margin) detailRow("Margin", orbit.boundary_margin);
  detailRow("Source", provenanceLabel(orbit));
  if (orbit.coordinate_kind) detailRow("Coordinates", orbit.coordinate_kind === "numeric" ? "numeric" : "exact rational");
  if (orbit.discovery_method) detailRow("Found by", orbit.discovery_method);
  if (orbit.family_dimension) detailRow("Family", `${orbit.family_dimension}D representative`);
  if (orbit.axis_direction.length) detailRow("Axis", orbit.axis_direction.join(", "), "mono");
  if (orbit.faces.length && !isSingular(orbit)) detailRow("Faces", orbit.faces.join(" "), "mono");
  if (isSingular(orbit) && orbit.barycentric_points) detailRow("Strata", displayWord(orbit), "mono");
  detailRow("Denominator", orbit.barycentric_denominator, "mono");
  renderPointRows(orbit);
}

function renderSummary() {
  const summary = inventory.summary;
  const counts = Object.entries(summary.period_counts)
    .map(([period, count]) => `${period}:${count}`)
    .join("  ");
  els.status.textContent = `${summary.ordinary_orbits} ordinary + ${summary.singular_orbits} singular paths loaded`;
  els.summary.innerHTML = "";
  const coverage = `Exhaustive through level ${summary.exhaustive_checked_through ?? 40}; ${summary.extra_verified_orbits ?? 0} exactified extra paths`;
  const singularSummary = inventory.singular_inventory?.summary;
  const singularCoverage = singularSummary
    ? `Exhaustive through period ${singularSummary.exhaustive_checked_through}; ${singularSummary.exact_rational_orbits ?? 0} exact rational, ${singularSummary.numeric_orbits ?? 0} numeric`
    : "Singular data unavailable";
  const rows = [
    ["Generated", inventory.generated_at],
    ["Ordinary", coverage],
    ["Singular", singularCoverage],
    ["Catalogue", `${summary.total_orbits} paths (${summary.ordinary_orbits} ordinary, ${summary.singular_orbits} singular)`],
    ["Max period", String(summary.max_period)],
    ["Counts", counts],
    ["Notebook", inventory.source_notebook ?? "Observable source linked in data"],
    ["Data", `${DATA_URL}, ${SINGULAR_DATA_URL}`]
  ];
  for (const [term, value] of rows) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = value;
    els.summary.append(dt, dd);
  }
}

function populatePeriods() {
  const existing = els.period.value || "all";
  els.period.innerHTML = '<option value="all">All periods</option>';
  for (const period of Object.keys(inventory.summary.period_counts).map(Number).sort((a, b) => a - b)) {
    const option = document.createElement("option");
    option.value = String(period);
    option.textContent = String(period);
    els.period.appendChild(option);
  }
  if ([...els.period.options].some(option => option.value === existing)) {
    els.period.value = existing;
  }
}

function populateStratumFilters() {
  const existing = els.kind.value || "all";
  const counts = Object.fromEntries(Object.keys(FILTER_LABELS).map(key => [key, 0]));
  counts.all = inventory.orbits.length;
  inventory.orbits.forEach(orbit => {
    counts[stratumFilterKey(orbit)] += 1;
  });
  for (const option of els.kind.options) {
    const count = counts[option.value] ?? 0;
    option.textContent = `${FILTER_LABELS[option.value] ?? option.textContent} (${count})`;
    option.disabled = option.value !== "all" && count === 0;
  }
  const existingOption = [...els.kind.options].find(option => option.value === existing);
  els.kind.value = existingOption && !existingOption.disabled ? existing : "all";
}

async function loadInventory() {
  try {
    const timestamp = Date.now();
    const [ordinaryResponse, singularResponse] = await Promise.all([
      fetch(`${DATA_URL}?t=${timestamp}`, { cache: "no-store" }),
      fetch(`${SINGULAR_DATA_URL}?t=${timestamp}`, { cache: "no-store" })
    ]);
    if (!ordinaryResponse.ok) throw new Error(`${DATA_URL}: HTTP ${ordinaryResponse.status}`);
    if (!singularResponse.ok) throw new Error(`${SINGULAR_DATA_URL}: HTTP ${singularResponse.status}`);
    const ordinaryInventory = await ordinaryResponse.json();
    const singularInventory = await singularResponse.json();
    const ordinaryOrbits = ordinaryInventory.orbits.map(orbit => normalizeOrbit(orbit, "ordinary"));
    const singularOrbits = singularInventory.orbits.map(orbit => normalizeOrbit({ ...orbit, singular: true }, "singular_normal_cone"));
    const orbits = [...ordinaryOrbits, ...singularOrbits]
      .sort((a, b) => a.period - b.period || a.length_numeric - b.length_numeric || displayWord(a).localeCompare(displayWord(b)));
    inventory = {
      ...ordinaryInventory,
      ordinary_inventory: ordinaryInventory,
      singular_inventory: singularInventory,
      orbits,
      summary: {
        ...ordinaryInventory.summary,
        ordinary_orbits: ordinaryOrbits.length,
        singular_orbits: singularOrbits.length,
        total_orbits: orbits.length,
        max_period: Math.max(...orbits.map(orbit => orbit.period)),
        period_counts: combinedPeriodCounts(orbits)
      }
    };
    populatePeriods();
    populateStratumFilters();
    renderSummary();
    selectDefaultVisibleOrbit();
    stopAnimation();
    renderRows();
    drawSelectedOrbit();
  } catch (error) {
    els.status.textContent = "Could not load tetrahedron billiards data";
    els.empty.textContent = String(error);
    els.empty.classList.remove("is-hidden");
  }
}

els.period.addEventListener("change", () => {
  stopAnimation();
  selectDefaultVisibleOrbit();
  renderRows();
  drawSelectedOrbit();
});
els.kind.addEventListener("change", () => {
  stopAnimation();
  selectDefaultVisibleOrbit();
  renderRows();
  drawSelectedOrbit();
});
els.search.addEventListener("input", () => {
  stopAnimation();
  selectDefaultVisibleOrbit();
  renderRows();
  drawSelectedOrbit();
});
els.sort.addEventListener("change", () => {
  sortKey = els.sort.value;
  sortDirection = "asc";
  renderRows();
});
els.sortHeaders.forEach(button => {
  button.addEventListener("click", () => {
    const nextKey = button.dataset.sortKey;
    if (sortKey === nextKey) {
      sortDirection = sortDirection === "asc" ? "desc" : "asc";
    } else {
      sortKey = nextKey;
      sortDirection = "asc";
    }
    renderRows();
  });
});
els.play.addEventListener("click", toggleAnimation);
els.reset.addEventListener("click", () => resetCamera());

loadInventory();
