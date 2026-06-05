import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DATA_URL = "data/tetra/billiards_inventory.json";
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

const els = {
  status: document.getElementById("inventoryStatus"),
  period: document.getElementById("periodSelect"),
  search: document.getElementById("wordSearch"),
  sort: document.getElementById("sortSelect"),
  folded: document.getElementById("foldedBtn"),
  unfolded: document.getElementById("unfoldedBtn"),
  summary: document.getElementById("summaryDetails"),
  title: document.getElementById("orbitTitle"),
  subtitle: document.getElementById("orbitSubtitle"),
  reset: document.getElementById("resetCamera"),
  stage: document.getElementById("stage"),
  canvas: document.getElementById("tetraCanvas"),
  empty: document.getElementById("stageEmpty"),
  rows: document.getElementById("orbitRows"),
  sortHeaders: [...document.querySelectorAll(".sort-header")],
  count: document.getElementById("recordCount"),
  details: document.getElementById("orbitDetails")
};

let inventory = null;
let selectedId = null;
let viewMode = "folded";
let currentBounds = null;
let sortKey = "period";
let sortDirection = "asc";

const renderer = new THREE.WebGLRenderer({ canvas: els.canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0xffffff, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;

const root = new THREE.Group();
scene.add(root);

const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x15181d, transparent: true, opacity: 0.82 });
const faintEdgeMaterial = new THREE.LineBasicMaterial({ color: 0x6e747c, transparent: true, opacity: 0.28 });
const cubeMaterial = new THREE.LineBasicMaterial({ color: 0x3f4c5a, transparent: true, opacity: 0.28 });
const pathMaterial = new THREE.LineBasicMaterial({ color: 0xd0342c });
const unfoldedPathMaterial = new THREE.LineBasicMaterial({ color: 0xd0342c, linewidth: 2 });

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

function resizeRenderer() {
  const rect = els.stage.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(300, Math.floor(rect.height));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

new ResizeObserver(resizeRenderer).observe(els.stage);
window.addEventListener("resize", resizeRenderer);
resizeRenderer();

function bigIntFrom(value) {
  return BigInt(String(value));
}

function ratio(value, denominator) {
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
  const aa = String(a).replace(/^-/, "");
  const bb = String(b).replace(/^-/, "");
  if (aa.length !== bb.length) return aa.length - bb.length;
  return aa.localeCompare(bb);
}

function parseFractionApprox(text) {
  const parts = String(text).split("/");
  if (parts.length === 1) return Number(parts[0]);
  return Number(parts[0]) / Number(parts[1]);
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

function clearRoot() {
  while (root.children.length) {
    const child = root.children.pop();
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

function addTetrahedron(group, vertices, opacity = 0.15, material = edgeMaterial) {
  for (const face of VERTEX_NAMES) addFace(group, vertices, face, opacity);
  addEdges(group, vertices, material);
}

function addPointMarker(group, point, face, radius) {
  const material = new THREE.MeshBasicMaterial({ color: FACE_COLORS[face] ?? FACE_COLORS["?"] });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 18, 12), material);
  mesh.position.copy(point);
  group.add(mesh);
}

function addPath(group, points, material = pathMaterial) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  group.add(new THREE.Line(geometry, material));
}

function reflectPoint(point, faceVertices) {
  const plane = new THREE.Plane().setFromCoplanarPoints(faceVertices[0], faceVertices[1], faceVertices[2]);
  return point.clone().addScaledVector(plane.normal, -2 * plane.distanceToPoint(point));
}

function reflectedVertices(vertices, face) {
  const faceVertices = FACES[face].map(name => vertices[name]);
  return Object.fromEntries(
    VERTEX_NAMES.map(name => [name, reflectPoint(vertices[name], faceVertices)])
  );
}

function foldedPoints(orbit) {
  return orbit.points_xyz.map(point => new THREE.Vector3(point[0], point[1], point[2]));
}

function unfoldedChain(orbit) {
  let vertices = vertexMapFromInventory();
  const copies = [];
  const points = [];
  const word = orbit.word;
  for (let i = 0; i < orbit.barycentric_points.length; i++) {
    copies.push({
      vertices: Object.fromEntries(VERTEX_NAMES.map(name => [name, vertices[name].clone()])),
      hitFace: word[i % word.length]
    });
    points.push(baryPoint(orbit.barycentric_points[i], orbit.barycentric_denominator, vertices));
    vertices = reflectedVertices(vertices, word[i % word.length]);
  }
  copies.push({
    vertices: Object.fromEntries(VERTEX_NAMES.map(name => [name, vertices[name].clone()])),
    hitFace: word[0]
  });
  points.push(baryPoint(orbit.barycentric_points[0], orbit.barycentric_denominator, vertices));
  return { copies, points };
}

function updateBoundsAndCamera(group) {
  const box = new THREE.Box3().setFromObject(group);
  if (box.isEmpty()) {
    currentBounds = null;
    return;
  }
  currentBounds = box;
  resetCamera();
}

function fitDistanceForBox(box, direction, padding) {
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

function resetCamera() {
  if (!currentBounds) {
    camera.position.set(2.2, 2.4, 2.0);
    controls.target.set(0.5, 0.5, 0.5);
    controls.update();
    return;
  }
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  currentBounds.getCenter(center);
  currentBounds.getSize(size);
  const maxSize = Math.max(size.x, size.y, size.z, 1);
  const direction = viewMode === "unfolded"
    ? new THREE.Vector3(0.72, -0.9, 0.46)
    : new THREE.Vector3(1.35, 1.45, 1.2);
  direction.normalize();
  const padding = viewMode === "unfolded" ? 1.08 : 1.12;
  const fitDistance = Math.max(fitDistanceForBox(currentBounds, direction, padding), maxSize * 0.82);
  camera.position.copy(center).addScaledVector(direction, fitDistance);
  camera.near = Math.max(0.01, maxSize / 500);
  camera.far = Math.max(1000, maxSize * 10);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function drawFolded(orbit) {
  const vertices = vertexMapFromInventory();
  addUnitCubeWireframe(root);
  addTetrahedron(root, vertices, 0.16, edgeMaterial);
  const points = foldedPoints(orbit);
  addPath(root, [...points, points[0]], pathMaterial);
  points.forEach((point, i) => addPointMarker(root, point, orbit.faces[i] ?? "?", 0.022));
}

function drawUnfolded(orbit) {
  addUnitCubeWireframe(root);
  const { copies, points } = unfoldedChain(orbit);
  copies.forEach((copy, i) => {
    addEdges(root, copy.vertices, faintEdgeMaterial);
    if (i < copies.length - 1) addFace(root, copy.vertices, copy.hitFace, 0.11);
  });
  addPath(root, points, unfoldedPathMaterial);
  points.slice(0, -1).forEach((point, i) => addPointMarker(root, point, orbit.faces[i] ?? "?", 0.028));
}

function selectedOrbit() {
  if (!inventory) return null;
  return inventory.orbits.find(orbit => orbit.id === selectedId) ?? visibleOrbits()[0] ?? null;
}

function drawSelectedOrbit() {
  clearRoot();
  const orbit = selectedOrbit();
  if (!orbit) {
    els.empty.textContent = "No path matches the current filters.";
    els.empty.classList.remove("is-hidden");
    els.title.textContent = "Inventory";
    els.subtitle.textContent = "";
    els.details.innerHTML = "";
    currentBounds = null;
    resetCamera();
    return;
  }
  els.empty.classList.add("is-hidden");
  selectedId = orbit.id;
  if (viewMode === "unfolded") drawUnfolded(orbit);
  else drawFolded(orbit);
  updateBoundsAndCamera(root);
  renderDetails(orbit);
  renderRows();
}

function orbitLengthLabel(orbit) {
  return orbit.length_exact || String(orbit.length_numeric.toFixed(8));
}

function compareBySortKey(a, b) {
  if (sortKey === "length") return a.length_numeric - b.length_numeric || a.period - b.period;
  if (sortKey === "height") return compareIntegerStrings(a.height, b.height) || a.period - b.period;
  if (sortKey === "margin") return parseFractionApprox(a.boundary_margin) - parseFractionApprox(b.boundary_margin);
  if (sortKey === "word") return a.word.localeCompare(b.word) || a.period - b.period;
  return a.period - b.period || a.length_numeric - b.length_numeric || a.word.localeCompare(b.word);
}

function visibleOrbits() {
  if (!inventory) return [];
  const period = els.period.value;
  const query = els.search.value.trim().toUpperCase();
  let rows = inventory.orbits.filter(orbit => {
    if (period !== "all" && String(orbit.period) !== period) return false;
    if (query && !`${orbit.id} ${orbit.word}`.toUpperCase().includes(query)) return false;
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

function renderRows() {
  const rows = visibleOrbits();
  els.count.textContent = `${rows.length} shown`;
  els.rows.innerHTML = "";
  renderSortHeaders();
  for (const orbit of rows) {
    const row = document.createElement("tr");
    row.className = `orbit-row${orbit.id === selectedId ? " is-selected" : ""}`;
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-label", `Select period ${orbit.period} path ${orbit.word}`);
    row.innerHTML = `
      <td class="period-cell"></td>
      <td class="word-cell mono"></td>
      <td class="numeric-cell"></td>
      <td class="exact-cell"></td>
      <td class="numeric-cell"></td>
      <td class="numeric-cell"></td>
    `;
    row.children[0].textContent = `p${String(orbit.period).padStart(2, "0")}`;
    row.children[1].textContent = orbit.word;
    row.children[2].textContent = orbit.length_numeric.toFixed(10);
    row.children[3].textContent = orbitLengthLabel(orbit);
    row.children[4].textContent = orbit.height;
    row.children[5].textContent = orbit.boundary_margin;
    const selectRow = () => {
      selectedId = orbit.id;
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

function renderDetails(orbit) {
  els.title.textContent = `${orbit.word}`;
  els.subtitle.textContent = `period ${orbit.period} | ${viewMode} view | ${orbit.id}`;
  els.details.innerHTML = "";
  detailRow("Length", `${orbit.length_exact} (${orbit.length_numeric.toFixed(10)})`);
  detailRow("Height", orbit.height);
  detailRow("Margin", orbit.boundary_margin);
  detailRow("Axis", orbit.axis_direction.join(", "), "mono");
  detailRow("Initial dir", orbit.initial_direction.join(", "), "mono");
  detailRow("Faces", orbit.faces.join(" "), "mono");
  detailRow("Denominator", orbit.barycentric_denominator, "mono");
  const points = orbit.points_xyz_exact.map((point, i) => `${i}: ${orbit.faces[i]} (${point.join(", ")})`).join("\n");
  detailRow("Points", points, "mono");
}

function renderSummary() {
  const summary = inventory.summary;
  const counts = Object.entries(summary.period_counts)
    .map(([period, count]) => `${period}:${count}`)
    .join("  ");
  els.status.textContent = `${summary.total_orbits} ordinary orbit classes loaded`;
  els.summary.innerHTML = "";
  const rows = [
    ["Generated", inventory.generated_at],
    ["Max period", String(summary.max_period)],
    ["Counts", counts],
    ["Notebook", "Observable source linked in data"],
    ["Data", DATA_URL]
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

async function loadInventory() {
  try {
    const response = await fetch(`${DATA_URL}?t=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    inventory = await response.json();
    populatePeriods();
    renderSummary();
    selectedId = inventory.orbits[0]?.id ?? null;
    renderRows();
    drawSelectedOrbit();
  } catch (error) {
    els.status.textContent = `Could not load ${DATA_URL}`;
    els.empty.textContent = String(error);
    els.empty.classList.remove("is-hidden");
  }
}

els.period.addEventListener("change", () => {
  selectedId = visibleOrbits()[0]?.id ?? null;
  renderRows();
  drawSelectedOrbit();
});
els.search.addEventListener("input", () => {
  selectedId = visibleOrbits()[0]?.id ?? null;
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
els.folded.addEventListener("click", () => {
  viewMode = "folded";
  els.folded.classList.add("is-active");
  els.unfolded.classList.remove("is-active");
  drawSelectedOrbit();
});
els.unfolded.addEventListener("click", () => {
  viewMode = "unfolded";
  els.unfolded.classList.add("is-active");
  els.folded.classList.remove("is-active");
  drawSelectedOrbit();
});
els.reset.addEventListener("click", resetCamera);

loadInventory();
