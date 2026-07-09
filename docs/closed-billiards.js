import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SCENE_COPY = {
  face: {
    title: "Smooth face reflection",
    active: "1",
    cone: "one ray",
    jump: "supported by the face normal"
  },
  edge: {
    title: "Edge normal-cone reflection",
    active: "2",
    cone: "wedge spanned by two normals",
    jump: "any positive combination of the two active normals"
  },
  vertex: {
    title: "Vertex normal-cone reflection",
    active: "3",
    cone: "trihedral cone",
    jump: "any positive combination of the three active normals"
  }
};

const COLORS = {
  background: 0xffffff,
  faceA: 0x4e79a7,
  faceB: 0x59a14f,
  faceC: 0xf28e2b,
  path: 0xd0342c,
  normal: 0x0b6f77,
  cone: 0xa56500,
  edge: 0x1f252b,
  grid: 0x79828a,
  hit: 0x111111
};

const els = {
  canvas: document.getElementById("normalConeCanvas"),
  tabs: [...document.querySelectorAll(".scene-tab")],
  title: document.getElementById("sceneTitle"),
  active: document.getElementById("sceneActive"),
  cone: document.getElementById("sceneCone"),
  jump: document.getElementById("sceneJump")
};

const renderer = new THREE.WebGLRenderer({ canvas: els.canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(COLORS.background, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
camera.position.set(3.8, 2.8, 3.6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.45;
controls.screenSpacePanning = true;
controls.target.set(0.25, 0.25, 0.2);

const root = new THREE.Group();
scene.add(root);

scene.add(new THREE.HemisphereLight(0xffffff, 0xd7ddd4, 1.35));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
keyLight.position.set(4, 5, 3);
scene.add(keyLight);

let activeScene = "face";
let currentBounds = null;

const faceMaterials = [
  new THREE.MeshBasicMaterial({ color: COLORS.faceA, transparent: true, opacity: 0.2, side: THREE.DoubleSide }),
  new THREE.MeshBasicMaterial({ color: COLORS.faceB, transparent: true, opacity: 0.18, side: THREE.DoubleSide }),
  new THREE.MeshBasicMaterial({ color: COLORS.faceC, transparent: true, opacity: 0.18, side: THREE.DoubleSide })
];
const coneMaterial = new THREE.MeshBasicMaterial({
  color: COLORS.cone,
  transparent: true,
  opacity: 0.24,
  side: THREE.DoubleSide,
  depthWrite: false
});
const pathMaterial = new THREE.LineBasicMaterial({ color: COLORS.path });
const edgeMaterial = new THREE.LineBasicMaterial({ color: COLORS.edge, transparent: true, opacity: 0.82 });
const coneLineMaterial = new THREE.LineBasicMaterial({ color: COLORS.cone, transparent: true, opacity: 0.84 });
const gridMaterial = new THREE.LineBasicMaterial({ color: COLORS.grid, transparent: true, opacity: 0.22 });

function vector(values) {
  return new THREE.Vector3(values[0], values[1], values[2]);
}

function clearRoot() {
  while (root.children.length) {
    const child = root.children.pop();
    child.traverse?.(node => {
      node.geometry?.dispose();
    });
  }
}

function triangleGeometry(points, indices) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function quadMesh(points, material) {
  return new THREE.Mesh(
    triangleGeometry(points.map(vector), [0, 1, 2, 0, 2, 3]),
    material
  );
}

function addLine(points, material = edgeMaterial) {
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points.map(vector)),
    material
  );
  root.add(line);
  return line;
}

function addArrow(start, end, color, headLength = 0.16, headWidth = 0.09) {
  const a = vector(start);
  const b = vector(end);
  const direction = b.clone().sub(a);
  const length = direction.length();
  const arrow = new THREE.ArrowHelper(direction.normalize(), a, length, color, headLength, headWidth);
  root.add(arrow);
  return arrow;
}

function addPath(incomingStart, hit, outgoingEnd) {
  addLine([incomingStart, hit, outgoingEnd], pathMaterial);
  addArrow(incomingStart, hit, COLORS.path, 0.18, 0.1);
  addArrow(hit, outgoingEnd, COLORS.path, 0.18, 0.1);
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.055, 24, 16),
    new THREE.MeshBasicMaterial({ color: COLORS.hit })
  );
  marker.position.copy(vector(hit));
  root.add(marker);
}

function addGridOnFace(size = 2.4, step = 0.4) {
  const points = [];
  const half = size / 2;
  for (let t = -half; t <= half + 0.001; t += step) {
    points.push(new THREE.Vector3(-half, 0.002, t), new THREE.Vector3(half, 0.002, t));
    points.push(new THREE.Vector3(t, 0.002, -half), new THREE.Vector3(t, 0.002, half));
  }
  const grid = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points), gridMaterial);
  root.add(grid);
}

function addConeFan(points) {
  const geometry = triangleGeometry(points.map(vector), points.length === 4 ? [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3] : [0, 1, 2]);
  const mesh = new THREE.Mesh(geometry, coneMaterial);
  root.add(mesh);
  const edgePoints = points.length === 4
    ? [points[0], points[1], points[2], points[0], points[3], points[1], points[3], points[2]]
    : [points[0], points[1], points[2], points[0]];
  addLine(edgePoints, coneLineMaterial);
}

function addFaceScene() {
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(3.2, 2.35),
    faceMaterials[0]
  );
  plane.rotation.x = -Math.PI / 2;
  root.add(plane);
  addGridOnFace(2.4, 0.4);
  addPath([-1.15, 0.9, -0.65], [0, 0, 0], [1.15, 0.9, -0.65]);
  addArrow([0, 0, 0], [0, 1.1, 0], COLORS.normal, 0.2, 0.11);
  addArrow([0.12, 0.02, 0.12], [0.12, 0.82, 0.12], COLORS.cone, 0.16, 0.09);
  addLine([[-1.45, 0.006, 0], [1.45, 0.006, 0]], edgeMaterial);
  addLine([[0, 0.006, -1.05], [0, 0.006, 1.05]], edgeMaterial);
}

function addEdgeScene() {
  root.add(quadMesh([[0, 0, -1.2], [1.8, 0, -1.2], [1.8, 0, 1.2], [0, 0, 1.2]], faceMaterials[0]));
  root.add(quadMesh([[0, 0, -1.2], [0, 1.8, -1.2], [0, 1.8, 1.2], [0, 0, 1.2]], faceMaterials[1]));
  addLine([[0, 0, -1.24], [0, 0, 1.24]], edgeMaterial);
  addPath([0.75, 1.18, -0.92], [0, 0, 0], [1.32, 0.54, 0.9]);
  addArrow([0, 0, 0], [0.95, 0, 0], COLORS.normal, 0.17, 0.09);
  addArrow([0, 0, 0], [0, 0.95, 0], COLORS.normal, 0.17, 0.09);
  addArrow([0.03, 0.03, 0.05], [0.76, 0.68, 0.05], COLORS.cone, 0.17, 0.09);
  addConeFan([[0, 0, 0], [1.05, 0, 0], [0, 1.05, 0]]);
}

function addVertexScene() {
  root.add(quadMesh([[0, 0, 0], [1.75, 0, 0], [1.75, 1.75, 0], [0, 1.75, 0]], faceMaterials[0]));
  root.add(quadMesh([[0, 0, 0], [1.75, 0, 0], [1.75, 0, 1.75], [0, 0, 1.75]], faceMaterials[1]));
  root.add(quadMesh([[0, 0, 0], [0, 1.75, 0], [0, 1.75, 1.75], [0, 0, 1.75]], faceMaterials[2]));
  addLine([[0, 0, 0], [1.8, 0, 0], [0, 0, 0], [0, 1.8, 0], [0, 0, 0], [0, 0, 1.8]], edgeMaterial);
  addPath([1.18, 0.74, 0.88], [0, 0, 0], [0.58, 1.24, 1.04]);
  addArrow([0, 0, 0], [0.92, 0, 0], COLORS.normal, 0.16, 0.085);
  addArrow([0, 0, 0], [0, 0.92, 0], COLORS.normal, 0.16, 0.085);
  addArrow([0, 0, 0], [0, 0, 0.92], COLORS.normal, 0.16, 0.085);
  addArrow([0.04, 0.04, 0.04], [0.68, 0.62, 0.74], COLORS.cone, 0.17, 0.09);
  addConeFan([[0, 0, 0], [1.02, 0, 0], [0, 1.02, 0], [0, 0, 1.02]]);
}

function updateReadout(name) {
  const copy = SCENE_COPY[name];
  els.title.textContent = copy.title;
  els.active.textContent = copy.active;
  els.cone.textContent = copy.cone;
  els.jump.textContent = copy.jump;
  els.tabs.forEach(button => {
    const selected = button.dataset.scene === name;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", selected ? "true" : "false");
  });
}

function updateBounds() {
  currentBounds = new THREE.Box3().setFromObject(root);
  if (currentBounds.isEmpty()) return;
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  currentBounds.getCenter(center);
  currentBounds.getSize(size);
  const radius = Math.max(size.x, size.y, size.z, 1.7);
  const cameraDirection = activeScene === "face"
    ? new THREE.Vector3(2.8, 2.25, 2.5)
    : new THREE.Vector3(3.1, 2.55, 2.8);
  cameraDirection.normalize();
  camera.position.copy(center).addScaledVector(cameraDirection, radius * 2.05);
  controls.target.copy(center);
  camera.near = 0.01;
  camera.far = 100;
  camera.updateProjectionMatrix();
  controls.update();
}

function drawScene(name) {
  activeScene = name;
  clearRoot();
  if (name === "edge") addEdgeScene();
  else if (name === "vertex") addVertexScene();
  else addFaceScene();
  updateReadout(name);
  updateBounds();
}

function resizeRenderer() {
  const rect = els.canvas.parentElement.getBoundingClientRect();
  const width = Math.max(300, Math.floor(rect.width));
  const height = Math.max(320, Math.floor(rect.height));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

els.tabs.forEach(button => {
  button.addEventListener("click", () => drawScene(button.dataset.scene));
});

new ResizeObserver(resizeRenderer).observe(els.canvas.parentElement);
window.addEventListener("resize", resizeRenderer);

resizeRenderer();
drawScene(activeScene);
animate();
