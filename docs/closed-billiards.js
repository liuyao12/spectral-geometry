import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const COLORS = {
  background: 0xffffff,
  faceA: 0x4e79a7,
  faceB: 0x59a14f,
  faceC: 0xf28e2b,
  faceD: 0xe15759,
  path: 0xd0342c,
  normal: 0x0b6f77,
  cone: 0xa56500,
  edge: 0x1f252b,
  grid: 0x79828a,
  hit: 0x111111,
  smooth: 0x8ecae6
};

const FACE_COLORS = {
  A: COLORS.faceA,
  B: COLORS.faceB,
  C: COLORS.faceC,
  D: COLORS.faceD
};

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

const baseVertices = {
  A: new THREE.Vector3(1, 0, 0).subScalar(0.5),
  B: new THREE.Vector3(0, 1, 0).subScalar(0.5),
  C: new THREE.Vector3(0, 0, 1).subScalar(0.5),
  D: new THREE.Vector3(1, 1, 1).subScalar(0.5)
};

const materials = {
  edge: new THREE.LineBasicMaterial({ color: COLORS.edge, transparent: true, opacity: 0.82 }),
  faintEdge: new THREE.LineBasicMaterial({ color: COLORS.edge, transparent: true, opacity: 0.22 }),
  path: new THREE.LineBasicMaterial({ color: COLORS.path }),
  faintPath: new THREE.LineBasicMaterial({ color: COLORS.path, transparent: true, opacity: 0.18 }),
  limitPath: new THREE.LineBasicMaterial({ color: COLORS.cone, transparent: true, opacity: 0.92 }),
  coneLine: new THREE.LineBasicMaterial({ color: COLORS.cone, transparent: true, opacity: 0.84 }),
  grid: new THREE.LineBasicMaterial({ color: COLORS.grid, transparent: true, opacity: 0.22 }),
  smoothLine: new THREE.LineBasicMaterial({ color: COLORS.smooth, transparent: true, opacity: 0.85 }),
  coneMesh: new THREE.MeshBasicMaterial({
    color: COLORS.cone,
    transparent: true,
    opacity: 0.24,
    side: THREE.DoubleSide,
    depthWrite: false
  }),
  smoothMesh: new THREE.MeshBasicMaterial({
    color: COLORS.smooth,
    transparent: true,
    opacity: 0.42,
    side: THREE.DoubleSide,
    depthWrite: false
  })
};

function vector(values) {
  return new THREE.Vector3(values[0], values[1], values[2]);
}

function makeFaceMaterial(face, opacity = 0.18) {
  return new THREE.MeshBasicMaterial({
    color: FACE_COLORS[face] ?? COLORS.faceA,
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
    depthWrite: false
  });
}

function makeScene(canvas, options = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(COLORS.background, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 120);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = options.autoRotate ?? false;
  controls.autoRotateSpeed = options.autoRotateSpeed ?? 0.35;
  controls.screenSpacePanning = true;
  controls.addEventListener("start", () => {
    controls.autoRotate = false;
  });

  const root = new THREE.Group();
  scene.add(root);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xd7ddd4, 1.35));
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
  keyLight.position.set(4, 5, 3);
  scene.add(keyLight);

  const controller = { renderer, scene, camera, controls, root, canvas };
  const resize = () => {
    const rect = canvas.parentElement.getBoundingClientRect();
    const width = Math.max(300, Math.floor(rect.width));
    const height = Math.max(320, Math.floor(rect.height));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvas.parentElement);
  window.addEventListener("resize", resize);
  resize();
  return controller;
}

function clearRoot(root) {
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

function addFace(group, vertices, face, opacity = 0.18) {
  const mesh = new THREE.Mesh(
    triangleGeometry(FACES[face].map(name => vertices[name]), [0, 1, 2]),
    makeFaceMaterial(face, opacity)
  );
  group.add(mesh);
  return mesh;
}

function addEdges(group, vertices, material = materials.edge) {
  const points = [];
  for (const [a, b] of EDGES) points.push(vertices[a], vertices[b]);
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  group.add(new THREE.LineSegments(geometry, material));
}

function addLine(group, points, material = materials.edge) {
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points.map(point => Array.isArray(point) ? vector(point) : point)),
    material
  );
  group.add(line);
  return line;
}

function addArrow(group, start, end, color, headLength = 0.16, headWidth = 0.09) {
  const a = Array.isArray(start) ? vector(start) : start.clone();
  const b = Array.isArray(end) ? vector(end) : end.clone();
  const direction = b.clone().sub(a);
  const length = direction.length();
  const arrow = new THREE.ArrowHelper(direction.normalize(), a, length, color, headLength, headWidth);
  group.add(arrow);
  return arrow;
}

function addMarker(group, point, color, radius = 0.045) {
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 24, 16),
    new THREE.MeshBasicMaterial({ color })
  );
  marker.position.copy(Array.isArray(point) ? vector(point) : point);
  group.add(marker);
  return marker;
}

function addPath(group, incomingStart, hit, outgoingEnd) {
  addLine(group, [incomingStart, hit, outgoingEnd], materials.path);
  addArrow(group, incomingStart, hit, COLORS.path, 0.18, 0.1);
  addArrow(group, hit, outgoingEnd, COLORS.path, 0.18, 0.1);
  addMarker(group, hit, COLORS.hit, 0.055);
}

function addConeFan(group, points) {
  const vectors = points.map(point => Array.isArray(point) ? vector(point) : point);
  const geometry = triangleGeometry(
    vectors,
    vectors.length === 4 ? [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3] : [0, 1, 2]
  );
  group.add(new THREE.Mesh(geometry, materials.coneMesh));
  const edgePoints = vectors.length === 4
    ? [vectors[0], vectors[1], vectors[2], vectors[0], vectors[3], vectors[1], vectors[3], vectors[2]]
    : [vectors[0], vectors[1], vectors[2], vectors[0]];
  addLine(group, edgePoints, materials.coneLine);
}

function fitCameraToBox(controller, box, padding = 2.15, direction = new THREE.Vector3(3.2, 2.6, 3.1)) {
  if (box.isEmpty()) return;
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const radius = Math.max(size.x, size.y, size.z, 1.6);
  controller.camera.position.copy(center).addScaledVector(direction.clone().normalize(), radius * padding);
  controller.controls.target.copy(center);
  controller.camera.near = 0.01;
  controller.camera.far = 120;
  controller.camera.updateProjectionMatrix();
  controller.controls.update();
}

function fitCamera(controller, padding = 2.15, direction = new THREE.Vector3(3.2, 2.6, 3.1)) {
  fitCameraToBox(controller, new THREE.Box3().setFromObject(controller.root), padding, direction);
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

function baryPoint(row, vertices) {
  const out = new THREE.Vector3();
  row.forEach((value, index) => {
    out.addScaledVector(vertices[VERTEX_NAMES[index]], value / 10);
  });
  return out;
}

function weightedBaryPoint(row, vertices = baseVertices) {
  const out = new THREE.Vector3();
  const total = row.reduce((sum, value) => sum + value, 0);
  row.forEach((value, index) => {
    out.addScaledVector(vertices[VERTEX_NAMES[index]], value / total);
  });
  return out;
}

function setupUnfoldingScene() {
  const canvas = document.getElementById("unfoldCanvas");
  if (!canvas) return null;
  const controller = makeScene(canvas);
  const title = document.getElementById("unfoldTitle");
  const view = document.getElementById("unfoldView");
  const path = document.getElementById("unfoldPath");
  const foldButton = document.getElementById("foldStep");
  const unfoldButton = document.getElementById("unfoldStep");
  const word = ["D", "A", "B", "C"];
  const unfoldFaces = ["A", "B", "C", "D"];
  const rows = [
    [3, 4, 3, 0],
    [0, 3, 4, 3],
    [3, 0, 3, 4],
    [4, 3, 0, 3]
  ];
  const liftRows = [rows[0], rows[1], rows[2], rows[3], rows[0], rows[1]];

  const reflectedCopies = [];
  const stateVertices = [];
  let vertices = Object.fromEntries(VERTEX_NAMES.map(name => [name, baseVertices[name].clone()]));
  for (let i = 0; i <= unfoldFaces.length; i++) {
    stateVertices.push(Object.fromEntries(VERTEX_NAMES.map(name => [name, vertices[name].clone()])));
    reflectedCopies.push({ face: unfoldFaces[i] ?? unfoldFaces[unfoldFaces.length - 1], vertices: stateVertices[i] });
    if (i < unfoldFaces.length) vertices = reflectedVertices(vertices, unfoldFaces[i]);
  }

  let step = 0;

  const unfoldingBounds = () => {
    const box = new THREE.Box3();
    reflectedCopies.forEach(copy => {
      VERTEX_NAMES.forEach(name => box.expandByPoint(copy.vertices[name]));
    });
    for (let stateIndex = 0; stateIndex < stateVertices.length; stateIndex++) {
      rows.forEach(row => box.expandByPoint(baryPoint(row, stateVertices[stateIndex])));
    }
    liftRows.forEach((row, index) => {
      const stateIndex = Math.min(Math.max(index - 1, 0), stateVertices.length - 1);
      box.expandByPoint(baryPoint(row, stateVertices[stateIndex]));
    });
    return box;
  };

  const fitUnfoldingCamera = () => {
    fitCameraToBox(controller, unfoldingBounds(), 2.0, new THREE.Vector3(3.2, 2.8, 2.45));
  };

  const draw = () => {
    clearRoot(controller.root);
    const visibleCopies = step + 1;
    for (let index = 0; index < visibleCopies; index++) {
      const copy = reflectedCopies[index];
      addFace(controller.root, copy.vertices, copy.face, 0.12);
      addEdges(controller.root, copy.vertices, index === 0 ? materials.edge : materials.faintEdge);
    }
    const visibleRows = step === 0 ? liftRows.slice(0, -1) : liftRows;
    const points = visibleRows.map((row, index) => {
      const stateIndex = index <= 1 ? 0 : Math.min(index - 1, step);
      return baryPoint(row, stateVertices[stateIndex]);
    });
    addLine(controller.root, points, materials.path);
    points.slice(0, -1).forEach(point => {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.035, 18, 12),
        new THREE.MeshBasicMaterial({ color: COLORS.path })
      );
      marker.position.copy(point);
      controller.root.add(marker);
    });
    foldButton.disabled = step === 0;
    unfoldButton.disabled = step === unfoldFaces.length;
    title.textContent = step === 0
      ? "Closed path DABC"
      : step === unfoldFaces.length
        ? "Straight-line lift"
        : `Unfolded at F_${unfoldFaces[step - 1]}`;
    view.textContent = step === 0
      ? "folded tetrahedron"
      : `${step + 1} tetrahedra`;
    path.textContent = step === 0
      ? "closed broken line"
      : step === unfoldFaces.length
        ? "identified end segments"
        : "first segment fixed";
  };

  foldButton.addEventListener("click", () => {
    step = Math.max(0, step - 1);
    draw();
  });
  unfoldButton.addEventListener("click", () => {
    step = Math.min(unfoldFaces.length, step + 1);
    draw();
  });

  draw();
  fitUnfoldingCamera();
  return {
    controller,
    update() {
      controller.controls.update();
      controller.renderer.render(controller.scene, controller.camera);
    }
  };
}

function roundedEdgeGeometry(radius, length = 2.2, segments = 28) {
  const points = [];
  for (let i = 0; i <= segments; i++) {
    const theta = (i / segments) * Math.PI / 2;
    const x = radius - radius * Math.cos(theta);
    const y = radius - radius * Math.sin(theta);
    points.push(new THREE.Vector3(x, y, -length / 2), new THREE.Vector3(x, y, length / 2));
  }
  const indices = [];
  for (let i = 0; i < segments; i++) {
    const a = i * 2;
    indices.push(a, a + 1, a + 3, a, a + 3, a + 2);
  }
  return triangleGeometry(points, indices);
}

function roundedVertexGeometry(radius, segments = 18) {
  const points = [];
  for (let u = 0; u <= segments; u++) {
    const phi = (u / segments) * Math.PI / 2;
    for (let v = 0; v <= segments; v++) {
      const theta = (v / segments) * Math.PI / 2;
      const nx = Math.sin(phi) * Math.cos(theta);
      const ny = Math.sin(phi) * Math.sin(theta);
      const nz = Math.cos(phi);
      points.push(new THREE.Vector3(
        radius - radius * nx,
        radius - radius * ny,
        radius - radius * nz
      ));
    }
  }
  const row = segments + 1;
  const indices = [];
  for (let u = 0; u < segments; u++) {
    for (let v = 0; v < segments; v++) {
      const a = u * row + v;
      indices.push(a, a + 1, a + row + 1, a, a + row + 1, a + row);
    }
  }
  return triangleGeometry(points, indices);
}

function setupSmoothingScene() {
  const canvas = document.getElementById("smoothCanvas");
  if (!canvas) return null;
  const controller = makeScene(canvas, { autoRotate: true, autoRotateSpeed: 0.28 });
  const tabs = [...document.querySelectorAll(".smooth-tab")];
  const title = document.getElementById("smoothTitle");
  const stratum = document.getElementById("smoothStratum");
  const radiusText = document.getElementById("smoothRadius");
  const reflection = document.getElementById("smoothReflection");
  const play = document.getElementById("smoothPlay");
  const slider = document.getElementById("smoothSlider");

  let mode = "edge";
  let sharpness = 0;
  let playing = true;

  const fitSmoothingCamera = () => {
    fitCamera(controller, 2.25, new THREE.Vector3(3.0, 2.6, 2.7));
  };

  const setMode = next => {
    mode = next;
    tabs.forEach(button => {
      const selected = button.dataset.smooth === mode;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-selected", selected ? "true" : "false");
    });
    draw();
    fitSmoothingCamera();
  };

  const drawEdge = radius => {
    const group = controller.root;
    group.add(quadMesh([[0, radius, -1.1], [0, 1.7, -1.1], [0, 1.7, 1.1], [0, radius, 1.1]], makeFaceMaterial("A", 0.18)));
    group.add(quadMesh([[radius, 0, -1.1], [1.7, 0, -1.1], [1.7, 0, 1.1], [radius, 0, 1.1]], makeFaceMaterial("B", 0.18)));
    group.add(new THREE.Mesh(roundedEdgeGeometry(radius), materials.smoothMesh));
    addLine(group, [[0, 0, -1.16], [0, 0, 1.16]], materials.edge);
    const c = Math.SQRT1_2;
    const hit = new THREE.Vector3(radius - radius * c, radius - radius * c, 0);
    addPath(group, [1.15, 0.7, -0.85], hit, [0.72, 1.15, 0.85]);
    addArrow(group, hit, hit.clone().add(new THREE.Vector3(0.55, 0.55, 0)), COLORS.normal, 0.16, 0.09);
    addConeFan(group, [[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]]);
  };

  const drawVertex = radius => {
    const group = controller.root;
    group.add(quadMesh([[radius, 0, 0], [1.55, 0, 0], [1.55, 1.55, 0], [0, 1.55, 0]], makeFaceMaterial("A", 0.16)));
    group.add(quadMesh([[radius, 0, 0], [1.55, 0, 0], [1.55, 0, 1.55], [0, 0, 1.55]], makeFaceMaterial("B", 0.15)));
    group.add(quadMesh([[0, radius, 0], [0, 1.55, 0], [0, 1.55, 1.55], [0, 0, 1.55]], makeFaceMaterial("C", 0.15)));
    group.add(new THREE.Mesh(roundedVertexGeometry(radius), materials.smoothMesh));
    addLine(group, [[0, 0, 0], [1.65, 0, 0], [0, 0, 0], [0, 1.65, 0], [0, 0, 0], [0, 0, 1.65]], materials.edge);
    const d = 1 / Math.sqrt(3);
    const hit = new THREE.Vector3(radius - radius * d, radius - radius * d, radius - radius * d);
    addPath(group, [1.18, 0.68, 0.95], hit, [0.6, 1.2, 1.08]);
    addArrow(group, hit, hit.clone().add(new THREE.Vector3(0.45, 0.45, 0.45)), COLORS.normal, 0.16, 0.09);
    addConeFan(group, [[0, 0, 0], [0.86, 0, 0], [0, 0.86, 0], [0, 0, 0.86]]);
  };

  const draw = () => {
    clearRoot(controller.root);
    const radius = 0.56 - sharpness * 0.48;
    if (mode === "vertex") drawVertex(radius);
    else drawEdge(radius);
    slider.value = String(Math.round(sharpness * 100));
    title.textContent = mode === "vertex" ? "Rounding a vertex" : "Rounding an edge";
    stratum.textContent = mode;
    radiusText.textContent = radius.toFixed(2);
    reflection.textContent = sharpness > 0.82 ? "limiting singular bounce" : "ordinary on the rounded surface";
  };

  tabs.forEach(button => button.addEventListener("click", () => setMode(button.dataset.smooth)));
  play.addEventListener("click", () => {
    playing = !playing;
    play.textContent = playing ? "Pause" : "Play";
  });
  slider.addEventListener("input", () => {
    playing = false;
    play.textContent = "Play";
    sharpness = Number(slider.value) / 100;
    draw();
  });

  draw();
  fitSmoothingCamera();
  return {
    controller,
    update(now) {
      if (playing) {
        sharpness = (Math.sin(now * 0.00072) + 1) / 2;
        draw();
      }
      controller.controls.update();
      controller.renderer.render(controller.scene, controller.camera);
    }
  };
}

function addArrowedCycle(group, points, material = materials.path) {
  addLine(group, [...points, points[0]], material);
  if (material === materials.faintPath) return;
  points.forEach((point, index) => {
    const next = points[(index + 1) % points.length];
    if (point.distanceTo(next) < 0.08) return;
    const start = point.clone().lerp(next, 0.34);
    const end = point.clone().lerp(next, 0.62);
    addArrow(group, start, end, COLORS.path, 0.08, 0.045);
  });
}

function addTetrahedronShell(group, faceOpacity = 0.11) {
  for (const face of Object.keys(FACES)) addFace(group, baseVertices, face, faceOpacity);
  addEdges(group, baseVertices);
}

function fixedEdgePoints(margin) {
  const half = 0.5 - margin / 2;
  return [
    weightedBaryPoint([half, 0, margin, half]),
    weightedBaryPoint([0, half, half, margin]),
    weightedBaryPoint([half, margin, 0, half]),
    weightedBaryPoint([margin, half, half, 0])
  ];
}

function fixedVertexPoints(margin) {
  return [
    weightedBaryPoint([0, margin, margin, 1 - 2 * margin]),
    weightedBaryPoint([margin, 0, margin, 1 - 2 * margin]),
    weightedBaryPoint([margin, margin, 0, 1 - 2 * margin])
  ];
}

function setupFixedLimitScene() {
  const canvas = document.getElementById("fixedLimitCanvas");
  if (!canvas) return null;
  const controller = makeScene(canvas, { autoRotate: true, autoRotateSpeed: 0.3 });
  const tabs = [...document.querySelectorAll(".fixed-limit-tab")];
  const title = document.getElementById("fixedLimitTitle");
  const stratum = document.getElementById("fixedLimitStratum");
  const marginText = document.getElementById("fixedLimitMargin");
  const body = document.getElementById("fixedLimitBody");
  const play = document.getElementById("fixedLimitPlay");
  const slider = document.getElementById("fixedLimitSlider");

  let mode = "edge";
  let approach = 0;
  let playing = true;

  const fitFixedLimitCamera = () => {
    fitCamera(controller, 2.15, new THREE.Vector3(3.15, 2.45, 2.9));
  };

  const setMode = next => {
    mode = next;
    tabs.forEach(button => {
      const selected = button.dataset.fixedLimit === mode;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-selected", selected ? "true" : "false");
    });
    draw();
    fitFixedLimitCamera();
  };

  const drawEdgeLimit = margin => {
    const group = controller.root;
    addTetrahedronShell(group, 0.1);
    addLine(group, [baseVertices.A, baseVertices.D], materials.limitPath);
    addLine(group, [baseVertices.B, baseVertices.C], materials.limitPath);

    const edgeAD = weightedBaryPoint([0.5, 0, 0, 0.5]);
    const edgeBC = weightedBaryPoint([0, 0.5, 0.5, 0]);
    addLine(group, [edgeAD, edgeBC], materials.limitPath);
    addMarker(group, edgeAD, COLORS.cone, 0.06);
    addMarker(group, edgeBC, COLORS.cone, 0.06);

    addArrowedCycle(group, fixedEdgePoints(0.28), materials.faintPath);
    const points = fixedEdgePoints(margin);
    addArrowedCycle(group, points, materials.path);
    points.forEach(point => addMarker(group, point, COLORS.path, 0.038));
  };

  const drawVertexLimit = margin => {
    const group = controller.root;
    addTetrahedronShell(group, 0.1);
    addLine(group, [baseVertices.D, baseVertices.A], materials.limitPath);
    addLine(group, [baseVertices.D, baseVertices.B], materials.limitPath);
    addLine(group, [baseVertices.D, baseVertices.C], materials.limitPath);
    addMarker(group, baseVertices.D, COLORS.cone, 0.075);

    addArrowedCycle(group, fixedVertexPoints(0.25), materials.faintPath);
    const points = fixedVertexPoints(margin);
    addArrowedCycle(group, points, materials.path);
    points.forEach(point => addMarker(group, point, COLORS.path, 0.038));
  };

  const draw = () => {
    clearRoot(controller.root);
    const margin = 0.28 - approach * 0.268;
    if (mode === "vertex") drawVertexLimit(margin);
    else drawEdgeLimit(margin);
    slider.value = String(Math.round(approach * 100));
    title.textContent = mode === "vertex" ? "Approaching a vertex" : "Approaching opposite edges";
    stratum.textContent = mode === "vertex" ? "D" : "AD BC";
    marginText.textContent = margin.toFixed(3);
    body.textContent = approach > 0.88 ? "fixed, singular closure" : "fixed, ordinary face hits";
  };

  tabs.forEach(button => button.addEventListener("click", () => setMode(button.dataset.fixedLimit)));
  play.addEventListener("click", () => {
    playing = !playing;
    play.textContent = playing ? "Pause" : "Play";
  });
  slider.addEventListener("input", () => {
    playing = false;
    play.textContent = "Play";
    approach = Number(slider.value) / 100;
    draw();
  });

  draw();
  fitFixedLimitCamera();
  return {
    controller,
    update(now) {
      if (playing) {
        approach = (Math.sin(now * 0.00068) + 1) / 2;
        draw();
      }
      controller.controls.update();
      controller.renderer.render(controller.scene, controller.camera);
    }
  };
}

const NORMAL_COPY = {
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

function addGridOnFace(group, size = 2.4, step = 0.4) {
  const points = [];
  const half = size / 2;
  for (let s = -half; s <= half + 0.001; s += step) {
    points.push(new THREE.Vector3(-half, 0.002, s), new THREE.Vector3(half, 0.002, s));
    points.push(new THREE.Vector3(s, 0.002, -half), new THREE.Vector3(s, 0.002, half));
  }
  group.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points), materials.grid));
}

function setupNormalScene() {
  const canvas = document.getElementById("normalConeCanvas");
  if (!canvas) return null;
  const controller = makeScene(canvas, { autoRotate: true, autoRotateSpeed: 0.45 });
  const tabs = [...document.querySelectorAll(".scene-tab")];
  const title = document.getElementById("sceneTitle");
  const active = document.getElementById("sceneActive");
  const cone = document.getElementById("sceneCone");
  const jump = document.getElementById("sceneJump");
  let activeScene = "face";

  const addFaceScene = () => {
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 2.35), makeFaceMaterial("A", 0.2));
    plane.rotation.x = -Math.PI / 2;
    controller.root.add(plane);
    addGridOnFace(controller.root);
    addPath(controller.root, [-1.15, 0.9, -0.65], [0, 0, 0], [1.15, 0.9, -0.65]);
    addArrow(controller.root, [0, 0, 0], [0, 1.1, 0], COLORS.normal, 0.2, 0.11);
    addArrow(controller.root, [0.12, 0.02, 0.12], [0.12, 0.82, 0.12], COLORS.cone, 0.16, 0.09);
  };

  const addEdgeScene = () => {
    controller.root.add(quadMesh([[0, 0, -1.2], [1.8, 0, -1.2], [1.8, 0, 1.2], [0, 0, 1.2]], makeFaceMaterial("A", 0.2)));
    controller.root.add(quadMesh([[0, 0, -1.2], [0, 1.8, -1.2], [0, 1.8, 1.2], [0, 0, 1.2]], makeFaceMaterial("B", 0.18)));
    addLine(controller.root, [[0, 0, -1.24], [0, 0, 1.24]], materials.edge);
    addPath(controller.root, [0.75, 1.18, -0.92], [0, 0, 0], [1.32, 0.54, 0.9]);
    addArrow(controller.root, [0, 0, 0], [0.95, 0, 0], COLORS.normal, 0.17, 0.09);
    addArrow(controller.root, [0, 0, 0], [0, 0.95, 0], COLORS.normal, 0.17, 0.09);
    addArrow(controller.root, [0.03, 0.03, 0.05], [0.76, 0.68, 0.05], COLORS.cone, 0.17, 0.09);
    addConeFan(controller.root, [[0, 0, 0], [1.05, 0, 0], [0, 1.05, 0]]);
  };

  const addVertexScene = () => {
    controller.root.add(quadMesh([[0, 0, 0], [1.75, 0, 0], [1.75, 1.75, 0], [0, 1.75, 0]], makeFaceMaterial("A", 0.18)));
    controller.root.add(quadMesh([[0, 0, 0], [1.75, 0, 0], [1.75, 0, 1.75], [0, 0, 1.75]], makeFaceMaterial("B", 0.16)));
    controller.root.add(quadMesh([[0, 0, 0], [0, 1.75, 0], [0, 1.75, 1.75], [0, 0, 1.75]], makeFaceMaterial("C", 0.16)));
    addLine(controller.root, [[0, 0, 0], [1.8, 0, 0], [0, 0, 0], [0, 1.8, 0], [0, 0, 0], [0, 0, 1.8]], materials.edge);
    addPath(controller.root, [1.18, 0.74, 0.88], [0, 0, 0], [0.58, 1.24, 1.04]);
    addArrow(controller.root, [0, 0, 0], [0.92, 0, 0], COLORS.normal, 0.16, 0.085);
    addArrow(controller.root, [0, 0, 0], [0, 0.92, 0], COLORS.normal, 0.16, 0.085);
    addArrow(controller.root, [0, 0, 0], [0, 0, 0.92], COLORS.normal, 0.16, 0.085);
    addArrow(controller.root, [0.04, 0.04, 0.04], [0.68, 0.62, 0.74], COLORS.cone, 0.17, 0.09);
    addConeFan(controller.root, [[0, 0, 0], [1.02, 0, 0], [0, 1.02, 0], [0, 0, 1.02]]);
  };

  const draw = name => {
    activeScene = name;
    clearRoot(controller.root);
    if (name === "edge") addEdgeScene();
    else if (name === "vertex") addVertexScene();
    else addFaceScene();
    const copy = NORMAL_COPY[name];
    title.textContent = copy.title;
    active.textContent = copy.active;
    cone.textContent = copy.cone;
    jump.textContent = copy.jump;
    tabs.forEach(button => {
      const selected = button.dataset.scene === name;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-selected", selected ? "true" : "false");
    });
    fitCamera(controller, 2.05, activeScene === "face" ? new THREE.Vector3(2.8, 2.25, 2.5) : new THREE.Vector3(3.1, 2.55, 2.8));
  };

  tabs.forEach(button => button.addEventListener("click", () => draw(button.dataset.scene)));
  draw(activeScene);
  return {
    controller,
    update() {
      controller.controls.update();
      controller.renderer.render(controller.scene, controller.camera);
    }
  };
}

const controllers = [
  setupUnfoldingScene(),
  setupSmoothingScene(),
  setupFixedLimitScene(),
  setupNormalScene()
].filter(Boolean);

function animate(now) {
  controllers.forEach(controller => controller.update(now));
  requestAnimationFrame(animate);
}

requestAnimationFrame(animate);
