import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const EXAMPLES = [
  {
    id: "tetra-t-cone",
    name: "Tetrahedron: Taylor T cone",
    family: "tetrahedron",
    status: "area-minimizing cone",
    note: "The canonical tetrahedral soap-film singularity.",
    file: "data/minimal_surface_fe/tetraflm.fe",
    source: "https://kenbrakke.com/cones/tetraflm.fe"
  },
  {
    id: "tetra-skew-quad",
    name: "Tetrahedron: four-edge skew cycle",
    family: "tetrahedron subset",
    status: "relaxed discrete Plateau disk",
    note: "A non-planar film spanning the edge cycle A-B-C-D-A of a regular tetrahedron.",
    file: "data/minimal_surface_fe/tetra_skew_quad_relaxed.json",
    format: "mesh",
    source: "data/minimal_surface_fe/tetra_skew_quad_relaxed.json"
  },
  {
    id: "brakke-skew-quad",
    name: "Skew quadrilateral: Evolver seed",
    family: "quadrilateral subset",
    status: "Surface Evolver example",
    note: "Brakke's minimal skew-quadrilateral seed: the simplest fixed-edge Plateau setup.",
    file: "data/minimal_surface_fe/quad.fe",
    source: "https://kenbrakke.com/evolver/downloads/quad.fe"
  },
  {
    id: "cube-petrie-hexagon",
    name: "Cube: Petrie hexagon",
    family: "cube subset",
    status: "relaxed discrete Plateau disk",
    note: "Six retained cube edges form a non-planar Petrie loop.",
    file: "data/minimal_surface_fe/cube_petrie_hexagon_relaxed.json",
    format: "mesh",
    source: "data/minimal_surface_fe/cube_petrie_hexagon_relaxed.json"
  },
  {
    id: "cube-two-square-caps",
    name: "Cube: top and bottom caps",
    family: "cube two-component boundary",
    status: "disconnected area minimizer",
    note: "Two separate planar disks span the two square loops; this beats the connected annulus in area.",
    file: "data/minimal_surface_fe/cube_two_square_caps.json",
    format: "mesh",
    source: "data/minimal_surface_fe/cube_two_square_caps.json"
  },
  {
    id: "cube-two-square-annulus",
    name: "Cube: connected square annulus",
    family: "cube two-component boundary",
    status: "connected annulus branch",
    note: "A square-catenoid analogue spanning the same two boundary components.",
    file: "data/minimal_surface_fe/cube_two_square_annulus_relaxed.json",
    format: "mesh",
    source: "data/minimal_surface_fe/cube_two_square_annulus_relaxed.json"
  },
  {
    id: "cube-two-square-annulus-sep-1p2",
    name: "Cube: square annulus, separation 1.2",
    family: "cube two-component boundary",
    status: "shorter connected annulus",
    note: "The same two square loops moved closer together; the neck opens up but caps still slightly win in this mesh.",
    file: "data/minimal_surface_fe/cube_two_square_annulus_sep_1p2.json",
    format: "mesh",
    source: "data/minimal_surface_fe/cube_two_square_annulus_sep_1p2.json"
  },
  {
    id: "cube-two-square-annulus-sep-0p7",
    name: "Cube: square annulus, separation 0.7",
    family: "cube two-component boundary",
    status: "cylinder-like annulus",
    note: "At this shorter separation the connected annulus is below the two-cap area and looks cylinder-like.",
    file: "data/minimal_surface_fe/cube_two_square_annulus_sep_0p7.json",
    format: "mesh",
    source: "data/minimal_surface_fe/cube_two_square_annulus_sep_0p7.json"
  },
  {
    id: "live-square-annulus",
    name: "Live: square annulus controls",
    family: "interactive two-component boundary",
    status: "browser area relaxation",
    note: "Move the two square loops by changing separation and twist; the annulus keeps relaxing in real time.",
    format: "liveSquareAnnulus",
    source: "#"
  },
  {
    id: "cube-film",
    name: "Cube: central square film",
    family: "cube",
    status: "Evolver minimal film",
    note: "A cubical frame state with a flat central square.",
    file: "data/minimal_surface_fe/cubefilm.fe",
    source: "https://kenbrakke.com/cones/cubefilm.fe"
  },
  {
    id: "cube-cone",
    name: "Cube: symmetric cone",
    family: "cube",
    status: "unstable cone",
    note: "The symmetric cone is a useful foil for the lower-area cubical film.",
    file: "data/minimal_surface_fe/cubecone.fe",
    source: "https://kenbrakke.com/cones/cubecone.fe"
  },
  {
    id: "octa-skew-hexagon",
    name: "Octahedron: skew hexagon",
    family: "octahedron subset",
    status: "relaxed discrete Plateau disk",
    note: "A six-edge cycle through alternating octahedral directions.",
    file: "data/minimal_surface_fe/octa_skew_hexagon_relaxed.json",
    format: "mesh",
    source: "data/minimal_surface_fe/octa_skew_hexagon_relaxed.json"
  },
  {
    id: "octa-best",
    name: "Octahedron: tetrahedral center",
    family: "octahedron",
    status: "Evolver absolute-minimum candidate",
    note: "The octahedral frame state with a tetrahedral junction near the center.",
    file: "data/minimal_surface_fe/octabest.fe",
    source: "https://kenbrakke.com/evolver/downloads/octabest.fe"
  },
  {
    id: "octa-hex",
    name: "Octahedron: central hexagon",
    family: "octahedron",
    status: "stable branch seed",
    note: "One of the classic octahedral-frame branches.",
    file: "data/minimal_surface_fe/octahex.fe",
    source: "https://kenbrakke.com/evolver/downloads/octahex.fe"
  },
  {
    id: "octa-pent",
    name: "Octahedron: central pentagon",
    family: "octahedron",
    status: "stable branch seed",
    note: "A nearby octahedral topology with a pentagonal center.",
    file: "data/minimal_surface_fe/octapent.fe",
    source: "https://kenbrakke.com/evolver/downloads/octapent.fe"
  },
  {
    id: "octa-quad",
    name: "Octahedron: central quadrilateral",
    family: "octahedron",
    status: "stable branch seed",
    note: "The central polygon has collapsed from the hexagonal branch.",
    file: "data/minimal_surface_fe/octaquad.fe",
    source: "https://kenbrakke.com/evolver/downloads/octaquad.fe"
  },
  {
    id: "octa-square",
    name: "Octahedron: central square",
    family: "octahedron",
    status: "stable branch seed",
    note: "A square-centered octahedral film state.",
    file: "data/minimal_surface_fe/octasq.fe",
    source: "https://kenbrakke.com/evolver/downloads/octasq.fe"
  },
  {
    id: "octa-cone",
    name: "Octahedron: symmetric cone",
    family: "octahedron",
    status: "unstable cone",
    note: "The symmetric cone has illegal high-valence interior singularities for a soap film.",
    file: "data/minimal_surface_fe/octacone.fe",
    source: "https://kenbrakke.com/evolver/downloads/octacone.fe"
  },
  {
    id: "icosa-skew-decagon",
    name: "Icosahedron: skew decagon",
    family: "icosahedron subset",
    status: "relaxed discrete Plateau disk",
    note: "Ten retained icosahedral edges chosen as a strongly non-planar loop.",
    file: "data/minimal_surface_fe/icosahedron_skew_decagon_relaxed.json",
    format: "mesh",
    source: "data/minimal_surface_fe/icosahedron_skew_decagon_relaxed.json"
  },
  {
    id: "dodeca-skew-decagon",
    name: "Dodecahedron: skew decagon",
    family: "dodecahedron subset",
    status: "relaxed discrete Plateau disk",
    note: "Ten retained dodecahedral edges spanning a saddle-like disk.",
    file: "data/minimal_surface_fe/dodecahedron_skew_decagon_relaxed.json",
    format: "mesh",
    source: "data/minimal_surface_fe/dodecahedron_skew_decagon_relaxed.json"
  },
  {
    id: "dodeca-cone",
    name: "Dodecahedron: conical seed",
    family: "dodecahedron",
    status: "nonminimal cone seed",
    note: "A dodecahedral cone datafile meant to be popped and evolved.",
    file: "data/minimal_surface_fe/dodecone.fe",
    source: "https://kenbrakke.com/cones/dodecone.fe"
  }
];

const el = {
  select: document.querySelector("#exampleSelect"),
  details: document.querySelector("#exampleDetails"),
  status: document.querySelector("#loadStatus"),
  title: document.querySelector("#surfaceTitle"),
  subtitle: document.querySelector("#surfaceSubtitle"),
  source: document.querySelector("#sourceLink"),
  canvas: document.querySelector("#sceneCanvas"),
  error: document.querySelector("#errorBox"),
  filmToggle: document.querySelector("#filmToggle"),
  wireToggle: document.querySelector("#wireToggle"),
  junctionToggle: document.querySelector("#junctionToggle"),
  resetView: document.querySelector("#resetView"),
  livePanel: document.querySelector("#livePanel"),
  separationSlider: document.querySelector("#separationSlider"),
  separationValue: document.querySelector("#separationValue"),
  twistSlider: document.querySelector("#twistSlider"),
  twistValue: document.querySelector("#twistValue"),
  relaxSlider: document.querySelector("#relaxSlider"),
  relaxValue: document.querySelector("#relaxValue"),
  resetLiveMesh: document.querySelector("#resetLiveMesh")
};

const fileModeWarning = document.querySelector("#fileModeWarning");
if (fileModeWarning) fileModeWarning.classList.add("is-hidden");

const scene = new THREE.Scene();
scene.background = null;

const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
camera.position.set(4.2, 3.6, 3.2);

const renderer = new THREE.WebGLRenderer({
  canvas: el.canvas,
  antialias: true,
  alpha: true,
  preserveDrawingBuffer: true
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

const controls = new OrbitControls(camera, el.canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

const filmGroup = new THREE.Group();
const boundaryGroup = new THREE.Group();
const junctionGroup = new THREE.Group();
const vertexGroup = new THREE.Group();
scene.add(filmGroup, boundaryGroup, junctionGroup, vertexGroup);

const ambient = new THREE.AmbientLight(0xffffff, 0.72);
const key = new THREE.DirectionalLight(0xffffff, 1.4);
key.position.set(5, 6, 7);
const fill = new THREE.DirectionalLight(0xe5f1ef, 0.6);
fill.position.set(-4, -2, 3);
scene.add(ambient, key, fill);

const boundaryMaterial = new THREE.MeshStandardMaterial({
  color: 0x171b20,
  roughness: 0.55,
  metalness: 0.08
});

const junctionMaterial = new THREE.MeshStandardMaterial({
  color: 0xa16b18,
  roughness: 0.62,
  metalness: 0.04
});

const freeVertexMaterial = new THREE.MeshStandardMaterial({
  color: 0x9b4d55,
  roughness: 0.5
});

const fixedVertexMaterial = new THREE.MeshStandardMaterial({
  color: 0x171b20,
  roughness: 0.5
});

let currentMesh = null;
const liveState = {
  active: false,
  example: null,
  model: null,
  mesh: null,
  bounds: null,
  stepHint: 0.08,
  area: 0,
  steps: 0,
  detailFrame: 0
};

function populateExamples() {
  el.select.innerHTML = "";
  for (const example of EXAMPLES) {
    const option = document.createElement("option");
    option.value = example.id;
    option.textContent = example.name;
    el.select.append(option);
  }
}

function clearGroup(group, disposeMaterials = false) {
  for (const child of group.children) {
    child.traverse((node) => {
      if (node.geometry) node.geometry.dispose();
      if (disposeMaterials && node.material && !Array.isArray(node.material)) node.material.dispose?.();
    });
  }
  group.clear();
}

function setDetails(rows) {
  el.details.innerHTML = "";
  for (const [key, value] of rows) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = value;
    el.details.append(dt, dd);
  }
}

function numericToken(token, defs) {
  const clean = token.trim();
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(clean)) {
    return Number(clean);
  }
  if (clean.startsWith("-")) return -numericToken(clean.slice(1), defs);
  if (clean.startsWith("+")) return numericToken(clean.slice(1), defs);
  if (Object.prototype.hasOwnProperty.call(defs, clean)) return defs[clean];
  throw new Error(`Unknown coordinate token: ${token}`);
}

function parseDefinitions(line, defs) {
  const define = line.match(/^#define\s+([A-Za-z_]\w*)\s+(.+)$/);
  if (define) {
    defs[define[1]] = numericToken(define[2].trim().split(/\s+/)[0], defs);
    return true;
  }

  const parameter = line.match(/^PARAMETER\s+([A-Za-z_]\w*)\s*=\s*([^\s]+)/i);
  if (parameter) {
    defs[parameter[1]] = numericToken(parameter[2], defs);
    return true;
  }

  return false;
}

function parseFe(text) {
  const defs = {};
  const vertices = new Map();
  const edges = new Map();
  const faces = [];
  let section = null;

  const noBlocks = text.replace(/\r/g, "").replace(/\/\*[\s\S]*?\*\//g, " ");
  for (const rawLine of noBlocks.split("\n")) {
    const asciiLine = rawLine.replace(/[^\x20-\x7E]/g, " ");
    const line = asciiLine.replace(/\/\/.*$/, "").trim();
    if (!line) continue;

    if (parseDefinitions(line, defs)) continue;

    const lower = line.toLowerCase();
    if (lower === "vertices" || lower.startsWith("vertices ")) {
      section = "vertices";
      continue;
    }
    if (lower === "edges" || lower.startsWith("edges ")) {
      section = "edges";
      continue;
    }
    if (lower === "faces" || lower.startsWith("faces ")) {
      section = "faces";
      continue;
    }
    if (lower.startsWith("read") || lower.startsWith("gogo")) {
      section = null;
      continue;
    }

    if (!/^[+-]?\d+/.test(line)) continue;
    const tokens = line.split(/\s+/);

    if (section === "vertices") {
      if (tokens.length < 4) continue;
      const id = Number(tokens[0]);
      vertices.set(id, {
        id,
        position: [
          numericToken(tokens[1], defs),
          numericToken(tokens[2], defs),
          numericToken(tokens[3], defs)
        ],
        fixed: /\bfixed\b/i.test(line)
      });
    } else if (section === "edges") {
      if (tokens.length < 3) continue;
      const id = Number(tokens[0]);
      edges.set(id, {
        id,
        a: Number(tokens[1]),
        b: Number(tokens[2]),
        fixed: /\bfixed\b/i.test(line)
      });
    } else if (section === "faces") {
      const id = Number(tokens[0]);
      const edgeIds = [];
      for (const token of tokens.slice(1)) {
        if (!/^[+-]?\d+$/.test(token)) break;
        edgeIds.push(Number(token));
      }
      if (edgeIds.length >= 3) faces.push({ id, edgeIds });
    }
  }

  return { vertices, edges, faces, showVertices: true };
}

function parseMeshJson(mesh) {
  const vertices = new Map();
  const edges = new Map();
  const fixed = new Set(mesh.fixedVertices || []);
  const boundaryKeys = new Set();
  const junctionKeys = new Set();

  for (const [a, b] of mesh.boundaryEdges || []) {
    boundaryKeys.add(`${Math.min(a, b)}:${Math.max(a, b)}`);
  }
  for (const [a, b] of mesh.junctionEdges || []) {
    junctionKeys.add(`${Math.min(a, b)}:${Math.max(a, b)}`);
  }

  mesh.vertices.forEach((position, zeroIndex) => {
    const id = zeroIndex + 1;
    vertices.set(id, {
      id,
      position,
      fixed: fixed.has(zeroIndex)
    });
  });

  let edgeId = 1;
  for (const [a, b] of mesh.boundaryEdges || []) {
    edges.set(edgeId, {
      id: edgeId,
      a: a + 1,
      b: b + 1,
      fixed: true,
      junction: false
    });
    edgeId += 1;
  }
  for (const [a, b] of mesh.junctionEdges || []) {
    edges.set(edgeId, {
      id: edgeId,
      a: a + 1,
      b: b + 1,
      fixed: false,
      junction: true
    });
    edgeId += 1;
  }

  return {
    vertices,
    edges,
    faces: mesh.triangles.map((triangle, index) => ({
      id: index + 1,
      vertexIds: triangle.map((zeroIndex) => zeroIndex + 1)
    })),
    meshMeta: mesh.meta || {},
    hiddenMeshEdges: countHiddenMeshEdges(mesh.triangles, boundaryKeys, junctionKeys),
    showVertices: false,
    smoothFilm: true
  };
}

function countHiddenMeshEdges(triangles, boundaryKeys, junctionKeys) {
  const keys = new Set();
  for (const triangle of triangles || []) {
    const pairs = [
      [triangle[0], triangle[1]],
      [triangle[1], triangle[2]],
      [triangle[2], triangle[0]]
    ];
    for (const [a, b] of pairs) {
      const key = `${Math.min(a, b)}:${Math.max(a, b)}`;
      if (!boundaryKeys.has(key) && !junctionKeys.has(key)) keys.add(key);
    }
  }
  return keys.size;
}

function vAdd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vSub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vMul(scale, a) {
  return [scale * a[0], scale * a[1], scale * a[2]];
}

function vDot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vCross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}

function vNorm(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

function vLerp(a, b, t) {
  return vAdd(vMul(1 - t, a), vMul(t, b));
}

function squareLoop(z, angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [
    [1, 1, z],
    [-1, 1, z],
    [-1, -1, z],
    [1, -1, z]
  ].map(([x, y, zz]) => [c * x - s * y, s * x + c * y, zz]);
}

function subdivideLoop(points, perEdge) {
  const boundary = [];
  for (let i = 0; i < points.length; i += 1) {
    const start = points[i];
    const end = points[(i + 1) % points.length];
    for (let j = 0; j < perEdge; j += 1) {
      boundary.push(vLerp(start, end, j / perEdge));
    }
  }
  return boundary;
}

function makeLiveSquareAnnulusMesh(separation, twistDegrees) {
  const perEdge = 8;
  const levels = 18;
  const half = separation / 2;
  const twist = twistDegrees * Math.PI / 180;
  const bottom = subdivideLoop(squareLoop(-half, 0), perEdge);
  const top = subdivideLoop(squareLoop(half, twist), perEdge);
  const count = bottom.length;
  const vertices = [];
  const fixed = new Set();
  const index = new Map();

  for (let level = 0; level <= levels; level += 1) {
    const t = level / levels;
    for (let i = 0; i < count; i += 1) {
      index.set(`${level}:${i}`, vertices.length);
      vertices.push(vLerp(bottom[i], top[i], t));
      if (level === 0 || level === levels) fixed.add(vertices.length - 1);
    }
  }

  const triangles = [];
  for (let level = 0; level < levels; level += 1) {
    for (let i = 0; i < count; i += 1) {
      const a = index.get(`${level}:${i}`);
      const b = index.get(`${level}:${(i + 1) % count}`);
      const c = index.get(`${level + 1}:${i}`);
      const d = index.get(`${level + 1}:${(i + 1) % count}`);
      triangles.push([a, c, d], [a, d, b]);
    }
  }

  const boundaryEdges = [];
  for (const level of [0, levels]) {
    for (let i = 0; i < count; i += 1) {
      boundaryEdges.push([index.get(`${level}:${i}`), index.get(`${level}:${(i + 1) % count}`)]);
    }
  }

  return { vertices, triangles, fixed, boundaryEdges, separation, twistDegrees, iterations: 0 };
}

function liveMeshToModel(mesh) {
  const vertices = new Map();
  const edges = new Map();
  mesh.vertices.forEach((position, zeroIndex) => {
    const id = zeroIndex + 1;
    vertices.set(id, {
      id,
      position,
      fixed: mesh.fixed.has(zeroIndex)
    });
  });

  mesh.boundaryEdges.forEach(([a, b], index) => {
    edges.set(index + 1, {
      id: index + 1,
      a: a + 1,
      b: b + 1,
      fixed: true,
      junction: false
    });
  });

  return {
    vertices,
    edges,
    faces: mesh.triangles.map((triangle, index) => ({
      id: index + 1,
      vertexIds: triangle.map((zeroIndex) => zeroIndex + 1)
    })),
    meshMeta: {
      boundary_components: 2,
      components: 1,
      separation: mesh.separation,
      twist: mesh.twistDegrees,
      iterations: mesh.iterations,
      area: liveArea(mesh)
    },
    hiddenMeshEdges: countHiddenMeshEdges(mesh.triangles, new Set(mesh.boundaryEdges.map(([a, b]) => `${Math.min(a, b)}:${Math.max(a, b)}`)), new Set()),
    showVertices: false,
    smoothFilm: true,
    liveMesh: mesh
  };
}

function liveTriangleAreaGradient(a, b, c) {
  const normal = vCross(vSub(b, a), vSub(c, a));
  const length = vNorm(normal);
  if (length < 1e-12) {
    return { area: 0, ga: [0, 0, 0], gb: [0, 0, 0], gc: [0, 0, 0] };
  }
  const unit = vMul(1 / length, normal);
  return {
    area: 0.5 * length,
    ga: vMul(0.5, vCross(vSub(b, c), unit)),
    gb: vMul(0.5, vCross(vSub(c, a), unit)),
    gc: vMul(0.5, vCross(vSub(a, b), unit))
  };
}

function liveAreaAndGradient(mesh) {
  const gradient = mesh.vertices.map(() => [0, 0, 0]);
  let area = 0;
  for (const [ia, ib, ic] of mesh.triangles) {
    const { area: triArea, ga, gb, gc } = liveTriangleAreaGradient(
      mesh.vertices[ia],
      mesh.vertices[ib],
      mesh.vertices[ic]
    );
    area += triArea;
    if (!mesh.fixed.has(ia)) gradient[ia] = vAdd(gradient[ia], ga);
    if (!mesh.fixed.has(ib)) gradient[ib] = vAdd(gradient[ib], gb);
    if (!mesh.fixed.has(ic)) gradient[ic] = vAdd(gradient[ic], gc);
  }
  return { area, gradient };
}

function liveArea(mesh) {
  let area = 0;
  for (const [ia, ib, ic] of mesh.triangles) {
    area += liveTriangleAreaGradient(mesh.vertices[ia], mesh.vertices[ib], mesh.vertices[ic]).area;
  }
  return area;
}

function orientedEdge(edge, sign) {
  return sign > 0 ? [edge.a, edge.b] : [edge.b, edge.a];
}

function faceVertexLoop(face, edges) {
  if (face.vertexIds) return face.vertexIds;

  const loop = [];
  for (const signedId of face.edgeIds) {
    const edge = edges.get(Math.abs(signedId));
    if (!edge) throw new Error(`Face ${face.id} references missing edge ${signedId}`);
    const [a, b] = orientedEdge(edge, signedId);
    if (!loop.length) {
      loop.push(a, b);
    } else {
      const last = loop[loop.length - 1];
      if (last === a) {
        loop.push(b);
      } else if (last === b) {
        loop.push(a);
      } else if (loop[0] === b) {
        loop.unshift(a);
      } else if (loop[0] === a) {
        loop.unshift(b);
      } else {
        loop.push(a, b);
      }
    }
  }
  if (loop.length > 1 && loop[0] === loop[loop.length - 1]) loop.pop();
  return loop;
}

function boundsFor(vertices) {
  const box = new THREE.Box3();
  for (const vertex of vertices.values()) {
    box.expandByPoint(new THREE.Vector3(...vertex.position));
  }
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const radius = Math.max(size.x, size.y, size.z) || 1;
  return { center, scale: 3.2 / radius };
}

function transformVertex(position, bounds) {
  return new THREE.Vector3(...position).sub(bounds.center).multiplyScalar(bounds.scale);
}

function originalArea(model) {
  let area = 0;
  for (const face of model.faces) {
    const loop = faceVertexLoop(face, model.edges);
    if (loop.length < 3) continue;
    const p0 = new THREE.Vector3(...model.vertices.get(loop[0]).position);
    for (let i = 1; i < loop.length - 1; i += 1) {
      const p1 = new THREE.Vector3(...model.vertices.get(loop[i]).position);
      const p2 = new THREE.Vector3(...model.vertices.get(loop[i + 1]).position);
      area += triangleArea(p0, p1, p2);
    }
  }
  return area;
}

function triangleArea(a, b, c) {
  return new THREE.Vector3().subVectors(b, a)
    .cross(new THREE.Vector3().subVectors(c, a))
    .length() * 0.5;
}

function makeFilmMesh(model, bounds) {
  const positions = [];
  const colors = [];
  const color = new THREE.Color();
  const palette = [0.51, 0.56, 0.03, 0.61, 0.97, 0.09, 0.45, 0.82];

  for (let faceIndex = 0; faceIndex < model.faces.length; faceIndex += 1) {
    const loop = faceVertexLoop(model.faces[faceIndex], model.edges);
    if (loop.length < 3) continue;
    if (model.smoothFilm) {
      color.setHSL(0.51, 0.34, 0.64);
    } else {
      color.setHSL(palette[faceIndex % palette.length], 0.5, 0.58);
    }
    const p0 = transformVertex(model.vertices.get(loop[0]).position, bounds);

    for (let i = 1; i < loop.length - 1; i += 1) {
      const tri = [
        p0,
        transformVertex(model.vertices.get(loop[i]).position, bounds),
        transformVertex(model.vertices.get(loop[i + 1]).position, bounds)
      ];
      for (const point of tri) {
        positions.push(point.x, point.y, point.z);
        colors.push(color.r, color.g, color.b);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.58,
    roughness: 0.82,
    metalness: 0.02,
    side: THREE.DoubleSide,
    depthWrite: false
  });

  return new THREE.Mesh(geometry, material);
}

function makeTube(a, b, radius, material) {
  const start = new THREE.Vector3(...a);
  const end = new THREE.Vector3(...b);
  const direction = new THREE.Vector3().subVectors(end, start);
  const length = direction.length();
  if (length <= 1e-8) return null;

  const geometry = new THREE.CylinderGeometry(radius, radius, length, 12, 1);
  const orientation = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.clone().normalize()
  );
  geometry.applyQuaternion(orientation);
  geometry.translate(
    (start.x + end.x) * 0.5,
    (start.y + end.y) * 0.5,
    (start.z + end.z) * 0.5
  );
  return new THREE.Mesh(geometry, material);
}

function makeSphere(position, radius, material) {
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 18, 12),
    material
  );
  sphere.position.copy(position);
  return sphere;
}

function updateModelDetails(example, model, areaValue = null) {
  const fixedEdges = [...model.edges.values()].filter((edge) => edge.fixed).length;
  const junctionEdges = [...model.edges.values()].filter((edge) => !edge.fixed && edge.junction !== false).length;
  const fixedVertices = [...model.vertices.values()].filter((vertex) => vertex.fixed).length;
  const freeVertices = model.vertices.size - fixedVertices;
  const area = areaValue ?? model.meshMeta?.area ?? originalArea(model);
  const extraRows = [];

  if (model.meshMeta?.twist !== undefined) {
    extraRows.push(["twist", `${model.meshMeta.twist.toFixed(0)} deg`]);
  }
  if (model.meshMeta?.iterations !== undefined) {
    extraRows.push(["iterations", String(model.meshMeta.iterations)]);
  }

  setDetails([
    ["solid", example.family],
    ["status", example.status],
    ["boundary", model.meshMeta?.boundary_components ? `${model.meshMeta.boundary_components} components` : "1 component"],
    ["surface", model.meshMeta?.components ? `${model.meshMeta.components} components` : "1 component"],
    ["separation", model.meshMeta?.separation ? model.meshMeta.separation.toFixed(3) : "n/a"],
    ...extraRows,
    ["vertices", `${model.vertices.size} total, ${freeVertices} interior`],
    ["edges", `${model.edges.size} shown, ${fixedEdges} boundary, ${junctionEdges} junction`],
    ["sheets", `${model.faces.length}`],
    ["mesh area", area.toFixed(6)],
    ["mesh edges", `${model.hiddenMeshEdges || 0} hidden triangulation`],
    ["data file", example.file ? example.file.split("/").pop() : "browser-generated"]
  ]);
}

function renderModel(example, model) {
  clearGroup(filmGroup, true);
  clearGroup(boundaryGroup);
  clearGroup(junctionGroup);
  clearGroup(vertexGroup);

  const bounds = boundsFor(model.vertices);
  currentMesh = makeFilmMesh(model, bounds);
  filmGroup.add(currentMesh);

  for (const edge of model.edges.values()) {
    const a = transformVertex(model.vertices.get(edge.a).position, bounds);
    const b = transformVertex(model.vertices.get(edge.b).position, bounds);
    const group = edge.fixed ? boundaryGroup : junctionGroup;
    if (!edge.fixed && edge.junction === false) continue;
    const tube = makeTube(a.toArray(), b.toArray(), edge.fixed ? 0.018 : 0.011, edge.fixed ? boundaryMaterial : junctionMaterial);
    if (tube) group.add(tube);
  }

  if (model.showVertices !== false) {
    for (const vertex of model.vertices.values()) {
      const p = transformVertex(vertex.position, bounds);
      vertexGroup.add(makeSphere(p, vertex.fixed ? 0.044 : 0.034, vertex.fixed ? fixedVertexMaterial : freeVertexMaterial));
    }
  }

  el.title.textContent = example.name;
  el.subtitle.textContent = `${example.status}: ${example.note}`;
  el.source.href = example.source;
  updateModelDetails(example, model);

  if (model.liveMesh) {
    liveState.active = true;
    liveState.example = example;
    liveState.model = model;
    liveState.mesh = model.liveMesh;
    liveState.bounds = bounds;
    liveState.area = model.meshMeta.area;
    liveState.steps = model.meshMeta.iterations || 0;
    liveState.detailFrame = 0;
  }

  applyLayerVisibility();
  frameObject();
}

function frameObject() {
  const box = new THREE.Box3();
  box.expandByObject(filmGroup);
  box.expandByObject(boundaryGroup);
  box.expandByObject(junctionGroup);
  if (box.isEmpty()) return;

  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const distance = maxDim * 1.55;
  camera.position.copy(center.clone().add(new THREE.Vector3(distance, distance * 0.82, distance * 0.72)));
  camera.near = Math.max(distance / 100, 0.01);
  camera.far = distance * 20;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function applyLayerVisibility() {
  filmGroup.visible = el.filmToggle.checked;
  boundaryGroup.visible = el.wireToggle.checked;
  junctionGroup.visible = el.junctionToggle.checked;
  vertexGroup.visible = el.wireToggle.checked || el.junctionToggle.checked;
}

function updateLiveLabels() {
  el.separationValue.textContent = Number(el.separationSlider.value).toFixed(2);
  el.twistValue.textContent = `${Number(el.twistSlider.value).toFixed(0)} deg`;
  el.relaxValue.textContent = String(Number(el.relaxSlider.value));
}

function resetLiveSurface() {
  if (!liveState.example) return;
  updateLiveLabels();
  const mesh = makeLiveSquareAnnulusMesh(
    Number(el.separationSlider.value),
    Number(el.twistSlider.value)
  );
  const model = liveMeshToModel(mesh);
  liveState.stepHint = 0.08;
  renderModel(liveState.example, model);
}

function relaxLiveStep() {
  const mesh = liveState.mesh;
  const { area, gradient } = liveAreaAndGradient(mesh);
  let norm2 = 0;
  for (let i = 0; i < gradient.length; i += 1) {
    if (!mesh.fixed.has(i)) norm2 += vDot(gradient[i], gradient[i]);
  }
  if (Math.sqrt(norm2) < 1e-8) {
    liveState.area = area;
    return false;
  }

  let step = liveState.stepHint;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    const trial = mesh.vertices.map((point, index) => (
      mesh.fixed.has(index) ? point : vAdd(point, vMul(-step, gradient[index]))
    ));
    const trialMesh = { ...mesh, vertices: trial };
    const trialArea = liveArea(trialMesh);
    if (trialArea < area) {
      mesh.vertices = trial;
      mesh.iterations += 1;
      liveState.steps = mesh.iterations;
      liveState.area = trialArea;
      liveState.stepHint = Math.min(step * 1.05, 0.18);
      return true;
    }
    step *= 0.5;
  }

  liveState.area = area;
  liveState.stepHint *= 0.5;
  return false;
}

function syncLiveModelPositions() {
  for (let i = 0; i < liveState.mesh.vertices.length; i += 1) {
    liveState.model.vertices.get(i + 1).position = liveState.mesh.vertices[i];
  }
  liveState.model.meshMeta.area = liveState.area;
  liveState.model.meshMeta.iterations = liveState.steps;
}

function syncLiveGeometry() {
  if (!currentMesh || !liveState.bounds) return;
  const position = currentMesh.geometry.getAttribute("position");
  let cursor = 0;
  for (const [ia, ib, ic] of liveState.mesh.triangles) {
    for (const index of [ia, ib, ic]) {
      const point = transformVertex(liveState.mesh.vertices[index], liveState.bounds);
      position.setXYZ(cursor, point.x, point.y, point.z);
      cursor += 1;
    }
  }
  position.needsUpdate = true;
  currentMesh.geometry.computeVertexNormals();
}

function tickLiveSolver() {
  if (!liveState.active || !liveState.mesh) return;
  const steps = Number(el.relaxSlider.value);
  let changed = false;
  for (let i = 0; i < steps; i += 1) {
    changed = relaxLiveStep() || changed;
  }
  if (!changed && steps > 0) return;
  syncLiveModelPositions();
  syncLiveGeometry();
  liveState.detailFrame += 1;
  if (liveState.detailFrame % 12 === 0 || steps === 0) {
    updateModelDetails(liveState.example, liveState.model, liveState.area);
  }
}

async function loadExample(id) {
  const example = EXAMPLES.find((item) => item.id === id) || EXAMPLES[0];
  el.error.classList.add("is-hidden");
  liveState.active = false;
  liveState.example = null;
  liveState.mesh = null;
  liveState.model = null;
  el.livePanel.classList.toggle("is-hidden", example.format !== "liveSquareAnnulus");
  updateLiveLabels();

  if (example.format === "liveSquareAnnulus") {
    liveState.example = example;
    el.status.textContent = "Running browser-side area relaxation";
    resetLiveSurface();
    el.status.textContent = `${EXAMPLES.length} local examples`;
    return;
  }

  el.status.textContent = "Loading Surface Evolver data...";
  const response = await fetch(example.file);
  if (!response.ok) throw new Error(`Could not load ${example.file}`);
  const model = example.format === "mesh"
    ? parseMeshJson(await response.json())
    : parseFe(await response.text());
  renderModel(example, model);
  el.status.textContent = `${EXAMPLES.length} local examples`;
}

function resize() {
  const rect = el.canvas.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function animate() {
  tickLiveSolver();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

function init() {
  populateExamples();
  resize();
  loadExample(EXAMPLES[0].id).catch(showError);
  animate();
}

function showError(error) {
  el.error.textContent = error.message;
  el.error.classList.remove("is-hidden");
  el.status.textContent = "Example load failed";
}

el.select.addEventListener("change", () => {
  loadExample(el.select.value).catch(showError);
});

el.filmToggle.addEventListener("change", applyLayerVisibility);
el.wireToggle.addEventListener("change", applyLayerVisibility);
el.junctionToggle.addEventListener("change", applyLayerVisibility);
el.resetView.addEventListener("click", frameObject);
el.separationSlider.addEventListener("input", resetLiveSurface);
el.twistSlider.addEventListener("input", resetLiveSurface);
el.relaxSlider.addEventListener("input", updateLiveLabels);
el.resetLiveMesh.addEventListener("click", resetLiveSurface);

window.addEventListener("resize", resize);
init();
