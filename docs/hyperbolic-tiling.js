const canvas = document.querySelector("#tilingCanvas");
const symmetrySelect = document.querySelector("#symmetrySelect");
const modelButtons = Array.from(document.querySelectorAll("[data-model]"));
const zoomInButton = document.querySelector("#zoomIn");
const zoomOutButton = document.querySelector("#zoomOut");
const resetButton = document.querySelector("#resetView");
const scaleReadout = document.querySelector("#scaleReadout");
const crossingReadout = document.querySelector("#crossingReadout");
const errorBox = document.querySelector("#webglError");

const gl = canvas.getContext("webgl", {
  antialias: false,
  alpha: false,
  depth: false,
  preserveDrawingBuffer: false
});

const vertexSource = `
  attribute vec2 a_position;

  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

const fragmentSource = `
  #extension GL_OES_standard_derivatives : enable
  precision highp float;

  uniform vec2 u_resolution;
  uniform float u_p;
  uniform float u_q;
  uniform float u_model;
  uniform float u_orientation;
  uniform float u_globalParity;
  uniform float u_viewZoom;
  uniform vec2 u_viewPan;
  uniform vec2 u_a;
  uniform vec2 u_b;
  uniform vec2 u_c;
  uniform vec2 u_d;

  const float PI = 3.141592653589793;

  vec2 cMul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
  }

  vec2 cDiv(vec2 a, vec2 b) {
    float den = max(dot(b, b), 1.0e-18);
    return vec2(
      (a.x * b.x + a.y * b.y) / den,
      (a.y * b.x - a.x * b.y) / den
    );
  }

  vec3 rgb(float r, float g, float b) {
    return vec3(r, g, b) / 255.0;
  }

  float ellipse(vec2 p, vec2 center, vec2 radii) {
    vec2 d = (p - center) / radii;
    return length(d) - 1.0;
  }

  void main() {
    vec2 screenPixel = gl_FragCoord.xy;
    vec2 pixel = 0.5 * u_resolution +
      (screenPixel - 0.5 * u_resolution - u_viewPan) / u_viewZoom;
    vec2 local;
    float frameDistance;

    if (u_model < 0.5) {
      float radius = min(u_resolution.x, u_resolution.y) * 0.455;
      local = (pixel - 0.5 * u_resolution) / radius;
      frameDistance = 1.0 - length(local);
    } else {
      float scale = min(u_resolution.x * 0.245, u_resolution.y * 0.275);
      vec2 halfPlane = vec2(
        (pixel.x - 0.5 * u_resolution.x) / scale,
        (pixel.y - 0.075 * u_resolution.y) / scale
      );
      frameDistance = halfPlane.y;
      vec2 numerator = halfPlane - vec2(0.0, 1.0);
      vec2 denominator = halfPlane + vec2(0.0, 1.0);
      local = cDiv(numerator, denominator);
    }

    vec3 outside = rgb(13.0, 31.0, 34.0);
    if (frameDistance < 0.0) {
      float edgeGlow = exp(-abs(frameDistance) * 125.0);
      gl_FragColor = vec4(mix(outside, rgb(205.0, 183.0, 138.0), edgeGlow * 0.5), 1.0);
      return;
    }

    if (u_orientation > 0.5) local.y = -local.y;
    vec2 numerator = cMul(u_a, local) + u_b;
    vec2 denominator = cMul(u_c, local) + u_d;
    vec2 point = cDiv(numerator, denominator);
    float pointRadius = length(point);
    if (pointRadius >= 0.9999995) point *= 0.9999995 / pointRadius;

    float angle = PI / u_p;
    float cosAngle = cos(angle);
    float sinAngle = sin(angle);
    float coshRadius = (cos(angle) / sin(angle)) * (cos(PI / u_q) / sin(PI / u_q));
    float vertexRadius = sqrt(max(0.0001, (coshRadius - 1.0) / (coshRadius + 1.0)));
    float sideCenter = (vertexRadius * vertexRadius + 1.0) / (2.0 * vertexRadius * cosAngle);
    float sideRadius2 = sideCenter * sideCenter - 1.0;
    vec2 sideOrigin = vec2(sideCenter, 0.0);
    vec2 lineRotation = vec2(cos(2.0 * angle), sin(2.0 * angle));

    float parity = u_globalParity;
    float reflectionCount = 0.0;
    float sideCrossings = 0.0;

    for (int iteration = 0; iteration < 96; iteration += 1) {
      if (point.y < 0.0) {
        point.y = -point.y;
        parity = 1.0 - parity;
        reflectionCount += 1.0;
        continue;
      }

      if (cosAngle * point.y - sinAngle * point.x > 0.0) {
        point = cMul(lineRotation, vec2(point.x, -point.y));
        parity = 1.0 - parity;
        reflectionCount += 1.0;
        continue;
      }

      vec2 sideDelta = point - sideOrigin;
      float sideDistance2 = dot(sideDelta, sideDelta);
      if (sideDistance2 < sideRadius2) {
        point = sideOrigin + sideDelta * sideRadius2 / max(sideDistance2, 1.0e-12);
        parity = 1.0 - parity;
        reflectionCount += 1.0;
        sideCrossings += 1.0;
        continue;
      }
      break;
    }

    float theta = clamp(atan(point.y, point.x), 0.0, angle);
    float rootTerm = max(sideCenter * sideCenter * cos(theta) * cos(theta) - 1.0, 0.0);
    float radialEdge = sideCenter * cos(theta) - sqrt(rootTerm);
    vec2 motif = vec2(theta / angle, length(point) / max(radialEdge, 0.0001));

    vec3 teal = rgb(20.0, 79.0, 82.0);
    vec3 deepTeal = rgb(10.0, 42.0, 45.0);
    vec3 ochre = rgb(210.0, 157.0, 78.0);
    vec3 coral = rgb(177.0, 72.0, 51.0);
    vec3 parchment = rgb(239.0, 224.0, 191.0);
    vec3 ink = rgb(12.0, 35.0, 37.0);

    vec3 base = parity < 0.5 ? ochre : teal;
    vec3 creature = parity < 0.5 ? deepTeal : coral;
    vec3 highlight = parity < 0.5 ? parchment : ochre;

    float body = ellipse(motif, vec2(0.52, 0.54), vec2(0.43, 0.24));
    float wingA = ellipse(motif, vec2(0.67, 0.57), vec2(0.22, 0.38));
    float wingB = ellipse(motif, vec2(0.34, 0.53), vec2(0.16, 0.31));
    float hookedHead = ellipse(motif, vec2(0.23, 0.35), vec2(0.17, 0.14));
    float tailCut = abs(motif.x - (0.62 + 0.23 * motif.y)) - (0.17 - 0.08 * motif.y);
    float bodyMask = smoothstep(0.025, -0.018, min(body, min(wingA + 0.12, hookedHead + 0.04)));
    bodyMask = max(bodyMask, smoothstep(0.02, -0.02, wingB + 0.12));
    bodyMask *= smoothstep(-0.02, 0.035, tailCut);
    vec3 color = mix(base, creature, bodyMask);

    float plume = abs(motif.x - (0.49 + 0.13 * sin(5.4 * motif.y + 0.4))) - 0.016;
    float feather1 = abs(motif.y - (0.45 + 0.11 * sin(6.2 * motif.x))) - 0.012;
    float feather2 = abs(motif.y - (0.63 + 0.08 * sin(7.0 * motif.x + 0.7))) - 0.010;
    float featherWidth = max(fwidth(plume), 0.005);
    float featherLines = 1.0 - smoothstep(0.0, featherWidth * 1.5, min(plume, min(feather1, feather2)));
    featherLines *= bodyMask;
    color = mix(color, highlight, featherLines * 0.78);

    float eye = ellipse(motif, vec2(0.205, 0.335), vec2(0.032, 0.025));
    float eyeMask = smoothstep(0.18, -0.08, eye) * bodyMask;
    color = mix(color, parchment, eyeMask);
    float pupil = ellipse(motif, vec2(0.205, 0.335), vec2(0.012, 0.012));
    color = mix(color, ink, smoothstep(0.18, -0.05, pupil));

    float rayEdgeA = abs(point.y);
    float rayEdgeB = abs(cosAngle * point.y - sinAngle * point.x);
    float circleEdge = abs(dot(point - sideOrigin, point - sideOrigin) - sideRadius2);
    float geometricEdge = min(rayEdgeA, min(rayEdgeB, circleEdge * 0.35));
    float edgeWidth = max(fwidth(geometricEdge), 0.00015);
    float edgeLine = 1.0 - smoothstep(edgeWidth * 0.55, edgeWidth * 1.7, geometricEdge);

    float bodyOutline = abs(body);
    float outlineWidth = max(fwidth(bodyOutline), 0.006);
    float motifOutline = (1.0 - smoothstep(0.0, outlineWidth * 1.6, bodyOutline)) * 0.72;
    color = mix(color, ink, max(edgeLine * 0.9, motifOutline * bodyMask));

    float texture = 0.018 * sin((motif.x * 47.0 + motif.y * 31.0 + sideCrossings * 2.0) * PI);
    color += texture * (0.35 + 0.65 * bodyMask);

    float frameWidth = u_model < 0.5 ? fwidth(frameDistance) : fwidth(frameDistance) * 0.8;
    float frameLine = 1.0 - smoothstep(frameWidth * 0.7, frameWidth * 2.2, frameDistance);
    color = mix(color, parchment, frameLine * 0.78);

    float vignette = smoothstep(1.18, 0.18, length((screenPixel - 0.5 * u_resolution) / u_resolution.y));
    color *= 0.82 + 0.18 * vignette;
    gl_FragColor = vec4(color, 1.0);
  }
`;

const state = {
  model: "disk",
  p: 8,
  q: 3,
  orientation: 0,
  globalParity: 0,
  matrix: {
    a: { x: 1, y: 0 },
    b: { x: 0, y: 0 },
    c: { x: 0, y: 0 },
    d: { x: 1, y: 0 }
  },
  viewZoom: 1,
  viewPan: { x: 0, y: 0 },
  lastZoomAnchor: null,
  drag: null,
  program: null,
  uniforms: null
};

function complex(x, y = 0) {
  return { x, y };
}

function add(a, b) {
  return complex(a.x + b.x, a.y + b.y);
}

function sub(a, b) {
  return complex(a.x - b.x, a.y - b.y);
}

function mul(a, b) {
  return complex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

function scale(a, amount) {
  return complex(a.x * amount, a.y * amount);
}

function conjugate(a) {
  return complex(a.x, -a.y);
}

function divide(a, b) {
  const denominator = b.x * b.x + b.y * b.y;
  return complex(
    (a.x * b.x + a.y * b.y) / denominator,
    (a.y * b.x - a.x * b.y) / denominator
  );
}

function magnitude(a) {
  return Math.hypot(a.x, a.y);
}

function matrixMultiply(left, right) {
  return {
    a: add(mul(left.a, right.a), mul(left.b, right.c)),
    b: add(mul(left.a, right.b), mul(left.b, right.d)),
    c: add(mul(left.c, right.a), mul(left.d, right.c)),
    d: add(mul(left.c, right.b), mul(left.d, right.d))
  };
}

function canonicalizeMatrix() {
  const center = divide(state.matrix.b, state.matrix.d);
  const determinant = sub(
    mul(state.matrix.a, state.matrix.d),
    mul(state.matrix.b, state.matrix.c)
  );
  const derivative = divide(determinant, mul(state.matrix.d, state.matrix.d));
  const derivativeLength = magnitude(derivative);
  const phase = derivativeLength > 1e-12
    ? scale(derivative, 1 / derivativeLength)
    : complex(1);
  const translation = divide(center, phase);

  state.matrix = {
    a: phase,
    b: center,
    c: conjugate(translation),
    d: complex(1)
  };
}

function cameraPoint(point) {
  const source = state.orientation ? conjugate(point) : point;
  return divide(
    add(mul(state.matrix.a, source), state.matrix.b),
    add(mul(state.matrix.c, source), state.matrix.d)
  );
}

function postReflectLine(angle) {
  const phase = complex(Math.cos(2 * angle), Math.sin(2 * angle));
  state.matrix = {
    a: mul(phase, conjugate(state.matrix.a)),
    b: mul(phase, conjugate(state.matrix.b)),
    c: conjugate(state.matrix.c),
    d: conjugate(state.matrix.d)
  };
  state.orientation = 1 - state.orientation;
  state.globalParity = 1 - state.globalParity;
  canonicalizeMatrix();
}

function postReflectCircle(center) {
  const reflection = {
    a: complex(center),
    b: complex(-1),
    c: complex(1),
    d: complex(-center)
  };
  const conjugated = {
    a: conjugate(state.matrix.a),
    b: conjugate(state.matrix.b),
    c: conjugate(state.matrix.c),
    d: conjugate(state.matrix.d)
  };
  state.matrix = matrixMultiply(reflection, conjugated);
  state.orientation = 1 - state.orientation;
  state.globalParity = 1 - state.globalParity;
  canonicalizeMatrix();
}

function tilingGeometry() {
  const angle = Math.PI / state.p;
  const coshRadius = (1 / Math.tan(angle)) * (1 / Math.tan(Math.PI / state.q));
  const vertexRadius = Math.sqrt((coshRadius - 1) / (coshRadius + 1));
  const sideCenter = (vertexRadius * vertexRadius + 1) / (2 * vertexRadius * Math.cos(angle));
  return {
    angle,
    sideCenter,
    sideRadius2: sideCenter * sideCenter - 1
  };
}

function recenterCamera() {
  const geometry = tilingGeometry();
  const epsilon = 2e-10;

  for (let iteration = 0; iteration < 80; iteration += 1) {
    const point = cameraPoint(complex(0));
    if (point.y < -epsilon) {
      postReflectLine(0);
      continue;
    }

    const upperSide = Math.cos(geometry.angle) * point.y - Math.sin(geometry.angle) * point.x;
    if (upperSide > epsilon) {
      postReflectLine(geometry.angle);
      continue;
    }

    const delta = sub(point, complex(geometry.sideCenter));
    if (delta.x * delta.x + delta.y * delta.y < geometry.sideRadius2 - epsilon) {
      postReflectCircle(geometry.sideCenter);
      continue;
    }
    break;
  }
}

function translateCamera(direction, distance) {
  const length = magnitude(direction);
  if (length < 1e-8 || distance === 0) return;
  const unit = scale(direction, 1 / length);
  const amount = Math.tanh(distance / 2);
  let translation = scale(unit, amount);
  if (state.orientation) translation = conjugate(translation);

  const localTransform = {
    a: complex(1),
    b: translation,
    c: conjugate(translation),
    d: complex(1)
  };
  state.matrix = matrixMultiply(state.matrix, localTransform);
  canonicalizeMatrix();
  recenterCamera();
  updateStatus();
  draw();
}

function formatMagnification(value) {
  if (value < 1000) {
    const digits = value < 10 ? 1 : 0;
    return `×${value.toFixed(digits)}`;
  }
  const exponent = Math.floor(Math.log10(value));
  const mantissa = value / 10 ** exponent;
  return `×${mantissa.toFixed(1)}·10^${exponent}`;
}

function updateStatus() {
  scaleReadout.textContent = `magnification ${formatMagnification(state.viewZoom)}`;
  crossingReadout.textContent = `boundary fixed · {${state.p}, ${state.q}}`;
}

function resetOpticalView() {
  state.viewZoom = 1;
  state.viewPan = { x: 0, y: 0 };
  state.lastZoomAnchor = null;
}

function resetCamera() {
  state.orientation = 0;
  state.globalParity = 0;
  state.matrix = {
    a: complex(1),
    b: complex(0),
    c: complex(0),
    d: complex(1)
  };
  resetOpticalView();
  updateStatus();
  draw();
}

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function initializeWebGL() {
  if (!gl || !gl.getExtension("OES_standard_derivatives")) {
    throw new Error("WebGL standard derivatives are unavailable.");
  }

  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,
     1, -1,
    -1,  1,
    -1,  1,
     1, -1,
     1,  1
  ]), gl.STATIC_DRAW);

  gl.useProgram(program);
  const position = gl.getAttribLocation(program, "a_position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

  state.program = program;
  state.uniforms = {
    resolution: gl.getUniformLocation(program, "u_resolution"),
    p: gl.getUniformLocation(program, "u_p"),
    q: gl.getUniformLocation(program, "u_q"),
    model: gl.getUniformLocation(program, "u_model"),
    orientation: gl.getUniformLocation(program, "u_orientation"),
    globalParity: gl.getUniformLocation(program, "u_globalParity"),
    viewZoom: gl.getUniformLocation(program, "u_viewZoom"),
    viewPan: gl.getUniformLocation(program, "u_viewPan"),
    a: gl.getUniformLocation(program, "u_a"),
    b: gl.getUniformLocation(program, "u_b"),
    c: gl.getUniformLocation(program, "u_c"),
    d: gl.getUniformLocation(program, "u_d")
  };
}

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function sendComplex(location, value) {
  gl.uniform2f(location, value.x, value.y);
}

function draw() {
  if (!state.program) return;
  resizeCanvas();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.useProgram(state.program);
  gl.uniform2f(state.uniforms.resolution, canvas.width, canvas.height);
  gl.uniform1f(state.uniforms.p, state.p);
  gl.uniform1f(state.uniforms.q, state.q);
  gl.uniform1f(state.uniforms.model, state.model === "disk" ? 0 : 1);
  gl.uniform1f(state.uniforms.orientation, state.orientation);
  gl.uniform1f(state.uniforms.globalParity, state.globalParity);
  gl.uniform1f(state.uniforms.viewZoom, state.viewZoom);
  const rect = canvas.getBoundingClientRect();
  const pixelRatio = rect.width > 0 ? canvas.width / rect.width : 1;
  gl.uniform2f(
    state.uniforms.viewPan,
    state.viewPan.x * pixelRatio,
    -state.viewPan.y * pixelRatio
  );
  sendComplex(state.uniforms.a, state.matrix.a);
  sendComplex(state.uniforms.b, state.matrix.b);
  sendComplex(state.uniforms.c, state.matrix.c);
  sendComplex(state.uniforms.d, state.matrix.d);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
    width: rect.width,
    height: rect.height
  };
}

function defaultZoomAnchor() {
  const rect = canvas.getBoundingClientRect();
  const radius = Math.min(rect.width, rect.height) * 0.455;
  return {
    x: rect.width / 2 + radius,
    y: rect.height / 2,
    width: rect.width,
    height: rect.height
  };
}

function zoomAnchorFromEvent(event) {
  const point = canvasPoint(event);
  const center = {
    x: point.width / 2 + state.viewPan.x,
    y: point.height / 2 + state.viewPan.y
  };

  if (state.model === "disk") {
    const radius = Math.min(point.width, point.height) * 0.455 * state.viewZoom;
    const dx = point.x - center.x;
    const dy = point.y - center.y;
    const distance = Math.hypot(dx, dy);
    const snapDistance = Math.max(18, Math.min(72, radius * 0.08));
    if (distance > 1e-6 && Math.abs(distance - radius) <= snapDistance) {
      return {
        ...point,
        x: center.x + dx * radius / distance,
        y: center.y + dy * radius / distance
      };
    }
    return point;
  }

  const boundaryY = point.height / 2 + state.viewPan.y +
    state.viewZoom * (point.height * 0.925 - point.height / 2);
  if (Math.abs(point.y - boundaryY) <= 36) {
    return { ...point, y: boundaryY };
  }
  return point;
}

function zoomViewAt(factor, anchor = state.lastZoomAnchor || defaultZoomAnchor()) {
  const previousZoom = state.viewZoom;
  const nextZoom = Math.max(0.55, Math.min(1e6, previousZoom * factor));
  const appliedFactor = nextZoom / previousZoom;
  const centerX = anchor.width / 2;
  const centerY = anchor.height / 2;
  state.viewPan = {
    x: anchor.x - centerX - appliedFactor * (anchor.x - centerX - state.viewPan.x),
    y: anchor.y - centerY - appliedFactor * (anchor.y - centerY - state.viewPan.y)
  };
  state.viewZoom = nextZoom;
  state.lastZoomAnchor = anchor;
  updateStatus();
  draw();
}

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const factor = Math.max(0.55, Math.min(1.85, Math.exp(-event.deltaY * 0.0018)));
  zoomViewAt(factor, zoomAnchorFromEvent(event));
}, { passive: false });

canvas.addEventListener("dblclick", (event) => {
  zoomViewAt(3, zoomAnchorFromEvent(event));
});

canvas.addEventListener("pointerdown", (event) => {
  canvas.setPointerCapture(event.pointerId);
  state.drag = {
    id: event.pointerId,
    x: event.clientX,
    y: event.clientY
  };
  canvas.classList.add("is-dragging");
});

canvas.addEventListener("pointermove", (event) => {
  if (!state.drag || state.drag.id !== event.pointerId) return;
  const rect = canvas.getBoundingClientRect();
  const dx = event.clientX - state.drag.x;
  const dy = event.clientY - state.drag.y;
  state.drag.x = event.clientX;
  state.drag.y = event.clientY;
  const direction = complex(-dx / rect.width, dy / rect.height);
  const distance = Math.min(0.24, magnitude(direction) * 4.8);
  if (distance > 0.002) translateCamera(direction, distance);
});

function finishDrag(event) {
  if (!state.drag || state.drag.id !== event.pointerId) return;
  state.drag = null;
  canvas.classList.remove("is-dragging");
}

canvas.addEventListener("pointerup", finishDrag);
canvas.addEventListener("pointercancel", finishDrag);

modelButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.model = button.dataset.model;
    modelButtons.forEach((candidate) => {
      const active = candidate === button;
      candidate.classList.toggle("is-active", active);
      candidate.setAttribute("aria-pressed", String(active));
    });
    resetOpticalView();
    updateStatus();
    draw();
  });
});

symmetrySelect.addEventListener("change", () => {
  const [p, q] = symmetrySelect.value.split(",").map(Number);
  state.p = p;
  state.q = q;
  resetCamera();
});

zoomInButton.addEventListener("click", () => zoomViewAt(1.8));
zoomOutButton.addEventListener("click", () => zoomViewAt(1 / 1.8));
resetButton.addEventListener("click", resetCamera);

const resizeObserver = new ResizeObserver(draw);
resizeObserver.observe(canvas);
window.addEventListener("orientationchange", draw);

try {
  initializeWebGL();
  updateStatus();
  draw();
} catch (error) {
  errorBox.hidden = false;
  console.error(error);
}
