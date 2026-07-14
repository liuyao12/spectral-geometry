const canvas = document.querySelector("#tilingCanvas");
const designSelect = document.querySelector("#designSelect");
const symmetrySelect = document.querySelector("#symmetrySelect");
const edgePlayInput = document.querySelector("#edgePlay");
const edgeReadout = document.querySelector("#edgeReadout");
const modelButtons = Array.from(document.querySelectorAll("[data-model]"));
const zoomInButton = document.querySelector("#zoomIn");
const zoomOutButton = document.querySelector("#zoomOut");
const resetButton = document.querySelector("#resetView");
const scaleReadout = document.querySelector("#scaleReadout");
const crossingReadout = document.querySelector("#crossingReadout");
const errorBox = document.querySelector("#webglError");
const designStudio = document.querySelector("#designStudio");
const designForm = document.querySelector("#designForm");
const designNameInput = document.querySelector("#designName");
const motifEditor = document.querySelector("#motifEditor");
const motifShapeLayer = document.querySelector("#motifShapeLayer");
const shapeCount = document.querySelector("#shapeCount");
const shapeSelect = document.querySelector("#shapeSelect");
const shapeTypeSelect = document.querySelector("#shapeType");
const shapeColorSelect = document.querySelector("#shapeColor");
const shapeXInput = document.querySelector("#shapeX");
const shapeYInput = document.querySelector("#shapeY");
const shapeWidthInput = document.querySelector("#shapeWidth");
const shapeHeightInput = document.querySelector("#shapeHeight");
const shapeRotationInput = document.querySelector("#shapeRotation");
const shapeXReadout = document.querySelector("#shapeXReadout");
const shapeYReadout = document.querySelector("#shapeYReadout");
const shapeWidthReadout = document.querySelector("#shapeWidthReadout");
const shapeHeightReadout = document.querySelector("#shapeHeightReadout");
const shapeRotationReadout = document.querySelector("#shapeRotationReadout");
const harmonicFitInput = document.querySelector("#harmonicFit");
const harmonicFitReadout = document.querySelector("#harmonicFitReadout");
const edgeCurveInputs = [
  document.querySelector("#edgeCurveA"),
  document.querySelector("#edgeCurveB"),
  document.querySelector("#edgeCurveC")
];
const edgeCurveReadouts = [
  document.querySelector("#edgeCurveAReadout"),
  document.querySelector("#edgeCurveBReadout"),
  document.querySelector("#edgeCurveCReadout")
];
const paletteInputs = [
  document.querySelector("#colorTileA"),
  document.querySelector("#colorTileB"),
  document.querySelector("#colorFigure"),
  document.querySelector("#colorAccent"),
  document.querySelector("#colorInk"),
  document.querySelector("#colorFrame")
];
const addShapeButton = document.querySelector("#addShape");
const duplicateShapeButton = document.querySelector("#duplicateShape");
const moveShapeBackButton = document.querySelector("#moveShapeBack");
const deleteShapeButton = document.querySelector("#deleteShape");
const saveDesignButton = document.querySelector("#saveDesign");
const shareDesignButton = document.querySelector("#shareDesign");
const exportDesignButton = document.querySelector("#exportDesign");
const importDesignInput = document.querySelector("#importDesign");
const resetDesignButton = document.querySelector("#resetDesign");
const studioStatus = document.querySelector("#studioStatus");

const MAX_CUSTOM_SHAPES = 12;
const CUSTOM_DESIGN_STORAGE_KEY = "hyperbolic-escher-design-v1";
const SHAPE_TYPES = ["ellipse", "capsule", "ring"];
const BUILT_IN_SYMMETRIES = [
  [6, 4],
  [5, 4],
  [8, 3],
  [7, 3]
];

function starterCustomDesign() {
  return {
    version: 2,
    name: "My interlocking creature",
    symmetry: [6, 4],
    edgePlay: 0.58,
    harmonicFit: 1,
    edges: [0.8, -0.35, 0.55],
    palette: ["#d29d4e", "#145052", "#f0dfbc", "#b14833", "#0c2325", "#efe0bf"],
    shapes: [
      { type: "ellipse", x: 0.42, y: 0.13, width: 0.2, height: 0.16, rotation: 0.06, color: 2 },
      { type: "capsule", x: 0.67, y: 0.31, width: 0.48, height: 0.17, rotation: 0.54, color: 2 },
      { type: "ellipse", x: 0.76, y: 0.53, width: 0.48, height: 0.16, rotation: 0.38, color: 3 },
      { type: "capsule", x: 0.82, y: 0.36, width: 0.36, height: 0.08, rotation: -0.34, color: 2 },
      { type: "ring", x: 0.39, y: 0.11, width: 0.055, height: 0.045, rotation: 0, color: 4 }
    ]
  };
}

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
  uniform float u_design;
  uniform float u_edgePlay;
  uniform float u_model;
  uniform float u_orientation;
  uniform float u_globalParity;
  uniform float u_viewZoom;
  uniform vec2 u_viewPan;
  uniform vec2 u_a;
  uniform vec2 u_b;
  uniform vec2 u_c;
  uniform vec2 u_d;
  uniform sampler2D u_harmonicMap;
  uniform vec2 u_harmonicBounds;
  uniform float u_harmonicMix;
  uniform vec3 u_customColors[6];
  uniform vec3 u_customEdges;
  uniform vec4 u_customGeometry[12];
  uniform vec4 u_customStyle[12];

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

  float rotatedEllipse(vec2 p, vec2 center, vec2 radii, float rotation) {
    vec2 q = p - center;
    float cs = cos(rotation);
    float sn = sin(rotation);
    q = vec2(cs * q.x + sn * q.y, -sn * q.x + cs * q.y);
    return length(q / radii) - 1.0;
  }

  float capsule(vec2 p, vec2 start, vec2 end, float radius) {
    vec2 segment = end - start;
    float t = clamp(dot(p - start, segment) / max(dot(segment, segment), 1.0e-8), 0.0, 1.0);
    return length(p - start - segment * t) - radius;
  }

  vec3 customPalette(float index) {
    if (index < 0.5) return u_customColors[0];
    if (index < 1.5) return u_customColors[1];
    if (index < 2.5) return u_customColors[2];
    if (index < 3.5) return u_customColors[3];
    if (index < 4.5) return u_customColors[4];
    return u_customColors[5];
  }

  float customPrimitiveDistance(vec2 p, vec4 geometry, vec4 style) {
    vec2 local = p - geometry.xy;
    float cs = cos(style.y);
    float sn = sin(style.y);
    local = vec2(cs * local.x + sn * local.y, -sn * local.x + cs * local.y);
    vec2 size = max(geometry.zw, vec2(0.002));

    if (style.x < 0.5) {
      return length(local / (0.5 * size)) - 1.0;
    }
    if (style.x < 1.5) {
      float halfSegment = max(0.0, 0.5 * (size.x - size.y));
      return capsule(local, vec2(-halfSegment, 0.0), vec2(halfSegment, 0.0), 0.5 * size.y);
    }
    return abs(length(local / (0.5 * size)) - 1.0) - 0.16;
  }

  void main() {
    vec2 screenPixel = gl_FragCoord.xy;
    vec2 pixel = 0.5 * u_resolution +
      (screenPixel - 0.5 * u_resolution - u_viewPan) / u_viewZoom;
    float radius = min(u_resolution.x, u_resolution.y) * 0.455;
    vec2 displayPoint = (pixel - 0.5 * u_resolution) / radius;

    // This continuous family of Mobius maps fixes displayPoint = -i. Its
    // circular boundary grows upward from the unit disk and becomes the
    // horizontal line Im(z) = -1 when u_model reaches one.
    vec2 modelDenominator =
      cMul(vec2(0.0, -u_model), displayPoint) + vec2(1.0 + u_model, 0.0);
    vec2 local = cDiv(displayPoint, modelDenominator);
    float frameDistance = 1.0 - length(local);

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
    float radialFraction = length(point) / max(radialEdge, 0.0001);
    float angularFraction = theta / angle;
    vec2 radialTriangle = vec2(radialFraction, radialFraction * angularFraction);
    vec2 mapUv = clamp(point / u_harmonicBounds, vec2(0.0), vec2(1.0));
    vec2 harmonicTriangle = texture2D(u_harmonicMap, mapUv).rg;
    vec2 sourceTriangle = mix(radialTriangle, harmonicTriangle, u_harmonicMix);
    vec2 motif = vec2(
      sourceTriangle.y / max(sourceTriangle.x, 0.0001),
      sourceTriangle.x
    );

    // Each mirror of the fundamental triangle gets the same symmetric wave.
    // Reflection reverses the signed normal, so every edited edge is also the
    // exact partner of its neighbour: the curves cannot leave gaps or overlap.
    float insideA = point.y;
    float insideB = sinAngle * point.x - cosAngle * point.y;
    float insideC = (dot(point - sideOrigin, point - sideOrigin) - sideRadius2) * 0.35;
    float edgeParameterA = clamp(point.x / max(vertexRadius, 0.0001), 0.0, 1.0);
    float edgeParameterB = clamp(
      dot(point, vec2(cosAngle, sinAngle)) / max(vertexRadius, 0.0001), 0.0, 1.0
    );
    float edgeParameterC = clamp(theta / angle, 0.0, 1.0);

    // Every profile has a sin(pi t) envelope, so it vanishes at both mirror
    // vertices. The neighbouring reflected cell therefore inherits precisely
    // the same contour in reverse: one line is simultaneously fin, wing,
    // shoulder, or tail, with no background left between figures.
    float edgeAmplitude = u_edgePlay * 0.064;
    float waveA = edgeAmplitude * sin(PI * edgeParameterA) * (
      0.74 + 0.30 * sin(2.0 * PI * edgeParameterA) -
      0.20 * cos(4.0 * PI * edgeParameterA)
    );
    float waveB = edgeAmplitude * sin(PI * edgeParameterB) * (
      -0.50 + 0.48 * cos(2.0 * PI * edgeParameterB) +
      0.18 * sin(6.0 * PI * edgeParameterB)
    );
    float waveC = edgeAmplitude * sin(PI * edgeParameterC) * (
      0.22 + 0.64 * sin(2.0 * PI * edgeParameterC) -
      0.16 * cos(6.0 * PI * edgeParameterC)
    );

    if (u_design > 0.5 && u_design < 1.5) {
      // Long, directional S-curves: snout, back, and split koi tail.
      waveA = edgeAmplitude * sin(PI * edgeParameterA) * (
        -0.28 + 0.92 * sin(2.0 * PI * edgeParameterA)
      );
      waveB = edgeAmplitude * sin(PI * edgeParameterB) * (
        0.18 - 0.82 * sin(2.0 * PI * edgeParameterB) +
        0.18 * sin(4.0 * PI * edgeParameterB)
      );
      waveC = edgeAmplitude * sin(PI * edgeParameterC) * (
        0.66 - 0.44 * cos(2.0 * PI * edgeParameterC)
      );
    } else if (u_design > 1.5 && u_design < 2.5) {
      // Bilateral scallops read as paired velvet wings.
      waveA = edgeAmplitude * sin(PI * edgeParameterA) * (
        0.34 + 0.62 * cos(2.0 * PI * edgeParameterA)
      );
      waveB = edgeAmplitude * sin(PI * edgeParameterB) * (
        -0.34 - 0.62 * cos(2.0 * PI * edgeParameterB)
      );
      waveC = edgeAmplitude * sin(PI * edgeParameterC) * (
        0.16 + 0.82 * cos(4.0 * PI * edgeParameterC)
      );
    } else if (u_design > 2.5 && u_design < 3.5) {
      // Alternating bumps give the salamanders feet and a hooked tail.
      waveA = edgeAmplitude * sin(PI * edgeParameterA) * (
        0.28 + 0.62 * sin(4.0 * PI * edgeParameterA)
      );
      waveB = edgeAmplitude * sin(PI * edgeParameterB) * (
        -0.20 - 0.68 * sin(4.0 * PI * edgeParameterB)
      );
      waveC = edgeAmplitude * sin(PI * edgeParameterC) * (
        -0.54 + 0.50 * sin(2.0 * PI * edgeParameterC) +
        0.22 * sin(6.0 * PI * edgeParameterC)
      );
    } else if (u_design > 3.5) {
      waveA = edgeAmplitude * sin(PI * edgeParameterA) * (
        0.72 + 0.52 * u_customEdges.x * cos(2.0 * PI * edgeParameterA)
      );
      waveB = edgeAmplitude * sin(PI * edgeParameterB) * (
        -0.54 + 0.58 * u_customEdges.y * sin(2.0 * PI * edgeParameterB)
      );
      waveC = edgeAmplitude * sin(PI * edgeParameterC) * (
        0.24 + 0.66 * u_customEdges.z * cos(4.0 * PI * edgeParameterC)
      );
    }

    float paritySign = parity < 0.5 ? 1.0 : -1.0;
    float signedA = paritySign * insideA;
    float signedB = paritySign * insideB;
    float signedC = paritySign * insideC;
    float boundaryA = abs(signedA - waveA);
    float boundaryB = abs(signedB - waveB);
    float boundaryC = abs(signedC - waveC);
    float deformedEdge = boundaryA;
    float ownership = 1.0 - step(waveA, signedA);
    if (boundaryB < deformedEdge) {
      deformedEdge = boundaryB;
      ownership = 1.0 - step(waveB, signedB);
    }
    if (boundaryC < deformedEdge) {
      deformedEdge = boundaryC;
      ownership = 1.0 - step(waveC, signedC);
    }

    vec3 teal = rgb(20.0, 79.0, 82.0);
    vec3 deepTeal = rgb(10.0, 42.0, 45.0);
    vec3 ochre = rgb(210.0, 157.0, 78.0);
    vec3 coral = rgb(177.0, 72.0, 51.0);
    vec3 parchment = rgb(239.0, 224.0, 191.0);
    vec3 ink = rgb(12.0, 35.0, 37.0);

    if (u_design > 0.5 && u_design < 1.5) {
      teal = rgb(17.0, 91.0, 103.0);
      deepTeal = rgb(7.0, 48.0, 60.0);
      ochre = rgb(224.0, 119.0, 72.0);
      coral = rgb(244.0, 188.0, 96.0);
      parchment = rgb(238.0, 228.0, 199.0);
      ink = rgb(7.0, 34.0, 43.0);
    } else if (u_design > 1.5 && u_design < 2.5) {
      teal = rgb(61.0, 47.0, 89.0);
      deepTeal = rgb(26.0, 24.0, 50.0);
      ochre = rgb(184.0, 137.0, 68.0);
      coral = rgb(190.0, 80.0, 92.0);
      parchment = rgb(236.0, 218.0, 171.0);
      ink = rgb(23.0, 20.0, 38.0);
    } else if (u_design > 2.5) {
      teal = rgb(31.0, 96.0, 72.0);
      deepTeal = rgb(13.0, 51.0, 42.0);
      ochre = rgb(198.0, 133.0, 61.0);
      coral = rgb(172.0, 67.0, 46.0);
      parchment = rgb(234.0, 220.0, 174.0);
      ink = rgb(15.0, 41.0, 34.0);
    }

    float celestialType = 1.0 - ownership;
    vec3 base = mix(ochre, teal, celestialType);
    vec3 creature = mix(deepTeal, parchment, celestialType);
    vec3 highlight = mix(coral, ochre, celestialType);

    float celestialHead = 1.0 - smoothstep(-0.04, 0.08,
      ellipse(motif, vec2(0.25, 0.31), vec2(0.105, 0.095)));
    float celestialBody = 1.0 - smoothstep(-0.04, 0.08,
      rotatedEllipse(motif, vec2(0.40, 0.55), vec2(0.145, 0.29), -0.12));
    float celestialRobe = 1.0 - smoothstep(-0.04, 0.08,
      rotatedEllipse(motif, vec2(0.43, 0.73), vec2(0.22, 0.22), 0.10));
    float celestialWingA = 1.0 - smoothstep(-0.04, 0.08,
      rotatedEllipse(motif, vec2(0.67, 0.48), vec2(0.34, 0.115), -0.28));
    float celestialWingB = 1.0 - smoothstep(-0.04, 0.08,
      rotatedEllipse(motif, vec2(0.67, 0.65), vec2(0.31, 0.105), 0.18));
    float celestialMask = max(max(celestialHead, celestialBody),
      max(celestialRobe, max(celestialWingA, celestialWingB)));

    float demonHead = 1.0 - smoothstep(-0.04, 0.08,
      ellipse(motif, vec2(0.29, 0.36), vec2(0.14, 0.125)));
    float demonBody = 1.0 - smoothstep(-0.04, 0.08,
      rotatedEllipse(motif, vec2(0.49, 0.58), vec2(0.24, 0.25), -0.08));
    float demonWingA = 1.0 - smoothstep(-0.04, 0.08,
      rotatedEllipse(motif, vec2(0.70, 0.47), vec2(0.31, 0.13), 0.32));
    float demonWingB = 1.0 - smoothstep(-0.04, 0.08,
      rotatedEllipse(motif, vec2(0.70, 0.69), vec2(0.30, 0.12), -0.24));
    float demonHornA = 1.0 - smoothstep(-0.010, 0.018,
      capsule(motif, vec2(0.23, 0.29), vec2(0.12, 0.13), 0.032));
    float demonHornB = 1.0 - smoothstep(-0.010, 0.018,
      capsule(motif, vec2(0.33, 0.29), vec2(0.41, 0.13), 0.030));
    float demonTailA = 1.0 - smoothstep(-0.010, 0.018,
      capsule(motif, vec2(0.54, 0.72), vec2(0.76, 0.82), 0.025));
    float demonTailB = 1.0 - smoothstep(-0.010, 0.018,
      capsule(motif, vec2(0.76, 0.82), vec2(0.87, 0.72), 0.020));
    float demonMask = max(max(demonHead, demonBody), max(max(demonWingA, demonWingB),
      max(max(demonHornA, demonHornB), max(demonTailA, demonTailB))));

    float koiHead = 1.0 - smoothstep(-0.04, 0.08,
      ellipse(motif, vec2(0.24, 0.43), vec2(0.15, 0.13)));
    float koiBody = 1.0 - smoothstep(-0.04, 0.08,
      rotatedEllipse(motif, vec2(0.49, 0.55), vec2(0.34, 0.16), 0.16));
    float koiTailA = 1.0 - smoothstep(-0.03, 0.07,
      rotatedEllipse(motif, vec2(0.78, 0.48), vec2(0.23, 0.09), 0.55));
    float koiTailB = 1.0 - smoothstep(-0.03, 0.07,
      rotatedEllipse(motif, vec2(0.80, 0.66), vec2(0.23, 0.09), -0.48));
    float koiFin = 1.0 - smoothstep(-0.03, 0.07,
      rotatedEllipse(motif, vec2(0.50, 0.69), vec2(0.18, 0.07), -0.35));
    float koiMask = max(max(koiHead, koiBody), max(koiFin, max(koiTailA, koiTailB)));

    float mothBody = 1.0 - smoothstep(-0.025, 0.05,
      capsule(motif, vec2(0.34, 0.27), vec2(0.48, 0.75), 0.065));
    float mothWingA = 1.0 - smoothstep(-0.04, 0.08,
      rotatedEllipse(motif, vec2(0.61, 0.43), vec2(0.31, 0.16), -0.43));
    float mothWingB = 1.0 - smoothstep(-0.04, 0.08,
      rotatedEllipse(motif, vec2(0.66, 0.67), vec2(0.29, 0.14), 0.36));
    float mothHead = 1.0 - smoothstep(-0.03, 0.06,
      ellipse(motif, vec2(0.31, 0.23), vec2(0.09, 0.08)));
    float mothAntennaA = 1.0 - smoothstep(-0.008, 0.018,
      capsule(motif, vec2(0.28, 0.18), vec2(0.18, 0.08), 0.018));
    float mothAntennaB = 1.0 - smoothstep(-0.008, 0.018,
      capsule(motif, vec2(0.34, 0.18), vec2(0.42, 0.07), 0.018));
    float mothMask = max(max(mothBody, mothHead),
      max(max(mothWingA, mothWingB), max(mothAntennaA, mothAntennaB)));

    float salamanderHead = 1.0 - smoothstep(-0.035, 0.07,
      ellipse(motif, vec2(0.23, 0.34), vec2(0.14, 0.12)));
    float salamanderBody = 1.0 - smoothstep(-0.04, 0.08,
      rotatedEllipse(motif, vec2(0.47, 0.55), vec2(0.30, 0.13), 0.38));
    float salamanderTailA = 1.0 - smoothstep(-0.015, 0.035,
      capsule(motif, vec2(0.64, 0.67), vec2(0.82, 0.80), 0.055));
    float salamanderTailB = 1.0 - smoothstep(-0.012, 0.03,
      capsule(motif, vec2(0.82, 0.80), vec2(0.92, 0.72), 0.035));
    float salamanderLegA = 1.0 - smoothstep(-0.01, 0.025,
      capsule(motif, vec2(0.38, 0.49), vec2(0.22, 0.61), 0.032));
    float salamanderLegB = 1.0 - smoothstep(-0.01, 0.025,
      capsule(motif, vec2(0.52, 0.61), vec2(0.39, 0.78), 0.032));
    float salamanderMask = max(max(salamanderHead, salamanderBody),
      max(max(salamanderTailA, salamanderTailB), max(salamanderLegA, salamanderLegB)));

    float creatureMask = mix(demonMask, celestialMask, celestialType);
    if (u_design > 0.5 && u_design < 1.5) {
      creatureMask = koiMask;
    } else if (u_design > 1.5 && u_design < 2.5) {
      creatureMask = mothMask;
    } else if (u_design > 2.5) {
      creatureMask = salamanderMask;
    }
    // The whole ownership region is the creature. Broad motif masks now act as
    // anatomy painted inside that body, rather than as a figure floating on a
    // separate background.
    vec3 color = mix(creature, base, 0.18 + 0.42 * creatureMask);

    float celestialFeatherA = abs(motif.y - (0.43 + 0.17 * motif.x + 0.025 * sin(13.0 * motif.x))) - 0.010;
    float celestialFeatherB = abs(motif.y - (0.54 + 0.13 * motif.x + 0.020 * sin(11.0 * motif.x))) - 0.010;
    float celestialFeatherC = abs(motif.y - (0.70 - 0.10 * motif.x + 0.018 * sin(12.0 * motif.x))) - 0.010;
    float demonRibA = abs(motif.y - (0.40 + 0.23 * motif.x)) - 0.010;
    float demonRibB = abs(motif.y - (0.74 - 0.17 * motif.x)) - 0.010;
    float demonRibC = abs(motif.x - (0.56 + 0.08 * sin(8.0 * motif.y))) - 0.012;
    float celestialDetail = min(celestialFeatherA, min(celestialFeatherB, celestialFeatherC));
    float demonDetail = min(demonRibA, min(demonRibB, demonRibC));
    float detailDistance = mix(demonDetail, celestialDetail, celestialType);
    if (u_design > 0.5 && u_design < 1.5) {
      float koiSpine = abs(motif.y - (0.48 + 0.15 * motif.x + 0.025 * sin(12.0 * motif.x))) - 0.010;
      float koiStripeA = abs(motif.x - 0.41) - 0.013;
      float koiStripeB = abs(motif.x - 0.58) - 0.013;
      detailDistance = min(koiSpine, min(koiStripeA, koiStripeB));
    } else if (u_design > 1.5 && u_design < 2.5) {
      float mothVeinA = abs(motif.y - (0.31 + 0.35 * motif.x)) - 0.010;
      float mothVeinB = abs(motif.y - (0.76 - 0.18 * motif.x)) - 0.010;
      float mothVeinC = abs(motif.x - (0.55 + 0.04 * sin(11.0 * motif.y))) - 0.010;
      detailDistance = min(mothVeinA, min(mothVeinB, mothVeinC));
    } else if (u_design > 2.5) {
      float salamanderSpine = abs(motif.y - (0.27 + 0.66 * motif.x - 0.18 * motif.x * motif.x)) - 0.011;
      float salamanderSpotA = abs(ellipse(motif, vec2(0.38, 0.48), vec2(0.045, 0.035)));
      float salamanderSpotB = abs(ellipse(motif, vec2(0.53, 0.59), vec2(0.040, 0.032)));
      detailDistance = min(salamanderSpine, min(salamanderSpotA, salamanderSpotB));
    }
    float detailWidth = max(fwidth(detailDistance), 0.004);
    float detailLines = (1.0 - smoothstep(0.0, detailWidth * 1.7, detailDistance)) * creatureMask;
    color = mix(color, highlight, detailLines * 0.82);

    float ornamentDistance = abs(ellipse(motif, vec2(0.25, 0.205), vec2(0.12, 0.042)));
    float ornamentPresence = celestialType;
    float eyeDistance = mix(
      min(ellipse(motif, vec2(0.255, 0.35), vec2(0.020, 0.016)),
        ellipse(motif, vec2(0.315, 0.35), vec2(0.020, 0.016))),
      ellipse(motif, vec2(0.225, 0.31), vec2(0.016, 0.014)),
      celestialType
    );
    if (u_design > 0.5 && u_design < 1.5) {
      ornamentDistance = abs(ellipse(motif, vec2(0.54, 0.54), vec2(0.115, 0.075)));
      ornamentPresence = creatureMask;
      eyeDistance = ellipse(motif, vec2(0.185, 0.40), vec2(0.018, 0.015));
    } else if (u_design > 1.5 && u_design < 2.5) {
      ornamentDistance = abs(ellipse(motif, vec2(0.65, 0.48), vec2(0.095, 0.075)));
      ornamentPresence = creatureMask;
      eyeDistance = min(
        ellipse(motif, vec2(0.285, 0.22), vec2(0.014, 0.012)),
        ellipse(motif, vec2(0.335, 0.22), vec2(0.014, 0.012))
      );
    } else if (u_design > 2.5) {
      ornamentDistance = min(
        abs(ellipse(motif, vec2(0.38, 0.48), vec2(0.045, 0.035))),
        abs(ellipse(motif, vec2(0.53, 0.59), vec2(0.040, 0.032)))
      );
      ornamentPresence = creatureMask;
      eyeDistance = min(
        ellipse(motif, vec2(0.20, 0.32), vec2(0.016, 0.014)),
        ellipse(motif, vec2(0.25, 0.35), vec2(0.016, 0.014))
      );
    }
    float ornamentWidth = max(fwidth(ornamentDistance), 0.015);
    float ornamentMask = (1.0 - smoothstep(
      0.02, 0.02 + ornamentWidth * 1.8, ornamentDistance
    )) * ornamentPresence;
    color = mix(color, ochre, ornamentMask);

    float eyeMask = (1.0 - smoothstep(-0.10, 0.20, eyeDistance)) * creatureMask;
    color = mix(color, celestialType > 0.5 ? ink : coral, eyeMask);

    if (u_design > 3.5) {
      color = mix(u_customColors[0], u_customColors[1], ownership);
      creatureMask = 0.0;
      for (int customIndex = 0; customIndex < 12; customIndex += 1) {
        vec4 geometry = u_customGeometry[customIndex];
        vec4 style = u_customStyle[customIndex];
        float shapeDistance = customPrimitiveDistance(sourceTriangle, geometry, style);
        float shapeWidth = max(fwidth(shapeDistance), 0.0025);
        float shapeMask = (1.0 - smoothstep(-shapeWidth, shapeWidth, shapeDistance)) * style.w;
        color = mix(color, customPalette(style.z), shapeMask);
        creatureMask = max(creatureMask, shapeMask);
      }
      ink = u_customColors[4];
      parchment = u_customColors[5];
    }

    float edgeWidth = max(fwidth(deformedEdge), 0.00015);
    float edgeLine = 1.0 - smoothstep(edgeWidth * 0.55, edgeWidth * 1.7, deformedEdge);

    float maskEdge = abs(creatureMask - 0.5);
    float outlineWidth = max(fwidth(maskEdge), 0.025);
    float motifOutline = 1.0 - smoothstep(0.08, 0.08 + outlineWidth * 1.7, maskEdge);
    float motifOutlineStrength = u_design > 3.5 ? 0.72 : 0.14;
    color = mix(color, ink, max(edgeLine * 0.96, motifOutline * motifOutlineStrength));

    float texture = 0.018 * sin((motif.x * 47.0 + motif.y * 31.0 + sideCrossings * 2.0) * PI);
    color += texture * (0.35 + 0.65 * creatureMask);

    float frameWidth = max(fwidth(frameDistance), 0.00001);
    float frameLine = 1.0 - smoothstep(frameWidth * 0.7, frameWidth * 2.2, frameDistance);
    color = mix(color, parchment, frameLine * 0.78);

    float vignette = smoothstep(1.18, 0.18, length((screenPixel - 0.5 * u_resolution) / u_resolution.y));
    color *= 0.82 + 0.18 * vignette;
    gl_FragColor = vec4(color, 1.0);
  }
`;

const state = {
  model: "disk",
  modelMix: 0,
  modelAnimation: null,
  design: 0,
  edgePlay: 0.58,
  p: 6,
  q: 4,
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
  customDesign: starterCustomDesign(),
  harmonicFit: 1,
  harmonicTexture: null,
  harmonicBounds: { x: 1, y: 1 },
  selectedShape: 0,
  editorDrag: null,
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

function harmonicMapData(size = 128) {
  const geometry = tilingGeometry();
  const { angle, sideCenter, sideRadius2 } = geometry;
  const sinAngle = Math.sin(angle);
  const cosAngle = Math.cos(angle);
  const sideRadius = Math.sqrt(sideRadius2);
  const coshRadius = (1 / Math.tan(angle)) * (1 / Math.tan(Math.PI / state.q));
  const vertexRadius = Math.sqrt((coshRadius - 1) / (coshRadius + 1));
  const mirrorVertex = {
    x: vertexRadius * cosAngle,
    y: vertexRadius * sinAngle
  };
  const axisVertex = sideCenter - sideRadius;
  const bounds = {
    x: Math.max(axisVertex, mirrorVertex.x),
    y: mirrorVertex.y
  };
  const count = size * size;
  const active = new Uint8Array(count);
  const fixed = new Uint8Array(count);
  let sourceU = new Float32Array(count);
  let sourceV = new Float32Array(count);
  let nextU = new Float32Array(count);
  let nextV = new Float32Array(count);

  const indexOf = (x, y) => y * size + x;
  const diskPoint = (x, y) => ({
    x: (x + 0.5) * bounds.x / size,
    y: (y + 0.5) * bounds.y / size
  });
  const insideTriangle = (point) => (
    point.y >= 0 &&
    sinAngle * point.x - cosAngle * point.y >= 0 &&
    (point.x - sideCenter) ** 2 + point.y ** 2 >= sideRadius2
  );
  const radialCoordinates = (point) => {
    const theta = Math.max(0, Math.min(angle, Math.atan2(point.y, point.x)));
    const root = Math.sqrt(Math.max(
      sideCenter * sideCenter * Math.cos(theta) ** 2 - 1,
      0
    ));
    const radialEdge = sideCenter * Math.cos(theta) - root;
    const radialFraction = Math.max(0, Math.min(1, Math.hypot(point.x, point.y) / radialEdge));
    const angularFraction = theta / angle;
    return [radialFraction, radialFraction * angularFraction];
  };

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const index = indexOf(x, y);
      const point = diskPoint(x, y);
      const [u, v] = radialCoordinates(point);
      sourceU[index] = u;
      sourceV[index] = v;
      active[index] = insideTriangle(point) ? 1 : 0;
    }
  }

  const neighbourOffsets = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const index = indexOf(x, y);
      if (!active[index]) continue;
      const touchesBoundary = neighbourOffsets.some(([dx, dy]) => {
        const nx = x + dx;
        const ny = y + dy;
        return nx < 0 || nx >= size || ny < 0 || ny >= size || !active[indexOf(nx, ny)];
      });
      if (!touchesBoundary) continue;

      fixed[index] = 1;
      const point = diskPoint(x, y);
      const distanceAxis = point.y;
      const distanceRay = Math.abs(sinAngle * point.x - cosAngle * point.y);
      const distanceArc = Math.abs(Math.hypot(point.x - sideCenter, point.y) - sideRadius);

      if (distanceAxis <= distanceRay && distanceAxis <= distanceArc) {
        sourceU[index] = Math.max(0, Math.min(1, point.x / axisVertex));
        sourceV[index] = 0;
      } else if (distanceRay <= distanceArc) {
        const t = Math.max(0, Math.min(1, Math.hypot(point.x, point.y) / vertexRadius));
        sourceU[index] = t;
        sourceV[index] = t;
      } else {
        sourceU[index] = 1;
        sourceV[index] = Math.max(0, Math.min(1, Math.atan2(point.y, point.x) / angle));
      }
    }
  }

  // Solve two Dirichlet problems on the folded Schwarz triangle. In two
  // dimensions the Poincare metric is conformal to the disk, so this ordinary
  // discrete Laplace solve also gives hyperbolic harmonic coordinates.
  for (let iteration = 0; iteration < 240; iteration += 1) {
    nextU.set(sourceU);
    nextV.set(sourceV);
    for (let y = 1; y < size - 1; y += 1) {
      for (let x = 1; x < size - 1; x += 1) {
        const index = indexOf(x, y);
        if (!active[index] || fixed[index]) continue;
        const left = index - 1;
        const right = index + 1;
        const down = index - size;
        const up = index + size;
        nextU[index] = 0.25 * (
          sourceU[left] + sourceU[right] + sourceU[down] + sourceU[up]
        );
        nextV[index] = 0.25 * (
          sourceV[left] + sourceV[right] + sourceV[down] + sourceV[up]
        );
      }
    }
    [sourceU, nextU] = [nextU, sourceU];
    [sourceV, nextV] = [nextV, sourceV];
  }

  const pixels = new Uint8Array(count * 4);
  for (let index = 0; index < count; index += 1) {
    pixels[index * 4] = Math.round(255 * Math.max(0, Math.min(1, sourceU[index])));
    pixels[index * 4 + 1] = Math.round(255 * Math.max(0, Math.min(1, sourceV[index])));
    pixels[index * 4 + 2] = active[index] ? 255 : 0;
    pixels[index * 4 + 3] = 255;
  }

  return { size, bounds, pixels };
}

function rebuildHarmonicMap() {
  if (!state.harmonicTexture) return;
  const map = harmonicMapData();
  state.harmonicBounds = map.bounds;
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, state.harmonicTexture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    map.size,
    map.size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    map.pixels
  );
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
    design: gl.getUniformLocation(program, "u_design"),
    edgePlay: gl.getUniformLocation(program, "u_edgePlay"),
    model: gl.getUniformLocation(program, "u_model"),
    orientation: gl.getUniformLocation(program, "u_orientation"),
    globalParity: gl.getUniformLocation(program, "u_globalParity"),
    viewZoom: gl.getUniformLocation(program, "u_viewZoom"),
    viewPan: gl.getUniformLocation(program, "u_viewPan"),
    a: gl.getUniformLocation(program, "u_a"),
    b: gl.getUniformLocation(program, "u_b"),
    c: gl.getUniformLocation(program, "u_c"),
    d: gl.getUniformLocation(program, "u_d"),
    harmonicMap: gl.getUniformLocation(program, "u_harmonicMap"),
    harmonicBounds: gl.getUniformLocation(program, "u_harmonicBounds"),
    harmonicMix: gl.getUniformLocation(program, "u_harmonicMix"),
    customColors: gl.getUniformLocation(program, "u_customColors[0]"),
    customEdges: gl.getUniformLocation(program, "u_customEdges"),
    customGeometry: gl.getUniformLocation(program, "u_customGeometry[0]"),
    customStyle: gl.getUniformLocation(program, "u_customStyle[0]")
  };

  state.harmonicTexture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, state.harmonicTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.uniform1i(state.uniforms.harmonicMap, 0);
  rebuildHarmonicMap();
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

function hexToRgbVector(hex) {
  const normalized = /^#[0-9a-f]{6}$/i.test(hex) ? hex.slice(1) : "000000";
  return [
    Number.parseInt(normalized.slice(0, 2), 16) / 255,
    Number.parseInt(normalized.slice(2, 4), 16) / 255,
    Number.parseInt(normalized.slice(4, 6), 16) / 255
  ];
}

function sendCustomDesign() {
  const colorData = state.customDesign.palette.flatMap(hexToRgbVector);
  const geometryData = new Float32Array(MAX_CUSTOM_SHAPES * 4);
  const styleData = new Float32Array(MAX_CUSTOM_SHAPES * 4);

  state.customDesign.shapes.forEach((shape, index) => {
    if (index >= MAX_CUSTOM_SHAPES) return;
    geometryData.set([shape.x, shape.y, shape.width, shape.height], index * 4);
    styleData.set([
      SHAPE_TYPES.indexOf(shape.type),
      shape.rotation,
      shape.color,
      1
    ], index * 4);
  });

  gl.uniform3fv(state.uniforms.customColors, new Float32Array(colorData));
  gl.uniform3fv(state.uniforms.customEdges, new Float32Array(state.customDesign.edges));
  gl.uniform4fv(state.uniforms.customGeometry, geometryData);
  gl.uniform4fv(state.uniforms.customStyle, styleData);
}

function draw() {
  if (!state.program) return;
  resizeCanvas();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.useProgram(state.program);
  gl.uniform2f(state.uniforms.resolution, canvas.width, canvas.height);
  gl.uniform1f(state.uniforms.p, state.p);
  gl.uniform1f(state.uniforms.q, state.q);
  gl.uniform1f(state.uniforms.design, state.design);
  gl.uniform1f(state.uniforms.edgePlay, state.edgePlay);
  gl.uniform1f(state.uniforms.model, state.modelMix);
  gl.uniform1f(state.uniforms.orientation, state.orientation);
  gl.uniform1f(state.uniforms.globalParity, state.globalParity);
  gl.uniform1f(state.uniforms.viewZoom, state.viewZoom);
  gl.uniform1f(state.uniforms.harmonicMix, state.harmonicFit);
  gl.uniform2f(
    state.uniforms.harmonicBounds,
    state.harmonicBounds.x,
    state.harmonicBounds.y
  );
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, state.harmonicTexture);
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
  sendCustomDesign();
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

function clampNumber(value, minimum, maximum, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback;
}

function sanitizeHex(value, fallback) {
  return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value) ? value.toLowerCase() : fallback;
}

function sanitizeCustomDesign(value) {
  const fallback = starterCustomDesign();
  const source = value && typeof value === "object" ? value : {};
  const palette = Array.isArray(source.palette) ? source.palette : [];
  const edges = Array.isArray(source.edges) ? source.edges : [];
  const shapes = Array.isArray(source.shapes) ? source.shapes : fallback.shapes;
  const requestedSymmetry = Array.isArray(source.symmetry) ? source.symmetry.map(Number) : fallback.symmetry;
  const supportedSymmetries = Array.from(symmetrySelect.options, (option) => option.value);
  const symmetryValue = `${requestedSymmetry[0]},${requestedSymmetry[1]}`;
  const symmetry = supportedSymmetries.includes(symmetryValue)
    ? requestedSymmetry.slice(0, 2)
    : fallback.symmetry;
  const migrateRadialChart = Array.isArray(source.shapes) && Number(source.version || 1) < 2;

  return {
    version: 2,
    name: typeof source.name === "string" && source.name.trim()
      ? source.name.trim().slice(0, 60)
      : fallback.name,
    symmetry,
    edgePlay: clampNumber(source.edgePlay, 0, 1, fallback.edgePlay),
    harmonicFit: clampNumber(source.harmonicFit, 0, 1, fallback.harmonicFit),
    edges: fallback.edges.map((edge, index) => clampNumber(edges[index], -1, 1, edge)),
    palette: fallback.palette.map((color, index) => sanitizeHex(palette[index], color)),
    shapes: shapes.slice(0, MAX_CUSTOM_SHAPES).map((shape, index) => {
      const sourceShape = shape && typeof shape === "object" ? shape : {};
      const defaultShape = fallback.shapes[index % fallback.shapes.length];
      const oldX = clampNumber(sourceShape.x, 0, 1, defaultShape.x);
      const oldY = clampNumber(sourceShape.y, 0, 1, defaultShape.y);
      const mappedX = migrateRadialChart ? oldY : oldX;
      const mappedY = migrateRadialChart ? oldY * oldX : oldY;
      const mappedWidth = migrateRadialChart
        ? clampNumber(sourceShape.width, 0.02, 1, defaultShape.width) * Math.max(oldY, 0.25)
        : clampNumber(sourceShape.width, 0.02, 1, defaultShape.width);
      return {
        type: SHAPE_TYPES.includes(sourceShape.type) ? sourceShape.type : defaultShape.type,
        x: mappedX,
        y: Math.min(mappedX, mappedY),
        width: Math.max(0.02, mappedWidth),
        height: clampNumber(sourceShape.height, 0.02, 1, defaultShape.height),
        rotation: clampNumber(sourceShape.rotation, -Math.PI, Math.PI, defaultShape.rotation),
        color: Math.round(clampNumber(sourceShape.color, 0, 5, defaultShape.color))
      };
    })
  };
}

function setStudioStatus(message, isError = false) {
  studioStatus.textContent = message;
  studioStatus.classList.toggle("is-error", isError);
}

function formatSigned(value) {
  const number = Number(value);
  if (Math.abs(number) < 0.005) return "0.00";
  return `${number > 0 ? "+" : "−"}${Math.abs(number).toFixed(2)}`;
}

function createSvgElement(name) {
  return document.createElementNS("http://www.w3.org/2000/svg", name);
}

function renderMotifEditor() {
  motifShapeLayer.replaceChildren();
  state.customDesign.shapes.forEach((shape, index) => {
    const svgY = (1 - shape.y) * 100;
    let node;
    if (shape.type === "capsule") {
      node = createSvgElement("rect");
      node.setAttribute("x", String((shape.x - shape.width / 2) * 100));
      node.setAttribute("y", String(svgY - shape.height * 50));
      node.setAttribute("width", String(shape.width * 100));
      node.setAttribute("height", String(shape.height * 100));
      node.setAttribute("rx", String(shape.height * 50));
    } else {
      node = createSvgElement("ellipse");
      node.setAttribute("cx", String(shape.x * 100));
      node.setAttribute("cy", String(svgY));
      node.setAttribute("rx", String(shape.width * 50));
      node.setAttribute("ry", String(shape.height * 50));
      if (shape.type === "ring") node.setAttribute("fill-opacity", "0.22");
    }

    node.classList.add("motif-shape");
    if (index === state.selectedShape) node.classList.add("is-selected");
    node.dataset.shapeIndex = String(index);
    node.setAttribute("fill", state.customDesign.palette[shape.color]);
    node.setAttribute(
      "transform",
      `rotate(${-shape.rotation * 180 / Math.PI} ${shape.x * 100} ${svgY})`
    );
    node.setAttribute("role", "img");
    node.setAttribute("aria-label", `${shape.type} layer ${index + 1}`);
    motifShapeLayer.append(node);
  });
}

function syncSelectedShapeControls() {
  const shapes = state.customDesign.shapes;
  const hasShape = shapes.length > 0;
  state.selectedShape = hasShape
    ? Math.max(0, Math.min(shapes.length - 1, state.selectedShape))
    : -1;

  shapeSelect.replaceChildren();
  shapes.forEach((shape, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index + 1} · ${shape.type}`;
    shapeSelect.append(option);
  });

  const controls = [
    shapeSelect,
    shapeTypeSelect,
    shapeColorSelect,
    shapeXInput,
    shapeYInput,
    shapeWidthInput,
    shapeHeightInput,
    shapeRotationInput,
    duplicateShapeButton,
    moveShapeBackButton,
    deleteShapeButton
  ];
  controls.forEach((control) => {
    control.disabled = !hasShape;
  });
  addShapeButton.disabled = shapes.length >= MAX_CUSTOM_SHAPES;
  shapeCount.textContent = `${shapes.length} / ${MAX_CUSTOM_SHAPES} shapes`;

  if (!hasShape) {
    [shapeXReadout, shapeYReadout, shapeWidthReadout, shapeHeightReadout, shapeRotationReadout]
      .forEach((output) => { output.textContent = "—"; });
    return;
  }

  const shape = shapes[state.selectedShape];
  shapeSelect.value = String(state.selectedShape);
  shapeTypeSelect.value = shape.type;
  shapeColorSelect.value = String(shape.color);
  shapeXInput.value = String(Math.round(shape.x * 100));
  shapeYInput.value = String(Math.round(shape.y * 100));
  shapeWidthInput.value = String(Math.round(shape.width * 100));
  shapeHeightInput.value = String(Math.round(shape.height * 100));
  shapeRotationInput.value = String(Math.round(shape.rotation * 180 / Math.PI));
  shapeXReadout.textContent = shape.x.toFixed(2);
  shapeYReadout.textContent = shape.y.toFixed(2);
  shapeWidthReadout.textContent = shape.width.toFixed(2);
  shapeHeightReadout.textContent = shape.height.toFixed(2);
  shapeRotationReadout.textContent = `${Math.round(shape.rotation * 180 / Math.PI)}°`;
  moveShapeBackButton.disabled = state.selectedShape <= 0;
}

function syncCustomControls() {
  designNameInput.value = state.customDesign.name;
  paletteInputs.forEach((input, index) => {
    input.value = state.customDesign.palette[index];
  });
  edgeCurveInputs.forEach((input, index) => {
    input.value = String(Math.round(state.customDesign.edges[index] * 100));
    edgeCurveReadouts[index].textContent = formatSigned(state.customDesign.edges[index]);
  });
  harmonicFitInput.value = String(Math.round(state.customDesign.harmonicFit * 100));
  harmonicFitReadout.textContent = `${harmonicFitInput.value}%`;
  syncSelectedShapeControls();
  renderMotifEditor();
}

function applyCustomViewSettings() {
  const [p, q] = state.customDesign.symmetry;
  state.p = p;
  state.q = q;
  symmetrySelect.value = `${p},${q}`;
  state.edgePlay = state.customDesign.edgePlay;
  state.harmonicFit = state.customDesign.harmonicFit;
  edgePlayInput.value = String(Math.round(state.edgePlay * 100));
  edgeReadout.textContent = `${edgePlayInput.value}%`;
  rebuildHarmonicMap();
}

function activateCustomDesign({ openStudio = true, resetView = false } = {}) {
  state.design = 4;
  designSelect.value = "4";
  if (openStudio) designStudio.open = true;
  applyCustomViewSettings();
  if (resetView) resetCamera();
  else {
    updateStatus();
    draw();
  }
}

function setCustomDesign(value, { activate = true, message = "Design loaded." } = {}) {
  state.customDesign = sanitizeCustomDesign(value);
  state.selectedShape = state.customDesign.shapes.length ? 0 : -1;
  syncCustomControls();
  if (activate) activateCustomDesign({ openStudio: true, resetView: true });
  setStudioStatus(message);
}

function markCustomDesignChanged(message = "Unsaved changes · preview updated.") {
  state.design = 4;
  designSelect.value = "4";
  setStudioStatus(message);
  draw();
}

function updateSelectedShapeFromControls() {
  const shape = state.customDesign.shapes[state.selectedShape];
  if (!shape) return;
  shape.type = SHAPE_TYPES.includes(shapeTypeSelect.value) ? shapeTypeSelect.value : "ellipse";
  shape.color = Number(shapeColorSelect.value);
  shape.x = Math.max(0, Math.min(1, Number(shapeXInput.value) / 100));
  shape.y = Math.max(0, Math.min(shape.x, Number(shapeYInput.value) / 100));
  shape.width = Number(shapeWidthInput.value) / 100;
  shape.height = Number(shapeHeightInput.value) / 100;
  shape.rotation = Number(shapeRotationInput.value) * Math.PI / 180;
  syncSelectedShapeControls();
  renderMotifEditor();
  markCustomDesignChanged();
}

function editorPoint(event) {
  const rect = motifEditor.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    y: Math.max(0, Math.min(1, 1 - (event.clientY - rect.top) / rect.height))
  };
}

function encodeDesign(design) {
  const bytes = new TextEncoder().encode(JSON.stringify(design));
  let binary = "";
  bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/u, "");
}

function decodeDesign(encoded) {
  const base64 = encoded.replaceAll("-", "+").replaceAll("_", "/");
  const padded = base64 + "=".repeat((4 - base64.length % 4) % 4);
  const binary = atob(padded);
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  return JSON.parse(new TextDecoder().decode(bytes));
}

function sharedDesignFromHash() {
  const encoded = new URLSearchParams(window.location.hash.slice(1)).get("design");
  return encoded ? decodeDesign(encoded) : null;
}

function designFileName() {
  const slug = state.customDesign.name
    .toLowerCase()
    .replace(/[^a-z0-9]+/gu, "-")
    .replace(/^-|-$/gu, "")
    .slice(0, 48);
  return `${slug || "hyperbolic-design"}.json`;
}

function downloadDesign() {
  const blob = new Blob([`${JSON.stringify(state.customDesign, null, 2)}\n`], {
    type: "application/json"
  });
  const link = document.createElement("a");
  const objectUrl = URL.createObjectURL(blob);
  link.href = objectUrl;
  link.download = designFileName();
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
  setStudioStatus(`Exported ${link.download}.`);
}

async function copyShareLink() {
  const url = new URL(window.location.href);
  url.hash = `design=${encodeDesign(state.customDesign)}`;
  window.history.replaceState(null, "", url);
  setStudioStatus("Share link ready in the address bar; copying it now…");
  try {
    const clipboardCopy = navigator.clipboard?.writeText
      ? navigator.clipboard.writeText(url.toString())
      : Promise.reject(new Error("Clipboard API unavailable"));
    await Promise.race([
      clipboardCopy,
      new Promise((_, reject) => {
        window.setTimeout(() => reject(new Error("Clipboard write timed out")), 900);
      })
    ]);
    setStudioStatus("Share link copied. It contains the complete vector design.");
  } catch (error) {
    const fallback = document.createElement("textarea");
    fallback.value = url.toString();
    fallback.setAttribute("readonly", "");
    fallback.style.position = "fixed";
    fallback.style.opacity = "0";
    document.body.append(fallback);
    fallback.select();
    const copied = document.execCommand("copy");
    fallback.remove();
    setStudioStatus(
      copied
        ? "Share link copied."
        : "Share link is ready in the address bar; copy it from there."
    );
  }
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
  const baseRadius = Math.min(point.width, point.height) * 0.455;
  const fixedBoundaryY = point.height / 2 + state.viewPan.y +
    baseRadius * state.viewZoom;

  if (state.modelMix < 0.999) {
    const inverseGap = 1 / (1 - state.modelMix);
    const radius = baseRadius * state.viewZoom * inverseGap;
    const center = {
      x: point.width / 2 + state.viewPan.x,
      y: fixedBoundaryY - radius
    };
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

  if (Math.abs(point.y - fixedBoundaryY) <= 36) {
    return { ...point, y: fixedBoundaryY };
  }
  return point;
}

function animateModel(model) {
  const target = model === "disk" ? 0 : 1;
  const start = state.modelMix;
  state.model = model;

  if (state.modelAnimation !== null) {
    cancelAnimationFrame(state.modelAnimation);
    state.modelAnimation = null;
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const distance = Math.abs(target - start);
  if (reducedMotion || distance < 0.0001) {
    state.modelMix = target;
    draw();
    return;
  }

  const startedAt = performance.now();
  const duration = Math.max(280, 1050 * distance);
  const step = (now) => {
    const progress = Math.min(1, (now - startedAt) / duration);
    const eased = progress * progress * (3 - 2 * progress);
    state.modelMix = start + (target - start) * eased;
    draw();

    if (progress < 1) {
      state.modelAnimation = requestAnimationFrame(step);
    } else {
      state.modelMix = target;
      state.modelAnimation = null;
    }
  };
  state.modelAnimation = requestAnimationFrame(step);
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
    modelButtons.forEach((candidate) => {
      const active = candidate === button;
      candidate.classList.toggle("is-active", active);
      candidate.setAttribute("aria-pressed", String(active));
    });
    animateModel(button.dataset.model);
    updateStatus();
  });
});

symmetrySelect.addEventListener("change", () => {
  const [p, q] = symmetrySelect.value.split(",").map(Number);
  state.p = p;
  state.q = q;
  if (state.design === 4) {
    state.customDesign.symmetry = [p, q];
    setStudioStatus("Unsaved changes · symmetry updated.");
  }
  rebuildHarmonicMap();
  resetCamera();
});

designSelect.addEventListener("change", () => {
  state.design = Number(designSelect.value);
  if (state.design === 4) {
    activateCustomDesign({ openStudio: true, resetView: true });
  } else {
    [state.p, state.q] = BUILT_IN_SYMMETRIES[state.design];
    symmetrySelect.value = `${state.p},${state.q}`;
    rebuildHarmonicMap();
    resetCamera();
  }
});

edgePlayInput.addEventListener("input", () => {
  state.edgePlay = Number(edgePlayInput.value) / 100;
  edgeReadout.textContent = `${edgePlayInput.value}%`;
  if (state.design === 4) {
    state.customDesign.edgePlay = state.edgePlay;
    setStudioStatus("Unsaved changes · edge depth updated.");
  }
  draw();
});

designStudio.addEventListener("toggle", () => {
  if (designStudio.open && state.design !== 4) {
    activateCustomDesign({ openStudio: false });
  }
});

designForm.addEventListener("submit", (event) => event.preventDefault());

designNameInput.addEventListener("input", () => {
  state.customDesign.name = designNameInput.value.slice(0, 60) || "Untitled design";
  markCustomDesignChanged();
});

paletteInputs.forEach((input, index) => {
  input.addEventListener("input", () => {
    state.customDesign.palette[index] = input.value;
    renderMotifEditor();
    markCustomDesignChanged();
  });
});

edgeCurveInputs.forEach((input, index) => {
  input.addEventListener("input", () => {
    state.customDesign.edges[index] = Number(input.value) / 100;
    edgeCurveReadouts[index].textContent = formatSigned(state.customDesign.edges[index]);
    markCustomDesignChanged();
  });
});

harmonicFitInput.addEventListener("input", () => {
  state.harmonicFit = Number(harmonicFitInput.value) / 100;
  state.customDesign.harmonicFit = state.harmonicFit;
  harmonicFitReadout.textContent = `${harmonicFitInput.value}%`;
  markCustomDesignChanged("Unsaved changes · curvature fit updated.");
});

shapeSelect.addEventListener("change", () => {
  state.selectedShape = Number(shapeSelect.value);
  syncSelectedShapeControls();
  renderMotifEditor();
});

[shapeTypeSelect, shapeColorSelect].forEach((input) => {
  input.addEventListener("change", updateSelectedShapeFromControls);
});

[shapeXInput, shapeYInput, shapeWidthInput, shapeHeightInput, shapeRotationInput]
  .forEach((input) => input.addEventListener("input", updateSelectedShapeFromControls));

addShapeButton.addEventListener("click", () => {
  if (state.customDesign.shapes.length >= MAX_CUSTOM_SHAPES) return;
  state.customDesign.shapes.push({
    type: "ellipse",
    x: 0.62,
    y: 0.28,
    width: 0.26,
    height: 0.16,
    rotation: 0,
    color: 2
  });
  state.selectedShape = state.customDesign.shapes.length - 1;
  syncCustomControls();
  markCustomDesignChanged("Shape added · drag it in the motif editor.");
});

duplicateShapeButton.addEventListener("click", () => {
  const shape = state.customDesign.shapes[state.selectedShape];
  if (!shape || state.customDesign.shapes.length >= MAX_CUSTOM_SHAPES) return;
  state.customDesign.shapes.push({
    ...shape,
    x: Math.min(1, shape.x + 0.05),
    y: Math.min(Math.min(1, shape.x + 0.05), shape.y + 0.03)
  });
  state.selectedShape = state.customDesign.shapes.length - 1;
  syncCustomControls();
  markCustomDesignChanged("Shape duplicated.");
});

moveShapeBackButton.addEventListener("click", () => {
  if (state.selectedShape <= 0) return;
  const shapes = state.customDesign.shapes;
  [shapes[state.selectedShape - 1], shapes[state.selectedShape]] =
    [shapes[state.selectedShape], shapes[state.selectedShape - 1]];
  state.selectedShape -= 1;
  syncCustomControls();
  markCustomDesignChanged("Layer moved behind the previous shape.");
});

deleteShapeButton.addEventListener("click", () => {
  if (state.selectedShape < 0) return;
  state.customDesign.shapes.splice(state.selectedShape, 1);
  state.selectedShape = Math.min(state.selectedShape, state.customDesign.shapes.length - 1);
  syncCustomControls();
  markCustomDesignChanged("Shape deleted.");
});

motifEditor.addEventListener("pointerdown", (event) => {
  const target = event.target.closest(".motif-shape");
  if (!target) return;
  const index = Number(target.dataset.shapeIndex);
  const shape = state.customDesign.shapes[index];
  if (!shape) return;
  const point = editorPoint(event);
  state.selectedShape = index;
  state.editorDrag = {
    id: event.pointerId,
    index,
    offsetX: point.x - shape.x,
    offsetY: point.y - shape.y
  };
  motifEditor.setPointerCapture(event.pointerId);
  syncSelectedShapeControls();
  renderMotifEditor();
});

motifEditor.addEventListener("pointermove", (event) => {
  if (!state.editorDrag || state.editorDrag.id !== event.pointerId) return;
  const shape = state.customDesign.shapes[state.editorDrag.index];
  if (!shape) return;
  const point = editorPoint(event);
  shape.x = Math.max(0, Math.min(1, point.x - state.editorDrag.offsetX));
  shape.y = Math.max(0, Math.min(shape.x, point.y - state.editorDrag.offsetY));
  shapeXInput.value = String(Math.round(shape.x * 100));
  shapeYInput.value = String(Math.round(shape.y * 100));
  shapeXReadout.textContent = shape.x.toFixed(2);
  shapeYReadout.textContent = shape.y.toFixed(2);
  renderMotifEditor();
  markCustomDesignChanged("Unsaved changes · shape moved.");
});

function finishEditorDrag(event) {
  if (!state.editorDrag || state.editorDrag.id !== event.pointerId) return;
  state.editorDrag = null;
}

motifEditor.addEventListener("pointerup", finishEditorDrag);
motifEditor.addEventListener("pointercancel", finishEditorDrag);

saveDesignButton.addEventListener("click", () => {
  try {
    localStorage.setItem(CUSTOM_DESIGN_STORAGE_KEY, JSON.stringify(state.customDesign));
    setStudioStatus("Saved in this browser. Select “Your design” whenever you return.");
  } catch (error) {
    setStudioStatus("This browser would not allow local saving. Export the JSON instead.", true);
    console.error(error);
  }
});

shareDesignButton.addEventListener("click", copyShareLink);
exportDesignButton.addEventListener("click", downloadDesign);

importDesignInput.addEventListener("change", async () => {
  const [file] = importDesignInput.files;
  if (!file) return;
  try {
    const imported = JSON.parse(await file.text());
    setCustomDesign(imported, { message: `Imported ${file.name}.` });
  } catch (error) {
    setStudioStatus("That file is not a valid Hyperbolic Escher design.", true);
    console.error(error);
  } finally {
    importDesignInput.value = "";
  }
});

resetDesignButton.addEventListener("click", () => {
  setCustomDesign(starterCustomDesign(), { message: "Started a fresh design." });
});

zoomInButton.addEventListener("click", () => zoomViewAt(1.8));
zoomOutButton.addEventListener("click", () => zoomViewAt(1 / 1.8));
resetButton.addEventListener("click", resetCamera);

const resizeObserver = new ResizeObserver(draw);
resizeObserver.observe(canvas);
window.addEventListener("orientationchange", draw);

let sharedDesignLoaded = false;
try {
  const sharedDesign = sharedDesignFromHash();
  if (sharedDesign) {
    state.customDesign = sanitizeCustomDesign(sharedDesign);
    state.selectedShape = state.customDesign.shapes.length ? 0 : -1;
    state.design = 4;
    designSelect.value = "4";
    designStudio.open = true;
    applyCustomViewSettings();
    sharedDesignLoaded = true;
  } else {
    const savedDesign = localStorage.getItem(CUSTOM_DESIGN_STORAGE_KEY);
    if (savedDesign) state.customDesign = sanitizeCustomDesign(JSON.parse(savedDesign));
  }
} catch (error) {
  setStudioStatus("The saved or shared design could not be read; the starter design is ready instead.", true);
  console.error(error);
}

syncCustomControls();
if (sharedDesignLoaded) setStudioStatus("Shared design loaded from this link.");

try {
  initializeWebGL();
  updateStatus();
  draw();
} catch (error) {
  errorBox.hidden = false;
  console.error(error);
}
