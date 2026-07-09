const state = {
  manifest: null,
  surface: null,
  form: null,
  formData: null,
  geodesicData: null,
  traceData: null,
  formError: "",
  geodesicError: "",
  selectedGeodesic: null,
  mode: "form"
};

const el = {
  sourceStatus: document.querySelector("#sourceStatus"),
  surfaceSelect: document.querySelector("#surfaceSelect"),
  formSelect: document.querySelector("#formSelect"),
  formViewSelect: document.querySelector("#formViewSelect"),
  geoViewSelect: document.querySelector("#geoViewSelect"),
  geoLimit: document.querySelector("#geoLimit"),
  formPanel: document.querySelector("#formPanel"),
  geoPanel: document.querySelector("#geoPanel"),
  tracePanel: document.querySelector("#tracePanel"),
  tabForm: document.querySelector("#tabForm"),
  tabGeo: document.querySelector("#tabGeo"),
  tabTrace: document.querySelector("#tabTrace"),
  formDetails: document.querySelector("#formDetails"),
  geoSummary: document.querySelector("#geoSummary"),
  traceDetails: document.querySelector("#traceDetails"),
  viewerTitle: document.querySelector("#viewerTitle"),
  viewerSubtitle: document.querySelector("#viewerSubtitle"),
  mainImage: document.querySelector("#mainImage"),
  geoOverlay: document.querySelector("#geoOverlay"),
  emptyState: document.querySelector("#emptyState"),
  recordList: document.querySelector("#recordList"),
  recordCount: document.querySelector("#recordCount"),
  tableTitle: document.querySelector("#tableTitle"),
  selectedJson: document.querySelector("#selectedJson"),
  resetSelection: document.querySelector("#resetSelection")
};

function fmt(value, digits = 6) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "") : "";
  return String(value);
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

async function fetchJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Could not load ${path}`);
  return res.json();
}

function setMode(mode) {
  state.mode = mode;
  el.formPanel.classList.toggle("is-hidden", mode !== "form");
  el.geoPanel.classList.toggle("is-hidden", mode !== "geo");
  el.tracePanel.classList.toggle("is-hidden", mode !== "trace");
  el.tabForm.classList.toggle("is-active", mode === "form");
  el.tabGeo.classList.toggle("is-active", mode === "geo");
  el.tabTrace.classList.toggle("is-active", mode === "trace");
  render();
}

function populateSurfaces() {
  el.surfaceSelect.innerHTML = "";
  for (const surface of state.manifest.surfaces) {
    const option = document.createElement("option");
    option.value = surface.id;
    option.textContent = surface.name;
    el.surfaceSelect.append(option);
  }
  state.surface = state.manifest.surfaces[0];
}

function populateForms() {
  el.formSelect.innerHTML = "";
  for (const form of state.surface.forms || []) {
    const option = document.createElement("option");
    option.value = form.label;
    option.textContent = form.label;
    el.formSelect.append(option);
  }
  state.form = state.surface.forms?.[0] || null;
}

async function loadSurface(surfaceId) {
  state.surface = state.manifest.surfaces.find((surface) => surface.id === surfaceId);
  state.formData = null;
  state.geodesicData = null;
  state.formError = "";
  state.geodesicError = "";
  state.selectedGeodesic = null;
  populateForms();
  await Promise.all([loadCurrentForm(), loadGeodesics()]);
  render();
}

async function loadCurrentForm() {
  if (!state.form) return;
  try {
    state.formData = await fetchJson(state.form.json);
    state.formError = "";
  } catch (error) {
    state.formData = null;
    state.formError = error.message;
  }
}

async function loadGeodesics() {
  if (!state.surface?.geodesics?.json) return;
  try {
    state.geodesicData = await fetchJson(state.surface.geodesics.json);
    state.geodesicError = "";
  } catch (error) {
    state.geodesicData = null;
    state.geodesicError = error.message;
  }
}

async function loadTraceData() {
  const path = state.manifest?.traceFormula?.sample;
  if (!path) return;
  try {
    state.traceData = await fetchJson(path);
  } catch {
    state.traceData = null;
  }
}

function selectedFormImage() {
  const view = el.formViewSelect.value;
  return state.form?.images?.[view] || "";
}

function selectedGeodesicImage() {
  const view = el.geoViewSelect.value;
  return state.surface?.geodesics?.images?.[view] || "";
}

function showImage(src, alt) {
  el.geoOverlay.innerHTML = "";
  if (!src) {
    el.mainImage.removeAttribute("src");
    el.mainImage.alt = "";
    el.mainImage.classList.add("is-hidden");
    el.emptyState.classList.remove("is-hidden");
    el.emptyState.textContent = "No rendered image is available for this selection yet. The JSON record is still available below.";
    return;
  }
  el.mainImage.src = src;
  el.mainImage.alt = alt;
  el.mainImage.classList.remove("is-hidden");
  el.emptyState.classList.add("is-hidden");
}

function renderFormDetails() {
  if (!state.formData) {
    setDetails(el.formDetails, [
      ["status", state.formError ? "data missing" : "no form selected"],
      ["source", state.form?.json || ""],
      ["note", state.formError || ""]
    ]);
    return;
  }
  const r = state.formData.spectral_parameter;
  setDetails(el.formDetails, [
    ["label", state.formData.label],
    ["surface", state.formData.surface_id],
    ["r", fmt(r, 9)],
    ["lambda", fmt(state.formData.laplace_eigenvalue ?? (0.25 + r * r), 9)],
    ["parity", state.formData.parity],
    ["Fricke", fmt(state.formData.fricke_sign, 0)],
    ["coefficients", fmt(state.formData.coefficients?.length, 0)]
  ]);
}

function renderFormList() {
  el.tableTitle.textContent = "LMFDB forms";
  el.recordList.innerHTML = "";
  const forms = state.surface.forms || [];
  el.recordCount.textContent = `${forms.length} records`;
  for (const form of forms) {
    const row = document.createElement("button");
    row.className = "record-row";
    row.type = "button";
    if (state.form?.label === form.label) row.classList.add("is-selected");
    row.innerHTML = `<strong>${form.label}</strong><span>${form.json}</span><span>${Object.keys(form.images || {}).length} images</span>`;
    row.addEventListener("click", async () => {
      state.form = form;
      el.formSelect.value = form.label;
      await loadCurrentForm();
      render();
    });
    el.recordList.append(row);
  }
}

function renderGeodesicSummary() {
  const geodesics = state.geodesicData?.geodesics || [];
  const limit = Math.min(Number(el.geoLimit.value), geodesics.length);
  setDetails(el.geoSummary, [
    ["surface", state.surface.id],
    ["available", fmt(geodesics.length, 0)],
    ["shown", fmt(limit, 0)],
    ["method", state.geodesicData?.enumeration?.method || state.geodesicError || ""]
  ]);
}

function renderGeodesicList() {
  const geodesics = state.geodesicData?.geodesics || [];
  const limit = Math.min(Number(el.geoLimit.value), geodesics.length);
  el.tableTitle.textContent = "Closed geodesics";
  el.recordCount.textContent = `${limit} of ${geodesics.length}`;
  el.recordList.innerHTML = "";
  geodesics.slice(0, limit).forEach((geo, index) => {
    const row = document.createElement("button");
    row.className = "record-row";
    row.type = "button";
    if (state.selectedGeodesic?.word === geo.word) row.classList.add("is-selected");
    row.innerHTML = `<strong>${geo.word}</strong><span>tr ${geo.trace_abs}; ${geo.length_expr}</span><span>${fmt(geo.length_numeric, 4)}</span>`;
    row.addEventListener("click", () => {
      state.selectedGeodesic = geo;
      render();
    });
    el.recordList.append(row);
  });
}

function renderTracePanel() {
  const terms = state.traceData?.terms;
  if (!terms) {
    setDetails(el.traceDetails, [
      ["status", "no sample"],
      ["test", "Gaussian h(r)=exp(-t*r^2)"],
      ["warning", "Run scripts/trace_formula_sandbox.py to generate diagnostics."]
    ]);
    return;
  }
  setDetails(el.traceDetails, [
    ["surface", state.traceData.surface_id],
    ["t", fmt(state.traceData.test_function.t, 4)],
    ["cusp sum", fmt(terms.spectral_cusp_sum.total, 8)],
    ["identity", fmt(terms.identity_term, 8)],
    ["hyperbolic", fmt(terms.hyperbolic_term.total, 8)],
    ["partial gap", fmt(terms.partial_gap_spectral_minus_identity_minus_hyperbolic, 8)]
  ]);
}

function renderSelectedJson() {
  let selected = {};
  if (state.mode === "form") selected = state.formData || {};
  if (state.mode === "geo") selected = state.selectedGeodesic || state.geodesicData?.geodesics?.[0] || {};
  if (state.mode === "trace") selected = state.traceData || { note: "Run scripts/trace_formula_sandbox.py to generate numerical diagnostics." };
  el.selectedJson.textContent = JSON.stringify(selected, null, 2);
}

function render() {
  if (!state.surface) return;
  if (state.mode === "form") {
    renderFormDetails();
    renderFormList();
    const image = state.formData ? selectedFormImage() : "";
    el.viewerTitle.textContent = `${state.surface.name}`;
    el.viewerSubtitle.textContent = state.formData ? `Eigenfunction ${state.formData.label}` : "Eigenfunction";
    showImage(image, "Maass eigenfunction render");
    if (!state.formData && state.formError) {
      el.emptyState.textContent = `${state.formError}. The manifest entry is preserved, but the referenced data is not checked in.`;
    }
  } else if (state.mode === "geo") {
    renderGeodesicSummary();
    renderGeodesicList();
    el.viewerTitle.textContent = `${state.surface.name}`;
    el.viewerSubtitle.textContent = "Closed geodesic data with exact matrices and folded paths";
    showImage(state.geodesicData ? selectedGeodesicImage() : "", "Closed geodesic render");
    if (!state.geodesicData && state.geodesicError) {
      el.emptyState.textContent = `${state.geodesicError}. The manifest entry is preserved, but the referenced data is not checked in.`;
    }
  } else {
    renderTracePanel();
    el.tableTitle.textContent = "Trace formula";
    el.recordCount.textContent = state.traceData ? "sample loaded" : "";
    el.recordList.innerHTML = "";
    const top = state.traceData?.terms?.hyperbolic_term?.top_contributions || [];
    for (const item of top) {
      const row = document.createElement("button");
      row.className = "record-row";
      row.type = "button";
      row.innerHTML = `<strong>${item.word}</strong><span>tr ${item.trace_abs}; length ${fmt(item.length, 4)}</span><span>${fmt(item.contribution, 3)}</span>`;
      row.addEventListener("click", () => {
        el.selectedJson.textContent = JSON.stringify(item, null, 2);
      });
      el.recordList.append(row);
    }
    el.viewerTitle.textContent = "Trace formula diagnostic";
    el.viewerSubtitle.textContent = "A partial numerical check: useful, but not yet the full Selberg trace formula.";
    showImage("", "Trace formula diagnostic");
    el.emptyState.textContent = "Run the trace formula sandbox script to create a JSON diagnostic, then add it to the manifest.";
  }
  renderSelectedJson();
}

async function init() {
  state.manifest = await fetchJson("manifest.json");
  el.sourceStatus.textContent = `${state.manifest.surfaces.length} surfaces in local manifest`;
  populateSurfaces();
  await loadTraceData();
  await loadSurface(state.surface.id);
}

el.surfaceSelect.addEventListener("change", () => loadSurface(el.surfaceSelect.value));
el.formSelect.addEventListener("change", async () => {
  state.form = state.surface.forms.find((form) => form.label === el.formSelect.value);
  await loadCurrentForm();
  render();
});
el.formViewSelect.addEventListener("change", render);
el.geoViewSelect.addEventListener("change", render);
el.geoLimit.addEventListener("input", render);
el.tabForm.addEventListener("click", () => setMode("form"));
el.tabGeo.addEventListener("click", () => setMode("geo"));
el.tabTrace.addEventListener("click", () => setMode("trace"));
el.resetSelection.addEventListener("click", () => {
  state.selectedGeodesic = null;
  render();
});

init().catch((error) => {
  el.sourceStatus.textContent = "Manifest load failed";
  el.emptyState.classList.remove("is-hidden");
  el.emptyState.textContent = error.message;
});
