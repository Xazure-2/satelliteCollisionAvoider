// End-to-end browser checks for the Reflex dashboard.
// Start the app first (cd dashboard && reflex run), then:
//   npm install playwright@1 && npx playwright install chromium   (once)
//   node tests/e2e_browser.mjs [http://localhost:3000]
import { chromium } from "playwright";

const BASE = process.argv[2] || "http://localhost:3000";
const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok: !!ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
};
const settle = (page, ms = 1200) => page.waitForTimeout(ms);
const text = async (page, sel) => (await page.locator(sel).first().innerText()).trim();

const browser = await chromium.launch({
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, acceptDownloads: true });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

// ---------------------------------------------------------------- overview + datasets
await page.goto(BASE + "/", { waitUntil: "networkidle" });
await page.waitForSelector('[data-testid="kpi-total"]');
await settle(page, 2500);
check("real data badge shown by default", await page.locator('[data-testid="badge-real"]').isVisible());
check("no data-source selector or synthetic demo", !(await page.locator('[data-testid="source-select"]').count()) &&
  !(await page.locator('[data-testid="badge-synthetic"]').count()));
check("globe labelled synthetic", (await text(page, '[data-testid="globe-stamp"]')).includes("SYNTHETIC"));
const totalA = await text(page, '[data-testid="kpi-total"] .oi-kpi-value');
const critA = await text(page, '[data-testid="kpi-critical"] .oi-kpi-value');
check("real outputs loaded", Number(totalA.replace(/,/g, "")) > 0, `total=${totalA} critical=${critA}`);
const predPc = await page.locator('[data-testid="queue-row"] td:nth-child(3)').allInnerTexts();
check("queue Pred. Pc column filled and readable", predPc.length > 0 && predPc.every((t) => t.trim() && t.trim() !== "—" && !/e-\d/.test(t)),
  predPc.slice(0, 3).join(" | "));

await page.click('[data-testid="kpi-critical"]');
await settle(page);
const tiers = await page.locator('[data-testid="queue-row"] td:nth-child(2) .tier:last-child').allInnerTexts();
check("critical card filters queue", tiers.length > 0 && tiers.every((t) => t.includes("CRITICAL")), `${tiers.length} rows`);
await page.click('[data-testid="kpi-total"]');
await settle(page);

// ---------------------------------------------------------------- risk page: filters, sort, export
await page.click('[data-testid="nav-risk"]');
await page.waitForSelector('[data-testid="tier-elevated"]');
await settle(page, 1500);
await page.click('[data-testid="tier-elevated"]');
await settle(page);
const elev = await page.locator('[data-testid="queue-row"] td:nth-child(5)').allInnerTexts();
check("tier filter Elevated", elev.length > 0 && elev.every((t) => t.includes("ELEVATED")), `${elev.length} rows`);
await page.click('[data-testid="tier-all"]');
await page.click('[data-testid="win-2-3"]');
await settle(page);
const cds = await page.locator('[data-testid="queue-row"] td:nth-child(8)').allInnerTexts();
check("time-to-TCA filter 2–3 d", cds.length > 0 && cds.every((t) => t.startsWith("T−2 d")), `${cds.length} rows`);
await page.click('[data-testid="win-all"]');
await page.fill('[data-testid="search"]', "12");
await settle(page, 1500);
const ids = await page.locator('[data-testid="queue-row"] td:nth-child(1)').allInnerTexts();
check("event ID search", ids.length > 0 && ids.every((t) => t.includes("12")), ids.slice(0, 6).join(","));
await page.fill('[data-testid="search"]', "");
await page.click('[data-testid="sort-pc"]');
await settle(page);
const pcs = (await page.locator('[data-testid="queue-row"] td:nth-child(6)').allInnerTexts()).map((t) =>
  t === "≈ 0" ? 0 : t === "—" ? -1 : parseFloat(t));
const sortedDesc = pcs.filter((x) => x >= 0).every((v, i, a) => i === 0 || a[i - 1] >= v);
check("sort by predicted final Pc", sortedDesc, pcs.slice(0, 4).join(", "));
await page.click('[data-testid="sort-priority"]');
await settle(page);
const [download] = await Promise.all([page.waitForEvent("download", { timeout: 10000 }), page.click('[data-testid="export"]')]);
check("CSV export", download.suggestedFilename().endsWith(".csv"), download.suggestedFilename());

// ---------------------------------------------------------------- encounter analysis (visual workspace)
const STEPS = ["Refer for human manoeuvre assessment", "Prioritise further monitoring or review", "Routine monitoring",
  "Request additional or corrected information"];
const read = async () => ({
  id: await text(page, '[data-testid="event-id"]'),
  observed: await text(page, '[data-testid="side-observed"] .tier'),
  predicted: await text(page, '[data-testid="side-predicted"] .tier'),
  escalation: await text(page, '[data-testid="escalation"]'),
  countdown: await text(page, '[data-testid="countdown"]'),
  pc: await text(page, '[data-testid="pred-pc"]'),
  obsMiss: await text(page, '[data-testid="obs-miss"]'),
  predMiss: await text(page, '[data-testid="pred-miss"]'),
  why: await text(page, '[data-testid="why-summary"]'),
  hits: await page.locator('[data-testid="threshold-row"] .inv-hit.on').count(),
  step: await text(page, '[data-testid="next-step"]'),
  unc: await text(page, '[data-testid="uncertainty"]'),
  inspector: await text(page, '[data-testid="cdm-inspector"]'),
  panel: await text(page, '[data-testid="summary-panel"]'),
});
const markers = () => page.locator('[data-testid="cdm-marker"]');
// centre of each visible dot (the selection ring would change a group's bounding box)
const markerPos = () => markers().evaluateAll((els) => els.map((e) => { const c = e.querySelector("circle:last-of-type");
  return `${Math.round(+c.getAttribute("cx"))},${Math.round(+c.getAttribute("cy"))}`; }).join(" "));
const consistent = (e, label) => {
  const severe = ["CRITICAL", "ELEVATED"].some((t) => e.predicted.includes(t));
  const ok = e.id.startsWith("Encounter #") && e.countdown.startsWith("T−") && STEPS.includes(e.step) &&
    e.why.length > 10 && (severe ? e.hits >= 1 : e.hits === 0) && e.unc.length > 0;
  check(`encounter ${label}: decision panel complete`, ok,
    `${e.id} ${e.observed}→${e.predicted} (${e.escalation}) ${e.countdown} Pc ${e.pc} | ${e.step}`);
};
const noOverlap = async (label) => {
  const r = await page.evaluate(() => {
    const box = (sel) => { const el = document.querySelector(sel); if (!el) return null; const b = el.getBoundingClientRect(); return { l: b.left, r: b.right, t: b.top, b: b.bottom }; };
    const inter = (a, b) => a && b && a.l < b.r - 1 && b.l < a.r - 1 && a.t < b.b - 1 && b.t < a.b - 1;
    const left = box(".enc-left"), right = box(".enc-right"), scene = box('[data-testid="encounter-scene"]');
    const overlays = [".es-tca", ".es-tools", ".es-legend", ".es-triad", ".es-scale"].map(box);
    const inside = overlays.every((o) => !o || !scene || (o.l >= scene.l - 1 && o.r <= scene.r + 1 && o.t >= scene.t - 1 && o.b <= scene.b + 1));
    const overlayClash = overlays.some((a, i) => overlays.some((b, j) => i < j && inter(a, b)));
    const head = [...document.querySelectorAll(".enc-head > *")].map((e) => e.getBoundingClientRect()).filter((b) => b.width > 0)
      .map((b) => ({ l: b.left, r: b.right, t: b.top, b: b.bottom }));
    const headClash = head.some((a, i) => head.some((b, j) => i < j && inter(a, b)));
    const hscroll = document.documentElement.scrollWidth > window.innerWidth + 1;
    const clipped = [...document.querySelectorAll(".inv-stat-value, .enc-step-label, .enc-title")]
      .some((e) => e.scrollWidth > e.clientWidth + 2);
    return { panels: inter(left, right), inside, overlayClash, headClash, hscroll, clipped };
  });
  check(`layout ${label}: no overlap or clipping`, !r.panels && r.inside && !r.overlayClash && !r.headClash && !r.hscroll && !r.clipped,
    JSON.stringify(r));
};

const firstId = (await page.locator('[data-testid="queue-row"] td:nth-child(1)').first().innerText()).trim();
await page.locator('[data-testid="queue-row"]').first().click();
await page.waitForURL("**/encounter");
await page.waitForSelector('[data-testid="encounter-scene"]');
await settle(page, 2500);
let enc = await read();
check("row click opens encounter", enc.id === "Encounter #" + firstId, enc.id);
const headText = await text(page, ".enc-head");
check("header: object type, data source, model version",
  /PAYLOAD|DEBRIS|ROCKET BODY|UNKNOWN/.test(headText) && headText.includes("Model outputs") &&
  headText.includes("final-tier"), headText.replace(/\s+/g, " ").slice(0, 90));
check("current observed vs AI-predicted final shown separately",
  enc.panel.includes("CURRENT OBSERVED") && enc.panel.includes("PREDICTED FINAL") && enc.escalation.length > 0, enc.escalation);
consistent(enc, "1");

// illustrative view labelling and data fidelity
const note = await text(page, '[data-testid="view-note"]');
const legend = await text(page, '[data-testid="scene-legend"]');
check("illustrative view labelled (projection, not to scale)", note.includes("ILLUSTRATIVE") && note.includes("Measured positions projected")
  && note.includes("not to scale"), note.slice(0, 110));
check("legend: estimates are not a path; arcs not real trajectories",
  legend.includes("not a path") && legend.includes("not real trajectories"));
await page.click('[data-testid="telemetry"] summary');
await settle(page, 600);
const nCdm = await page.locator('[data-testid="cdm-table"] tbody tr').count();
check("one marker per measured CDM", (await markers().count()) === nCdm, `${nCdm} CDMs`);
const fullHead = (await page.locator('[data-testid="cdm-table"] thead').innerText()).toLowerCase();
check("telemetry columns carry units", ["(d)", "(m)", "(m/s)", "r (m)", "t (m)", "n (m)"].every((u) => fullHead.includes(u)), fullHead.replace(/\s+/g, " "));
await page.click('[data-testid="telemetry"] summary');
check("separation = latest observed miss distance",
  (await page.locator('[data-testid="separation-label"]').textContent()).trim() === `${enc.obsMiss} (observed)`, enc.obsMiss);
check("predicted ring only when model supplies final miss distance",
  ((await page.locator('[data-testid="predicted-ring"]').count()) > 0) === (enc.predMiss !== "Not provided"), enc.predMiss);
check("inspector defaults to latest eligible CDM", enc.inspector.includes("Latest eligible CDM"));

// hover, click, zoom, pan, reset in the illustrative view
// pick a marker whose centre is not covered by a neighbouring estimate
const reachable = await page.evaluate(() => {
  for (const g of document.querySelectorAll('[data-testid="cdm-marker"]')) {
    const c = g.querySelector("circle").getBoundingClientRect();
    const hit = document.elementFromPoint(c.x + c.width / 2, c.y + c.height / 2);
    if (hit && hit.closest('[data-testid="cdm-marker"]') === g) return g.getAttribute("data-id");
  }
  return null;
});
const first = page.locator(`[data-testid="cdm-marker"][data-id="${reachable}"]`);
await first.hover();
await settle(page, 500);
const tip = await page.locator(".es-tip").innerText().catch(() => "");
check("hover CDM shows measurements", tip.includes("Miss distance") && tip.includes("Observed Pc") && /R .* T .* N /.test(tip), tip.replace(/\n/g, " | "));
await first.click();
await settle(page, 1200);
const insp = await text(page, '[data-testid="cdm-inspector"]');
check("click CDM opens it in the inspector", insp.includes(tip.split(" · ")[0].replace(/\n.*/s, "")) && insp.includes("time_to_tca") && insp.includes("observed Pc"), insp.slice(0, 80));
await page.click('[data-testid="cdm-prev"]');
await settle(page, 900);
const insp2 = await text(page, '[data-testid="cdm-inspector"]');
check("inspector steps through CDMs (overlapping markers)", insp2 !== insp && insp2.includes("time_to_tca"), insp2.slice(0, 40));
const p0 = await markerPos();
await page.click('[data-testid="scene-zoom-in"]');
await settle(page, 400);
const p1 = await markerPos();
const sb = await page.locator('[data-testid="encounter-scene"]').boundingBox();
await page.mouse.move(sb.x + sb.width * 0.55, sb.y + sb.height * 0.35);
await page.mouse.wheel(0, -300);
await settle(page, 400);
await page.mouse.down();
await page.mouse.move(sb.x + sb.width * 0.55 + 80, sb.y + sb.height * 0.35 + 40, { steps: 6 });
await page.mouse.up();
await settle(page, 400);
const p2 = await markerPos();
await page.click('[data-testid="scene-reset"]');
await settle(page, 400);
const p3 = await markerPos();
check("illustrative view zooms, pans and resets", p0 !== p1 && p1 !== p2 && p3 === p0, [p0, p1, p2, p3].map((x) => x.slice(0, 16)).join(" → "));
await page.click('[data-testid="scene-zoom-in"]');
await settle(page, 400);
const zoomed = await markerPos();
await page.click('[data-testid="cdm-next"]');
await settle(page, 1200);
check("zoom is kept when inspecting another CDM", (await markerPos()) === zoomed && zoomed !== p0);
await page.click('[data-testid="scene-reset"]');
await settle(page, 400);

// measured R/T/N geometry view
await page.click('[data-testid="mode-measured"]');
await page.waitForSelector('[data-testid="geometry-plot"] .js-plotly-plot');
await settle(page, 2500);
check("measured view labelled", (await text(page, '[data-testid="view-note"]')).includes("MEASURED R/T/N GEOMETRY"));
const gd = '[data-testid="geometry-plot"] .js-plotly-plot';
const measured = await page.evaluate((sel) => {
  const g = document.querySelector(sel);
  return g.data.filter((t) => t.type === "scatter3d").map((t) => ({ n: t.name, k: (t.x || []).length }));
}, gd);
const inputs = measured.filter((t) => /Input CDMs|After cutoff|Latest eligible/.test(t.n));
check("measured view plots the actual CDM positions", inputs.length > 0, measured.map((t) => `${t.n}:${t.k}`).join(" | "));
const eye = () => page.evaluate((sel) => { const e = document.querySelector(sel)._fullLayout.scene._scene.getCamera().eye; return { x: e.x, y: e.y, r: Math.hypot(e.x, e.y, e.z) }; }, gd);
const gb = await page.locator(gd).boundingBox();
const e0 = await eye();
await page.mouse.move(gb.x + gb.width * 0.6, gb.y + gb.height * 0.5);
await page.mouse.down();
await page.mouse.move(gb.x + gb.width * 0.6 + 150, gb.y + gb.height * 0.5 + 20, { steps: 10 });
await page.mouse.up();
await settle(page, 1000);
const e1 = await eye();
await page.mouse.wheel(0, -400);
await settle(page, 1000);
const e2 = await eye();
check("measured view rotates and zooms", Math.abs(e1.x - e0.x) + Math.abs(e1.y - e0.y) > 0.2 && e2.r < e1.r - 0.05,
  `eye ${e0.x.toFixed(2)},${e0.y.toFixed(2)} → ${e1.x.toFixed(2)},${e1.y.toFixed(2)}; dist ${e1.r.toFixed(2)} → ${e2.r.toFixed(2)}`);
await page.evaluate((sel) => { window.__hov = []; document.querySelector(sel).on("plotly_hover", (d) => window.__hov.push(d.points[0].data.name)); }, gd);
let cdmSpot = null;
for (let gx = 0.1; gx <= 0.9 && !cdmSpot; gx += 0.04) {
  for (let gy = 0.1; gy <= 0.9; gy += 0.04) {
    await page.mouse.move(gb.x + gb.width * gx, gb.y + gb.height * gy);
    await settle(page, 90);
    const h = await page.evaluate(() => window.__hov.slice(-1)[0]);
    if (h && /Input CDMs|Latest eligible|After cutoff/.test(h)) { cdmSpot = { x: gb.x + gb.width * gx, y: gb.y + gb.height * gy, name: h }; break; }
  }
}
check("measured view: hover a CDM", !!cdmSpot, cdmSpot ? cdmSpot.name : (await page.evaluate(() => window.__hov.slice(-3).join(" | "))));
if (cdmSpot) {
  const before = await text(page, '[data-testid="cdm-inspector"]');
  await page.mouse.click(cdmSpot.x, cdmSpot.y);
  await settle(page, 1500);
  const after = await text(page, '[data-testid="cdm-inspector"]');
  check("measured view: click CDM updates inspector", after.includes("time_to_tca") && (after !== before || after.includes(cdmSpot.name.includes("Latest") ? "Latest" : "CDM")),
    after.slice(0, 60));
}
await page.click('[data-testid="geometry-reset"]');
await settle(page, 1500);
const e3 = await eye();
check("measured view resets camera", Math.abs(e3.x - 1.5) < 0.05 && Math.abs(e3.y - 1.5) < 0.05, `eye ${e3.x.toFixed(2)},${e3.y.toFixed(2)}`);
await page.click('[data-testid="mode-illustrative"]');
await settle(page, 1500);

// evidence charts: units and honest prediction markers
const axes = async (id) => (await page.locator(`[data-testid="${id}"] .xtitle, [data-testid="${id}"] .ytitle`)
  .evaluateAll((els) => els.map((e) => e.textContent))).join(" | ");
check("risk chart units (log10 Pc vs days)", (await axes("chart-risk")).includes("log10(Pc)") && (await axes("chart-risk")).includes("days"));
check("miss chart units (metres vs days)", (await axes("chart-miss")).includes("(m, log scale)") && (await axes("chart-miss")).includes("days"));
check("compact CDM history ≤ 6 rows", (await page.locator('[data-testid="cdm-mini"] tbody tr').count()) <= 6);
await noOverlap("1920×1080");

// navigation: everything follows the selected encounter
const seen = new Set([enc.id]);
let pos = await markerPos();
for (let k = 2; k <= 4; k++) {
  await page.click('[data-testid="next-event"]');
  await settle(page, 2200);
  enc = await read();
  const np = await markerPos();
  check(`next encounter ${k}: header, panel and visualisation update`, !seen.has(enc.id) && np !== pos, enc.id);
  consistent(enc, String(k));
  seen.add(enc.id);
  pos = np;
}
const options = await page.locator('[data-testid="event-select"] option').evaluateAll((o) => o.map((x) => x.value));
await page.selectOption('[data-testid="event-select"]', options[60]);
await settle(page, 2000);
consistent(await read(), "picker");

// held-out encounter
await page.click('[data-testid="enc-scope-held"]');
await settle(page, 2500);
enc = await read();
consistent(enc, "held-out");
const hollow = await page.evaluate(() => [...document.querySelectorAll('[data-testid="cdm-marker"] circle')].some((c) => c.getAttribute("fill") === "none" && c.getAttribute("r") !== "9"));
check("held-out: post-cutoff CDMs drawn distinctly (hollow)", hollow);

await page.setViewportSize({ width: 1440, height: 900 });
await noOverlap("1440×900");
await page.setViewportSize({ width: 1920, height: 1080 });

// ---------------------------------------------------------------- validation
await page.click('[data-testid="nav-validation"]');
await page.waitForSelector('[data-testid="confusion-matrix"]');
await settle(page);
check("validation computed from held-out events", await page.locator('[data-testid="validation-real"]').isVisible());
const cells = (await page.locator('[data-testid="confusion-matrix"] td').allInnerTexts()).map(Number);
const n = await text(page, ".oi-kpi .oi-kpi-value");
check("confusion matrix sums to evaluated events", cells.reduce((a, b) => a + b, 0) === Number(n), `sum=${cells.reduce((a, b) => a + b, 0)} n=${n}`);

// ---------------------------------------------------------------- globe
await page.click('[data-testid="nav-tracking"]');
await page.waitForFunction(() => window.__orbitGlobe && window.__orbitGlobe.count() > 0, null, { timeout: 30000 });
await settle(page, 3000);
await page.getByText("Rotation on").click();
await settle(page, 800);
const canvas = await page.locator('[data-testid="orbit-globe"] canvas').first().boundingBox();
const pov0 = await page.evaluate(() => window.__orbitGlobe.pov());

// rotate by dragging
await page.mouse.move(canvas.x + canvas.width / 2, canvas.y + canvas.height / 2);
await page.mouse.down();
await page.mouse.move(canvas.x + canvas.width / 2 + 220, canvas.y + canvas.height / 2, { steps: 12 });
await page.mouse.up();
await settle(page, 1500);
const pov1 = await page.evaluate(() => window.__orbitGlobe.pov());
check("drag rotates globe", Math.abs(pov1.lng - pov0.lng) > 5, `lng ${pov0.lng.toFixed(1)} → ${pov1.lng.toFixed(1)}`);

// zoom with wheel
await page.mouse.wheel(0, -900);
await settle(page, 1500);
const pov2 = await page.evaluate(() => window.__orbitGlobe.pov());
check("wheel zooms in", pov2.altitude < pov1.altitude - 0.2, `alt ${pov1.altitude.toFixed(2)} → ${pov2.altitude.toFixed(2)}`);
await page.click('[data-testid="globe-home"]');
await settle(page, 2000);

// find visible, front-facing markers; hover until a tooltip confirms one, then click it
await settle(page, 2500);
const candidates = await page.evaluate(() => {
  const g = window.__orbitGlobe, pov = g.pov(), rad = Math.PI / 180, out = [];
  const canvas = document.querySelector('[data-testid="orbit-globe"] canvas').getBoundingClientRect();
  for (const id of g.ids()) {
    const o = g.object(id);
    if (o.regime !== "LEO") continue;
    const c = Math.sin(o.lat * rad) * Math.sin(pov.lat * rad) +
      Math.cos(o.lat * rad) * Math.cos(pov.lat * rad) * Math.cos((o.lng - pov.lng) * rad);
    const s = g.screenCoords(id);
    if (!(c > 0.8 && s && s.x > 300 && s.x < 1300 && s.y > 200 && s.y < 750)) continue;
    const hit = document.elementFromPoint(canvas.x + s.x, canvas.y + s.y);
    if (hit && hit.tagName === "CANVAS") out.push(id);
    if (out.length >= 12) break;
  }
  return out;
});
check("visible markers found", candidates.length > 0, `${candidates.length} candidates`);
const box = canvas;
let hovered = null, at = null;
// Software WebGL (no GPU) can run at a few fps; hover is evaluated once per frame.
const frameMs = await page.evaluate(() => new Promise((r) => { let n = 0; const t = performance.now();
  const f = () => { n++; performance.now() - t < 1500 ? requestAnimationFrame(f) : r(1500 / n); }; requestAnimationFrame(f); }));
const wait = Math.min(4000, Math.max(700, frameMs * 8));
console.log(`      (globe frame time ${frameMs.toFixed(0)} ms; interaction wait ${wait.toFixed(0)} ms)`);
for (const id of candidates.concat(candidates)) {
  const s = await page.evaluate((i) => window.__orbitGlobe.screenCoords(i), id);
  await page.mouse.move(box.x + s.x - 8, box.y + s.y - 8);
  await page.mouse.move(box.x + s.x, box.y + s.y, { steps: 8 });
  await settle(page, wait);
  const tip = await page.evaluate(() => [...document.querySelectorAll(".float-tooltip-kap")]
    .filter((e) => getComputedStyle(e).visibility !== "hidden").map((e) => e.innerText).join(" "));
  if (tip.includes("Synthetic orbital object")) { hovered = tip.split("\n")[0].trim(); at = s; break; }
}
check("hover shows tooltip", !!hovered, hovered || "no tooltip");
if (hovered) {
  await page.mouse.down();
  await settle(page, 100);
  await page.mouse.up();
  await page.waitForSelector('[data-testid="object-panel"]', { timeout: 10000 }).catch(() => {});
  const panelVisible = await page.locator('[data-testid="object-panel"]').isVisible();
  const panelText = panelVisible ? await page.locator('[data-testid="object-panel"]').innerText() : "";
  check("click marker opens its detail panel", panelVisible && panelText.includes(hovered), `${hovered} → ${panelText.split("\n")[0]}`);
  check("panel labels object synthetic, no Airbus link",
    panelText.includes("SYNTHETIC ORBITAL OBJECT") && panelText.includes("not linked"));
  const before = await page.evaluate(() => window.__orbitGlobe.pov());
  await page.click('[data-testid="focus-object"]');
  await settle(page, 1600 + wait * 2);
  const after = await page.evaluate(() => window.__orbitGlobe.pov());
  check("focus zooms to object", after.altitude < 1 && after.altitude < before.altitude,
    `alt ${before.altitude.toFixed(2)} → ${after.altitude.toFixed(2)}`);
  check("other markers still shown while focused", (await page.evaluate(() => window.__orbitGlobe.count())) > 1000);
  await page.click('[data-testid="reset-view"]');
  await settle(page, 1400 + wait * 2);
  const home = await page.evaluate(() => window.__orbitGlobe.pov());
  check("return to global view", home.altitude > 2.4 && !(await page.locator('[data-testid="object-panel"]').count()),
    `alt ${home.altitude.toFixed(2)}`);
}

const all = await page.evaluate(() => window.__orbitGlobe.count());
await page.click('[data-testid="gtype-PAYLOAD"]');
await settle(page, 1500);
const payload = await page.evaluate(() => window.__orbitGlobe.count());
check("type filter reduces markers", payload < all && payload > 0, `${all} → ${payload}`);
await page.click('[data-testid="gtype-PAYLOAD"]');
await page.click('[data-testid="gprox-close"]');
await settle(page, 1500);
const close = await page.evaluate(() => window.__orbitGlobe.count());
check("proximity filter", close > 0 && close < 40, `${close} close objects`);
await page.click('[data-testid="gprox-close"]');

await page.selectOption('[data-testid="globe-size"]', "12000");
await page.waitForFunction(() => window.__orbitGlobe.count() > 11000, null, { timeout: 60000 });
await settle(page, 2000);
const fps = await page.evaluate(() => new Promise((res) => {
  let n = 0; const t0 = performance.now();
  const tick = () => { n++; if (performance.now() - t0 < 3000) requestAnimationFrame(tick); else res(n / 3); };
  requestAnimationFrame(tick);
}));
const t0 = Date.now();
await page.click('[data-testid="nav-home"]');
await page.waitForSelector('[data-testid="kpi-total"]');
check("12,000 markers: page stays responsive", fps > 0.5 && Date.now() - t0 < 15000,
  `${fps.toFixed(1)} fps (software WebGL), navigation ${Date.now() - t0} ms`);

check("no browser errors", errors.length === 0, errors.slice(0, 3).join(" | "));
await browser.close();
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
