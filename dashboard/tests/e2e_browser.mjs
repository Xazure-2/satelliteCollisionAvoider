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
check("synthetic badge shown", await page.locator('[data-testid="badge-synthetic"]').isVisible());
check("synthetic data strip shown", await page.locator('[data-testid="synthetic-strip"]').isVisible());
check("globe labelled synthetic", (await text(page, '[data-testid="globe-stamp"]')).includes("SYNTHETIC"));
const totalA = await text(page, '[data-testid="kpi-total"] .oi-kpi-value');
const critA = await text(page, '[data-testid="kpi-critical"] .oi-kpi-value');
check("dataset A totals", totalA === "220", `total=${totalA} critical=${critA}`);

await page.click('[data-testid="kpi-critical"]');
await settle(page);
const tiers = await page.locator('[data-testid="queue-row"] td:nth-child(2) .tier:last-child').allInnerTexts();
check("critical card filters queue", tiers.length > 0 && tiers.every((t) => t.includes("CRITICAL")), `${tiers.length} rows`);
await page.click('[data-testid="kpi-total"]');
await settle(page);

await page.selectOption('[data-testid="source-select"]', "synthetic_b");
await settle(page, 2000);
const totalB = await text(page, '[data-testid="kpi-total"] .oi-kpi-value');
const critB = await text(page, '[data-testid="kpi-critical"] .oi-kpi-value');
check("cards update for dataset B", totalB === "90" && critB !== critA, `total=${totalB} critical=${critB}`);

await page.selectOption('[data-testid="source-select"]', "real");
await settle(page, 2000);
const err = page.locator('[data-testid="data-error"]');
check("missing real outputs → clear error", (await err.isVisible()) && (await err.innerText()).includes("event_predictions.csv"),
  (await err.innerText()).split("\n")[0]);
check("no synthetic substitution for real source", !(await page.locator('[data-testid="badge-synthetic"]').count()));
await page.getByText("Show synthetic demo A instead").click();
await settle(page, 2000);
check("recover to synthetic A", (await text(page, '[data-testid="kpi-total"] .oi-kpi-value')) === "220");

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

// ---------------------------------------------------------------- encounter analysis
const firstId = (await page.locator('[data-testid="queue-row"] td:nth-child(1)').first().innerText()).trim();
await page.locator('[data-testid="queue-row"]').first().click();
await page.waitForURL("**/encounter");
await page.waitForSelector('[data-testid="event-id"]');
await settle(page, 2000);
check("row click opens encounter", (await text(page, '[data-testid="event-id"]')) === "#" + firstId, firstId);
check("observed and predicted shown separately",
  (await text(page, '[data-testid="side-observed"]')).includes("CURRENT OBSERVED") &&
  (await text(page, '[data-testid="side-predicted"]')).includes("PREDICTED FINAL"));
const plots = await page.locator(".js-plotly-plot").count();
check("risk, miss-distance and R/T/N charts render", plots >= 3, `${plots} plots`);
const axes = (await page.locator(".js-plotly-plot .xtitle, .js-plotly-plot .ytitle").evaluateAll(
  (els) => els.map((e) => e.textContent))).join(" | ");
check("chart units labelled (days, m)", axes.includes("days") && axes.includes("(m)"), axes);
const options = await page.locator('[data-testid="event-select"] option').evaluateAll((o) => o.map((x) => x.value));
await page.selectOption('[data-testid="event-select"]', options[3]);
await settle(page, 1500);
const newId = await text(page, '[data-testid="event-id"]');
check("selecting another event changes details", newId !== "#" + firstId, `${firstId} → ${newId}`);

// ---------------------------------------------------------------- validation
await page.click('[data-testid="nav-validation"]');
await page.waitForSelector('[data-testid="confusion-matrix"]');
await settle(page);
check("validation labelled as demonstration", await page.locator('[data-testid="validation-demo"]').isVisible());
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
