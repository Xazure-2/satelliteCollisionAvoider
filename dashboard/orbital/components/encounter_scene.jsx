// Illustrative encounter view. CDM markers sit at their MEASURED relative
// positions (projected onto a 2-D plane, km). The Earth backdrop, icons and
// faint approach arcs are decoration only and are labelled as such.
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

const CYAN = "#38bdf8";
const AMBER = "#f5a524";
const RED = "#ef4444";
const MUTED = "#7d91b6";

function niceStep(km) {
  const p = Math.pow(10, Math.floor(Math.log10(km)));
  for (const m of [1, 2, 5, 10]) if (m * p >= km) return m * p;
  return 10 * p;
}
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), Math.max(lo, hi));
const fmtKm = (km) => (km >= 1 ? `${+km.toFixed(km >= 10 ? 0 : 1)} km` : `${Math.round(km * 1000)} m`);

function Satellite({ x, y, size = 1 }) {
  return (
    <g transform={`translate(${x},${y}) rotate(-18) scale(${size})`} filter="url(#glowCyan)">
      <rect x="-30" y="-7" width="20" height="14" rx="1.5" fill="#0e3a5c" stroke={CYAN} strokeWidth="1.4" />
      <line x1="-24" y1="-7" x2="-24" y2="7" stroke={CYAN} strokeWidth="0.6" opacity="0.6" />
      <line x1="-17" y1="-7" x2="-17" y2="7" stroke={CYAN} strokeWidth="0.6" opacity="0.6" />
      <rect x="10" y="-7" width="20" height="14" rx="1.5" fill="#0e3a5c" stroke={CYAN} strokeWidth="1.4" />
      <line x1="17" y1="-7" x2="17" y2="7" stroke={CYAN} strokeWidth="0.6" opacity="0.6" />
      <line x1="24" y1="-7" x2="24" y2="7" stroke={CYAN} strokeWidth="0.6" opacity="0.6" />
      <line x1="-10" y1="0" x2="10" y2="0" stroke={CYAN} strokeWidth="1.4" />
      <rect x="-7" y="-9" width="14" height="18" rx="2.5" fill="#cfefff" stroke={CYAN} strokeWidth="1.4" />
      <circle cx="0" cy="-12" r="2.4" fill={CYAN} />
    </g>
  );
}

function Debris({ x, y, size = 1 }) {
  return (
    <g transform={`translate(${x},${y}) rotate(24) scale(${size})`} filter="url(#glowAmber)">
      <path d="M-11,-4 L-4,-12 L8,-9 L12,1 L5,11 L-7,9 L-12,3 Z" fill="#5a3a06" stroke={AMBER} strokeWidth="1.6" />
      <path d="M-4,-12 L-1,-1 L12,1 M-1,-1 L-7,9" stroke={AMBER} strokeWidth="0.8" opacity="0.7" fill="none" />
      <rect x="10" y="-16" width="11" height="6" rx="1" fill="#5a3a06" stroke={AMBER} strokeWidth="1.1"
        transform="rotate(30 15 -13)" />
    </g>
  );
}

export function EncounterScene({ points = [], hAxis = "T", vAxis = "R", droppedAxis = "N", predictedMissKm = null,
  predictedMissLabel = "", separationLabel = "", countdown = "", selectedId = "", resetNonce = 0, onSelect }) {
  const wrap = useRef(null);
  const [size, setSize] = useState({ w: 800, h: 560 });
  const [view, setView] = useState({ k: 1, tx: 0, ty: 0 });
  const [hover, setHover] = useState(null);
  const drag = useRef(null);

  useEffect(() => {
    if (!wrap.current) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);
  const reset = useCallback(() => setView({ k: 1, tx: 0, ty: 0 }), []);
  // Reset only on an explicit reset or a different encounter, not on every state update.
  const signature = points.map((p) => p.id).join(",");
  useEffect(reset, [resetNonce, signature, reset]);

  // Fit the measured points, the primary (origin) and any predicted-distance ring into the view.
  const base = useMemo(() => {
    const xs = [0, ...points.map((p) => p.x)], ys = [0, ...points.map((p) => p.y)];
    const ring = predictedMissKm || 0;
    const minX = Math.min(...xs, -ring), maxX = Math.max(...xs, ring);
    const minY = Math.min(...ys, -ring), maxY = Math.max(...ys, ring);
    const spanX = Math.max(maxX - minX, 0.5), spanY = Math.max(maxY - minY, 0.5);
    const s = Math.min((size.w * 0.44) / spanX, (size.h * 0.48) / spanY);
    return { s, cx: size.w * 0.6 - ((minX + maxX) / 2) * s, cy: size.h * 0.45 + ((minY + maxY) / 2) * s };
  }, [points, predictedMissKm, size]);

  const s = base.s * view.k;
  const P = (x, y) => [base.cx + view.tx + x * s, base.cy + view.ty - y * s];
  const zoomAt = (factor, px = size.w / 2, py = size.h / 2) => setView((v) => {
    const k = Math.min(60, Math.max(0.25, v.k * factor));
    const f = k / v.k;
    // keep the point under the cursor fixed
    const ox = base.cx + v.tx, oy = base.cy + v.ty;
    return { k, tx: px - (px - ox) * f - base.cx, ty: py - (py - oy) * f - base.cy };
  });

  const onWheel = (e) => {
    e.preventDefault();
    const r = wrap.current.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
  };
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  });

  const ordered = points;
  const latest = ordered.find((p) => p.latest) || ordered[ordered.length - 1];
  const [ox, oy] = P(0, 0);
  const lp = latest ? P(latest.x, latest.y) : null;
  const scaleKm = niceStep((size.w * 0.16) / s);
  const fullscreen = () => (document.fullscreenElement ? document.exitFullscreen() : wrap.current.requestFullscreen());

  return (
    <div ref={wrap} className="es-wrap" data-testid="encounter-scene" data-points={points.length}
      onPointerDown={(e) => { if (e.target.closest(".es-ui")) return; drag.current = { x: e.clientX, y: e.clientY, ...view }; }}
      onPointerMove={(e) => { if (!drag.current) return; const d = drag.current;
        setView({ k: d.k, tx: d.tx + e.clientX - d.x, ty: d.ty + e.clientY - d.y }); }}
      onPointerUp={() => { drag.current = null; }} onPointerLeave={() => { drag.current = null; }}>
      <div className="es-earth" aria-hidden="true" />
      <svg width={size.w} height={size.h} className="es-svg">
        <defs>
          <filter id="glowCyan" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="3" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="glowAmber" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="3" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <radialGradient id="caGlow"><stop offset="0%" stopColor={RED} stopOpacity="0.12" /><stop offset="100%" stopColor={RED} stopOpacity="0" /></radialGradient>
        </defs>

        {/* Illustrative approach arcs: decoration only, not trajectories */}
        <path d={`M ${ox - size.w * 0.42} ${oy + size.h * 0.18} Q ${ox - size.w * 0.2} ${oy - size.h * 0.02} ${ox - 34} ${oy - 4}`}
          stroke={CYAN} strokeWidth="1.2" strokeDasharray="2 7" fill="none" opacity="0.45" />
        {lp && <path d={`M ${lp[0] + size.w * 0.4} ${lp[1] - size.h * 0.2} Q ${lp[0] + size.w * 0.18} ${lp[1] - size.h * 0.04} ${lp[0] + 22} ${lp[1] - 4}`}
          stroke={AMBER} strokeWidth="1.2" strokeDasharray="2 7" fill="none" opacity="0.45" />}

        {/* Predicted final miss distance (distance only, around the primary) */}
        {predictedMissKm ? (<g data-testid="predicted-ring">
          <circle cx={ox} cy={oy} r={predictedMissKm * s} fill="url(#caGlow)" />
          <circle cx={ox} cy={oy} r={predictedMissKm * s} fill="none" stroke={RED} strokeWidth="1.4" strokeDasharray="6 6" opacity="0.85" />
          <text x={clamp(ox, 190, size.w - 190)} y={Math.max(oy - predictedMissKm * s - 10, 96)} textAnchor="middle"
            className="es-ring-label">{predictedMissLabel}</text>
        </g>) : null}
        {s > 26 && (<g data-testid="km-ring">
          <circle cx={ox} cy={oy} r={1 * s} fill="none" stroke={RED} strokeWidth="0.8" strokeDasharray="1 4" opacity="0.5" />
          <text x={ox - 0.71 * s - 4} y={oy - 0.71 * s - 4} textAnchor="end" className="es-faint">1 km threshold</text>
        </g>)}

        {/* Separation at the latest eligible CDM (measured) */}
        {lp && <g>
          <line x1={ox} y1={oy} x2={lp[0]} y2={lp[1]} stroke="#e2e8f0" strokeWidth="1.2" strokeDasharray="5 5" opacity="0.8" />
          <text x={(ox + lp[0]) / 2 + 10} y={(oy + lp[1]) / 2 - 10} className="es-sep" data-testid="separation-label">{separationLabel}</text>
        </g>}

        {/* Successive CDM estimates (measured), connected in time order: not a path */}
        <polyline points={ordered.map((p) => P(p.x, p.y).join(",")).join(" ")} fill="none" stroke={AMBER}
          strokeWidth="1" opacity="0.55" />
        {ordered.map((p) => {
          const [x, y] = P(p.x, p.y);
          const sel = p.id === selectedId;
          return (<g key={p.id} className="es-pt" data-testid="cdm-marker" data-id={p.id}
            onPointerEnter={() => setHover({ p, x, y })} onPointerLeave={() => setHover(null)}
            onPointerDown={(e) => e.stopPropagation()} onClick={() => onSelect && onSelect(p.id)}>
            <circle cx={x} cy={y} r="7" fill="transparent" />
            {sel && <circle cx={x} cy={y} r="9" fill="none" stroke="#fff" strokeWidth="1.5" />}
            <circle cx={x} cy={y} r={p.latest ? 5.5 : 3.6} fill={p.used ? AMBER : "none"}
              stroke={p.used ? AMBER : MUTED} strokeWidth={p.used ? 0 : 1.4} opacity={p.used ? (p.latest ? 1 : 0.75) : 0.9} />
          </g>);
        })}

        {/* Icons (illustrative): primary at origin, secondary at the latest eligible CDM */}
        <Satellite x={ox} y={oy} />
        <text x={ox} y={oy + 34} textAnchor="middle" className="es-label" fill={CYAN}>PRIMARY</text>
        {lp && <g onClick={() => onSelect && onSelect(latest.id)} style={{ cursor: "pointer" }}
          onPointerEnter={() => setHover({ p: latest, x: lp[0], y: lp[1] })} onPointerLeave={() => setHover(null)}>
          <circle cx={lp[0]} cy={lp[1]} r="20" fill="none" stroke={AMBER} strokeWidth="1" opacity="0.5" />
          <Debris x={lp[0]} y={lp[1]} />
          <text x={clamp(lp[0], 120, size.w - 120)} y={lp[1] + 34} textAnchor="middle" className="es-label"
            fill={AMBER}>SECONDARY · latest eligible CDM</text>
        </g>}
      </svg>

      {hover && (<div className="es-tip" style={{ left: hover.x + 14, top: hover.y + 10 }}>
        <div className="es-tip-h">{hover.p.label}</div>
        <div>Miss distance {hover.p.miss}</div>
        <div>{hover.p.rtn}</div>
        <div>Observed Pc {hover.p.pc}</div>
        <div className="es-tip-r">{hover.p.role}</div>
      </div>)}

      <div className="es-ui es-tca"><div>Time to closest approach</div><b>{countdown}</b></div>
      <div className="es-ui es-tools">
        <button title="Zoom in" onClick={() => zoomAt(1.4)} data-testid="scene-zoom-in">+</button>
        <button title="Zoom out" onClick={() => zoomAt(1 / 1.4)} data-testid="scene-zoom-out">−</button>
        <button title="Reset view" onClick={reset} data-testid="scene-reset">⟲</button>
        <button title="Full screen" onClick={fullscreen}>⛶</button>
      </div>
      <div className="es-ui es-scale"><span style={{ width: scaleKm * s }} />{fmtKm(scaleKm)}</div>
      <div className="es-ui es-triad" title="Projection plane of the measured R/T/N positions">
        <svg width="78" height="70"><g transform="translate(18,52)">
          <line x1="0" y1="0" x2="44" y2="0" stroke={CYAN} strokeWidth="1.6" /><text x="48" y="4" fill={CYAN}>{hAxis}</text>
          <line x1="0" y1="0" x2="0" y2="-40" stroke={RED} strokeWidth="1.6" /><text x="-4" y="-44" fill={RED}>{vAxis}</text>
          <circle r="3" fill="#94a3b8" /><text x="6" y="14" fill="#94a3b8">{droppedAxis} ⊙</text>
        </g></svg>
      </div>
      <div className="es-ui es-legend" data-testid="scene-legend">
        <div><i className="sw-c" style={{ background: CYAN }} />Primary satellite (reference, at origin)</div>
        <div><i className="sw-c" style={{ background: AMBER }} />Secondary object at latest eligible CDM</div>
        <div><i className="sw-c sm" style={{ background: AMBER, opacity: 0.75 }} />Past CDM estimates (≥ 2 d, measured)</div>
        <div><i className="sw-c sm hollow" />After cutoff (held-out truth, if any)</div>
        <div><i className="sw-l" />Successive estimates in time order (not a path)</div>
        <div><i className="sw-l dash" style={{ borderColor: "#e2e8f0" }} />Observed separation</div>
        {predictedMissKm ? <div><i className="sw-ring" />Predicted final miss distance (distance only)</div> : null}
        <div><i className="sw-l dot" />Approach arcs: illustrative, not real trajectories</div>
      </div>
    </div>
  );
}
