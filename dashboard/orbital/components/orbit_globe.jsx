// Interactive WebGL globe for the SYNTHETIC orbital population.
// Wrapped for Reflex in globe.py. Positions come from synthetic orbital
// elements only; nothing here is derived from the Airbus encounter data.
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Globe from "react-globe.gl";
import * as THREE from "three";

const R_EARTH_KM = 6371;
const TYPE_COLORS = {
  PAYLOAD: "#22d3ee",
  "ROCKET BODY": "#a78bfa",
  DEBRIS: "#cbd5e1",
  UNKNOWN: "#7c8aa5",
};
const TYPE_SHAPES = { PAYLOAD: "circle", "ROCKET BODY": "square", DEBRIS: "diamond", UNKNOWN: "ring" };
const PROX_COLORS = { close: "#ef4444", near: "#f59e0b" };
const SELECT_COLOR = "#38bdf8";
const HOME = { lat: 18, lng: 20, altitude: 2.7 };

// Display altitude in globe radii. LEO is true to scale; above 2,000 km the
// scale is compressed (logarithmic) so MEO/GEO objects stay on screen.
export function displayAltitude(altKm) {
  if (altKm <= 2000) return altKm / R_EARTH_KM;
  const top = 2000 / R_EARTH_KM;
  return top + 0.62 * (Math.log10(altKm / 2000) / Math.log10(36000 / 2000));
}

const textureCache = {};
function markerTexture(shape, color) {
  const key = shape + color;
  if (textureCache[key]) return textureCache[key];
  const size = 64;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d");
  g.fillStyle = color;
  g.strokeStyle = color;
  g.lineWidth = 9;
  g.shadowColor = color;
  g.shadowBlur = 10;
  const m = size / 2;
  g.beginPath();
  if (shape === "circle") g.arc(m, m, 17, 0, Math.PI * 2);
  else if (shape === "square") g.rect(m - 15, m - 15, 30, 30);
  else if (shape === "diamond") {
    g.moveTo(m, m - 20); g.lineTo(m + 17, m); g.lineTo(m, m + 20); g.lineTo(m - 17, m); g.closePath();
  } else g.arc(m, m, 16, 0, Math.PI * 2);
  if (shape === "ring") g.stroke(); else g.fill();
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  textureCache[key] = tex;
  return tex;
}

const materialCache = {};
function spriteFor(d, selected) {
  const color = selected ? SELECT_COLOR : PROX_COLORS[d.proximity] || TYPE_COLORS[d.type] || "#ffffff";
  const shape = TYPE_SHAPES[d.type] || "circle";
  const key = shape + color;
  if (!materialCache[key]) {
    materialCache[key] = new THREE.SpriteMaterial({
      map: markerTexture(shape, color), transparent: true, depthWrite: false,
    });
  }
  const sprite = new THREE.Sprite(materialCache[key]);
  const s = selected ? 3.8 : d.proximity !== "clear" ? 2.5 : 1.6;
  sprite.scale.set(s, s, 1);
  return sprite;
}

function orbitPath(d, steps = 180) {
  const inc = (d.inc_deg * Math.PI) / 180;
  const raan = (d.raan_deg * Math.PI) / 180;
  const alt = displayAltitude(d.alt_km);
  const pts = [];
  for (let k = 0; k <= steps; k++) {
    const u = (2 * Math.PI * k) / steps;
    const x = Math.cos(raan) * Math.cos(u) - Math.sin(raan) * Math.sin(u) * Math.cos(inc);
    const y = Math.sin(raan) * Math.cos(u) + Math.cos(raan) * Math.sin(u) * Math.cos(inc);
    const z = Math.sin(u) * Math.sin(inc);
    pts.push([(Math.asin(z) * 180) / Math.PI, (Math.atan2(y, x) * 180) / Math.PI, alt]);
  }
  return pts;
}

function tooltip(d) {
  const prox = d.proximity !== "clear"
    ? `<div style="color:${PROX_COLORS[d.proximity]}">Simulated separation ${d.sep_km} km</div>` : "";
  return `<div class="og-tip"><div class="og-tip-id">${d.id}</div>
    <div>${d.type} · ${d.regime} · ${Math.round(d.alt_km).toLocaleString()} km</div>${prox}
    <div class="og-tip-syn">Synthetic orbital object</div></div>`;
}

export function OrbitGlobe({
  objects = [], visibleTypes = [], visibleProximity = [], selectedId = "",
  focusNonce = 0, resetNonce = 0, autoRotate = true, showOrbits = true, onSelect,
}) {
  const wrap = useRef(null);
  const globe = useRef(null);
  const [size, setSize] = useState({ w: 600, h: 500 });

  useEffect(() => {
    if (!wrap.current) return;
    const ro = new ResizeObserver(([e]) => {
      const { width, height } = e.contentRect;
      setSize({ w: Math.max(200, Math.floor(width)), h: Math.max(200, Math.floor(height)) });
    });
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  const data = useMemo(() => {
    const types = new Set(visibleTypes);
    const prox = new Set(visibleProximity);
    return objects
      .filter((d) => d.id === selectedId || ((!types.size || types.has(d.type)) && (!prox.size || prox.has(d.proximity))))
      .map((d) => ({ ...d, _alt: displayAltitude(d.alt_km) }));
  }, [objects, visibleTypes, visibleProximity, selectedId]);

  // Stable accessors: react-globe.gl rebuilds every marker when these change.
  const makeObject = useCallback((d) => spriteFor(d, d.id === selectedId), [selectedId]);
  const handleClick = useCallback((d) => onSelect && onSelect(d.id), [onSelect]);

  const byId = useMemo(() => Object.fromEntries(objects.map((d) => [d.id, d])), [objects]);
  const selected = byId[selectedId];

  const paths = useMemo(() => {
    if (!showOrbits || !selected) return [];
    const out = [{ pts: orbitPath(selected), color: SELECT_COLOR }];
    if (selected.partner && byId[selected.partner]) {
      out.push({ pts: orbitPath(byId[selected.partner]), color: PROX_COLORS[selected.proximity] });
    }
    return out;
  }, [selected, byId, showOrbits]);

  const rings = useMemo(() => {
    const out = objects
      .filter((d) => d.proximity === "close")
      .map((d) => ({ lat: d.lat, lng: d.lng, color: PROX_COLORS.close, alt: displayAltitude(d.alt_km), speed: 1.2 }));
    if (selected) out.push({ lat: selected.lat, lng: selected.lng, color: SELECT_COLOR, alt: displayAltitude(selected.alt_km), speed: 2 });
    return out;
  }, [objects, selected]);

  useEffect(() => {
    const g = globe.current;
    if (!g) return;
    const controls = g.controls();
    controls.autoRotate = autoRotate && !selectedId;
    controls.autoRotateSpeed = 0.35;
    controls.enableDamping = true;
    controls.minDistance = 101.5;
    controls.maxDistance = 900;
  }, [autoRotate, selectedId, size]);

  // Read-only hook used by the browser tests to locate markers on the canvas.
  useEffect(() => {
    window.__orbitGlobe = {
      screenCoords: (id) => {
        const d = byId[id];
        if (!d || !globe.current) return null;
        return globe.current.getScreenCoords(d.lat, d.lng, displayAltitude(d.alt_km));
      },
      pov: () => (globe.current ? globe.current.pointOfView() : null),
      count: () => data.length,
      ids: () => data.map((d) => d.id),
      object: (id) => byId[id] || null,
    };
  }, [byId, data]);

  useEffect(() => {
    if (!focusNonce || !globe.current || !selected) return;
    const alt = displayAltitude(selected.alt_km);
    globe.current.pointOfView({ lat: selected.lat, lng: selected.lng, altitude: Math.max(0.25, alt + 0.35) }, 1600);
  }, [focusNonce]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (globe.current) globe.current.pointOfView(HOME, resetNonce ? 1400 : 0);
  }, [resetNonce]);

  return (
    <div ref={wrap} className="og-wrap" data-testid="orbit-globe" data-object-count={data.length}>
      <Globe
        ref={globe}
        width={size.w}
        height={size.h}
        backgroundColor="rgba(0,0,0,0)"
        backgroundImageUrl="/globe/night-sky.png"
        globeImageUrl="/globe/earth-blue-marble.jpg"
        bumpImageUrl="/globe/earth-topology.png"
        showAtmosphere={true}
        atmosphereColor="#4cc9ff"
        atmosphereAltitude={0.2}
        objectsData={data}
        objectLat="lat"
        objectLng="lng"
        objectAltitude="_alt"
        objectFacesSurface={false}
        objectThreeObject={makeObject}
        objectLabel={tooltip}
        onObjectClick={handleClick}
        pathsData={paths}
        pathPoints="pts"
        pathPointLat={(p) => p[0]}
        pathPointLng={(p) => p[1]}
        pathPointAlt={(p) => p[2]}
        pathColor={(p) => p.color}
        pathStroke={1.2}
        pathDashLength={0.02}
        pathDashGap={0.008}
        pathDashAnimateTime={0}
        pathTransitionDuration={0}
        ringsData={rings}
        ringLat="lat"
        ringLng="lng"
        ringAltitude="alt"
        ringColor={(r) => (t) => `${r.color}${Math.round((1 - t) * 255).toString(16).padStart(2, "0")}`}
        ringMaxRadius={3}
        ringPropagationSpeed={(r) => r.speed}
        ringRepeatPeriod={1100}
      />
    </div>
  );
}
