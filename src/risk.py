"""Challenge risk rules as pure functions: tiers, readable reasons, actions, and the scored-row table.

Thresholds come from the hackathon brief (NASA/ESA-informed, not a certification):
Critical if Pc >= 1e-4 or miss distance < 1,000 m; Elevated if not Critical and (Pc >= 1e-6 or
miss distance <= 5,000 m); otherwise Safe. Log10(Pc) is compared with -4 and -6 directly to avoid
roundoff. Unusable inputs give needs_review, never safe, unless a valid input already proves Critical.
"""
import json

import numpy as np
import pandas as pd

CRITICAL_LOG_PC, ELEVATED_LOG_PC = -4.0, -6.0
CRITICAL_DISTANCE_M, ELEVATED_DISTANCE_M = 1000.0, 5000.0
TIERS = ['safe', 'elevated', 'critical', 'needs_review']
# Queue order: act first on Critical, then anything that needs a human look, then Elevated, then Safe.
PRIORITY = {'critical': 0, 'needs_review': 1, 'elevated': 2, 'safe': 3}
SEVERITY = {'safe': 0, 'elevated': 1, 'needs_review': 1, 'critical': 2}
ACTIONS = {'critical': 'Initiate maneuver planning',
           'needs_review': 'Check telemetry quality before relying on this assessment',
           'elevated': 'Review: request refined tracking and monitor updates',
           'safe': 'Monitor'}
PC_FLOOR_LOG = -6.001  # row model floor: "below 1e-6"


def format_pc(log_pc):
    if not np.isfinite(log_pc):
        return 'unavailable'
    return '< 1e-6' if log_pc <= PC_FLOOR_LOG + 1e-9 else f'{10 ** log_pc:.1e}'


def assess(log_pc, miss_distance_m):
    """Vectorized rules. Returns risk_tier, pc_trigger, distance_trigger, alert_reasons (lists), recommended_action."""
    p = np.asarray(log_pc, dtype=float).reshape(-1)
    d = np.asarray(miss_distance_m, dtype=float).reshape(-1)
    if p.shape != d.shape:
        raise ValueError('log_pc and miss_distance_m must have the same length.')
    vp = np.isfinite(p) & (p <= 0)
    vd = np.isfinite(d) & (d >= 0)
    pc_trigger = np.select([vp & (p >= CRITICAL_LOG_PC), vp & (p >= ELEVATED_LOG_PC)], ['critical', 'elevated'], None)
    distance_trigger = np.select([vd & (d < CRITICAL_DISTANCE_M), vd & (d <= ELEVATED_DISTANCE_M)], ['critical', 'elevated'], None)
    critical = (pc_trigger == 'critical') | (distance_trigger == 'critical')
    tier = np.select([critical, ~(vp & vd), (pc_trigger == 'elevated') | (distance_trigger == 'elevated')],
                     ['critical', 'needs_review', 'elevated'], 'safe')
    reasons = []
    for i in range(len(p)):
        r = []
        km = d[i] / 1000
        if pc_trigger[i] == 'critical':
            r.append(f'Estimated Pc {format_pc(p[i])} is at or above 1e-4')
        elif pc_trigger[i] == 'elevated':
            r.append(f'Estimated Pc {format_pc(p[i])} is between 1e-6 and 1e-4')
        if distance_trigger[i] == 'critical':
            r.append(f'Miss distance {km:.2f} km is below 1 km')
        elif distance_trigger[i] == 'elevated':
            r.append(f'Miss distance {km:.2f} km is within 5 km')
        if not vp[i]:
            r.append('Estimated Pc unavailable or invalid')
        if not vd[i]:
            r.append('Miss distance missing or invalid')
        if tier[i] == 'safe':
            r.append(f'Estimated Pc {format_pc(p[i])} is below 1e-6 and miss distance {km:.2f} km is above 5 km')
        reasons.append(r)
    return pd.DataFrame({'risk_tier': tier, 'pc_trigger': pc_trigger, 'distance_trigger': distance_trigger,
                         'alert_reasons': reasons, 'recommended_action': [ACTIONS[t] for t in tier]})


def most_severe(*tiers):
    """Element-wise most severe tier; Critical outranks needs_review, which ranks with Elevated."""
    stacked = np.column_stack([np.asarray(t, dtype=object) for t in tiers])
    out = []
    for row in stacked:
        valid = [t for t in row if isinstance(t, str)]
        out.append(max(valid, key=lambda t: (SEVERITY[t], -PRIORITY[t])) if valid else 'needs_review')
    return np.asarray(out, dtype=object)


def score_rows(frame, predicted_log10_pc, source='test', quality_flags=None):
    """One scored row per input message, in input order, using the shared output contract.

    Units are converted exactly once here (days -> hours, metres -> km). `risk` is never read.
    """
    p = np.asarray(predicted_log10_pc, dtype=float)
    if len(p) != len(frame):
        raise ValueError('One predicted log10(Pc) per input row is required.')
    if 'event_id' not in frame or frame.event_id.isna().any():
        raise ValueError('Every row needs an event_id.')
    distance = pd.to_numeric(frame.get('miss_distance'), errors='coerce').to_numpy(dtype=float)
    tca = pd.to_numeric(frame.get('time_to_tca'), errors='coerce').to_numpy(dtype=float)
    rules = assess(p, distance)
    out = pd.DataFrame({
        'source': source,
        'row_id': frame['row_id'].to_numpy() if 'row_id' in frame else np.arange(len(frame)),
        'event_id': frame.event_id.to_numpy(),
        'predicted_log10_pc': p, 'predicted_pc': 10.0 ** p, 'pc_display': [format_pc(v) for v in p],
        'miss_distance_m': distance, 'miss_distance_km': distance / 1000,
        'time_to_tca_days': tca, 'tca_hours': tca * 24,
        'c_object_type': frame.get('c_object_type', pd.Series(['UNKNOWN'] * len(frame))).fillna('UNKNOWN').astype(str).to_numpy(),
    })
    for axis in 'rtn':
        out[f'relative_position_{axis}_m'] = pd.to_numeric(frame.get(f'relative_position_{axis}'), errors='coerce').to_numpy(dtype=float)
    for col in rules:
        out[col] = rules[col].to_numpy()
    flags = [list(f) for f in quality_flags] if quality_flags is not None else [[] for _ in range(len(frame))]
    for i in np.flatnonzero(tca < 0):
        flags[i].append('after_tca_historical')
    out['quality_flags'] = flags
    return out


def to_csv_ready(frame):
    """JSON-encode list columns for CSV export."""
    out = frame.copy()
    for col in ['alert_reasons', 'quality_flags']:
        if col in out:
            out[col] = [json.dumps(v) if isinstance(v, list) else v for v in out[col]]
    return out
