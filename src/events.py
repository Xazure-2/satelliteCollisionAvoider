"""Event-level views for the dashboard: latest-message snapshots, priority tiers, time windows, action queue, insights.

There are no clock times in the data, so warning windows use time_to_tca (organizer guidance: e.g.
events with closest approach within the next 1 day). A snapshot keeps, per (source, event_id), the
newest message at or before a cutoff `time_to_tca` (cutoff 0 = latest pre-TCA message; a positive
cutoff replays what was known that many days before each event's TCA). Messages after TCA are
historical and never selected.
"""
import json

import numpy as np
import pandas as pd

from src.risk import ACTIONS, PRIORITY, assess, most_severe, score_rows, format_pc

TIER_NAME = {'Safe': 'safe', 'Elevated': 'elevated', 'Critical': 'critical'}


def snapshot(scored, cutoff_days=0.0):
    """Newest message per (source, event_id) with time_to_tca >= cutoff; ties broken by the later row."""
    keys = ['source', 'event_id'] if 'source' in scored else ['event_id']
    eligible = scored.loc[np.isfinite(scored.time_to_tca_days) & scored.time_to_tca_days.ge(max(cutoff_days, 0.0))]
    latest = (eligible.sort_values(keys + ['time_to_tca_days', 'row_id'], ascending=[True] * len(keys) + [True, False])
              .drop_duplicates(keys))
    latest = latest.assign(snapshot_cutoff_days=float(cutoff_days),
                           message_age_days=latest.time_to_tca_days - max(cutoff_days, 0.0))
    return latest.reset_index(drop=True)


def event_table(scored, event_predictions=None, cutoff_days=0.0, outcomes=None):
    """One row per event: current assessment (latest message), final-tier forecast, priority tier and reasons.

    The final-tier forecast uses messages at least 2 days before closest approach. While the latest
    message is itself 2+ days out, the forecast is current and priority_tier is the most severe of the
    current and forecast tiers (the forecast can raise attention, never lower it). Once newer messages
    exist, they supersede the forecast and priority_tier is the current tier; the forecast stays visible.
    """
    from src.final_tier import HORIZON_DAYS
    latest = snapshot(scored, cutoff_days)
    counts = scored.loc[scored.time_to_tca_days.ge(max(cutoff_days, 0.0))].groupby(['source', 'event_id']).size()
    events = latest.rename(columns={'risk_tier': 'current_tier', 'alert_reasons': 'current_reasons'})
    events['message_count'] = counts.reindex(pd.MultiIndex.from_frame(events[['source', 'event_id']])).to_numpy()
    if event_predictions is not None and len(event_predictions):
        f = event_predictions.rename(columns={'predicted_tier': 'forecast_tier'})
        f['forecast_tier'] = f.forecast_tier.map(lambda t: TIER_NAME.get(t, t))
        f['persistence_tier'] = f.persistence_tier.map(lambda t: TIER_NAME.get(t, t))
        cols = ['source', 'event_id', 'forecast_tier', 'persistence_tier'] + \
               [c for c in ['predicted_final_miss_distance_m', 'p_critical'] if c in f]
        events = events.merge(f[cols], on=['source', 'event_id'], how='left', validate='one_to_one')
    else:
        events['forecast_tier'] = None
        events['persistence_tier'] = None
    events['forecast_available'] = events.forecast_tier.notna()
    events['forecast_current'] = events.forecast_available & events.time_to_tca_days.ge(HORIZON_DAYS)
    events['priority_tier'] = np.where(events.forecast_current, most_severe(events.current_tier, events.forecast_tier),
                                       events.current_tier)
    reasons = []
    for row in events.itertuples():
        r = list(row.current_reasons)
        if isinstance(row.forecast_tier, str):
            distance = getattr(row, 'predicted_final_miss_distance_m', np.nan)
            detail = f' (forecast final miss distance {distance / 1000:.2f} km)' if np.isfinite(distance) else ''
            if not row.forecast_current:
                r.append(f'Earlier final-tier forecast from messages 2+ days out was {row.forecast_tier}{detail}; '
                         'superseded by newer messages')
            elif PRIORITY[row.forecast_tier] < PRIORITY[row.current_tier] and row.forecast_tier != 'needs_review':
                r.insert(0, f'Model forecasts a {row.forecast_tier} final tier{detail}, above the current {row.current_tier}')
            else:
                r.append(f'Model forecasts a {row.forecast_tier} final tier{detail}')
        else:
            r.append('No final-tier forecast: no message 2+ days before closest approach')
        reasons.append(r)
    events['reasons'] = reasons
    events['recommended_action'] = [ACTIONS[t] for t in events.priority_tier]
    if outcomes is not None:
        events = events.merge(outcomes, on=['source', 'event_id'], how='left', validate='one_to_one')
    return events.drop(columns=['current_reasons'])


def in_window(events, window_days=1.0):
    t = events.time_to_tca_days
    return t.ge(0) & t.le(window_days)


def window_counts(events, window_days=1.0):
    """Unique-event counts with closest approach within the window (0 <= time_to_tca <= window_days)."""
    w = events.loc[in_window(events, window_days)]
    return dict(window_days=float(window_days), encounters=int(len(w)),
                below_1km_now=int(w.distance_trigger.eq('critical').sum()),
                critical=int(w.priority_tier.eq('critical').sum()),
                elevated=int(w.priority_tier.eq('elevated').sum()),
                needs_review=int(w.priority_tier.eq('needs_review').sum()),
                critical_pc_driven=int(w.pc_trigger.eq('critical').sum()),
                critical_distance_driven=int(w.distance_trigger.eq('critical').sum()),
                forecast_upgrades=int((w.forecast_current & w.priority_tier.ne(w.current_tier)).sum()))


def action_queue(events):
    """Priority tier, then soonest closest approach, then highest estimated Pc; stable on event key."""
    q = events.assign(_rank=events.priority_tier.map(PRIORITY), _pc=events.predicted_log10_pc.fillna(-np.inf))
    q = q.sort_values(['_rank', 'time_to_tca_days', '_pc', 'source', 'event_id'],
                      ascending=[True, True, False, True, True], kind='mergesort')
    return q.drop(columns=['_rank', '_pc']).reset_index(drop=True)


def risk_trends(scored):
    """Per event: change in estimated log10(Pc) between consecutive messages (older -> newer), and the largest rise."""
    s = scored.loc[scored.time_to_tca_days.ge(0)].sort_values(['source', 'event_id', 'time_to_tca_days', 'row_id'],
                                                              ascending=[True, True, False, True])
    s = s.assign(log_pc_change=s.groupby(['source', 'event_id']).predicted_log10_pc.diff())
    rise = s.groupby(['source', 'event_id']).log_pc_change.max().rename('largest_log_pc_rise')
    return s, rise.reset_index()


def insights(events, scored, window_days=1.0):
    """Deterministic sentences for the dashboard; every number comes from the tables."""
    out = []
    n = len(events)
    tiers = events.priority_tier.value_counts()
    out.append(f"{n} events: {tiers.get('critical', 0)} critical, {tiers.get('elevated', 0)} elevated, "
               f"{tiers.get('safe', 0)} safe, {tiers.get('needs_review', 0)} need review (priority tier).")
    c = window_counts(events, window_days)
    if c['encounters'] == 0 and len(events):
        out.append(f"No event has closest approach within {window_days:g} day(s) of its latest message "
                   f"(nearest: {events.time_to_tca_days.min():.2f} days).")
    else:
        out.append(f"{c['encounters']} events have closest approach within {window_days:g} day(s): "
                   f"{c['critical']} critical ({c['critical_distance_driven']} by miss distance, "
                   f"{c['critical_pc_driven']} by estimated Pc), {c['elevated']} elevated.")
    upgrades = events.loc[events.forecast_current & events.priority_tier.ne(events.current_tier)]
    if len(upgrades):
        out.append(f"The final-tier model raised {len(upgrades)} events above their current tier "
                   f"({int(upgrades.priority_tier.eq('critical').sum())} to critical).")
    valid = events.loc[np.isfinite(events.predicted_log10_pc)]
    if len(valid):
        top = valid.loc[valid.predicted_log10_pc.idxmax()]
        out.append(f"Highest estimated Pc: event {top.event_id} ({format_pc(top.predicted_log10_pc)}, "
                   f"{top.c_object_type}, {top.time_to_tca_days:.2f} days to closest approach).")
    near = events.loc[np.isfinite(events.miss_distance_m)]
    if len(near):
        top = near.loc[near.miss_distance_m.idxmin()]
        out.append(f"Closest current miss distance: event {top.event_id} at {top.miss_distance_km:.3f} km "
                   f"({top.c_object_type}).")
    by_type = events.loc[events.priority_tier.eq('critical')].c_object_type.value_counts()
    if len(by_type):
        out.append('Critical events by object type: ' + ', '.join(f'{k} {v}' for k, v in by_type.items()) + '.')
    _, rise = risk_trends(scored)
    rise = rise.dropna(subset=['largest_log_pc_rise'])
    if len(rise) and rise.largest_log_pc_rise.max() > 0:
        top = rise.loc[rise.largest_log_pc_rise.idxmax()]
        out.append(f"Largest rise in estimated Pc between consecutive messages: event {top.event_id}, "
                   f"+{top.largest_log_pc_rise:.2f} orders of magnitude (a rising trend is not by itself a tier change).")
    return out


def build_tables(messages, row_bundle, tier_bundle=None, source='test', cutoff_days=0.0, outcomes=None,
                 window_days=1.0):
    """End to end: per-message scores, final-tier forecasts, event table, queue, window counts, insights.

    `messages` holds event_id plus permitted telemetry; any `risk` column is ignored. The final-tier
    model only sees messages at least 2 days before closest approach (and at or before the cutoff).
    """
    from src.final_tier import HORIZON_DAYS, predict_events
    from src.model import prediction_frame
    frame = messages.drop(columns=['risk'], errors='ignore').copy()
    if 'row_id' not in frame:
        frame['row_id'] = np.arange(len(frame))
    pred = prediction_frame(frame, row_bundle, source=source)
    flags = [json.loads(f) for f in pred.quality_flags]
    scored = score_rows(frame, pred.predicted_log10_pc.to_numpy(), source=source, quality_flags=flags)
    scored['model_version'] = pred.model_version.to_numpy()
    forecasts = None
    if tier_bundle is not None:
        visible = frame.loc[frame.time_to_tca.ge(max(HORIZON_DAYS, cutoff_days))]
        if len(visible):
            forecasts = predict_events(visible, tier_bundle).assign(source=source)
    events = event_table(scored, forecasts, cutoff_days, outcomes)
    return dict(rows=scored, events=events, queue=action_queue(events),
                counts=window_counts(events, window_days), insights=insights(events, scored, window_days))


__all__ = ['snapshot', 'event_table', 'window_counts', 'action_queue', 'risk_trends', 'insights', 'build_tables',
           'assess']
