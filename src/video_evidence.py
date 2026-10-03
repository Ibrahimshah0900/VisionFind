"""Summarize stored frame detections without inventing scene descriptions."""
import math
from statistics import median

def summarize(records, targets):
    if not records:
        raise ValueError('No frame detections are available.')
    targets = list(dict.fromkeys(targets))
    if not targets:
        raise ValueError('Specify the evaluated categories.')
    times, indices = [], []
    counts = {target: [] for target in targets}
    for record in records:
        timestamp = float(record['timestamp_seconds'])
        index = record['frame_index']
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Invalid timestamp.')
        if type(index) is not int or index < 0:
            raise ValueError('Invalid frame index.')
        if times and (timestamp <= times[-1] or index <= indices[-1]):
            raise ValueError('Frames must have strictly increasing indices and timestamps.')
        detections = record['detections']
        if record['detected_count'] != len(detections):
            raise ValueError('Stored count does not match stored detections.')
        times.append(timestamp)
        indices.append(index)
        for target in targets:
            counts[target].append(sum(d['label'] == target for d in detections))

    per_frame_steps = [(times[i]-times[i-1])/(indices[i]-indices[i-1])
                       for i in range(1,len(times))]
    step = median(per_frame_steps) if per_frame_steps else None
    categories = []
    for target in targets:
        values = counts[target]
        spans = []
        start = end = None
        for position, value in enumerate(values):
            adjacent = (end is not None and indices[position] == indices[end]+1
                        and (step is None or times[position]-times[end] <= 1.5*step+1e-6))
            if value > 0:
                if start is not None and not adjacent:
                    spans.append({'start_seconds':times[start],'end_seconds':times[end],
                                  'frames':end-start+1})
                    start = None
                if start is None:
                    start = position
                end = position
            elif start is not None:
                spans.append({'start_seconds':times[start],'end_seconds':times[end],
                              'frames':end-start+1})
                start = end = None
        if start is not None:
            spans.append({'start_seconds':times[start],'end_seconds':times[end],
                          'frames':end-start+1})
        categories.append({'category':target,'frames_with_predictions':sum(v>0 for v in values),
                           'prediction_count_range_per_frame':[min(values),max(values)],
                           'prediction_runs':spans})
    return {'evaluated_frames':len(records),'first_evaluated_timestamp':times[0],
            'last_evaluated_timestamp':times[-1],'categories':categories,
            'scope':'Stored detector predictions; no identity tracking or scene interpretation.',
            'limitations':'No prediction does not establish absence. Prediction runs do not prove continuous physical presence or that the same object persisted.'}

def render(report):
    lines = ['**Detection-evidence summary**',
        f"Evaluated {report['evaluated_frames']} frames, with timestamps from "
        f"{report['first_evaluated_timestamp']:.2f}s to {report['last_evaluated_timestamp']:.2f}s."]
    for entry in report['categories']:
        low, high = entry['prediction_count_range_per_frame']
        lines.append(f"**{entry['category']}**: predictions in {entry['frames_with_predictions']} "
            f"of {report['evaluated_frames']} evaluated frames. Per-frame prediction counts ranged from {low} to {high}.")
        runs = entry['prediction_runs']
        if runs:
            intervals = ', '.join(f"{r['start_seconds']:.2f}–{r['end_seconds']:.2f}s" for r in runs[:12])
            lines.append('Consecutive recorded prediction runs: '+intervals+'.')
            if len(runs)>12:
                lines.append(f"{len(runs)-12} additional runs are recorded in the JSON report.")
        else:
            lines.append('No predictions for this category were recorded in the evaluated frames; this does not establish absence.')
    lines.append('These are detector predictions, not a verified object total. '
                 'They do not establish unique identities, movement, weather, intentions or events between frames.')
    return '\n\n'.join(lines)
