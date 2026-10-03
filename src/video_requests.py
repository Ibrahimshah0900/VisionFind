"""Initial video action and category routing; no model changes."""
import re
ALIASES = {
    'person': ['person','persons','people','ppl','perosn','pedestrian','pedestrians'],
    'car': ['car','cars'], 'bicycle': ['bicycle','bicycles','bike','bikes'],
    'motorcycle': ['motorcycle','motorcycles','motorbike','motorbikes'],
    'bus': ['bus','buses'], 'truck': ['truck','trucks'],
    'dog': ['dog','dogs'], 'cat': ['cat','cats'], 'bird': ['bird','birds'],
    'boat': ['boat','boats'], 'bottle': ['bottle','bottles'],
    'cup': ['cup','cups'], 'bowl': ['bowl','bowls'], 'bed': ['bed','beds'],
    'dining table': ['dining table','dining tables'], 'chair': ['chair','chairs'],
}

def plan_video_request(text):
    query = re.sub(r'\s+', ' ', str(text).lower()).strip()
    query = re.sub(r'\b(?:bouding|bonding)\b', 'bounding', query)
    query = re.sub(r'\b(?:exolain|explainn)\b', 'explain', query)
    boxes = bool(re.search(r'\b(?:box|boxes|bounding|detect|locate)\b', query))
    explain = bool(re.search(r'\b(?:explain|describe|description|summarize|summary)\b', query))
    count = bool(re.search(r'\b(?:count|how many)\b', query))
    targets = [category for category, names in ALIASES.items()
               if any(re.search(r'\b'+re.escape(name)+r'\b', query) for name in names)]
    clarification = ''
    if (boxes or count) and not targets:
        clarification = 'Supported video targets include people, cars, bicycles, buses and trucks. Which objects should I locate?'
    elif not (boxes or explain or count):
        clarification = 'Would you like a video explanation or boxes around a named object?'
    return {'query':query, 'explain':explain, 'draw_boxes':boxes,
            'count_per_frame':count and bool(targets), 'targets':targets,
            'target':targets[0] if len(targets)==1 else None,
            'clarification':clarification, 'visible_ids':False,
            'unique_person_count':False}


# Detection evidence is separate from generated scene descriptions.
_base_video_plan = plan_video_request

def plan_video_request(text):
    plan = _base_video_plan(text)
    question = str(text).lower()
    timing = bool(re.search(r'\bwhen\b|\bat what (?:time|times|point)\b|\btimeline\b',question))
    evidence = bool(plan.get('count_per_frame') or timing)
    plan['evidence_summary'] = evidence and bool(plan.get('targets'))
    if plan['evidence_summary']:
        plan['clarification'] = ''
    elif evidence and not plan.get('targets'):
        plan['clarification'] = 'Which object category should I summarize from the detections?'
    return plan
