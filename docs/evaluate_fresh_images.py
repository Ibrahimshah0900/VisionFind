"""Download fresh project-evaluation photos and run inference only.

No optimizer, training call, model replacement, or chat-code mutation.
COCO128 is new to this project test suite, not proven unseen by pretrained models.
"""
import csv
import hashlib
import json
import random
import re
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

DATA_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip'
NAMES_URL = 'https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco128.yaml'
DOCS_URL = 'https://docs.ultralytics.com/datasets/detect/coco128/'
TERMS_URL = 'https://cocodataset.org/#termsofuse'
SEED = 20260930


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def download_once(url, path):
    path = Path(path)
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.part')
    print('Downloading:', path.name, flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'VisionFind-evaluation/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temp.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if temp.stat().st_size == 0:
            raise ValueError('Download returned an empty file.')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def extract_dataset(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as saved:
        bad = saved.testzip()
        if bad:
            raise ValueError(f'Dataset archive has a corrupt entry: {bad}')
        for entry in saved.infolist():
            if not (destination / entry.filename).resolve().is_relative_to(destination):
                raise ValueError('Invalid dataset archive path.')
        image_entries = [x for x in saved.namelist() if '/images/train2017/' in x and x.endswith('.jpg')]
        if not image_entries:
            raise ValueError('Archive does not contain the expected COCO128 photos.')
        if any(not (destination / x).is_file() for x in image_entries):
            saved.extractall(destination)
    return destination / 'coco128'


def parse_names(text):
    names = {}
    in_names = False
    for line in text.splitlines():
        if line.strip() == 'names:':
            in_names = True
            continue
        if in_names:
            match = re.fullmatch(r'\s+(\d+):\s*([A-Za-z][A-Za-z0-9 -]{0,40})\s*', line)
            if match:
                names[int(match.group(1))] = match.group(2).strip()
    if set(names) != set(range(80)):
        raise ValueError('Expected the official 80 COCO class names.')
    return names


def read_records(dataset, names):
    dataset = Path(dataset)
    records = []
    for image in sorted((dataset / 'images/train2017').glob('*.jpg')):
        label_file = dataset / 'labels/train2017' / f'{image.stem}.txt'
        if not label_file.exists():
            continue
        boxes = []
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f'Unexpected annotation row: {label_file}')
            category = int(values[0])
            coords = [float(x) for x in values[1:]]
            if category not in names or not all(0 <= x <= 1 for x in coords) or coords[2] <= 0 or coords[3] <= 0:
                raise ValueError(f'Invalid normalized annotation: {label_file}')
            boxes.append({'category_id': category, 'name': names[category], 'xywh': coords})
        if not boxes:
            continue
        primary = max(boxes, key=lambda x: x['xywh'][2] * x['xywh'][3])['name']
        records.append({'coco_image_id': image.stem, 'path': str(image.resolve()),
                        'label_path': str(label_file.resolve()), 'primary': primary,
                        'classes': sorted({x['name'] for x in boxes}), 'boxes': boxes})
    return records


def select_records(records, count=20, seed=SEED):
    """Freeze selection before inference; diversify largest annotated subjects."""
    if len(records) < count:
        raise ValueError(f'Need at least {count} annotated photos; found {len(records)}.')
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    selected, seen_classes, seen_images = [], set(), set()
    for record in shuffled:
        if record['primary'] not in seen_classes and record['coco_image_id'] not in seen_images:
            selected.append(record)
            seen_classes.add(record['primary'])
            seen_images.add(record['coco_image_id'])
            if len(selected) == count:
                return selected
    for record in shuffled:
        if record['coco_image_id'] not in seen_images:
            selected.append(record)
            seen_images.add(record['coco_image_id'])
            if len(selected) == count:
                return selected
    raise ValueError('Not enough unique photos.')


def category_queries(records):
    """Any annotated match in the batch is an acceptable retrieval source."""
    queries = []
    for category in sorted({r['primary'] for r in records}):
        expected = [i for i, record in enumerate(records, 1) if category in record['classes']]
        queries.append({'category': category, 'query': f'a photo containing {category}',
                        'expected_ids': expected})
    return queries


def run_fresh_eval(engine, project_root=None):
    from src.upload_rag import UploadedImageRAG
    root = Path(project_root or engine.root).resolve()
    data_root = root / 'data/evaluation'
    archive = data_root / 'downloads/coco128.zip'
    names_file = data_root / 'downloads/coco128.yaml'
    download_once(DATA_URL, archive)
    download_once(NAMES_URL, names_file)
    dataset = extract_dataset(archive, data_root)
    names = parse_names(names_file.read_text(encoding='utf-8'))
    records = read_records(dataset, names)
    selected = select_records(records)
    packs = {'A': selected[::2], 'B': selected[1::2]}
    selection_path = data_root / 'coco128_selection_seed20260930.json'
    manifest = {
        'dataset_url': DATA_URL, 'class_names_url': NAMES_URL, 'docs_url': DOCS_URL,
        'image_terms_url': TERMS_URL,
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'class_names_sha256': hashlib.sha256(names_file.read_bytes()).hexdigest(),
        'seed': SEED, 'available_annotated_images': len(records), 'selected': selected,
        'purpose': 'inference evaluation only; no weights updated',
        'scope': '20 photos in two separate ten-image conversations; not proven unseen by pretrained models',
    }
    if selection_path.exists():
        previous = json.loads(selection_path.read_text())
        for field in ['archive_sha256', 'class_names_sha256', 'selected']:
            if previous.get(field) != manifest[field]:
                raise ValueError('Evaluation dataset or selection changed. Preserve the prior evaluation and version a new set explicitly.')
    else:
        write_json(selection_path, manifest)
    print(f'Frozen evaluation: {len(selected)} photos; {len(set(r["primary"] for r in selected))} main-object categories.', flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'results/fresh_image_evaluation' / stamp
    output.mkdir(parents=True, exist_ok=True)
    report = {'source_manifest': str(selection_path), 'retrieval': [], 'visual_answers': [],
              'model_identity': None, 'model_weights_updated': False,
              'limitations': 'Small annotation-based evaluation. Missing annotations are not proof of absence. VLM answers need visual review.'}
    for pack, items in packs.items():
        print(f'Indexing evaluation pack {pack}: {len(items)} photos', flush=True)
        state = {'session_id': f'coco128_eval_{SEED}_{pack}',
                 'images': [{'id': i, 'path': item['path'], 'filename': Path(item['path']).name}
                            for i, item in enumerate(items, 1)]}
        rag = UploadedImageRAG.from_state(engine, state)
        report['model_identity'] = rag.model
        for query in category_queries(items):
            hits = rag.search(query['query'], top_k=3)
            expected = set(query['expected_ids'])
            row = dict(query, pack=pack, top_image=hits[0]['image_id'],
                       top_score=hits[0]['similarity'], top1_correct=hits[0]['image_id'] in expected,
                       top3_correct=any(h['image_id'] in expected for h in hits), matches=hits)
            report['retrieval'].append(row)
            write_json(output / 'results.json', report)
        reloaded = UploadedImageRAG.from_state(engine, state)
        if reloaded.cache_status != 'loaded':
            raise RuntimeError('Evaluation index did not reload from its cache.')
        # Two varied answer samples per pack; do not automatically mark quality passed.
        for i, item in enumerate(items[:2], 1):
            question = 'Describe the main visible subjects, their physical actions, and surroundings briefly. State uncertainty and avoid guesses about emotions, health, identity or exact breeds.'
            answer = engine.ask(item['path'], question, max_new_tokens=120)
            report['visual_answers'].append({'pack': pack, 'image_id': i,
                'path': item['path'], 'annotated_classes': item['classes'],
                'question': question, 'answer': answer['answer'], 'review_status': 'pending'})
            write_json(output / 'results.json', report)
        print(f'Pack {pack} complete; cache reload verified.', flush=True)
    rows = report['retrieval']
    report['summary'] = {'photos': len(selected), 'queries': len(rows),
        'top1_hits': sum(r['top1_correct'] for r in rows),
        'top3_hits': sum(r['top3_correct'] for r in rows),
        'top1_rate': sum(r['top1_correct'] for r in rows) / len(rows),
        'top3_rate': sum(r['top3_correct'] for r in rows) / len(rows),
        'caption_quality': 'visual review pending', 'weights_updated': False}
    write_json(output / 'results.json', report)
    with (output / 'retrieval.csv').open('w', newline='') as handle:
        fields = ['pack','category','expected_ids','top_image','top_score','top1_correct','top3_correct']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    print('Actual results:', report['summary'], flush=True)
    print('Saved:', output, flush=True)
    return report, output


if __name__ == '__main__':
    raise SystemExit('Import run_fresh_eval from your Colab cell and pass the loaded engine. This script does not load or train a model.')
