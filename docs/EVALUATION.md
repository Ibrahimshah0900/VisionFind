# Recorded evaluation

These results come from saved development runs. No model inference was
rerun while preparing this document.

## CLIP retrieval

| Evaluation | Result |
| --- | --- |
| Initial six-image retrieval sanity check | 6/6 expected top-1 matches |
| Twenty-image COCO128 baseline: top-1 | 14/20 (70%) |
| Twenty-image COCO128 baseline: top-3 | 17/20 (85%) |

The twenty-image baseline used two disjoint ten-image collections
and twenty category queries. Index cache reload was verified for both.

Recorded run: results/fresh_image_evaluation/20260930T171931_048075Z
in the development workspace.

Top-1 misses were bus, clock, bowl, oven, person, and refrigerator.
Bowl, oven, and person also missed the top three.

The six-image check is a small sanity check. The twenty-image result is
a small category-retrieval baseline, not a general accuracy estimate.
The images are not proven unseen by the pretrained models.

## Visual descriptions

Four caption samples from the twenty-image baseline were reviewed.
All four needed revision, including object identification and unsupported
visual details. Category annotations were not caption ground truth.

Later video-description experiments remained separate from the
accepted detection pipeline.

## Detection and playback

User playback reviews accepted person-box video playback and selected
integrated car and bottle detection clips during development.
These reviews were sample-specific; no general detection metric or
ground-truth IoU benchmark was recorded.

## Code checks

The eleven core regression scripts passed during repository preparation.
The separate fresh-evaluation plumbing test also passed.

The checks cover routing, upload handling, retrieval/cache behavior,
voice callbacks, video handling, detection evidence, and API behavior.
They primarily use fixtures or mocked inference.
