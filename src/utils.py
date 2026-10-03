"""Search visualization and experiment export utilities."""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

def plot_search_results(query, results, project_root, columns=3):
    """Create a labeled image grid in retrieval order."""
    if results.empty:
        raise ValueError("There are no results to display.")
    if not isinstance(columns, int) or columns < 1:
        raise ValueError("columns must be a positive integer.")

    cols = min(columns, len(results))
    rows = (len(results) + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 4 * rows),
        squeeze=False, layout="constrained"
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, row in zip(axes.flat, results.itertuples(index=False)):
        path = Path(project_root) / row.relative_path

        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))

        ax.set_title(
            f"{row.rank}. {row.filename}\n"
            f"Cosine similarity: {row.similarity:.4f}"
        )

    fig.suptitle(f"Query: {query}", fontsize=14)
    return fig


def save_search_run(query, results, figure, output_root, manifest, top_k):
    """Save a search table, experiment metadata, and image preview."""
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    results.assign(query=query).to_csv(
        run_dir / "results.csv",
        index=False,
        float_format="%.9g",
    )

    record = {
        "query": query,
        "requested_top_k": top_k,
        "returned_count": len(results),
        "score_type": "cosine_similarity",
        "created_at_utc": created_at.isoformat(),
        "index_manifest": manifest,
    }
    (run_dir / "run.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    figure.savefig(
        run_dir / "preview.png",
        dpi=200, bbox_inches="tight", facecolor="white"
    )
    return run_dir
