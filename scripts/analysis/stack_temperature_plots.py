"""Stack per-temperature plots side by side for visual comparison.

Given multiple ``temp_<T>`` directories, this script finds PNG files shared
across temperatures and creates one horizontal panel per plot name.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--temp-dir",
        action="append",
        default=[],
        help="Labelled input directory in the form T=DIR; repeat per temperature.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def _parse_temp_dir(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --temp-dir value '{spec}'; expected T=DIR")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label:
        raise ValueError(f"Invalid --temp-dir value '{spec}'; missing temperature label")
    return label, path


def _common_png_names(temp_dirs: list[Path]) -> list[str]:
    name_sets = []
    for d in temp_dirs:
        if not d.exists():
            name_sets.append(set())
            continue
        name_sets.append({p.name for p in d.glob("*.png") if p.is_file()})
    if not name_sets:
        return []
    common = set.intersection(*name_sets)
    return sorted(common)


def _stack_one(plot_name: str, labeled_dirs: list[tuple[str, Path]], out_dir: Path, dpi: int) -> None:
    images = []
    labels = []
    for label, d in labeled_dirs:
        p = d / plot_name
        if not p.exists():
            return
        images.append(mpimg.imread(p))
        labels.append(label)
    if not images:
        return

    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    max_h = max(heights)
    total_w = sum(widths)
    # Keep native-like aspect while avoiding huge figures.
    fig_w = max(12.0, min(36.0, total_w / 140.0))
    fig_h = max(4.0, min(18.0, max_h / 110.0))

    fig, axes = plt.subplots(1, len(images), figsize=(fig_w, fig_h))
    if len(images) == 1:
        axes = [axes]
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(f"T={label}", fontsize=11)
        ax.axis("off")
    fig.suptitle(plot_name, fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    out_path = out_dir / plot_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.temp_dir:
        raise ValueError("At least one --temp-dir T=DIR is required")

    labeled_dirs = [_parse_temp_dir(spec) for spec in args.temp_dir]
    temp_dirs = [d for _, d in labeled_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common_names = _common_png_names(temp_dirs)
    for name in common_names:
        _stack_one(name, labeled_dirs, args.output_dir, args.dpi)
    print(f"[stack] wrote {len(common_names)} comparison panel(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
