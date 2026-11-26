"""Render audited local protein/GO evidence without changing model scores.

The input is the JSON produced by this add-on's extractor. No model, network
request, or additional inference is used here. Heatmap cells display raw
projected cosine similarities, not probabilities or residue importance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import textwrap

import numpy as np


FORMAT = "mzsgo_fa_local_evidence_v1"
MAX_CASES = 2
MAX_FEATURE_LABELS = 8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_integer(value, name):
    require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


def _finite(value, name):
    require(not isinstance(value, bool) and isinstance(value, (int, float)),
            f"{name} must be a finite number")
    require(math.isfinite(float(value)), f"{name} must be finite")
    return float(value)


def _vector(value, name, size=2):
    require(isinstance(value, list) and len(value) == size,
            f"{name} must contain exactly {size} values")
    return np.array([_finite(x, name) for x in value], dtype=float)


def verified_features(case):
    """Only render feature coordinates whose sequence mapping was verified."""
    return [feature for feature in case.get("annotations", [])
            if feature.get("sequence_verified") is True]


def validate_evidence(data):
    """Validate the scientific display contract before creating any output."""
    require(isinstance(data, dict) and data.get("format") == FORMAT,
            f"Expected evidence format {FORMAT}")
    _positive_integer(data.get("n_candidates"), "n_candidates")
    _positive_integer(data.get("window_length"), "window_length")
    _positive_integer(data.get("window_stride"), "window_stride")
    require(_finite(data.get("mil_tau"), "mil_tau") > 0, "mil_tau must be positive")
    require(data.get("evaluation_mode") in ("general", "temporal"),
            "evaluation_mode must be general or temporal")
    cases = data.get("cases")
    require(isinstance(cases, list) and 1 <= len(cases) <= MAX_CASES,
            f"Render one or two cases at a time; received {len(cases) if isinstance(cases, list) else 'invalid cases'}")
    ids = []
    for case in cases:
        require(isinstance(case, dict), "Each case must be an object")
        for name in ("case_id", "protein_id", "go_id"):
            require(isinstance(case.get(name), str) and case[name], f"Missing case {name}")
        ids.append(case["case_id"])
        require(re.fullmatch(r"GO:\d{7}", case["go_id"]) is not None, "Expected canonical GO identifier")
        length = _positive_integer(case.get("protein_length"), "protein_length")
        rank = _positive_integer(case.get("rank"), "rank")
        require(rank <= data["n_candidates"], "Rank is outside the scored candidate universe")
        _finite(case.get("score_raw"), "score_raw")
        require(case.get("segment_names") == ["name", "definition"],
                "Segment order must be name, definition")
        present = case.get("segment_present")
        require(isinstance(present, list) and len(present) == 2
                and all(type(x) is bool for x in present) and present[0],
                "A name segment is required; segment_present must contain two booleans")
        weights = _vector(case.get("segment_weights"), "segment_weights")
        require(bool(np.all(weights >= 0)) and abs(float(weights.sum()) - 1) <= 1e-5,
                "GO segment weights must be nonnegative and sum to one")
        require(all(p or abs(float(w)) <= 1e-7 for p, w in zip(present, weights)),
                "Missing segments must receive zero weight")
        segment_scores = _vector(case.get("segment_scores"), "segment_scores")
        require(abs(float(weights @ segment_scores) - float(case["score_raw"])) <= 1e-5,
                "Weighted segment scores do not reproduce the final raw score")
        windows = case.get("windows")
        require(isinstance(windows, list) and windows, "Every case needs valid protein windows")
        centers, responsibility = [], []
        for window in windows:
            require(isinstance(window, dict), "Each window must be an object")
            start = _positive_integer(window.get("start"), "window start")
            end = _positive_integer(window.get("end"), "window end")
            require(start <= end <= length, "Window coordinates must be 1-based inclusive and within the protein")
            count = _positive_integer(window.get("valid_residue_count"), "valid_residue_count")
            require(count <= end - start + 1, "Valid residue count exceeds window coordinates")
            similarities = _vector(window.get("similarities"), "window similarities")
            require(bool(np.all(np.abs(similarities) <= 1.01)),
                    "Expected cosine similarities within numerical tolerance of [-1, 1]")
            resp = _vector(window.get("mil_responsibility"), "mil_responsibility")
            require(bool(np.all(resp >= 0)) and bool(np.all(resp <= 1 + 1e-6)),
                    "MIL responsibilities must be between zero and one")
            responsibility.append(resp)
            centers.append((start + end) / 2)
        require(len(centers) == 1 or bool(np.all(np.diff(centers) > 0)),
                "Windows must have strictly increasing centers")
        totals = np.sum(responsibility, axis=0)
        for index, is_present in enumerate(present):
            require((abs(float(totals[index]) - 1) <= 1e-4) if is_present
                    else min(abs(float(totals[index])), abs(float(totals[index]) - 1)) <= 1e-4,
                    "MIL responsibilities must sum to one per present segment")
        parity = case.get("parity")
        require(isinstance(parity, dict), "Missing replay parity information")
        delta = _finite(parity.get("max_abs_score_difference"), "max_abs_score_difference")
        require(0 <= delta <= 1e-5, "Replay parity did not meet the 1e-5 score tolerance")
        annotations = case.get("annotations", [])
        require(isinstance(annotations, list) and all(isinstance(f, dict) for f in annotations),
                "annotations must be a list of feature objects")
        for feature in verified_features(case):
            start = _positive_integer(feature.get("start"), "feature start")
            end = _positive_integer(feature.get("end"), "feature end")
            require(start <= end <= length, "Verified feature is outside the evaluated protein sequence")
            require(isinstance(feature.get("label"), str) and feature["label"],
                    "Verified features need labels")
            require(isinstance(feature.get("source"), str) and feature["source"],
                    "Verified features need an annotation source")
    require(len(ids) == len(set(ids)), "Duplicate case IDs")
    return data


def window_edges(case):
    """Display overlapping windows at their centers, on sequence coordinates."""
    windows = case["windows"]
    centers = np.array([(w["start"] + w["end"]) / 2 for w in windows], dtype=float)
    if len(centers) == 1:
        return np.array([windows[0]["start"] - .5, windows[0]["end"] + .5])
    return np.r_[windows[0]["start"] - .5, (centers[:-1] + centers[1:]) / 2,
                 windows[-1]["end"] + .5]


def _display_limits(cases):
    values = []
    for case in cases:
        for window in case["windows"]:
            values.extend(value for value, present in zip(window["similarities"], case["segment_present"])
                          if present)
    lo, hi = min(values), max(values)
    if hi - lo < .01:
        lo, hi = lo - .025, hi + .025
    lo = max(-1., math.floor(lo * 20) / 20)
    hi = min(1., math.ceil(hi * 20) / 20)
    if lo >= hi:  # Round-off at a cosine boundary.
        lo, hi = max(-1., lo - .05), min(1., hi + .05)
    return lo, hi


def _feature_text(case):
    features = verified_features(case)
    hidden = len(case.get("annotations", [])) - len(features)
    if not features:
        message = "No sequence-verified region annotations supplied"
    else:
        message = f"{len(features)} sequence-matched feature annotation(s)"
    if hidden:
        message += f"; {hidden} unverified feature(s) hidden"
    return message


def _draw_features(ax, case, panel_label):
    from matplotlib.patches import Rectangle

    length = case["protein_length"]
    features = sorted(verified_features(case), key=lambda f: (f["start"], f["end"], f["label"]))
    ax.set_xlim(.5, length + .5)
    ax.set_ylim(-.35, 1.4)
    ax.axhline(.1, color="#475569", lw=2.5, zorder=1)
    colors = ["#2563eb", "#d97706", "#059669", "#9333ea"]
    last_end = [-math.inf] * 3
    for index, feature in enumerate(features):
        available = [lane for lane, end in enumerate(last_end) if end < feature["start"]]
        lane = available[0] if available else int(np.argmin(last_end))
        last_end[lane] = feature["end"] + max(5, length * .025)
        y = .1 + lane * .28
        width = max(1, feature["end"] - feature["start"] + 1)
        ax.add_patch(Rectangle((feature["start"] - .5, y - .065), width, .13,
                               color=colors[lane], alpha=.8, zorder=3))
        if index < MAX_FEATURE_LABELS:
            label = textwrap.shorten(feature["label"], width=37, placeholder="...")
            ax.text((feature["start"] + feature["end"]) / 2, y + .085, label,
                    fontsize=7, ha="center", va="bottom", clip_on=True, color=colors[lane])
    ax.text(0, 1.04, f"{panel_label}  {case['protein_id']}  |  {length:,} amino acids",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=11, weight="bold")
    ax.text(0, -.05, _feature_text(case), transform=ax.transAxes, fontsize=7.5,
            va="top", color="#475569")
    if len(features) > MAX_FEATURE_LABELS:
        ax.text(1, 1.04, f"First {MAX_FEATURE_LABELS} feature labels shown",
                transform=ax.transAxes, ha="right", fontsize=7, color="#475569")
    ax.set_axis_off()


def _draw_info(ax, case, n_candidates):
    ax.set_axis_off()
    ax.set_facecolor("#f8fafc")
    ax.text(0, 1, case["go_id"], fontsize=11, weight="bold", va="top")
    name = case.get("go_name") or "GO name not supplied"
    name_lines = textwrap.wrap(str(name), width=36)[:3]
    ax.text(0, .90, "\n".join(name_lines), fontsize=9, va="top", linespacing=1.25)
    y = .89 - .065 * len(name_lines) - .06
    rows = [
        ("Full-universe rank", f"{case['rank']:,} / {n_candidates:,}"),
        ("Final raw score", f"{case['score_raw']:.4f}"),
        ("GO weights, name / def.", " / ".join(f"{v:.3f}" for v in case["segment_weights"])),
        ("MIL scores, name / def.", " / ".join(f"{v:.4f}" for v in case["segment_scores"])),
    ]
    for label, value in rows:
        ax.text(0, y, label, fontsize=8.2, va="top", color="#475569")
        ax.text(.99, y, value, fontsize=8.2, va="top", ha="right", weight="bold")
        y -= .105
    positive = case.get("label") == 1 or case.get("annotated_positive") is True
    status = "Annotated positive in evaluation data" if positive else "Annotation status not supplied"
    ax.text(0, y, status, fontsize=7.7, va="top", color="#166534" if positive else "#475569")
    y -= .10
    group = str(case.get("selection_group", "unspecified")).replace("_", " ")
    ax.text(0, y, f"Selection: {group}", fontsize=7.5, va="top", color="#475569")
    y -= .09
    ax.text(0, y, f"Replay score difference: {case['parity']['max_abs_score_difference']:.2g}",
            fontsize=7.3, va="top", color="#475569")


def build_caption(data, limits):
    """Return a factual caption and display notes, never a localization claim."""
    count = len(data["cases"])
    lines = [
        "Local protein evidence for GO retrieval.",
        f"{count} illustrative protein-GO case(s) from the {data['evaluation_mode']} evaluation are shown. "
        "Cases follow the recorded selection policy; this figure is not a localization benchmark.",
        f"Protein windows have nominal length {data['window_length']} and stride {data['window_stride']} "
        "amino acids; partial terminal windows retain their recorded valid coordinates. "
        "Each heatmap cell shows the raw projected cosine similarity between one protein window and "
        "the target GO name or definition representation. Overlapping windows are displayed at their "
        "sequence-coordinate centers, rather than as non-overlapping sequence partitions. "
        f"All panels share one color scale ({limits[0]:.3f} to {limits[1]:.3f}); no row-specific normalization is applied. "
        "An absent definition is shown in grey.",
        f"SmoothMIL combines window scores separately for each text segment (temperature {data['mil_tau']:g}), "
        "then GO-text-conditioned weights combine the two segment scores. These GO weights do not depend "
        "on the protein. Scores are not calibrated probabilities or residue-level biological importance.",
        f"Ranks use the complete scored universe of {data['n_candidates']:,} GO candidates. "
        "They are distinct from any temporal-only Fmax/AUPR calculation. "
        "The underlying final scores passed replay parity against the saved benchmark scores at tolerance 1e-5.",
    ]
    for letter, case in zip("AB", data["cases"]):
        name = case.get("go_name") or "name not supplied"
        lines.append(f"{letter}: {case['protein_id']}, {case['go_id']} ({name}); "
                     f"rank {case['rank']:,}, raw score {case['score_raw']:.6f}. {_feature_text(case)}.")
        definition = case.get("go_definition")
        if isinstance(definition, str) and definition.strip():
            lines.append(f"  Canonical GO definition for reference: {definition.strip()}")
        for feature in verified_features(case):
            lines.append(f"  Feature: {feature['start']}-{feature['end']} {feature['label']}; "
                         f"type={feature.get('type', 'unspecified')}; "
                         f"evidence={feature.get('evidence_kind', 'unspecified')}; source={feature['source']}.")
        if len(verified_features(case)) > MAX_FEATURE_LABELS:
            lines.append(f"  All verified feature intervals are drawn; only the first {MAX_FEATURE_LABELS} "
                         "features in coordinate order receive text labels.")
    if any(verified_features(case) for case in data["cases"]):
        lines.append("Feature coordinates are displayed only when sequence_verified=true. "
                     "Sequence matching validates coordinate mapping, not the biological evidence of an annotation; "
                     "the supplied evidence type and source are retained above.")
    else:
        lines.append("This rendering is a model-evidence preview. No functional domain boundaries were inferred "
                     "from the heatmap or added without independently supplied annotations.")
    return "\n\n".join(lines) + "\n"


def render_evidence(evidence, out_dir):
    """Create PDF, SVG, PNG and caption from audited JSON or a parsed object."""
    if isinstance(evidence, (str, Path)):
        data = json.loads(Path(evidence).read_text(encoding="utf-8"))
    else:
        data = evidence
    validate_evidence(data)
    out_dir = Path(out_dir).expanduser().resolve()
    require(not out_dir.exists() or (out_dir.is_dir() and not any(out_dir.iterdir())),
            f"Output directory must be new or empty: {out_dir}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.ticker import MaxNLocator

    cases = data["cases"]
    limits = _display_limits(cases)
    figure_height = 3.45 * len(cases) + 1.45
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.linewidth": .7, "savefig.facecolor": "white"}):
        fig = plt.figure(figsize=(12, figure_height))
        outer = GridSpec(len(cases), 1, figure=fig, left=.09, right=.975,
                         top=.86 if len(cases) == 1 else .90, bottom=.25 if len(cases) == 1 else .19,
                         hspace=.60)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#e5e7eb")
        norm = Normalize(vmin=limits[0], vmax=limits[1])
        heatmap = None
        for index, case in enumerate(cases):
            inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[index],
                                           width_ratios=[2.5, 1.4], height_ratios=[1, 1.05],
                                           hspace=.36, wspace=.23)
            track = fig.add_subplot(inner[0, 0])
            _draw_features(track, case, "AB"[index])
            ax = fig.add_subplot(inner[1, 0])
            values = np.array([window["similarities"] for window in case["windows"]], dtype=float).T
            mask = np.repeat(~np.array(case["segment_present"], dtype=bool)[:, None], values.shape[1], axis=1)
            matrix = np.ma.array(values, mask=mask)
            heatmap = ax.pcolormesh(window_edges(case), np.array([-.5, .5, 1.5]), matrix,
                                   shading="flat", cmap=cmap, norm=norm, rasterized=False)
            ax.set_xlim(.5, case["protein_length"] + .5)
            ax.set_ylim(1.5, -.5)
            ax.set_yticks([0, 1], ["Name", "Definition" if case["segment_present"][1] else "Definition\n(absent)"])
            ax.set_xlabel("Window center along the protein (amino-acid position)", fontsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=2))
            ax.tick_params(axis="both", labelsize=8, length=3)
            for spine in ax.spines.values():
                spine.set_visible(False)
            _draw_info(fig.add_subplot(inner[:, 1]), case, data["n_candidates"])
        fig.suptitle("Local protein evidence for GO retrieval", x=.09, y=.975,
                     ha="left", fontsize=15, weight="bold")
        fig.text(.09, .930 if len(cases) == 1 else .944,
                 "Illustrative cases selected by a recorded policy; not a localization benchmark.",
                 ha="left", va="top", fontsize=9, color="#475569")
        color_y = .125 if len(cases) == 1 else .095
        color_ax = fig.add_axes([.22, color_y, .43, .022])
        bar = fig.colorbar(heatmap, cax=color_ax, orientation="horizontal")
        bar.set_label("Raw window-GO cosine similarity, shared scale", fontsize=8, labelpad=4)
        bar.ax.tick_params(labelsize=7)
        fig.text(.09, .012, "GO segment weights depend on GO text only. Heatmap values are model similarities, not probabilities.",
                 ha="left", va="bottom", fontsize=8, color="#475569")
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        try:
            for extension in ("pdf", "svg", "png"):
                target = out_dir / f"local_evidence.{extension}"
                fig.savefig(target, dpi=300, bbox_inches="tight", pad_inches=.12)
                files[extension] = str(target)
        finally:
            plt.close(fig)
        caption = out_dir / "captions.txt"
        caption.write_text(build_caption(data, limits), encoding="utf-8")
        files["caption"] = str(caption)
    return {"files": files, "n_cases": len(cases), "shared_color_limits": list(limits),
            "verified_feature_counts": {case["case_id"]: len(verified_features(case)) for case in cases},
            "feature_annotations_inferred_from_heatmap": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    result = render_evidence(args.evidence, args.out_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
