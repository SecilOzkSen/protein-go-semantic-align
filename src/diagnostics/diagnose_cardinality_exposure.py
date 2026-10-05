import pandas as pd

PATH = (
    "/workspace/data_pfresgo/diagnostics/"
    "go_specificity_shift_bp/"
    "protein_annotation_specificity.csv"
)

BINS = [0, 5, 10, 20, 40, 80, 160, float("inf")]

LABELS = [
    "1_5",
    "6_10",
    "11_20",
    "21_40",
    "41_80",
    "81_160",
    "161plus",
]

df = pd.read_csv(PATH)

df["card_bin"] = pd.cut(
    df["n_go"],
    bins=BINS,
    labels=LABELS,
    include_lowest=True,
)

summary = (
    df.groupby(
        ["split", "card_bin"],
        observed=False,
    )
    .size()
    .reset_index(name="n")
)

totals = (
    df.groupby("split")
    .size()
    .rename("total")
)

summary = summary.merge(
    totals,
    on="split",
)

summary["pct"] = (
        100.0
        * summary["n"]
        / summary["total"]
)

table_n = summary.pivot(
    index="card_bin",
    columns="split",
    values="n",
)

table_pct = summary.pivot(
    index="card_bin",
    columns="split",
    values="pct",
)

print("\n==============================")
print("PROTEIN COUNTS")
print("==============================")
print(table_n.to_string())

print("\n==============================")
print("PERCENT OF EACH SPLIT")
print("==============================")
print(
    table_pct.to_string(
        float_format=lambda x: f"{x:.2f}%"
    )
)

print("\n==============================")
print("HIGH FUNCTIONAL LOAD")
print("==============================")

for split in ["train", "valid", "test"]:
    sub = df[df["split"] == split]

    for threshold in [21, 41, 81, 161]:
        pct = (
                100.0
                * (sub["n_go"] >= threshold).mean()
        )

        print(
            f"{split:5s} "
            f"n_go >= {threshold:3d}: "
            f"{pct:6.2f}%"
        )