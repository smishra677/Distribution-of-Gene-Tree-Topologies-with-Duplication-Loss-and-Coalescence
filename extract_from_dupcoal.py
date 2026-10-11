'''
 Code written by Sarthak Mishra. 
 Refactored and edited for readability with assistance from OpenAI Codex. 
'''

import csv
import gzip
from io import StringIO
from pathlib import Path

import ete3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo


BRANCHES = (
    "Tree1_parent", "Tree1_daughter",
    "Tree2_3_parent", "Tree2_3_daughter",
    "Tree4", "Tree6", "Tree8",
)
ROOT = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path("exp_all_results")
# Same lambda defaults as theory.py: 0.01 to 0.105, step 0.005.
FOLDERS = (
    (0.01, "dist_exte_0.01"),
    (0.015, "dist_exte_0.015"),
    (0.02, "dist_exte_0.02"),
    (0.025, "dist_exte_0.025"),
    (0.03, "dist_exte_0.03"),
    (0.035, "dist_exte_0.035"),
    (0.04, "dist_exte_0.04"),
    (0.045, "dist_exte_0.045"),
    (0.05, "dist_exte_0.05"),
    (0.055, "dist_exte_0.055"),
    (0.06, "dist_exte_0.06"),
    (0.065, "dist_exte_0.065"),
    (0.07, "dist_exte_0.07"),
    (0.075, "dist_exte_0.075"),
    (0.08, "dist_exte_0.08"),
    (0.085, "dist_exte_0.085"),
    (0.09, "dist_exte_0.09"),
    (0.095, "dist_exte_0.095"),
    (0.1, "dist_exte_0.1"),
)


def isequal(tree1, tree2):
    """Copied from dist_exte_1/exp.py."""
    if tree1 is None or tree2 is None:
        return False
    if tree1[-1] != ";":
        tree1 += ";"
    if tree2[-1] != ";":
        tree2 += ";"
    first_tree = ete3.Tree(tree1)
    second_tree = ete3.Tree(tree2)
    names1 = {node.name for node in first_tree.traverse() if node.name}
    names2 = {node.name for node in second_tree.traverse() if node.name}
    if len(tree1) <= 4 or len(tree2) <= 4:
        return names1 == names2
    rf, _, _, _, _, _, _ = first_tree.robinson_foulds(second_tree)
    return rf == 0 and names1 == names2


def add_matching_lengths(tree, topology, values):
    """The seven topology/clade tests from dist_exte_1/exp.py."""
    tests = []
    if isequal("((A_1,B_1),(A_2,B_2));", topology):
        tests.extend([
            (frozenset(("A_1", "B_1")), "Tree1_parent"),
            (frozenset(("A_2", "B_2")), "Tree1_daughter"),
        ])
    if isequal("((A_1,(A_2,B_2)),B_1);", topology):
        tests.extend([
            (frozenset(("A_1", "A_2", "B_2")), "Tree2_3_parent"),
            (frozenset(("A_2", "B_2")), "Tree2_3_daughter"),
        ])
    if isequal("(A_1,(B_1,B_2));", topology):
        tests.append((frozenset(("B_1", "B_2")), "Tree4"))
    if isequal("(B_2,(A_1,B_1));", topology):
        tests.append((frozenset(("A_1", "B_1")), "Tree6"))
    if isequal("(A_1,(A_2,B_1));", topology):
        tests.append((frozenset(("A_2", "B_1")), "Tree8"))

    # Keep every requested label for a clade.  This mirrors the independent
    # ``if`` blocks in exp.py even when ETE regards two target trees as equal.
    wanted = {}
    for leaves, branch in tests:
        wanted.setdefault(leaves, []).append(branch)
    for clade in tree.find_clades():
        if clade.name is None and clade.branch_length != 0.0:
            leaves = frozenset(leaf.name for leaf in clade.get_terminals())
            for branch in wanted.get(leaves, ()):
                values[branch].append(clade.branch_length)


def analyze_folder(folder, lam, raw_writer, summary_writer):
    print(f"Processing {folder} (lambda={lam:g})", flush=True)
    df = pd.read_csv(folder / "summary.csv", usecols=["replicate", "duplications"])
    ids = df.loc[df["duplications"] == 1, "replicate"].to_numpy()
    trees = (folder / "trees.tre").read_text().split(";")
    values = {branch: [] for branch in BRANCHES}

    for number, replicate in enumerate(ids, 1):
        newick = trees[int(replicate)]
        tree = Phylo.read(StringIO(newick), "newick")
        add_matching_lengths(tree, tree.format("newick"), values)
        if number % 10000 == 0:
            print(f"  duplication trees: {number:,}/{len(ids):,}", flush=True)

    # Save every value, not only the mean point.
    for branch in BRANCHES:
        array = np.asarray(values[branch], dtype=float)
        for value in array:
            raw_writer.writerow((f"{lam:g}", folder.name, branch, f"{value:.17g}"))
        mean = float(np.mean(array)) if len(array) else np.nan
        variance = float(np.var(array)) if len(array) else np.nan
        std = float(np.std(array)) if len(array) else np.nan
        summary_writer.writerow((f"{lam:g}", folder.name, branch, len(array), mean, variance, std))
    return values


def plot_curves(summary_path, output_path):
    data = pd.read_csv(summary_path)
    fig, ax = plt.subplots(figsize=(13, 8))
    for branch in BRANCHES:
        part = data[data["branch"] == branch].sort_values("lambda")
        # One marker is one folder; the line connects folder means.
        ax.plot(part["lambda"], part["mean"], marker="o", linewidth=2,
                label=branch)
    ax.set_xlabel(r"Duplication rate $\lambda$")
    ax.set_ylabel("Mean branch length")
    ax.set_title("Branch-length mean from each simulation folder")
    ax.set_xticks(sorted(data["lambda"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(folder_values, output_dir):
    labels = [f"{lam:g}\n{folder.name}" for lam, folder, _ in folder_values]
    fig, axes = plt.subplots(4, 2, figsize=(14, 17))
    for ax, branch in zip(axes.flat, BRANCHES):
        ax.boxplot([values[branch] for _, _, values in folder_values],
                   tick_labels=labels, showfliers=False)
        ax.set_title(branch)
        ax.set_xlabel(r"Duplication rate $\lambda$")
        ax.set_ylabel("Branch length")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "branch_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 8))
    centers = np.arange(len(folder_values), dtype=float)
    width = 0.10
    for j, branch in enumerate(BRANCHES):
        color = plt.get_cmap("tab10")(j)
        positions = centers + (j - 3) * width
        boxes = ax.boxplot([values[branch] for _, _, values in folder_values],
                           positions=positions, widths=width * 0.85,
                           showfliers=False, patch_artist=True, manage_ticks=False)
        for box in boxes["boxes"]:
            box.set_facecolor(color)
        ax.plot([], [], color=color, linewidth=8, label=branch)
    ax.set_xticks(centers, labels, rotation=45, fontsize=7)
    ax.set_xlabel(r"Duplication rate $\lambda$")
    ax.set_ylabel("Branch length")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "branch_boxplots_grouped.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    folders = [(lam, ROOT / name) for lam, name in FOLDERS]
    for _, folder in folders:
        if not (folder / "summary.csv").is_file() or not (folder / "trees.tre").is_file():
            raise FileNotFoundError(f"missing summary.csv or trees.tre in {folder}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / "all_branch_values.csv.gz"
    summary_path = OUTPUT_DIR / "folder_points.csv"
    folder_values = []
    with gzip.open(raw_path, "wt", newline="") as raw_file, summary_path.open("w", newline="") as summary_file:
        raw_writer = csv.writer(raw_file)
        summary_writer = csv.writer(summary_file)
        raw_writer.writerow(("lambda", "folder", "branch", "branch_length"))
        summary_writer.writerow(("lambda", "folder", "branch", "count", "mean", "variance", "std"))
        for lam, folder in folders:
            values = analyze_folder(folder, lam, raw_writer, summary_writer)
            folder_values.append((lam, folder, values))

    plot_path = OUTPUT_DIR / "seven_folder_point_curves.png"
    plot_curves(summary_path, plot_path)
    plot_boxplots(folder_values, OUTPUT_DIR)
    print(f"Raw values: {raw_path}")
    print(f"Folder points: {summary_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()