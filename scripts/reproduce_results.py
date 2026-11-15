"""Recompute main benchmark summaries from the published fixed predictions."""
from pathlib import Path
import argparse
import importlib.util
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


def read(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def classification(labels, scores):
    return {"auc": float(roc_auc_score(labels, scores)),
            "ap": float(average_precision_score(labels, scores))}


def el_results(root):
    directory = root / "Data_2/EL/pooled_tests_20261109"
    frame = read(directory / "hla63_scores.csv.gz")
    clean = frame.loc[frame.pair_clean.eq(True)].copy()
    assert len(clean) == 168086 and clean.label.sum() == 6146 and clean.mhc.nunique() == 63
    reference = read(directory / "pooled_common_metrics.csv")
    models = reference.model.tolist()
    common = clean.loc[clean[models].notna().all(axis=1)]
    assert len(common) == 143546 and common.label.sum() == 5277 and common.mhc.nunique() == 55
    rows = []
    for scope, population in [("source_clean_supported", clean), ("common_55", common)]:
        for model in models:
            matched = population.loc[population[model].notna()]
            labels = matched.label.to_numpy()
            scores = matched[model].to_numpy()
            metrics = classification(labels, scores)
            order = np.argsort(-scores, kind="stable")
            metrics["ppvn"] = float(labels[order[:int(labels.sum())]].mean())
            row = dict(scope=scope, model=model, n=len(matched), positive=int(labels.sum()),
                       n_hla=int(matched.mhc.nunique()), **metrics)
            if scope == "common_55":
                expected = reference.loc[reference.model.eq(model)].iloc[0]
                for metric in metrics:
                    np.testing.assert_allclose(row[metric], expected[metric], atol=1e-12, rtol=0)
            rows.append(row)
    return rows


def ba_results(root):
    directory = root / "Data_2/current_figure/ba"
    frame = read(directory / "predictions1541.csv")
    assert len(frame) == 1541 and frame.mhc.nunique() == 32
    reference = read(directory / "pooled_three_metrics_ci95.csv")
    threshold = 1 - np.log(500) / np.log(50000)
    binders = frame.affinity.ge(threshold)
    assert binders.sum() == 580
    rows = []
    for model in frame.columns[3:]:
        scores = frame[model]
        row = {"model": model, "n": len(frame), "binders": int(binders.sum()),
               "pearson": float(pearsonr(frame.affinity, scores).statistic),
               "spearman": float(spearmanr(frame.affinity, scores).statistic),
               "mae": float(mean_absolute_error(frame.affinity, scores)),
               **classification(binders, scores)}
        for metric in ["pearson", "spearman"]:
            expected = reference.loc[reference.model.eq(model) & reference.metric.eq(metric), "estimate"].item()
            np.testing.assert_allclose(row[metric], expected, atol=1e-12, rtol=0)
        rows.append(row)
    return rows


def structure_results(root):
    directory = root / "Data_3/current_figure/analysis"
    frame = read(directory / "paired_target_measurements.csv")
    reference = read(directory / "paired_comparisons.csv")
    ours = frame.loc[frame.model.eq("ours")].set_index("pdb_id").sort_index()
    assert len(ours) == 37
    rows = []
    for model, records in frame.groupby("model", sort=False):
        paired = records.set_index("pdb_id").reindex(ours.index)
        assert len(records) == 37 and paired[["ca", "allatom", "seconds"]].notna().all().all()
        row = {"model": model, "n": 37}
        for metric in ["ca", "allatom", "seconds"]:
            row[metric + "_median"] = float(paired[metric].median())
            if model != "ours":
                p = float(wilcoxon(ours[metric] - paired[metric], alternative="two-sided").pvalue)
                expected = reference.loc[reference.model.eq(model) & reference.metric.eq(metric), "paired_wilcoxon_p"].item()
                np.testing.assert_allclose(p, expected, atol=1e-12, rtol=0)
                row[metric + "_paired_wilcoxon_p"] = p
        row["median_target_paired_speedup"] = float((paired.seconds / ours.seconds).median())
        rows.append(row)
    return rows


def imm_results(root):
    directory = root / "Data_6"
    reproduction = directory / "reproduction"
    spec = importlib.util.spec_from_file_location("prism_released_imm", reproduction / "imm_ensemble.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, ensemble = module.read_spec(directory / "ensemble.json")
    result = {"cohorts": [], "patient_metrics": {}}
    for cohort in ["Mix", "P", "Patients"]:
        pairs = read(reproduction / f"{cohort}_pairs.csv.gz")
        scores = module.combine_logits(np.load(reproduction / f"{cohort}_member_logits.npy"), ensemble)
        if cohort == "Patients":
            assignments = read(reproduction / "Patients_assignments.csv.gz", dtype={"patient": str})
            scored = module.aggregate_patients(assignments, pairs, scores)
            keys = ["patient", "peptide"]
        else:
            scored = pairs.assign(score=scores)
            keys = ["peptide", "allele"]
        reference = read(directory / f"application/{cohort}_scores.csv.gz", dtype={"patient": str})
        joined = reference[keys + ["label", "PrismMHC-IMM"]].merge(
            scored[keys + ["score"]], on=keys, how="left", validate="one_to_one", sort=False)
        assert len(scored) == len(reference) and joined.score.notna().all()
        maximum_error = float(np.max(np.abs(joined.score - joined["PrismMHC-IMM"])))
        assert maximum_error < 1e-12
        result["cohorts"].append({"cohort": cohort, "n": len(joined),
                                  "positive": int(joined.label.sum()), "maximum_score_error": maximum_error,
                                  **classification(joined.label, joined.score)})
        if cohort == "Patients":
            assert len(joined) == 243764 and joined.label.sum() == 100 and joined.patient.nunique() == 70
            per_patient = []
            for patient, group in joined.groupby("patient", sort=False):
                ranked = group.sort_values("score", ascending=False, kind="stable")
                positive = int(group.label.sum())
                row = {"patient": patient, "positive": positive,
                       "ap": float(average_precision_score(group.label, group.score)) if positive else None}
                for k in [20, 50, 100]:
                    row[f"hits{k}"] = int(ranked.iloc[:k].label.sum())
                per_patient.append(row)
            patient_frame = pd.DataFrame(per_patient)
            positive_frame = patient_frame.loc[patient_frame.positive.gt(0)]
            assert len(positive_frame) == 53
            metrics = {"patients": 70, "positive_patients": 53,
                       "mean_ap": float(positive_frame.ap.mean()), "median_ap": float(positive_frame.ap.median())}
            for k, expected in [(20, 53), (50, 73), (100, 86)]:
                metrics[f"hits{k}"] = int(patient_frame[f"hits{k}"].sum())
                metrics[f"success{k}"] = int(positive_frame[f"hits{k}"].gt(0).sum())
                assert metrics[f"hits{k}"] == expected
            np.testing.assert_allclose(metrics["mean_ap"], 0.2667157655467254, atol=1e-12, rtol=0)
            result["patient_metrics"] = metrics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/supplementary_data")
    parser.add_argument("--section", choices=["all", "el", "ba", "structure", "imm"], default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    functions = {"el": el_results, "ba": ba_results, "structure": structure_results, "imm": imm_results}
    selected = functions if args.section == "all" else {args.section: functions[args.section]}
    result = {"status": "PASS", "sections": {name: run(args.data_dir.resolve()) for name, run in selected.items()}}
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content)


if __name__ == "__main__":
    main()

