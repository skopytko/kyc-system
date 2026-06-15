#!/usr/bin/env python3
"""
Объективная оценка связи бинарных признаков из описания с отказом capture_validation.

Читает: models/model_borders/runs/russia_test_capture_vs_annotation_detail.csv

Для каждого признака:
  - таблица 2×2 (признак × отказ)
  - Fisher exact (двусторонний), OR SciPy
  - разница рисков Δ = P(отказ|признак=1) − P(отказ|признак=0)
  - bootstrap 95% ДИ для Δ и для log(OR) с псевдочастотами Haldane +0.5

Многомерно: sklearn LogisticRegression(L2, C=0.3) — при малых n без штрафа бывает полное разделение классов;
bootstrap ДИ для exp(β).

Запуск из корня репозитория:
  PYTHONPATH=. python scripts/russia_annotation_rejection_stats.py

По умолчанию пишется только один файл отчёта (.md). Табличные CSV — опционально:
  PYTHONPATH=. python scripts/russia_annotation_rejection_stats.py --export-csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression

CLUSTER_KEYS = [
    "перспектива_наклон",
    "много_фона",
    "блики_водяной_знак",
    "низкое_разрешение",
    "скан",
    "мятый_заломы_углов",
    "грязный_документ",
    "блюр_текста",
]

LABEL_SHORT = {
    "перспектива_наклон": "перспектива / наклон",
    "много_фона": "много фона",
    "блики_водяной_знак": "блики / водяной знак",
    "низкое_разрешение": "низкое разрешение",
    "скан": "скан",
    "мятый_заломы_углов": "мятый / заломы",
    "грязный_документ": "грязный документ",
    "блюр_текста": "блюр текста",
}


def load_arrays(csv_path: Path):
    import csv

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    y_reject = []
    X_list = []
    names = []
    for r in rows:
        names.append(r.get("имя_файла", "").strip())
        rej = str(r.get("capture_ok", "")).strip().lower() == "false"
        y_reject.append(int(rej))
        feats = []
        for k in CLUSTER_KEYS:
            try:
                feats.append(int(float(r.get(k, 0) or 0)))
            except ValueError:
                feats.append(0)
        X_list.append(feats)
    return np.asarray(y_reject, dtype=np.int64), np.asarray(X_list, dtype=np.float64), names


def contingency(feat: np.ndarray, y_reject: np.ndarray):
    """feat,y binary 0/1. Таблица [reject,accept]× не используем — считаем a,b,c,d."""
    fj = feat.astype(bool)
    rj = y_reject.astype(bool)
    a = np.sum(fj & rj)
    b = np.sum(fj & ~rj)
    c = np.sum(~fj & rj)
    d = np.sum(~fj & ~rj)
    return int(a), int(b), int(c), int(d)


def risk_diff(a: int, b: int, c: int, d: int) -> float:
    n1 = a + b
    n0 = c + d
    if n1 == 0 or n0 == 0:
        return float("nan")
    return a / n1 - c / n0


def or_haldane(a: int, b: int, c: int, d: int) -> float:
    ha, hb, hc, hd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (ha * hd) / (hb * hc)


def bootstrap_univariate(feat: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int = 8000):
    """Bootstrap CI для Δ и для OR (Haldane по выборке)."""
    n = len(y)
    deltas = []
    logs_or = []
    idx = np.arange(n)
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        a, b, c, d = contingency(feat[samp], y[samp])
        deltas.append(risk_diff(a, b, c, d))
        lor = np.log(or_haldane(a, b, c, d))
        logs_or.append(lor)
    deltas = np.asarray(deltas)
    logs_or = np.asarray(logs_or)
    q = (2.5, 97.5)
    d_lo, d_hi = np.nanpercentile(deltas, q)
    or_lo, or_hi = np.exp(np.nanpercentile(logs_or, q))
    return d_lo, d_hi, or_lo, or_hi


def md_row(cells, esc=False):
    return "| " + " | ".join(str(c) for c in cells) + " |\n"


def _verdict_significance(row: dict, alpha: float = 0.05) -> str:
    """Одна строка: формальная значимость + ориентир по bootstrap ДИ."""
    parts: list[str] = []
    p_s = row.get("p_Fisher_two_sided", "")
    p = float(p_s) if p_s not in ("", None) else float("nan")

    if np.isfinite(p):
        if p < alpha:
            parts.append(f"да, p<{alpha} (Фишер)")
        else:
            parts.append(f"нет при α={alpha} (Фишер)")

    lo_s, hi_s = row.get("delta_CI95_lo"), row.get("delta_CI95_hi")
    if lo_s not in ("", None) and hi_s not in ("", None):
        try:
            lf, hf = float(lo_s), float(hi_s)
            if lf * hf > 0:
                parts.append("ДИ для Δ не содержит 0 (bootstrap)")
            else:
                parts.append("ДИ для Δ содержит 0")
        except ValueError:
            pass

    b = int(row["b"])
    a = int(row["a"])
    if b == 0 and a > 0:
        parts.append("при «1» нет принятых (b=0)")
    return "; ".join(parts)


def _delta_abs_key(row: dict) -> float:
    d = row.get("delta_риска", "")
    try:
        return abs(float(d)) if d != "" else -1.0
    except ValueError:
        return -1.0


def build_summary_table_md(
    uni_rows: list,
    multi_lookup: dict[str, dict],
) -> tuple[str, list[dict]]:
    """Итоговая таблица: сила влияния (|Δ|) + значимость + многомерный ориентир."""
    rows_sorted = sorted(uni_rows, key=_delta_abs_key, reverse=True)

    lines = [
        "| Признак | Влияние Δ (п.п.) | 95% ДИ Δ | OR (как в детализации) | p (Фишер) | Значимость и устойчивость | exp(β), мног.* | 95% ДИ exp(β) |\n",
        "| --- | ---: | --- | ---: | ---: | --- | ---: | --- |\n",
    ]

    export: list[dict] = []
    alpha = 0.05
    for r in rows_sorted:
        key = r["признак"]
        label = LABEL_SHORT[key]
        delta = r.get("delta_риска", "")
        d_pp = f"{100 * float(delta):+.1f}" if delta != "" else "—"
        lo, hi = r.get("delta_CI95_lo"), r.get("delta_CI95_hi")
        di = (
            f"[{100 * float(lo):+.1f}; {100 * float(hi):+.1f}]"
            if lo not in ("", None) and hi not in ("", None)
            else "—"
        )

        a, b, c, d_i = int(r["a"]), int(r["b"]), int(r["c"]), int(r["d"])
        oddsr, _pf = fisher_exact([[a, b], [c, d_i]], alternative="two-sided")
        or_disp = f"{oddsr:.3f}" if np.isfinite(oddsr) else f"{or_haldane(a, b, c, d_i):.3f}†"

        p_disp = f"{float(r['p_Fisher_two_sided']):.4f}" if r.get("p_Fisher_two_sided") != "" else "—"
        verdict = _verdict_significance(r, alpha=alpha)

        m = multi_lookup.get(key) or {}
        eb = m.get("exp_beta")
        elo = m.get("exp_beta_CI95_lo")
        ehi = m.get("exp_beta_CI95_hi")
        eb_s = f"{float(eb):.3f}" if eb is not None else "—"
        eb_ci = (
            f"[{float(elo):.3f}; {float(ehi):.3f}]"
            if elo is not None and ehi is not None
            else "—"
        )

        lines.append(md_row([label, d_pp, di, or_disp, p_disp, verdict, eb_s, eb_ci]))

        export.append(
            {
                "признак": key,
                "признак_ру": label,
                "Δ_проц_пунктов": d_pp,
                "ДИ95_Δ": di,
                "OR_отображение": or_disp,
                "p_Fisher": p_disp,
                "вывод_значимость": verdict,
                "exp_beta_L2": eb_s,
                "ДИ95_exp_beta": eb_ci,
            }
        )

    lines.append(
        "\n*Столбец «exp(β), мног.»* — одна логистическая модель по **всем восьми** признакам сразу с **L2**, `C=0.3`; "
        "это ориентир «при прочих признаках в строке», не замена одномерной оценки.\n\n"
    )
    return "".join(lines), export


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Сборка russia_annotation_rejection_significance.md (пояснения + таблицы)."
    )
    ap.add_argument(
        "--export-csv",
        action="store_true",
        help="Дополнительно записать CSV с одномерной/итоговой/многомерной таблицами.",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    csv_path = repo / "models/model_borders/runs/russia_test_capture_vs_annotation_detail.csv"
    out_md = repo / "models/model_borders/runs/russia_annotation_rejection_significance.md"
    out_csv = repo / "models/model_borders/runs/russia_annotation_rejection_significance.csv"
    tpl_path = repo / "models/model_borders/runs/russia_annotation_significance_narrative.template.md"

    y, X, _names = load_arrays(csv_path)
    n = len(y)
    rng = np.random.default_rng(42)

    import csv as _csv

    uni_rows = []
    lines_univ = []
    lines_univ.append(
        "| Признак | a (1∩отказ) | b (1∩принято) | c (0∩отказ) | d (0∩принято) | "
        "Δ риска | 95% ДИ Δ | OR (Fisher) | 95% ДИ OR | p (Fisher) |\n"
    )
    lines_univ.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |\n")

    for j, key in enumerate(CLUSTER_KEYS):
        feat = X[:, j]
        a, b, c, d = contingency(feat, y)
        delta = risk_diff(a, b, c, d)
        d_lo, d_hi, or_lo, or_hi = bootstrap_univariate(feat, y, rng)
        oddsr, p_fish = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        # SciPy OR = (a*d)/(b*c) при нулях inf; для таблицы показываем Haldane point
        or_point = or_haldane(a, b, c, d)

        label = LABEL_SHORT[key]
        lines_univ.append(
            md_row(
                [
                    label,
                    a,
                    b,
                    c,
                    d,
                    f"{100 * delta:.1f} п.п." if not np.isnan(delta) else "—",
                    f"[{100 * d_lo:.1f}; {100 * d_hi:.1f}] п.п." if np.isfinite(d_lo) else "—",
                    f"{oddsr:.3f}" if np.isfinite(oddsr) else f"{or_point:.3f}†",
                    f"[{or_lo:.3f}; {or_hi:.3f}]",
                    f"{p_fish:.4f}",
                ]
            )
        )
        uni_rows.append(
            {
                "признак": key,
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "delta_риска": round(delta, 4) if np.isfinite(delta) else "",
                "delta_CI95_lo": round(d_lo, 4) if np.isfinite(d_lo) else "",
                "delta_CI95_hi": round(d_hi, 4) if np.isfinite(d_hi) else "",
                "OR_Fisher": oddsr if np.isfinite(oddsr) else "",
                "OR_Haldane": round(or_point, 6),
                "OR_CI95_lo": round(or_lo, 6),
                "OR_CI95_hi": round(or_hi, 6),
                "p_Fisher_two_sided": round(p_fish, 6) if np.isfinite(p_fish) else "",
            }
        )

    lines_univ.append(
        "\n*† в ячейке 2×2 был нуль; см. раздел про OR и поправку Хелдейна выше в этом же документе.*\n"
    )

    table_univ_md = "".join(lines_univ)

    clf = LogisticRegression(
        penalty="l2",
        C=0.3,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-6,
        random_state=42,
    )
    clf.fit(X, y)
    coef = clf.coef_.ravel()
    intercept = clf.intercept_[0]
    # bootstrap CI для coef
    n_boot = 8000
    coef_samp = []
    idx = np.arange(n)
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        cf = LogisticRegression(
            penalty="l2",
            C=0.3,
            solver="lbfgs",
            max_iter=2000,
            tol=1e-6,
            random_state=None,
        )
        try:
            cf.fit(X[samp], y[samp])
            coef_samp.append(cf.coef_.ravel())
        except Exception:  # noqa: BLE001
            coef_samp.append(np.full_like(coef, np.nan))
    coef_samp = np.asarray(coef_samp)
    q = (2.5, 97.5)

    lines_mv: list[str] = []
    lines_mv.append("| Признак | β (log-OR) | exp(β) | 95% ДИ exp(β) |\n")
    lines_mv.append("| --- | ---: | ---: | --- |\n")
    multi_rows = []
    for j, key in enumerate(CLUSTER_KEYS):
        b = coef[j]
        expb = np.exp(b)
        lo, hi = np.nanpercentile(np.exp(coef_samp[:, j]), q)
        lines_mv.append(
            md_row(
                [
                    LABEL_SHORT[key],
                    f"{b:.3f}",
                    f"{expb:.3f}",
                    f"[{lo:.3f}; {hi:.3f}]",
                ]
            )
        )
        multi_rows.append(
            {
                "признак": key,
                "beta": round(float(b), 6),
                "exp_beta": round(float(expb), 6),
                "exp_beta_CI95_lo": round(float(lo), 6),
                "exp_beta_CI95_hi": round(float(hi), 6),
            }
        )

    table_mv_md = "".join(lines_mv)
    multi_lookup = {mr["признак"]: mr for mr in multi_rows}
    summary_md, summary_export = build_summary_table_md(uni_rows, multi_lookup)

    if tpl_path.is_file():
        template = tpl_path.read_text(encoding="utf-8")
        template = template.replace("{{N}}", str(n))
        template = template.replace("{{DETAIL_CSV_REL}}", csv_path.relative_to(repo).as_posix())
        template = template.replace("<<<SUMMARY_TABLE>>>", summary_md)
        template = template.replace("<<<TABLE_UNIVARIATE>>>", table_univ_md)
        template = template.replace("<<<TABLE_MULTIVARIATE>>>", table_mv_md)
        template = template.replace("<<<INTERCEPT>>>", f"{intercept:.4f}")
        body = template
    else:
        body = "".join([
            "# (шаблон не найден)\n\n",
            table_univ_md,
            "\n---\n",
            table_mv_md,
            "\nintercept ",
            str(intercept),
        ])

    out_md.write_text(body, encoding="utf-8")

    if args.export_csv:
        with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(uni_rows[0].keys()))
            w.writeheader()
            w.writerows(uni_rows)

        multi_csv = out_csv.with_name("russia_annotation_rejection_significance_multivariate.csv")
        with multi_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = _csv.DictWriter(
                f,
                fieldnames=["признак", "beta", "exp_beta", "exp_beta_CI95_lo", "exp_beta_CI95_hi"],
            )
            w.writeheader()
            w.writerows(multi_rows)

        summary_csv = out_csv.with_name("russia_annotation_rejection_summary.csv")
        if summary_export:
            with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(summary_export[0].keys()))
                w.writeheader()
                w.writerows(summary_export)
        print(out_csv.resolve())
        if summary_export:
            print(summary_csv.resolve())
        print(multi_csv.resolve())

    print(out_md.resolve())


if __name__ == "__main__":
    main()
