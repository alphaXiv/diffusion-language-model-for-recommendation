import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Masked denoising for movie recommendation

    This notebook is a self-contained walkthrough of a compact reproduction
    of **Diffusion Language Model for Recommendation** (arXiv:2607.21519).
    It opens with the already-produced Kubernetes evidence—nothing here
    requires retraining—and then explains the split, matched models,
    robustness test, and mechanism ablations.

    **Verdict: partially reproduced.** One-pass masked denoising strongly
    beat left-to-right generation, but iterative correction, progressive
    corruption, and stability voting did not behave as claimed in this
    reconstruction.

    Formal runs used Kubernetes on NVIDIA RTX PRO 6000 Blackwell GPUs,
    peaked at 16 concurrent GPUs, and spanned 0.1153 wall hours (6m55s)
    from first pod start to last pod finish.
    """)
    return


@app.cell
def _():
    results = {
        "Autoregressive": {
            "ndcg": [0.00714714, 0.00871069, 0.00756362, 0.00806048],
            "recall": [0.01561810, 0.01821192, 0.01583885, 0.01694261],
            "retention": [1.10547, 1.02906, 0.94200, 0.99272],
            "jaccard": [0.58371, 0.56901, 0.57360, 0.56815],
        },
        "Full 3-step diffusion": {
            "ndcg": [0.00911320, 0.00959021, 0.00661523, 0.00830479],
            "recall": [0.01561810, 0.01727373, 0.01324503, 0.01539735],
            "retention": [1.00314, 0.99786, 0.98932, 0.99978],
            "jaccard": [0.33271, 0.39725, 0.34649, 0.40914],
        },
        "One-step diffusion": {
            "ndcg": [0.01634483, 0.01762312, 0.01124030, 0.01592819],
            "recall": [0.03283665, 0.03460265, 0.02461369, 0.03394040],
            "retention": [1.0, 1.0, 1.0, 1.0],
            "jaccard": [0.52933, 0.55916, 0.56836, 0.48818],
        },
        "Five-step diffusion": {
            "ndcg": [0.00921320, 0.00979255, 0.00651812, 0.00827122],
            "recall": [0.01611479, 0.01766004, 0.01302428, 0.01528698],
            "retention": [0.98878, 0.98800, 0.99626, 1.00380],
            "jaccard": [0.32421, 0.38755, 0.34945, 0.41030],
        },
        "No collaborative tokens": {
            "ndcg": [0.00797401, 0.00883684, 0.00793891, 0.00750706],
            "recall": [0.01600442, 0.01688742, 0.01749448, 0.01368653],
            "retention": [0.98981, 0.99622, 0.99147, 0.99940],
            "jaccard": [0.26738, 0.31089, 0.28084, 0.40111],
        },
        "No progressive curriculum": {
            "ndcg": [0.00622882, 0.00753768, 0.00954721, 0.01066144],
            "recall": [0.01181015, 0.01528698, 0.01859823, 0.02036424],
            "retention": [1.00121, 0.99874, 1.00549, 0.99356],
            "jaccard": [0.64796, 0.42253, 0.34933, 0.41327],
        },
        "No stability voting": {
            "ndcg": [0.00897085, 0.00957055, 0.00654647, 0.00824183],
            "recall": [0.01545254, 0.01721854, 0.01313466, 0.01517660],
            "retention": [1.00116, 0.99741, 1.00371, 1.00320],
            "jaccard": [0.33098, 0.39656, 0.35306, 0.40856],
        },
        "Plain diffusion": {
            "ndcg": [0.00687185, 0.00638602, 0.01199552, 0.00890827],
            "recall": [0.01545254, 0.01330022, 0.02251656, 0.01738411],
            "retention": [1.00699, 1.00173, 0.99853, 1.00726],
            "jaccard": [0.76691, 0.64936, 0.41055, 0.41231],
        },
    }
    return (results,)


@app.cell
def _(mo):
    metric = mo.ui.dropdown(
        options={
            "NDCG@10": "ndcg",
            "Recall@10": "recall",
            "Early-error retention": "retention",
            "Top-10 consistency": "jaccard",
        },
        value="NDCG@10",
        label="Metric",
    )
    metric
    return (metric,)


@app.cell
def _():
    import math
    import statistics

    def summarize(values):
        return statistics.mean(values), statistics.stdev(values)

    def bars_svg(rows, title, percent=False):
        width, height = 760, 410
        left, bottom, top = 175, 355, 65
        values = [row[1] for row in rows]
        vmax = max(values) * 1.15
        pieces = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fbfaf7"/>',
            f'<text x="24" y="32" font-family="system-ui" font-size="20" font-weight="700">{title}</text>',
        ]
        for i, (label, value, sd) in enumerate(rows):
            y = top + i * 39
            bar_w = 500 * value / vmax
            color = "#e87547" if label == "One-step diffusion" else "#3d7b8c"
            pieces += [
                f'<text x="{left - 10}" y="{y + 16}" text-anchor="end" font-family="system-ui" font-size="12">{label}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_w}" height="23" rx="3" fill="{color}"/>',
                f'<text x="{left + bar_w + 8}" y="{y + 16}" font-family="system-ui" font-size="12" font-weight="700">{value * 100:.1f}%</text>'
                if percent
                else f'<text x="{left + bar_w + 8}" y="{y + 16}" font-family="system-ui" font-size="12" font-weight="700">{value:.5f}</text>',
            ]
        pieces.append("</svg>")
        return "".join(pieces)

    return bars_svg, summarize


@app.cell
def _(bars_svg, metric, mo, results, summarize):
    labels = [
        "Autoregressive",
        "One-step diffusion",
        "Full 3-step diffusion",
        "Five-step diffusion",
        "No collaborative tokens",
        "No progressive curriculum",
        "No stability voting",
        "Plain diffusion",
    ]
    selected = metric.value
    rows = [(name, *summarize(results[name][selected])) for name in labels]
    mo.Html(
        bars_svg(
            rows,
            {
                "ndcg": "Full-catalog NDCG@10",
                "recall": "Full-catalog Recall@10",
                "retention": "Downstream NDCG retained after an early error",
                "jaccard": "Stochastic top-10 Jaccard consistency",
            }[selected],
            percent=selected == "retention",
        )
    )
    return (rows,)


@app.cell
def _(mo, rows):
    mo.ui.table(
        [
            {"condition": name, "mean": round(mean, 6), "seed SD": round(sd, 6)}
            for name, mean, sd in rows
        ],
        pagination=False,
        selection=None,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The test in plain language

    A causal recommender sees a history and chooses the first future movie,
    then treats that choice as context for the second, and so on. A masked
    denoiser instead starts with three blank future positions and predicts
    them with bidirectional attention. Both models here use the same
    two-layer, 128-wide transformer and exactly **893,568 parameters**.

    MovieLens-1M is treated as implicit feedback. For every user, the final
    three timestamped interactions are held out and everything before them
    is training history. Ranking considers all unseen catalog items rather
    than a sampled set of negatives.

    The paper's absolute MovieLens result—DLMRec NDCG@10 0.1113—is not a
    target for this model. Its 8B language model uses item titles, learned
    stochastic tokens, and a different global timepoint split. The
    reproduction isolates the generative paradigm at small scale.
    """)
    return


@app.cell
def _(mo, results, summarize):
    ar = summarize(results["Autoregressive"]["ndcg"])[0]
    one = summarize(results["One-step diffusion"]["ndcg"])[0]
    _full_headline = summarize(results["Full 3-step diffusion"]["ndcg"])[0]
    mo.md(
        f"""
        ## 2. Headline result

        One-step masked denoising reaches **{one:.5f} NDCG@10**, compared with
        **{ar:.5f}** for autoregression: a **{(one / ar - 1) * 100:.1f}%**
        increase. Every matched seed difference is positive, and the paired 95%
        interval for the absolute change is `[0.00335, 0.01148]`.

        Three-step diffusion reaches **{_full_headline:.5f}**, only
        **{(_full_headline / ar - 1) * 100:.1f}%** above autoregression; its paired
        interval crosses zero. More passes are not automatically better because
        this reconstruction feeds hard predictions back into the model, which
        can preserve mistakes as easily as it corrects them.
        """
    )
    return


@app.cell
def _(mo, results, summarize):
    _full_mechanism = summarize(results["Full 3-step diffusion"]["ndcg"])[0]
    no_graph = summarize(results["No collaborative tokens"]["ndcg"])[0]
    no_curr = summarize(results["No progressive curriculum"]["ndcg"])[0]
    no_vote = summarize(results["No stability voting"]["ndcg"])[0]
    vote_j = summarize(results["Full 3-step diffusion"]["jaccard"])[0]
    no_vote_j = summarize(results["No stability voting"]["jaccard"])[0]
    mo.md(
        f"""
        ## 3. Do the mechanisms explain quality?

        - **Collaborative tokens:** full NDCG is **{(_full_mechanism/no_graph-1)*100:.1f}%**
          above the no-token ablation, but one of four seeds reverses the effect.
        - **Progressive curriculum:** removing it gives {no_curr:.5f}, versus
          {_full_mechanism:.5f} with it. The claimed benefit is not visible.
        - **Stability voting:** removing it gives {no_vote:.5f}. Consistency is
          {vote_j:.3f} with voting and {no_vote_j:.3f} without it, so voting is
          neutral under this diagnostic.

        These are paired interventions: architecture, data, optimizer, and seed
        stay fixed. The wide four-seed intervals still make small effects
        uncertain.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The robustness stress test

    At the first decoded position, evaluation replaces the top prediction
    with the tenth-ranked item. It then evaluates NDCG on the remaining two
    held-out positions. Autoregression retains 101.7% of clean downstream
    NDCG and three-step diffusion 99.5%; values around 100% are ranking
    noise. The diffusion model corrects the deliberately wrong token 0% of
    the time under hard updates.

    This is useful negative evidence. It says the compact hard-update
    mechanism does not recover early errors; it does not establish that the
    paper's confidence-guided soft update fails in its original 8B system.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. What would make this faithful?

    A full reproduction should use the released stochastic tokenizer,
    three learned collaborative hops, item text in the hybrid prompt,
    LLaDA-8B with LoRA, the paper's global timepoint split, and
    confidence-guided soft embedding updates. It should report both
    full-catalog and paper-protocol rankings across more seeds.

    The bounded conclusion here is narrower: **bidirectional masked
    prediction can substantially outperform causal generation at the same
    compact parameter budget, but this implementation does not support the
    claimed iterative correction, curriculum, or voting mechanisms.**
    """)
    return


if __name__ == "__main__":
    app.run()
