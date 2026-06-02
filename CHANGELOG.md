# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-01

First public release: the deterministic core, with no LLM required.

### Added
- `analyze(data, question)` and a `statskeptic analyze` CLI that profile a dataset,
  plan a vetted analysis, run it, critique it against a methodological rubric, revise
  what can be fixed, and report what the data shows and what it cannot conclude.
- Vetted statistical routines, each validated against hand-computed answers: Student's
  t, Welch's t, Mann-Whitney U, Pearson, Spearman, chi-square, Fisher's exact, one-way
  ANOVA, Kruskal-Wallis, OLS, and logistic regression. Every result carries an effect
  size, a confidence interval where one is defined, and the computation that produced it.
- A deterministic critique engine and revision loop covering assumption violations,
  multiple comparisons, confounding, low power, data leakage, and outlier sensitivity.
- Robust CSV reading via CleverCSV (dialect, quoting, and encoding detection), with
  infinities treated as missing data.
- Meaningful CLI exit codes (defensible / cannot-conclude / declined / usage / internal).

### Notes
- The model never produces a number; every statistic originates in executed, re-runnable
  code. "The data cannot support a reliable answer" is a first-class success state.
- An optional LLM critic and clinical/financial domain packs are planned; the check
  registry and planner are built as the extension points for them.
