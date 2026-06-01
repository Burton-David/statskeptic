# statskeptic

**A data-analysis agent that red-teams its own conclusions.**

Give it a dataset and a question. statskeptic profiles the data, picks a vetted
statistical method, runs it, and then turns on the result: it attacks its own analysis
against a methodological rubric (assumption violations, multiple comparisons,
confounding, underpowered samples, data leakage, outlier sensitivity), revises what it
can, and reports what the data shows **and what it cannot conclude**.

Two rules make it different from the fluent-but-wrong tools it competes with:

1. **The model never produces a number.** Every statistic comes from real, tested code
   (scipy / statsmodels) and ships with the exact call that produced it, so any figure
   can be re-run and checked. statskeptic selects methods and interprets them; it does
   not invent them.
2. **"Cannot conclude" is a success state.** Over-claiming is the cardinal sin here.
   When the data does not support a reliable answer, statskeptic says so plainly, and a
   non-zero exit code lets a pipeline act on it.

## A trap a naive tool walks into

`examples/skewed_trial.csv` is a two-arm trial where recovery time is heavily
right-skewed and there is no real difference between the arms. Point a tool that reaches
straight for a t-test at it and you get a confident false positive: `p = 0.014`,
"significant," ship it.

```
$ statskeptic analyze examples/skewed_trial.csv -q "Does the drug reduce recovery hours?"

## Mann-Whitney U
comparing 'recovery_hours' across 'arm': two groups, so a t-test is the usual first pass

- Result: U = 814, p = 0.110 (not significant at alpha=0.05)
- Effect: rank_biserial_r = -0.196
- location shift (drug - placebo): 95% CI [-17.5, 1.1]
- n = 90

### Revisions
- Switched from Student's t-test to Mann-Whitney U (assumption.normality): data is
  non-normal; the rank-based test is valid here. p 0.014 -> 0.110.

### Objections raised
- None outstanding.

## What this cannot conclude
- Nothing beyond the assumptions and caveats noted above.
```

statskeptic planned the same t-test a careful analyst would reach for first, then its
normality check fired, the revision loop switched to the rank-based test, and the
"significant" result evaporated. The audit trail shows the switch and the p-value before
and after. The false positive never leaves the building.

## What it catches

Each objection is grounded in the actual numbers and carries a concrete remedy. Some are
fixed automatically by re-running; others can only be flagged, and those push the verdict
toward "cannot conclude."

| Objection | What fires it | What statskeptic does |
| --- | --- | --- |
| Non-normality | Shapiro plus a real skew magnitude, not a trivial deviation | switch to the rank test (Mann-Whitney, Kruskal-Wallis, Spearman) |
| Unequal variance | Levene on a pooled-variance t-test | switch to Welch's t-test |
| Sparse contingency cells | expected counts below Cochran's threshold | switch a 2x2 to Fisher's exact test |
| Multiple comparisons | many tests run against one outcome | apply a Holm correction and re-read significance |
| Confounding | a causal question on observational data | name a candidate confounder; refuse the causal claim |
| Low power | a non-significant result where only a large effect was detectable | report the minimum detectable effect; refuse to read "no effect" |
| Data leakage | an identifier used as a predictor | drop it and re-fit |
| Outlier sensitivity | dropping extreme points flips significance | switch to a rank-based test |

The vetted toolset covers two-group comparisons (Student's t, Welch, Mann-Whitney),
k-group comparisons (one-way ANOVA, Kruskal-Wallis), association (Pearson, Spearman,
chi-square, Fisher's exact), and regression (OLS, logistic). Each routine reports an
effect size and, where one is defined, a confidence interval, and lists the assumptions
it checked against your data.

## Install

```
git clone https://github.com/Burton-David/statskeptic
cd statskeptic
pip install -e .
```

Python 3.10 or newer. The core needs no API key and makes no network calls.

## Usage

```
statskeptic analyze data.csv --question "Does the treatment change recovery?"
```

Options:

- `--json` emits the full typed report, every number traceable to its computation.
- `--outcome`, `--group` / `--by`, `--predictors` name columns when the question is
  ambiguous (the planner declines rather than guess).
- `--alpha` sets the significance level (default 0.05).
- `--quiet` suppresses the report body and returns only the exit code.

Exit codes make it scriptable as a gate:

| code | meaning |
| --- | --- |
| 0 | a defensible result (with caveats counts as defensible) |
| 2 | the data cannot support a reliable answer |
| 3 | the question does not map to a vetted method |
| 64 | usage error (bad flags, missing file, unknown column) |
| 70 | a statistical routine failed and the cause is reported, not hidden |

As a library:

```python
from statskeptic import analyze

report = analyze("data.csv", "Does exercise cause better health?")
print(report.explain())     # markdown
report.to_json()            # the full typed report
report.verdict              # defensible / defensible_with_caveats / cannot_conclude / declined
```

## Try the planted-trap corpus

`examples/` ships five datasets, each with one planted flaw, generated by a seeded script
so the numbers above are reproducible (`python examples/make_demo_data.py`):

```
statskeptic analyze examples/biomarker_screen.csv  -q "Which markers are associated with the outcome?"
statskeptic analyze examples/exercise_health.csv   -q "Does more exercise cause a better health score?"
statskeptic analyze examples/small_trial.csv       -q "Does the treatment change the test score?"
statskeptic analyze examples/clean_ab_test.csv     -q "Does the variant change order value?"
```

The biomarker screen finds 4 markers significant at `p<0.05`, then a Holm correction
across the 24 tests leaves only the one real signal standing. The exercise question
reports a strong correlation and still refuses to call it causal, naming age as the
likely confounder. The small trial returns "cannot conclude": at nine per arm, only a
large effect was ever detectable. The clean A/B test returns a plain, defensible yes.

## Honest limits

- Causal critique is a flag, not an engine. statskeptic names a candidate confounder and
  declines the causal claim; it does not estimate causal effects.
- The rule-based planner maps a question to a method by keywords and column structure. It
  declines ambiguous questions rather than guess, so you may need `--outcome` / `--group`
  to point it at the right columns.
- Independence is assumed and stated, not tested. It is a property of the study design,
  which the data alone cannot reveal.
- An optional LLM critic (for context-specific objections the static rubric cannot
  encode) and clinical / financial domain packs are planned extensions, not yet shipped.
  The check registry and the planner are built as the seams for them.

## License

MIT.
