# Badcase report

## Summary (penetrated holes only)

```json
{
  "train": {
    "n_penetrated": 820,
    "pct_within_3": 55.00000000000001,
    "pct_within_5": 68.29268292682927,
    "pct_over_10": 20.365853658536583,
    "n_missing_pred_or_true": 0,
    "n_within_3": 451,
    "n_within_5": 560,
    "n_over_10": 167
  },
  "test": {
    "n_penetrated": 205,
    "pct_within_3": 52.19512195121951,
    "pct_within_5": 64.8780487804878,
    "pct_over_10": 26.82926829268293,
    "n_missing_pred_or_true": 0,
    "n_within_3": 107,
    "n_within_5": 133,
    "n_over_10": 55
  },
  "combined": {
    "n_penetrated": 1025,
    "pct_within_3": 54.43902439024391,
    "pct_within_5": 67.60975609756098,
    "pct_over_10": 21.658536585365855,
    "n_missing_pred_or_true": 0,
    "n_within_3": 558,
    "n_within_5": 693,
    "n_over_10": 222
  },
  "decision_override": {
    "method": "topkmedian_unc",
    "best_params": {
      "k": 7,
      "min_thresh": 0.3,
      "beta": 0.25,
      "gate_mode": "pred",
      "var_thresh": 0.002,
      "win": 5,
      "gate_action": "veto",
      "fallback": "earliest_topk",
      "fallback_thresh": 0.75,
      "fallback_k": 5,
      "fallback_min_thresh": 0.75
    },
    "metrics": {
      "n_penetrated": 1025,
      "pct_within_3": 54.43902439024391,
      "pct_within_5": 67.60975609756098,
      "pct_over_10": 21.65853658536585
    },
    "source": "decision_grid_gate_hard/decision_grid_results_combined.json"
  },
  "notes": {
    "metric_scope": "penetrated holes only (true_label=1)",
    "error_definition": "abs(pred_idx-true_idx) if both exist else 999"
  }
}
```

## Files
- `scatter_*.png`: true vs pred scatter plots
- `err_hist_*.png`: error histograms
- `err_cdf_*.png`: error CDF plots
- `badcases_top.csv` / `badcases_top.json`: worst cases table
- `curves/*.png`: probability curves for selected cases (true/pred markers)
- `good_curves/*.png`: good-case curves (error <= 3) with prob/mean/var if available