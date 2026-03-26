# Decision grid-search (offline)

Using `inference_results_train.json` and `inference_results_test.json` probability curves.
All metrics are computed on penetrated holes only (true_label=1).

## Best on train

| method | n_pen | ≤3% | ≤5% | >10% | params | evals |
|---|---:|---:|---:|---:|---|---:|
| topkmedian_unc | 820 | 55.0 | 68.3 | 20.4 | `{"k": 7, "min_thresh": 0.3, "beta": 0.25, "gate_mode": "pred", "var_t...` | 498 |
| argmax_unc | 820 | 53.7 | 67.9 | 20.7 | `{"min_thresh": 0.7, "beta": 1.5, "gate_mode": "win", "var_thresh": 5e...` | 220 |
| two_stage_unc | 820 | 54.8 | 67.9 | 20.6 | `{"region_thresh": 0.65, "min_len": 2, "peak_thresh": 0.5, "beta": 2.0...` | 409 |
| first_thresh_unc | 820 | 53.5 | 67.0 | 20.7 | `{"thresh": 0.75, "beta": 1.25, "gate_mode": "pred", "var_thresh": 2e-...` | 174 |
| centroid_unc | 820 | 51.5 | 66.6 | 21.0 | `{"thresh": 0.7, "beta": 1.75, "gate_mode": "win", "var_thresh": 1e-05...` | 169 |
| smooth_first_unc | 820 | 50.4 | 63.9 | 25.0 | `{"window": 29, "thresh": 0.3, "beta": 0.25, "gate_mode": "pred", "var...` | 344 |

## Best on test

| method | n_pen | ≤3% | ≤5% | >10% | params | evals |
|---|---:|---:|---:|---:|---|---:|
| smooth_first_unc | 205 | 47.8 | 67.8 | 25.9 | `{"window": 21, "thresh": 0.45, "beta": 0.0, "gate_mode": "win", "var_...` | 495 |
| topkmedian_unc | 205 | 47.8 | 67.8 | 25.9 | `{"k": 11, "min_thresh": 0.65, "beta": 0.75, "gate_mode": "win", "var_...` | 59 |
| first_thresh_unc | 205 | 47.8 | 67.3 | 26.3 | `{"thresh": 0.6, "beta": 0.25, "gate_mode": "pred", "var_thresh": 2e-0...` | 155 |
| centroid_unc | 205 | 52.7 | 66.8 | 26.3 | `{"thresh": 0.9, "beta": 1.0, "gate_mode": "pred", "var_thresh": 2e-05...` | 289 |
| argmax_unc | 205 | 53.2 | 64.9 | 27.3 | `{"min_thresh": 0.35, "beta": 2.25, "gate_mode": "win", "var_thresh": ...` | 72 |
| two_stage_unc | 205 | 44.9 | 64.4 | 26.3 | `{"region_thresh": 0.85, "min_len": 4, "peak_thresh": 0.75, "beta": 0....` | 443 |

## Best on combined (weighted by n_penetrated)

| method | n_pen | ≤3% | ≤5% | >10% | params | evals |
|---|---:|---:|---:|---:|---|---:|
| topkmedian_unc | 1025 | 54.4 | 67.6 | 21.7 | `{"k": 7, "min_thresh": 0.3, "beta": 0.25, "gate_mode": "pred", "var_t...` | 498 |
| argmax_unc | 1025 | 53.7 | 67.2 | 22.0 | `{"min_thresh": 0.35, "beta": 2.25, "gate_mode": "win", "var_thresh": ...` | 72 |
| centroid_unc | 1025 | 51.1 | 66.3 | 22.0 | `{"thresh": 0.7, "beta": 1.75, "gate_mode": "win", "var_thresh": 1e-05...` | 169 |
| first_thresh_unc | 1025 | 52.2 | 65.9 | 22.0 | `{"thresh": 0.75, "beta": 1.25, "gate_mode": "pred", "var_thresh": 2e-...` | 174 |
| two_stage_unc | 1025 | 51.4 | 65.5 | 21.8 | `{"region_thresh": 0.65, "min_len": 2, "peak_thresh": 0.5, "beta": 2.0...` | 409 |
| smooth_first_unc | 1025 | 42.6 | 63.2 | 23.2 | `{"window": 5, "thresh": 0.6, "beta": 1.0, "gate_mode": "win", "var_th...` | 20 |

**Best combined (≤5%)**: `topkmedian_unc`  ≤5=67.61%  params={'k': 7, 'min_thresh': 0.3, 'beta': 0.25, 'gate_mode': 'pred', 'var_thresh': 0.002, 'win': 5, 'gate_action': 'veto', 'fallback': 'earliest_topk', 'fallback_thresh': 0.75, 'fallback_k': 5, 'fallback_min_thresh': 0.75}
