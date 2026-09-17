### egy_test

| Model | n | JSON valid | Intent acc | Slot F1 | Exact match | 95% CI | EM clean | p50 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E-D-camelbert-da | 200 | 1.00 | 0.825 | 0.665 | **0.625** | [0.56, 0.69] | 0.608 | 0.02 |
| D-qwen3-17b | 200 | 1.00 | 0.795 | 0.682 | **0.590** | [0.52, 0.66] | 0.569 | — |
| C-qwen3-17b | 200 | 1.00 | 0.775 | 0.649 | **0.575** | [0.51, 0.64] | 0.547 | — |
| E-C-camelbert-da | 200 | 1.00 | 0.805 | 0.628 | **0.565** | [0.49, 0.63] | 0.541 | 0.02 |
| D-qwen3-06b | 200 | 1.00 | 0.730 | 0.599 | **0.510** | [0.44, 0.57] | 0.492 | 0.55 |
| C-qwen3-06b | 200 | 1.00 | 0.725 | 0.588 | **0.485** | [0.42, 0.56] | 0.453 | 0.55 |
| A_sonnet5_zeroshot | 200 | 1.00 | 0.785 | 0.503 | **0.440** | [0.38, 0.51] | 0.420 | — |
| A_sonnet5_5shot | 200 | 1.00 | 0.815 | 0.500 | **0.435** | [0.37, 0.50] | 0.409 | — |
| B-qwen3-06b-5shot | 200 | 1.00 | 0.260 | 0.100 | **0.040** | [0.01, 0.07] | 0.039 | 1.06 |

19 items overlap training data and are excluded from EM clean.

### massive_ar_test

| Model | n | JSON valid | Intent acc | Slot F1 | Exact match | 95% CI | EM clean | p50 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C-qwen3-17b | 300 | 1.00 | 0.857 | 0.707 | **0.607** | [0.56, 0.66] | 0.581 | — |
| E-D-camelbert-da | 1000 | 1.00 | 0.835 | 0.684 | **0.586** | [0.56, 0.62] | 0.564 | 0.02 |
| D-qwen3-17b | 300 | 1.00 | 0.840 | 0.697 | **0.583** | [0.53, 0.64] | 0.559 | — |
| E-C-camelbert-da | 1000 | 1.00 | 0.829 | 0.665 | **0.567** | [0.54, 0.60] | 0.542 | 0.01 |
| C-qwen3-06b | 1000 | 1.00 | 0.791 | 0.678 | **0.563** | [0.53, 0.59] | 0.538 | 0.57 |
| D-qwen3-06b | 1000 | 1.00 | 0.783 | 0.652 | **0.539** | [0.51, 0.57] | 0.512 | 1.63 |
| A_sonnet5_5shot | 300 | 1.00 | 0.813 | 0.569 | **0.450** | [0.40, 0.50] | 0.434 | — |
| A_sonnet5_zeroshot | 300 | 1.00 | 0.807 | 0.554 | **0.447** | [0.39, 0.50] | 0.430 | — |
| B-qwen3-06b-5shot | 1000 | 1.00 | 0.310 | 0.122 | **0.030** | [0.02, 0.04] | 0.024 | 0.98 |

254 items overlap training data and are excluded from EM clean.

### Paired differences (exact match)

| Comparison | Eval set | n | Δ EM | 95% CI | Significant | Only A | Only B |
|---|---|---:|---:|---:|---|---:|---:|
| D-qwen3-06b − C-qwen3-06b | egy_test | 200 | +0.025 | [-0.030, +0.080] | no | 18 | 13 |
| D-qwen3-06b − C-qwen3-06b | massive_ar_test | 1000 | -0.024 | [-0.045, -0.004] | yes | 45 | 69 |
| E-D-camelbert-da − E-C-camelbert-da | egy_test | 200 | +0.060 | [+0.015, +0.105] | yes | 16 | 4 |
| E-D-camelbert-da − E-C-camelbert-da | massive_ar_test | 1000 | +0.019 | [-0.001, +0.039] | no | 59 | 40 |
| E-D-camelbert-da − D-qwen3-06b | egy_test | 200 | +0.115 | [+0.055, +0.175] | yes | 33 | 10 |
| E-D-camelbert-da − D-qwen3-06b | massive_ar_test | 1000 | +0.047 | [+0.017, +0.078] | yes | 136 | 89 |
| D-qwen3-17b − C-qwen3-17b | egy_test | 200 | +0.015 | [-0.030, +0.060] | no | 11 | 8 |
| D-qwen3-17b − C-qwen3-17b | massive_ar_test | 300 | -0.023 | [-0.060, +0.017] | no | 15 | 22 |
| D-qwen3-17b − D-qwen3-06b | egy_test | 200 | +0.080 | [+0.020, +0.145] | yes | 26 | 10 |
| D-qwen3-17b − D-qwen3-06b | massive_ar_test | 287 | +0.049 | [+0.003, +0.098] | yes | 31 | 17 |
| A_sonnet5_5shot − A_sonnet5_zeroshot | egy_test | 200 | -0.005 | [-0.055, +0.040] | no | 12 | 13 |
| A_sonnet5_5shot − A_sonnet5_zeroshot | massive_ar_test | 300 | +0.003 | [-0.040, +0.043] | no | 21 | 20 |
| D-qwen3-17b − A_sonnet5_5shot | egy_test | 200 | +0.155 | [+0.085, +0.225] | yes | 44 | 13 |
| D-qwen3-17b − A_sonnet5_5shot | massive_ar_test | 300 | +0.133 | [+0.073, +0.197] | yes | 66 | 26 |
| E-D-camelbert-da − A_sonnet5_5shot | egy_test | 200 | +0.190 | [+0.125, +0.260] | yes | 47 | 9 |
| E-D-camelbert-da − A_sonnet5_5shot | massive_ar_test | 287 | +0.077 | [+0.014, +0.139] | yes | 53 | 31 |
