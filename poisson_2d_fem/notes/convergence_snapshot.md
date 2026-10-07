# Saved convergence snapshot

Saved convergence data, including L², H¹, and Linf errors. These values need the qualifications below before they can be used to assess the solver.

[JSON data](../results/convergence_snapshot.json)

| h | Nodes | Elements | L2 error | H1 error | Linf error | L2 rate | H1 rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.31623 | 13 | 16 | 0.002448 | 0.02458 | 0.4551 | — | — |
| 0.22361 | 23 | 28 | 0.001228 | 0.0174 | 0.5119 | 1.990628 | 0.996816 |
| 0.15811 | 41 | 64 | 0.0006147 | 0.0123 | 0.5058 | 1.996488 | 1.000746 |
| 0.1118 | 77 | 123 | 0.0003077 | 0.008697 | 0.502 | 1.996682 | 1.000120 |
| 0.07906 | 147 | 260 | 0.0001539 | 0.006152 | 0.5029 | 1.999461 | 0.999124 |

Rates use log(e_previous/e_current) / log(h_previous/h_current). For comparison curves, anchor h² and h references at the first L2 and H1 samples.

The H1 computation in the current Python driver contains a placeholder. The Linf values here stay near 0.5 rather than tending to zero. These data do not establish convergence in either of those norms.
