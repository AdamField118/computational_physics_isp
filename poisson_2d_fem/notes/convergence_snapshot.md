# Saved convergence snapshot

This table preserves every numerical array from the original convergence display, including the L-infinity values omitted from its visible table. It is an archived numerical snapshot, not a new validation run.

[JSON data](../results/convergence_snapshot.json)

| h | Nodes | Elements | L2 error | H1 error | Linf error | L2 rate | H1 rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.31623 | 13 | 16 | 0.002448 | 0.02458 | 0.4551 | — | — |
| 0.22361 | 23 | 28 | 0.001228 | 0.0174 | 0.5119 | 1.990628 | 0.996816 |
| 0.15811 | 41 | 64 | 0.0006147 | 0.0123 | 0.5058 | 1.996488 | 1.000746 |
| 0.1118 | 77 | 123 | 0.0003077 | 0.008697 | 0.502 | 1.996682 | 1.000120 |
| 0.07906 | 147 | 260 | 0.0001539 | 0.006152 | 0.5029 | 1.999461 | 0.999124 |

Rates use log(e_previous/e_current) / log(h_previous/h_current). The display compared L2 with an h² reference and H1 with an h reference, anchored at the first sample.

The H1 computation in the current Python driver contains a placeholder. The Linf values here stay near 0.5 rather than tending to zero. Neither issue is resolved by converting the data to Markdown. Keep this snapshot distinct from future corrected convergence runs.
