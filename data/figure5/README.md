# Figure 5: success probability versus problem size

| File | Contents |
| --- | --- |
| `ck_results.csv` | CK results for `m=3,...,12`, or `n=9,13,...,45` |
| `graph_pool.jsonl` | The 2,000 CK-like graph instances used in the calculation: 200 at each size |
| `graph_pool_summary.csv` | One-row-per-graph index of the same instances |
| `ck_like_results.csv` | 6,000 results: one per graph and algorithm |
| `ck_like_summary.csv` | Median, minimum and maximum for each algorithm and size |

## Settings and plotted quantities

All algorithms use `gamma=10`, `p=2`, `q=4`, `Omega=1` and `r=n`. FK evolution uses exponential midpoint propagation: 16 steps per clock link for CK and 8 for CK-like graphs, with at least 200 steps.

`p_FK`, `p_NHHD` and `p_HHD` in the CK table correspond to FK QAA, HD QAA and HM QAA. The CK-like figure uses `median_success`, `min_success` and `max_success`; shading is the min-max range. HM uses the right logarithmic axis, and FK/HD use the left linear axis.

## Graph instances

`m` determines the CK parent graph, with `n=4*m-3`. Each record stores its exact `deleted_cross_edges` in `(left vertex, right triangle, vertex within triangle)` notation. The function `ck_like_edges()` in `code/generate_ck_like_random_deletion_graphs.py` reconstructs the full edge set.

- `graph_id`, `graph_hash`, `sample_id`: instance identifiers linking the graph pool and result tables.
- `seed`, `delete_probability`, `probability_bucket_index`, `attempt_index`: generation parameters.
- `x_star_bits`, `x_star_index`, `mis_size`: unique MIS, its little-endian integer index and its size.
- `num_edges`, `r`, `L`: edge count, circuit rounds and circuit length; `L=r*(n+num_edges)`.
- `success_probability`: individual algorithm result.
- `n_samples`: number of instances in a summary group.

Use the supplied graph pool to reproduce Figure 5 with the same instances. The generator also supports creating new pools with different parameters or seeds.
