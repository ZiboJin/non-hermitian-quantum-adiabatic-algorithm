# Figure 7: success probability under noise

| File | Contents |
| --- | --- |
| `realizations.csv` | 4,500 records: 3 algorithms, 15 noise strengths and 100 realizations per group |
| `summary.csv` | 45 statistical summaries |
| `baselines.csv` | No-noise baseline for each algorithm, extracted from the realization records |

## Settings

The graph is CK `G_2`, with `n=r=5`, `L=70`, `p=2`, `q=4`, `gamma=10`, `Omega=1` and full dimension 2,272. The FK Hamiltonian has no extra input term.

Each trajectory uses 70 independent piecewise-constant Ginibre noise segments, 8 integration substeps per segment, the `combined` integrator and 64 spectral-norm power iterations. Noise strengths are `1e-15,...,1e-1`.

The master seed is 1234. With `seed_mode=paired_noise`, algorithms share the same derived seed for a given noise strength and realization index. Different realization indices use independent samples.

## Fields

- `algorithm`: `NHMIS-FKQAA`, `NHMIS-HDQAA` and `HMIS-HDQAA` correspond to FK QAA, HD QAA and HM QAA.
- `epsilon`, `realization`, `seed`: noise strength, realization index (0-99) and random seed.
- `p_raw`: success probability plotted in Figure 7.
- `p0`: no-noise baseline; `p_normalized=p_raw/p0`.
- `n_real`: sample count requested by the originating run; `n_success` in the summary is the final group count.
- `n_noise`, `steps_per_segment`, `integrator`, `norm_iters`: numerical settings.
- `status`, `error_message`, `runtime_seconds`: run diagnostics.

The figure plots `median_p_raw`, with shading between `min_p_raw` and `max_p_raw`. The inset uses the same quantities. Quartiles and normalized statistics are also present in the summary.
