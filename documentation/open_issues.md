# Open issues

Everything here is a known gap, not a bug report against work already verified.
Each item says what is wrong or missing, why it matters, and what would close it.
Ordered by how much it can change a number the paper quotes.

---

## 1. Calibration

### 1.1 `theta = 1` is a placeholder, and one result depends on it discontinuously

`load_parameters.jl` sets `theta = 1.` with `1.768` commented out beside it, and
`nu_s = 1.5` for every sector. The variety-weight tail index is
`kappa_s = theta/(nu_s - 1)`, so the current calibration gives `kappa = 2`
**exactly** — the boundary at which `Xi(kappa) = Gamma(1-2/kappa)/Gamma(1-1/kappa)^2`
diverges and the variance of the variety expenditure weights ceases to be finite.

Consequences, in order of severity:

- The closed form `V = Xi(kappa_s)/N_s` has no content at this calibration.
  Section 5.1 says so and measures `V` from the draws instead. That is honest but
  it is a workaround, not a result.
- At `theta = 1.768`, `kappa = 3.56` and `Xi` is finite. The closed form becomes
  usable and the paragraph can be written on it rather than around it.
- `theta*alpha` (0.228 auto / 0.458 aero) rescales, so **every** `(alpha*theta)^2`
  statement moves: the level and cross-industry ranking of `1 - C`, the implied
  geometric term `<V>`, the distance-equivalence figure, and the `dH/d(alpha*theta)`
  derivative.

**To close**: decide `theta`, re-run, re-read every number in §4 and §5.1. The user
has explicitly deferred this. Nothing downstream should be treated as final until it
is done.

### 1.2 `nu_s` is uniform at 1.5 and unestimated

Both `theta` and `nu_s` are set by hand, and `kappa` depends on their ratio. A
sector-varying `nu_s` would give sector-varying `kappa`, which is what the
variety-tail table is built to report. At present that table has ten identical rows.

---

## 2. Measurement still open

### 2.1 `E[omega] = gamma` is asserted nowhere because it is not measured

`expected_network_check` reports a total-variation distance of 0.19 (auto) and 0.24
(aero) between the mean realised incidence and the sourcing probability `gamma`. Two
things are mixed in that number and they cannot be separated as it stands:

- **the CES cross term** — a cheaper winner takes a larger expenditure share, so the
  value-weighted network is genuinely not the count-weighted one;
- **sampling noise** at 50 replications, which enters a TV distance with a `1/sqrt(B)`
  bias that does not average out.

Raising replications 50 -> 500 cuts only the second. The clean estimator is the
**paired value-weighted against count-weighted** comparison off the parquet's
`variety` column: both sides are computed on the *same* draws, so the noise
differences out and what remains is the CES term alone. Cost is one pass over a file
already on disk.

Until that is run, no claim of the form "the expected network is the win probability"
should appear in the text. It currently does not.

### 2.2 The two routes to the four cells agree only loosely

`granular_cells` is checked against the realisation-by-realisation split (two routes
sharing no code). Agreement is 5.0% / 7.1% on `E[Hbar]` and 12.7% / 21.9% on the
granular **common** cell. The identity is exact, so the gap is finite-draw noise in
the realisation route — but 22% is large enough that the agreement check is currently
weak evidence rather than a gate. More replications would tighten it; nothing else
would.

### 2.3 `rho` is reported at four regimes but validated at two

The `alpha = 0` controls are exact by construction (`Q = 1`, `C = 1`, `rho = 1` to
machine precision) and `Both forces` is computed both ways. The `Distance only` and
`Uniform benchmark` rows come only from `simulate_granular_regime`, i.e. from one
route with no independent check.

---

## 3. Identification and inference

### 3.1 No interval on any counterfactual in §5.1

Every counterfactual quantity — the effective numbers, `C`, `rho`, the local shares,
the sourcing distances — is a function of `(alpha_hat, T_hat)` and inherits their
sampling uncertainty. All are reported as point estimates. The parametric bootstrap
that would attach an interval (redraw from the estimated sampling distribution,
recompute each counterfactual) has never been run. This is the one `\swann{}` note
left in §5.1.

### 3.2 `N_s` is calibrated on the moment its level then restates

`N_hat_s` solves `Gbar_s(0) = G_s(0)`, so the **level** of granular concentration is
a restatement of a targeted moment rather than a prediction. The defence, stated in
the text, is that the **split** between common and buyer-specific granularity is
governed by `Q_rr's` — a two-buyer object the estimation never looks at. That
defence is sound but it is an argument, not a test. A test would be an untargeted
two-buyer moment in the data; none is currently available.

### 3.3 `N_hat_s` carries large Monte-Carlo dispersion in at least one sector

The pinned-median diagnostic has reported a spread of ~45% (19 -> 27 varieties) in
one sector across independent draw sets at fixed theta. `N_rho` / `N_RHO_INFERENCE`
are the levers. This propagates into block 6, into `V`, and therefore into the whole
granular decomposition.

### 3.4 The AD Jacobian remains FKG-biased on alpha

Documented in `CLAUDE.md`'s critical caveats and unchanged: the closed-form
`reg_coef` uses the FKG product approximation, so the analytical/AD Jacobian is
biased for the alpha column. Alpha inference must use the simulation-based Jacobian.
`test/test_extensive_margin.jl` measures the size of the coefficient bias; the exact
union-of-boxes replacement was scoped but never wired into production.

---

## 4. Data inputs

### 4.1 `G_K.csv` has missing rows

Load-time warnings on both industries: `C33Z, K=3`, `G45Z, K=1,2` (auto) and
`C25D, K=1,2`, `C33Z, K=1`, `G46Z, K=1,2`, `M71Z, K=1,2,3` (aero). The CDF is carried
forward (no mass added) and the filled cells are named, which is the right default,
but the count-curve panels at those `(sector, K)` are reading an imputation rather
than data.

### 4.2 `G_K_var.csv` has no value-column name

The loader reports: *reading column 'G' as the VARIANCE of the increment, no var/se
column name to go on*. It is guessing. Rename the column to `var` or `se` in the data
pipeline. If it is in fact the variance of the **CDF** rather than of the increment,
every count-curve error band is wrong — the `K = 0` consistency gate against
`diag(Sigma_data)` is what would catch that, and it should be read on the next run.

### 4.3 `france.gpkg` carries two layers

`More than one layer found in 'france.gpkg': 'france_idf' (default), 'france'`. The
default is being taken silently. Pass an explicit `layer=` or drop the spare layer;
a silent default on a geography file is the kind of thing that moves a map without
moving an error message.

---

## 5. Exhibits and text

### 5.1 Two figures must be copied before the paper compiles

`concentration_decomposition_mu2.pdf` and `buyer_concentration_mu2.pdf` are written
by the notebook into `reporting_combined` and are referenced by
`structural_2026_endogeneous.tex`. They are not in `figures/`.

### 5.2 The §4 fit panels point at `_mu1` while §5.1 is `_mu2`

Noted previously and not resolved. The §4 panels are the `theta_hat_1` fit and may be
deliberate, but a re-run at `MU = 2` will not overwrite them, so the two halves of the
paper can silently describe two different economies.

### 5.3 One unverified empirical claim survives in §5.1

The Valence claim ("around 7% ... nearly triples") has never been checked against a
run.

---

## 6. Code hygiene

### 6.1 `second_stage` is a dead positional argument

Retained through four call layers (`full_SMM` <- `parallel_SMM` <- `parallel_SMM_safe`
<- `train_stage_one`) purely to avoid a positional-argument refactor. The branch it
selected is gone.

### 6.2 `test_extensive_margin.jl`'s `reg_coef_quad` is out of sync

It assumes the old `N_REG + 1` regression design, which changed when the log-z size
control was made conditional. Print-only and off the production path, so it is stale
rather than wrong, but it will mislead whoever runs it next.

### 6.3 The Julia side is unverified in this environment

No Julia is installed here. Every `.jl` change in recent sessions has been statically
verified only (block balance, call-site arity, format-string argument counts). Run
`test/test_granular_aa.jl` and `test/test_cloglog_streaming.jl` before trusting a new
fit.
