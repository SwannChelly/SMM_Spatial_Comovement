# old_version — the multi-specification Julia code (frozen)

The Julia estimator as it stood on 2026-10-28, before the light rewrite at the repository
root. It keeps every specification the root code dropped: GMM / analytical moments
(`main_gmm.jl`, `model_analytical.jl`), the continuum and ZE-level comparative-advantage
models, the LPM and firm-level cloglog regressions, the control-group design, the joint
(non-profiled) T search, CMA-ES and TikTak, and the diagnostic scripts in `test/` and
`extras/`. `CLAUDE.md` here holds the full changelog of that code.

Nothing here is maintained. The layout is unchanged, so relative includes still work;
the launchers now point at this folder:

    ./SMM_Spatial_Comovement/old_version/run.sh aero --n_coef=4 --n_tau=1 ...

run from the repository's parent directory (next to `baseline_<industry>/`). Old runs and
new runs write to the same `reporting_<…>` folder names: do not run both in one tree.
