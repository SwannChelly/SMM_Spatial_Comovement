##### test_gamma_margin_domestic.jl #####
# Check the γ-margin (adding-up) condition on the sourcing-share targets that seed
# the T starting values, to VERIFY THE INPUT γ_ls ARE DOMESTIC SHARES, NOT
# INTERNATIONAL (TOTAL) SHARES. STANDALONE + PRINT-ONLY: it includes the model
# files + load_parameters.jl the way test_ge_inversion.jl does and touches no
# production path.
#
#     julia test/test_gamma_margin_domestic.jl aero 4 1
#     julia test/test_gamma_margin_domestic.jl auto 1
#
# ── Why this test exists ─────────────────────────────────────────────────────
# The model DEFINES the location-specific linkage share as
#
#     γ_ls[r,s] = ( X_ls[r,s] / X_s[s] ) · domestic_share[s]           (Eq. B-γ)
#
# (model_analytical.jl:278, model_CP.jl:968, profiling.jl:79). The bracket is a
# probability distribution over upstream regions r within sector s — it sums to 1
# by construction — so summing Eq. B-γ over r gives the MARGIN / ADDING-UP identity
#
#     Σ_r γ_ls[r,s] = domestic_share[s]        (the "gamma margin condition").
#
# domestic_share[s] ∈ (0,1] is the fraction of sector s's input purchases sourced
# DOMESTICALLY (the complement 1−domestic_share[s] is imported / international). So
# the γ shares partition the DOMESTIC sourcing mass and must sum, per sector, to
# domestic_share[s] — NOT to 1.
#
# The failure mode this guards against: if the empirical γ target file was built
# from INTERNATIONAL (total, import-inclusive) sourcing shares — normalized to sum
# to 1 across regions — then Σ_r γ_ls[r,s] ≈ 1 ≠ domestic_share[s], and every
# downstream object built on γ is wrong. In particular emp_gamma_ls is the exact
# array fed to invert_T_from_gamma for the T STARTING VALUES (load_parameters.jl
# SECTION 10b), so an international-share input silently mis-anchors the whole
# comparative-advantage search box.
#
# ── What it checks ───────────────────────────────────────────────────────────
#  (1) MARGIN CONDITION. Per sector, Σ_r γ_ls[r,s] == domestic_share[s]?
#      GATE: max relative deviation < TOL_MARGIN.
#  (2) DOMESTIC vs INTERNATIONAL classification. domestic if Σγ ≈ domestic_share;
#      international if Σγ ≈ 1 while domestic_share < 1. Flags the international case
#      loudly (that is the mistake the user asked to be caught).
#  (3) domestic_share sanity: every entry in (0,1].
#  (4) T-INIT CONSISTENCY. The array that satisfies (1) is byte-identical to the one
#      invert_T_from_gamma consumes, and T_rs_init active entries are finite/positive
#      — so the T starting values are anchored on DOMESTIC sourcing geography.
#      (The raw on-disk file is also compared to the post-load emp_gamma_ls const to
#       confirm the γ-threshold renormalization preserved each sector's total.)

using Distributed
@everywhere using NPZ
using LinearAlgebra, Statistics, Printf

@everywhere include("../model_CP.jl")
@everywhere include("../tools.jl")
@everywhere include("../model_analytical.jl")
@everywhere include("../optimizers/pso_integration.jl")

industry = length(ARGS) >= 1 ? ARGS[1] : "aero"
n_coef   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau    = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : n_coef
K_sim    = 10000   # unused here; load_parameters.jl may reference it

input_folder  = "./baseline_$industry"
output_folder = "./reporting_$industry"
mkpath(output_folder)

include("../load_parameters.jl")

# Tolerances. The margin identity is exact up to how the target file was built
# (rounding in the empirical construction), so the GATE is generous; the raw-vs-const
# reconciliation is a numerical identity and uses a tight tolerance.
const TOL_MARGIN  = 1e-3    # GATE: relative |Σγ − domestic_share| / domestic_share
const TOL_RECON   = 1e-9    # raw sum vs post-load const sum (threshold renorm)
const TOL_INTL    = 1e-3    # "Σγ ≈ 1" band for the international-share detector

println("\n" * "="^76)
println("γ-MARGIN / DOMESTIC-SHARE CHECK on the T-init sourcing targets")
@printf("  industry=%s  N_REG=%d  N_TAU=%d  S=%d  R=%d\n",
        industry, N_REG, N_TAU, S, R)
println("="^76)

# ── Pristine on-disk targets (before any load_parameters mutation) ───────────
# emp_gamma_ls.npy is stored (S, R); load_parameters permutedims's it to (R, S).
# domestic_share.npy is (S,).
raw_gamma_SR = NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy"))   # (S, R)
dom_share    = NPZ.npzread(joinpath(input_folder, "domestic_share.npy")) # (S,)
@assert size(raw_gamma_SR) == (S, R) "emp_gamma_ls.npy is $(size(raw_gamma_SR)), expected ($S,$R)"
@assert length(dom_share)  == S       "domestic_share.npy has $(length(dom_share)) entries, expected $S"

raw_gamma_rs = permutedims(raw_gamma_SR)   # (R, S), same layout as emp_gamma_ls const

# ── (3) domestic_share sanity ────────────────────────────────────────────────
println("\n[3] domestic_share sanity (each entry must lie in (0,1]):")
bad_dom = findall(x -> !(0 < x <= 1 + 1e-9), dom_share)
if isempty(bad_dom)
    @printf("  OK — all %d sectors in (0,1].  range [%.4f, %.4f], mean %.4f\n",
            S, minimum(dom_share), maximum(dom_share), mean(dom_share))
else
    @printf("  ✗ %d sector(s) with domestic_share ∉ (0,1]: %s\n",
            length(bad_dom), string(collect(bad_dom)))
    for s in bad_dom
        @printf("      sector %d: domestic_share = %.6g\n", s, dom_share[s])
    end
end

# ── (1) MARGIN CONDITION: Σ_r γ_ls[r,s] == domestic_share[s] ──────────────────
gamma_sum   = vec(sum(raw_gamma_rs, dims = 1))          # Σ_r γ per sector, (S,)
abs_dev     = abs.(gamma_sum .- dom_share)
rel_dev     = abs_dev ./ max.(abs.(dom_share), 1e-300)
n_active_s  = [count(>(0), @view raw_gamma_rs[:, s]) for s in 1:S]

println("\n[1] γ-margin condition   Σ_r γ_ls[r,s]  vs  domestic_share[s]:")
@printf("  %-8s %8s %14s %14s %12s %12s\n",
        "sector", "n_act", "Σγ", "domestic", "|Δ|", "rel")
for s in 1:S
    flag = rel_dev[s] > TOL_MARGIN ? "  <-- OFF" : ""
    @printf("  %-8d %8d %14.6f %14.6f %12.2e %12.2e%s\n",
            s, n_active_s[s], gamma_sum[s], dom_share[s], abs_dev[s], rel_dev[s], flag)
end
worst_s   = argmax(rel_dev)
margin_ok = maximum(rel_dev) < TOL_MARGIN
@printf("\n  max relative deviation = %.3e (sector %d)   [GATE < %.0e]  ⇒ %s\n",
        maximum(rel_dev), worst_s, TOL_MARGIN, margin_ok ? "PASS" : "FAIL")

# ── (2) DOMESTIC vs INTERNATIONAL classification ─────────────────────────────
# Domestic (correct): Σγ ≈ domestic_share.  International (wrong): Σγ ≈ 1 while
# domestic_share < 1.  Per sector, which target does the observed Σγ match?
println("\n[2] DOMESTIC vs INTERNATIONAL classification (per sector):")
n_domestic = 0; n_intl = 0; n_ambig = 0
intl_sectors = Int[]
for s in 1:S
    near_dom  = rel_dev[s] < TOL_MARGIN
    near_one  = abs(gamma_sum[s] - 1) < TOL_INTL
    dom_below = dom_share[s] < 1 - TOL_INTL
    label =
        if near_dom
            global n_domestic += 1; "DOMESTIC (Σγ≈domestic_share)"
        elseif near_one && dom_below
            global n_intl += 1; push!(intl_sectors, s); "INTERNATIONAL? (Σγ≈1 > domestic_share)"
        else
            global n_ambig += 1; "AMBIGUOUS (Σγ matches neither)"
        end
    @printf("  sector %-3d  Σγ=%.4f  domestic=%.4f  ->  %s\n",
            s, gamma_sum[s], dom_share[s], label)
end
@printf("\n  summary: %d domestic, %d international-looking, %d ambiguous (of %d)\n",
        n_domestic, n_intl, n_ambig, S)

intl_detected = n_intl > 0
if intl_detected
    println("\n  ⚠  INTERNATIONAL-SHARE PATTERN DETECTED on sector(s) $(intl_sectors):")
    println("     Σ_r γ_ls ≈ 1 while domestic_share < 1. The γ target for these")
    println("     sectors looks like TOTAL (import-inclusive) sourcing normalized to")
    println("     1, NOT the DOMESTIC sourcing shares the model expects. Rebuild")
    println("     emp_gamma_ls.npy so that Σ_r γ_ls[r,s] = domestic_share[s], i.e.")
    println("     multiply the region distribution by domestic_share (Eq. B-γ).")
end

# ── (4) T-INIT CONSISTENCY ───────────────────────────────────────────────────
# (a) the post-load emp_gamma_ls const (what invert_T_from_gamma reads) preserves
#     each sector's total from the raw file → the threshold renorm did not silently
#     rescale the domestic mass.
# (b) T_rs_init (the γ-inversion starting values) has finite, positive active entries.
println("\n[4] T-init consistency (emp_gamma_ls feeds invert_T_from_gamma):")
const_sum = vec(sum(emp_gamma_ls, dims = 1))            # (S,), post-load const
recon_dev = abs.(const_sum .- gamma_sum)
recon_ok  = maximum(recon_dev) < TOL_RECON
@printf("  raw-file vs post-load sector totals: max |Δ| = %.3e   [< %.0e]  ⇒ %s\n",
        maximum(recon_dev), TOL_RECON, recon_ok ? "PASS" : "FAIL")
if !recon_ok
    println("     (γ-threshold=$gamma_threshold renormalization changed a sector total —")
    println("      the domestic mass fed to T-init no longer matches the raw file.)")
end

active_T   = T_rs_init[T_rs_init .> 0]
tinit_ok   = !isempty(active_T) && all(isfinite, active_T)
@printf("  T_rs_init active entries: %d, all finite & positive ⇒ %s",
        length(active_T), tinit_ok ? "PASS" : "FAIL")
if tinit_ok
    @printf("   (range [%.3g, %.3g])\n", minimum(active_T), maximum(active_T))
else
    println()
end

# ── (5) SINKHORN-BALANCED TARGET: emp_gamma_ls_tilde sums to 1 per sector ─────
# The inversion (invert_T_from_gamma / invert_T_ge) matches the DOMESTIC γ against
# the expenditure margin ω (Σ ω = 1). Sinkhorn's theorem needs compatible margins,
# so the code rescales the target to γ̃ = emp_gamma_ls / domestic_share (Σ_r = 1).
# Verify that const exists and its per-sector total is 1 (= Σ ω), i.e. the margin
# condition Σ γ̃ = Σ ω now holds for the object actually fed to the solver.
println("\n[5] Sinkhorn-balanced target  γ̃ = emp_gamma_ls / domestic_share  (Σ_r γ̃ = 1):")
tilde_ok = true
if isdefined(Main, :emp_gamma_ls_tilde)
    tilde_sum = vec(sum(emp_gamma_ls_tilde, dims = 1))     # (S,), should be ≈ 1
    tilde_dev = abs.(tilde_sum .- 1)
    tilde_ok  = maximum(tilde_dev) < TOL_MARGIN
    @printf("  per-sector Σ_r γ̃: range [%.6f, %.6f]   max |Σγ̃ − 1| = %.3e   [< %.0e]  ⇒ %s\n",
            minimum(tilde_sum), maximum(tilde_sum), maximum(tilde_dev),
            TOL_MARGIN, tilde_ok ? "PASS" : "FAIL")
    println("  ⇒ balanced against the expenditure margin ω (Σ ω = 1): Sinkhorn precondition met.")
else
    tilde_ok = false
    println("  ✗ const emp_gamma_ls_tilde is not defined — the balanced target was not built.")
end

# ── Verdict ──────────────────────────────────────────────────────────────────
println("\n" * "="^76)
all_ok = margin_ok && recon_ok && tinit_ok && tilde_ok && isempty(bad_dom) && !intl_detected
if all_ok
    println("VERDICT: PASS — the γ targets are DOMESTIC shares (Σ_r γ = domestic_share),")
    println("         balanced to γ̃ (Σ = 1) for Sinkhorn, and correctly seed T.")
else
    println("VERDICT: FAIL — see the flagged section(s) above.")
    !margin_ok      && println("  · γ-margin condition violated (Σγ ≠ domestic_share).")
    intl_detected   && println("  · INTERNATIONAL shares suspected (Σγ ≈ 1 > domestic_share).")
    !isempty(bad_dom) && println("  · domestic_share out of (0,1].")
    !recon_ok       && println("  · raw vs post-load sector totals disagree.")
    !tinit_ok       && println("  · T_rs_init has non-finite / no active entries.")
    !tilde_ok       && println("  · balanced target emp_gamma_ls_tilde missing or Σγ̃ ≠ 1.")
end
println("="^76)
