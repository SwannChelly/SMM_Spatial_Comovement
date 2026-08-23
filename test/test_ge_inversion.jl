##### test_ge_inversion.jl #####
# GE-Sinkhorn T-inversion: feasibility gate (Phase 0) AND the α × damping
# STABILITY MAP that explains the `T non-converged: n/98 particles` lines in a
# --profile_T=true PSO log. STANDALONE + PRINT-ONLY: includes the model files +
# load_parameters.jl the way test_profile_alpha_sweep.jl does. Touches no
# production path.
#
#     julia test/test_ge_inversion.jl aero            # run.sh defaults (see below)
#     julia test/test_ge_inversion.jl aero 4 1 ./reporting_aero_profiled_aa_gran_nlo1_pso
#  Args: industry [n_coef] [n_tau] [run_folder] [reg_method] [include_control]
#        [granular] [ca_level] [relax_n_lo]
#
# ⚠ THE CONFIG ARGS ARE NOT OPTIONAL DECORATION — and the DEFAULTS HERE MATCH
# `run.sh`, not the legacy model. `load_parameters.jl` reads the modelling flags as
# `granular_local = (@isdefined(granular)) ? … : false` and
# `ca_level_local = (@isdefined(ca_level)) ? … : :ze`, so a script that does not
# define them BEFORE the include silently loads the LEGACY ZE continuum model with
# no error. That is what happened to the earlier version of this file: it reported
# `GRANULAR = false ; CA_LEVEL = :ze`, `reg_method = :lpm`, `N_REG = 5` (the LPM
# target file has five bins) and 99 ZE T-columns instead of 72 AA columns — i.e. it
# gated a different model than the one being run. Pass the same flags you passed to
# run.sh; the defaults below already are run.sh's.
#
# ── WHAT THIS ANSWERS ────────────────────────────────────────────────────────────
#
#   (1) ROUND-TRIP. From a T_true, generate γ with the exact analytical map, then
#       check invert_T_ge recovers T_true. If this fails the solver is simply wrong.
#
#   (2) SPECTRUM AT THE FIXED POINT. The one-step map Ψ linearises as
#       DΨ = I − M, M = ∂log γ_model/∂log T. Reports ρ_frozen (≈ κ_S, the Birkhoff
#       contraction of the pure matrix scaling), ρ_full and ‖J_GE‖ — evaluated at
#       the ESTIMATED head and alpha_hat, not at a default α.
#
#   (3) UNIQUENESS. 10 random warm starts must land on the same T*.
#
#   (4) COST. invert_T_ge vs one full_SMM loss.
#
#   (5) α × DAMPING STABILITY MAP  ← the section that reproduces the PSO log.
#       For each α on a grid, at the REAL data target and the estimated head:
#         (a) find T*(α) with a very small δ (δ=0.05, long budget). Small δ is
#             stable for any M < 2/δ = 40, so it converges whenever a fixed point
#             EXISTS. If even this fails, the target is infeasible at that α — a
#             different diagnosis (Hall-type margin violation), and the map says so.
#         (b) at that T*(α), power-iterate ρ(I − δM) for δ ∈ {0.9,0.7,0.5,0.3,0.1}.
#             ρ ≥ 1 ⇒ that δ CANNOT converge, however long you run it.
#         (c) run the PRODUCTION invert_T_ge at each (α,δ) and record
#             converged / iters / resid, plus a TRACE-based classification of the
#             failure — SIGN-FLIP fraction on the worst coordinate separates
#             OSCILLATION (|1−δM|>1 with M>2/δ: alternating steps) from CREEP
#             (M≈0: monotone, just slow). Predicted vs observed is the gate.
#         (d) report λ_max(M), the STABILITY LIMIT δ* = 2/λ_max, and the OPTIMAL
#             relaxation δ_opt = 2/(λ_min+λ_max) with its rate
#             (λ_max−λ_min)/(λ_max+λ_min)  — the standard Richardson result, since
#             the damped Sinkhorn update IS Richardson iteration on log T.
#
#   (6) MECHANISM. At each α, decompose the DIAGONAL of M per (sector, T-column):
#           M_c = 1 − σ̄_c + G_c ,
#           σ̄_c = Σ_dr ω_{c,dr} σ_{c,dr}      (c's sales-weighted own market share)
#           G_c = −(1/θ) Σ_dr ω_{c,dr} η_E(dr) σ_{c,dr}      (the GE feedback)
#           η_E = (1−ν)(1−ψ_s) + (1−λ)ψ_s(1−χ) + ε·χ·ψ_s·(1−ξ)
#       with ψ_s = sector share of P_r, χ = (1−Ω^L)(P_r/c_r)^{1−λ} = D_r − 1,
#       ξ = the destination's weight in P_agg. The allocation term (1 − σ̄) is a
#       contraction; the GE term is a POSITIVE demand feedback (raise T in area c ⇒
#       cheaper inputs in the destinations c serves ⇒ with ε = −16 their sales jump
#       ⇒ demand flows back to c). σ̄_c rises with θα, so M_c rises with α and
#       crosses 2/δ at a critical α — that is the cliff. The FD diagonal of M is
#       measured exactly and compared against this formula, so the attribution is a
#       measurement, not an argument.
#
#   (7) COUNTERFACTUALS without rebinding the consts. ε enters only η_E, so the
#       critical α under ε' is a pure post-process. θ enters twice and in OPPOSITE
#       directions (σ̄ depends on θα, the price transmission on 1/θ), so the θ'
#       counterfactual is read off the grid at α = θ'α'/θ with G rescaled by θ/θ'.
#       Answers "would θ = 1.768 (the calibrated value load_parameters.jl:63
#       overrides with 1.0) move the wall out of the search box?"
#
# GO decision: round-trip OK, uniqueness OK, ρ_full < 1 AT alpha_hat. Section 5's verdict
# is separate and is the one that matters for the optimiser: it names the largest δ
# that is stable over the whole α box, and the α at which the production δ = 0.9
# stops working.

using Distributed
@everywhere using Random
@everywhere using NPZ
using LinearAlgebra, Statistics, Printf

@everywhere include("../model_CP.jl")
@everywhere include("../tools.jl")
@everywhere include("../model_analytical.jl")
@everywhere include("../optimizers/pso_integration.jl")

# ── Args. DEFAULTS = run.sh's (n_coef=4, n_tau=1, cloglog, no control group,
#    granular=true, ca_level=aa, relax_n_lo=true). Every modelling flag is defined
#    BEFORE the include, as load_parameters.jl's @isdefined probes require. ────────
industry = length(ARGS) >= 1 ? ARGS[1] : "aero"
n_coef   = length(ARGS) >= 2 && !isempty(strip(ARGS[2])) ? parse(Int, ARGS[2]) : 4
n_tau    = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : 1
K_sim    = 10000   # unused here; load_parameters.jl may reference it

_flag(i, dflt) = length(ARGS) >= i && !isempty(strip(ARGS[i])) ?
                 (lowercase(strip(ARGS[i])) in ("true", "1", "yes")) : dflt

reg_method      = length(ARGS) >= 5 && !isempty(strip(ARGS[5])) ? Symbol(strip(ARGS[5])) : :cloglog
@assert reg_method in (:lpm, :cloglog) "reg_method must be lpm|cloglog, got :$reg_method"
include_control = _flag(6, false)                                   # run.sh: --controls=false
granular        = _flag(7, true)                                    # run.sh: --granular=true
ca_level        = length(ARGS) >= 8 && !isempty(strip(ARGS[8])) ? Symbol(strip(ARGS[8])) : :aa
@assert ca_level in (:ze, :aa) "ca_level must be ze|aa, got :$ca_level"
relax_n_lo      = _flag(9, true)                                    # run.sh: --relax_n_lo=true
optimizer_backend = :pso
profile_T       = true
draw_method     = :sobol    # run.sh: --draws=sobol

input_folder  = "./baseline_$industry"
# Default run tree = exactly what main.jl builds from the run.sh defaults, so
# load_theta() finds the θ̂ of the run being explained instead of falling back to a
# default head at α = 0.1 (which is what made the earlier gate report GO).
default_tree = "./reporting_$industry" * (profile_T ? "_profiled" : "") *
               (ca_level == :aa ? "_aa" : "") * (granular ? "_gran" : "") *
               (relax_n_lo && granular ? "_nlo1" : "") * "_$optimizer_backend"
output_folder = length(ARGS) >= 4 && !isempty(strip(ARGS[4])) ? String(ARGS[4]) : default_tree
mkpath(output_folder)

include("../load_parameters.jl")
include("../profiling.jl")

@assert N_TAU == 1 "This diagnostic assumes the power-law single-α parametrization (N_TAU==1)."

println("\n" * "="^76)
println("GE-Sinkhorn T-inversion: feasibility gate + α × damping stability map")
@printf("  industry=%s  N_REG=%d  N_TAU=%d  |  ν=%.3g  λ=%.3g  θ=%.3g  ε=%.3g\n",
        industry, N_REG, N_TAU, nu, lambda, theta, epsilon)
@printf("  GRANULAR=%s  CA_LEVEL=:%s  RELAX_N_LO=%s  REG=%s  |  T_COL_DIM=%d  n_good=%d  n_T=%d\n",
        GRANULAR, CA_LEVEL, RELAX_N_LO, REG_METHOD, T_COL_DIM, n_good, count(T_MASK))
@printf("  anchoring θ̂ on: %s\n", output_folder)
println("="^76)

# ── θ̂: prefer a saved estimate (same config), else a documented default ───────
function load_theta()
    expected = 1 + S + R_downstream + N_TAU + count(T_MASK)
    for rel in (("step3", "theta_hat_2.npy"), ("step1", "theta_hat_1.npy"),
                ("step1", "best_params.npy"))
        p = joinpath(output_folder, rel...)
        isfile(p) || continue
        θ = NPZ.npzread(p); ndims(θ) > 1 && (θ = θ[:, 1])
        if length(θ) == expected
            @printf("Loaded θ̂ from %s (length %d)\n", p, length(θ)); return θ
        end
    end
    T_red = vec(permutedims(T_rs_init))[T_MASK]
    Ω_L   = 0.5; Ω_s = ones(S); A = ones(R_downstream)
    β     = N_TAU == 1 ? [P_alpha] : collect(range(0.05, 0.30, length = N_TAU))
    θ = vcat(Ω_L, Ω_s, A, β, T_red)
    @printf("⚠ No θ̂ under %s — using DEFAULT head (Ω^L=0.5, Ω^s/A=1, α=prior=%.4f).\n",
            output_folder, β[1])
    println("  The stability map still runs (it sweeps α itself), but ψ_s and χ — which")
    println("  set the GE gain — are then at the default head, not the estimated one.")
    @assert length(θ) == expected
    return θ
end

θ0                    = load_theta()
Ω_L, Ω_s, A, alpha_hat_v, _ = unpack_params(θ0)
T_hat_par             = unpack_T_par(θ0)          # (S, T_COL_DIM) — the COLUMN space
alpha_hat                     = alpha_hat_v[1]
tau0                  = build_tau(alpha_hat_v)
@printf("Head: Ω^L=%.4f  α̂=%.4f  |  ‖Ω^s‖₁=%.3f  A[1]=%.3f\n", Ω_L, alpha_hat, sum(Ω_s), A[1])

# Free (active, non-reference) coordinates in T-COLUMN space — the identified
# log-T directions. Under :ze a column IS a ZE; under :aa it is an attraction area.
free_coords = Tuple{Int,Int}[]
for s in 1:S, c in SECTOR_T_COLS[s]
    c == T_REF_REGION[s] && continue
    push!(free_coords, (s, c))
end
n_free = length(free_coords)
@printf("Active T free coordinates (ref dropped): %d over %d sectors\n", n_free, S)

# ── shared helpers, all in T-COLUMN space ─────────────────────────────────────
gamma_of(T_par, tau) =
    aggregate_gamma_to_T(gamma_ls_analytical(Ω_L, Ω_s, A, gather_T_to_ze(T_par), tau)[1])

logfree(T_par) = Float64[log(T_par[s, c]) for (s, c) in free_coords]

function refnorm!(T_par)
    for s in 1:S
        ref = T_REF_REGION[s]
        (ref > 0 && T_par[s, ref] > 0) && (T_par[s, :] ./= T_par[s, ref])
    end
    return T_par
end

function perturb(T_base, v, ε)
    Tp = copy(T_base)
    for (k, (s, c)) in enumerate(free_coords); Tp[s, c] *= exp(ε * v[k]); end
    return refnorm!(Tp)
end

# One UNDAMPED Sinkhorn step against `target` (ref-normalised), in column space.
function sink_update(T_par, gamma_model, target)
    Tn = copy(T_par)
    for s in 1:S
        cols = SECTOR_T_COLS[s]; isempty(cols) && continue
        ref  = T_REF_REGION[s] > 0 ? T_REF_REGION[s] : cols[1]
        for c in cols
            gm = gamma_model[c, s]; ge = target[c, s]
            Tn[s, c] = T_par[s, c] * ((gm > 1e-300 && ge > 0) ? ge / gm : 1.0)
        end
        rv = Tn[s, ref]; rv > 0 && (Tn[s, cols] ./= rv)
    end
    return Tn
end

# Central-difference directional derivative of a step map, in the free log basis.
function jvp(step_fn, T0, v; ε = 1e-6)
    (logfree(step_fn(perturb(T0, v, ε))) .- logfree(step_fn(perturb(T0, v, -ε)))) ./ (2ε)
end

# Power iteration for the spectral radius of a linear operator given its matvec.
function spectral_radius(matvec, n; iters = 60, seed = 1)
    rng = MersenneTwister(seed)
    v = randn(rng, n); v ./= norm(v)
    ρ = 0.0
    for _ in 1:iters
        w = matvec(v); nw = norm(w)
        nw < 1e-300 && return 0.0
        ρ = nw; v = w ./ nw
    end
    return ρ
end

# ═══════════════════════════════════════════════════════════════════════════════
# (1) ROUND-TRIP  — at the ESTIMATED head and alpha_hat
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "-"^76)
println("(1) ROUND-TRIP: recover T_true from its own γ  (perturbed start, at α̂)")
println("-"^76)
γ_target = gamma_of(T_hat_par, tau0)      # a fixed point exists BY CONSTRUCTION

rng_rt = MersenneTwister(777)
T_start_rt = copy(T_hat_par)
for (s, c) in free_coords
    T_start_rt[s, c] *= exp(log(2.0) * (2 * rand(rng_rt) - 1))
end
res_rt = invert_T_ge(alpha_hat_v, Ω_L, Ω_s, A; target = γ_target, T_init = T_start_rt,
                     max_iter = 5000, tol = 1e-12, damping = 0.5)
rt_err = maximum(abs.(logfree(res_rt.T) .- logfree(T_hat_par)))
@printf("  iters=%d  converged=%s  final resid=%.2e  (δ=0.5)\n",
        res_rt.iters, res_rt.converged, res_rt.resid)
@printf("  max|log T_rec − log T_true| over active free coords = %.3e  %s\n",
        rt_err, rt_err < 1e-8 ? "✓ OK" : "✗ FAIL")

T_star_hat = res_rt.T

# ═══════════════════════════════════════════════════════════════════════════════
# (2) SPECTRUM AT THE FIXED POINT (alpha_hat)
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "-"^76)
println("(2) SPECTRUM: linearised map at the fixed point (undamped Ψ), at α̂")
println("-"^76)

# Endogenous expenditure at T*, frozen for the pure-matrix-scaling comparison.
function exp_matrix(T_par, tau)
    pr = compute_prices_analytical(Ω_L, Ω_s, A, gather_T_to_ze(T_par), tau)
    E  = zeros(S, R_downstream)
    for s in 1:S, dr in 1:R_downstream
        E[s, dr] = Ω_s[s] * (pr.P_sr[s, dr] / pr.P_r[dr])^(1 - nu) *
                   (pr.P_r[dr] / pr.c_r[dr])^(1 - lambda) * (1 - Ω_L) * pr.mu * pr.Y_r[dr]
    end
    return E
end

# γ with expenditure held fixed (Φ still responds to T ⇒ pure matrix scaling).
function gamma_frozen(exp_mat, T_par, tau)
    T_ze = gather_T_to_ze(T_par)
    Phi  = compute_Phi(T_ze, tau)
    X_ls = zeros(R, S)
    for g in 1:n_good
        s = GOOD_S[g]; r_p = GOOD_R[g]; w = W_RS_FLAT[g]
        for dr in 1:R_downstream
            ph = Phi[s, dr]; ph < 1e-300 && continue
            X_ls[r_p, s] += T_ze[s, r_p] * (w * tau[r_p, dr])^(-theta) / ph * exp_mat[s, dr]
        end
    end
    Xs = vec(sum(X_ls, dims=1)); gz = zeros(R, S)
    for s in 1:S
        xs = Xs[s]; xs < 1e-300 && continue
        for r in 1:R; gz[r, s] = X_ls[r, s] / xs * domestic_share[s]; end
    end
    return aggregate_gamma_to_T(gz)
end

expE = exp_matrix(T_star_hat, tau0)
step_full(T)   = sink_update(T, gamma_of(T, tau0),                    γ_target)
step_frozen(T) = sink_update(T, gamma_frozen(expE, T, tau0),          γ_target)

ρ_full   = spectral_radius(v -> jvp(step_full,   T_star_hat, v), n_free)
ρ_frozen = spectral_radius(v -> jvp(step_frozen, T_star_hat, v), n_free)
jge_norm = spectral_radius(v -> (jvp(step_full, T_star_hat, v) .- jvp(step_frozen, T_star_hat, v)), n_free)

obs_rate = NaN
if length(res_rt.resid_hist) >= 4
    h = res_rt.resid_hist; tail = h[max(1, end-5):end]
    ratios = [tail[i+1] / tail[i] for i in 1:length(tail)-1 if tail[i] > 0]
    isempty(ratios) || (obs_rate = median(ratios))
end

@printf("  ρ_frozen (≈ κ_S, Birkhoff contraction of the matrix scaling) = %.4f\n", ρ_frozen)
@printf("  ρ_full   (endogenous-expenditure map, spectral radius)       = %.4f  %s\n",
        ρ_full, ρ_full < 1 ? "✓ (<1, converges)" : "✗ (≥1, DIVERGES)")
@printf("  ‖J_GE‖   (‖DΨ_full − DΨ_frozen‖₂, GE channel)                = %.4f\n", jge_norm)
@printf("  sufficient Sinkhorn bound κ_S + ‖J_GE‖                       = %.4f  %s\n",
        ρ_frozen + jge_norm, (ρ_frozen + jge_norm) < 1 ? "✓ (<1)" : "⚠ (≥1, only sufficient)")
@printf("  observed contraction (median residual ratio, tail)           = %.4f\n", obs_rate)
println("  NOTE: this is the map at α̂ only. Section 5 sweeps α — that is where the")
println("        PSO failures live, and ρ_full at α̂ says nothing about α > α̂.")

# ═══════════════════════════════════════════════════════════════════════════════
# (3) UNIQUENESS
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "-"^76)
println("(3) UNIQUENESS: 10 random warm starts → same fixed point?")
println("-"^76)
function uniqueness_check()
    rng = MersenneTwister(12345)
    sols = Vector{Matrix{Float64}}(); n_conv = 0
    for _ in 1:10
        Ti = copy(T_hat_par)
        for (s, c) in free_coords; Ti[s, c] *= exp(log(2.0) * (2 * rand(rng) - 1)); end
        rb = invert_T_ge(alpha_hat_v, Ω_L, Ω_s, A; target = γ_target, T_init = Ti,
                         max_iter = 5000, tol = 1e-11, damping = 0.5)
        rb.converged && (n_conv += 1); push!(sols, rb.T)
    end
    spread = 0.0
    for (s, c) in free_coords
        vals = [log(sol[s, c]) for sol in sols]
        spread = max(spread, maximum(vals) - minimum(vals))
    end
    return n_conv, spread
end
n_conv, max_spread = uniqueness_check()
@printf("  converged starts: %d/10\n", n_conv)
@printf("  max cross-start |Δlog T| over active coords = %.3e  %s\n",
        max_spread, max_spread < 1e-7 ? "✓ UNIQUE" : "✗ NON-UNIQUE / weakly identified")

# ═══════════════════════════════════════════════════════════════════════════════
# (4) COST
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "-"^76)
println("(4) COST: invert_T_ge vs one full_SMM loss vs one analytical eval")
println("-"^76)
invert_T_ge(alpha_hat_v, Ω_L, Ω_s, A; max_iter = 200, tol = 1e-9)   # JIT warm
t_inv = @elapsed for _ in 1:5
    invert_T_ge(alpha_hat_v, Ω_L, Ω_s, A; max_iter = 500, tol = 1e-9)
end
t_inv /= 5
@printf("  invert_T_ge (production default target):  %.4f s / call\n", t_inv)
let t_smm = NaN
    try
        full_SMM(θ0; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)
        t_smm = @elapsed full_SMM(θ0; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)
        @printf("  full_SMM loss (outer-loop unit):          %.4f s / call  (inversion = %.1f%%)\n",
                t_smm, 100 * t_inv / t_smm)
    catch e
        @printf("  full_SMM timing skipped (%s)\n", sprint(showerror, e))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# (5) α × DAMPING STABILITY MAP, at the REAL data target
#     Everything below lives inside functions: a top-level `for` in Julia is SOFT
#     scope, so assigning to a global inside one either warns or silently binds a
#     fresh local (the bug documented for main.jl's siren_counter).
# ═══════════════════════════════════════════════════════════════════════════════
const ALPHA_GRID  = sort(unique(round.(vcat(collect(0.10:0.10:1.20),
                                            alpha_hat, 0.45, 0.50, 0.55); digits = 4)))
const DAMP_GRID   = [0.9, 0.7, 0.5, 0.3, 0.1]
const DAMP_PROD   = 0.9                    # profiling.jl's invert_T_ge default
const REAL_TARGET = EMP_GAMMA_T_TILDE      # the production target

# Traced damped Sinkhorn: the same recursion as invert_T_ge, plus the SIGNED step on
# one tracked coordinate. A high sign-flip fraction is OSCILLATION (λ > 2/δ); a low
# one with a shrinking residual is CREEP (slow contraction). `resid` alone cannot
# tell them apart, and they call for opposite fixes.
function invert_traced(a::Float64, T_init::AbstractMatrix; damping::Float64,
                       max_iter::Int = 500, tol::Float64 = 1e-9)
    tau   = build_tau([a])
    T_par = copy(T_init); refnorm!(T_par)
    track_s = 0; track_c = 0; best1 = -1.0
    hist  = Float64[]; steps = Float64[]
    resid = Inf; it_used = 0
    for it in 1:max_iter
        it_used = it
        gm    = gamma_of(T_par, tau)
        T_new = copy(T_par)
        for s in 1:S
            cols = SECTOR_T_COLS[s]; isempty(cols) && continue
            ref  = T_REF_REGION[s] > 0 ? T_REF_REGION[s] : cols[1]
            for c in cols
                g0 = gm[c, s]; ge = REAL_TARGET[c, s]
                ratio = (g0 > 1e-300 && ge > 0) ? ge / g0 : 1.0
                T_new[s, c] = exp(log(T_par[s, c]) + damping * log(ratio))
            end
            rv = T_new[s, ref]; rv > 0 && (T_new[s, cols] ./= rv)
        end
        mx = 0.0
        for s in 1:S, c in SECTOR_T_COLS[s]
            d = abs(log(T_new[s, c]) - log(T_par[s, c]))
            d > mx && (mx = d)
            if it == 1 && d > best1
                best1 = d; track_s = s; track_c = c
            end
        end
        track_s > 0 && push!(steps,
            log(T_new[track_s, track_c]) - log(T_par[track_s, track_c]))
        T_par = T_new; resid = mx; push!(hist, mx)
        resid < tol && break
    end
    tailsteps = steps[max(1, length(steps) - 49):end]
    flips = 0
    for i in 2:length(tailsteps)
        tailsteps[i] * tailsteps[i-1] < 0 && (flips += 1)
    end
    flip_frac = length(tailsteps) > 1 ? flips / (length(tailsteps) - 1) : NaN
    ratio_tail = NaN
    if length(hist) >= 6
        t  = hist[max(1, end-10):end]
        rr = [t[i+1]/t[i] for i in 1:length(t)-1 if t[i] > 0]
        isempty(rr) || (ratio_tail = median(rr))
    end
    return (T = T_par, iters = it_used, converged = resid < tol, resid = resid,
            flip_frac = flip_frac, ratio_tail = ratio_tail)
end

function classify(r)
    r.converged            && return "OK"
    isnan(r.flip_frac)     && return "?"
    r.flip_frac  > 0.6     && return "OSCIL"
    r.ratio_tail > 1.02    && return "DIVERG"
    r.ratio_tail > 0.995   && return "STALL"
    return "CREEP"
end

# M = ∂log γ_model / ∂log T on the free coordinates.  DΨ_undamped = I − M, so
# M·v = v − DΨ·v; the damped iteration matrix is I − δM.
function M_operator(T_at, a::Float64)
    tau_a = build_tau([a])
    sf    = T -> sink_update(T, gamma_of(T, tau_a), REAL_TARGET)
    return v -> (v .- jvp(sf, T_at, v))
end

struct AlphaRow
    a::Float64; exists::Bool; iters::Int; resid::Float64
    lmax::Float64; lmin::Float64
end

# ── (5a) does a fixed point EXIST, and what is the spectrum of M there? ─────────
function stability_scan()
    println("\n(5a) Does a fixed point EXIST at each α?  (δ=0.05, 20000 iters — stable for λ<40)")
    println("      α      exists  iters      resid    |  λ_max(M)  λ_min(M)  δ*=2/λmax   δ_opt   rate")
    println("     " * "-"^94)
    rows   = AlphaRow[]
    Tstars = Dict{Float64, Matrix{Float64}}()
    for a in ALPHA_GRID
        r0 = invert_traced(a, T_rs_init; damping = 0.05, max_iter = 20000, tol = 1e-10)
        Tstars[a] = r0.T
        lmax = NaN; lmin = NaN
        if r0.converged
            Mv   = M_operator(r0.T, a)
            lmax = spectral_radius(Mv, n_free)
            # ρ(λmax·I − M) = λmax − λmin for a real spectrum bounded above by λmax.
            lmin = lmax - spectral_radius(v -> (lmax .* v .- Mv(v)), n_free)
        end
        @printf("   %6.3f   %-6s %6d   %.2e  |  %8.3f %8.3f  %8.3f  %6.3f  %5.3f\n",
                a, r0.converged ? "yes" : "NO", r0.iters, r0.resid,
                lmax, lmin, 2/lmax, 2/(lmin+lmax), (lmax-lmin)/(lmax+lmin))
        push!(rows, AlphaRow(a, r0.converged, r0.iters, r0.resid, lmax, lmin))
    end
    println("     λ_max(M) is the gain of the worst log-T direction. The damped update IS")
    println("     Richardson iteration on log T, so it is STABLE iff δ < δ* = 2/λ_max and")
    println("     FASTEST at δ_opt = 2/(λ_min+λ_max), rate (λmax−λmin)/(λmax+λmin).")
    println("     An `exists=NO` row is a DIFFERENT diagnosis: no δ helps, the target margin")
    println("     is infeasible at that α (a Hall-type transport condition fails).")
    return rows, Tstars
end

# ── (5b/c) predicted ρ(I − δM) vs what invert_T_ge actually does ────────────────
function damping_map(rows, Tstars)
    println("\n(5b/c) PREDICTED ρ(I−δM)  vs  OBSERVED invert_traced (max_iter=500, tol=1e-9, T_init=T_rs_init)")
    print("      α    ")
    for d in DAMP_GRID; @printf("|  δ=%.1f: ρ  class  it ", d); end
    println()
    println("     " * "-"^(11 + 22 * length(DAMP_GRID)))
    for row in rows
        @printf("   %6.3f ", row.a)
        Mv = row.exists ? M_operator(Tstars[row.a], row.a) : nothing
        for d in DAMP_GRID
            rho = Mv === nothing ? NaN : spectral_radius(v -> (v .- d .* Mv(v)), n_free)
            ro  = invert_traced(row.a, T_rs_init; damping = d, max_iter = 500, tol = 1e-9)
            @printf("| %5.2f %-6s %4d ", rho, classify(ro), ro.iters)
        end
        println()
    end
    println("     ρ < 1 predicts convergence; ρ ≥ 1 predicts failure AT ANY BUDGET.")
    println("     OSCIL = sign-flipping steps (λ_max > 2/δ — the instability). CREEP = monotone, slow.")
    println("     Column-by-column agreement between the ρ and the class IS the confirmation.")
end

rows, Tstars = stability_scan()
damping_map(rows, Tstars)

# ═══════════════════════════════════════════════════════════════════════════════
# (6) MECHANISM: the diagonal of M, measured exactly and decomposed
#         M_c = 1 − σ̄_c + G_c
#         σ̄_c = Σ_dr ω_{c,dr} σ_{c,dr}                     (allocation, a contraction)
#         G_c = −(1/θ) Σ_dr ω_{c,dr} η_E(dr) σ_{c,dr}      (GE feedback, positive)
#         η_E = (1−ν)(1−ψ_s) + (1−λ)ψ_s(1−χ) + ε·χ·ψ_s·(1−ξ)
# ═══════════════════════════════════════════════════════════════════════════════
function mechanism(T_par, tau)
    T_ze = gather_T_to_ze(T_par)
    pr   = compute_prices_analytical(Ω_L, Ω_s, A, T_ze, tau)
    Phi  = pr.Phi
    psi  = zeros(S, R_downstream)
    for dr in 1:R_downstream
        den = sum(Ω_s[s] * pr.P_sr[s, dr]^(1 - nu) for s in 1:S)
        for s in 1:S
            psi[s, dr] = Ω_s[s] * pr.P_sr[s, dr]^(1 - nu) / den
        end
    end
    chi = [(1 - Ω_L) * pr.P_r[dr]^(1-lambda) / pr.c_r[dr]^(1-lambda) for dr in 1:R_downstream]
    xi  = pr.Y_r ./ sum(pr.Y_r)
    E   = zeros(S, R_downstream)
    for s in 1:S, dr in 1:R_downstream
        E[s, dr] = Ω_s[s] * (pr.P_sr[s,dr]/pr.P_r[dr])^(1-nu) *
                   (pr.P_r[dr]/pr.c_r[dr])^(1-lambda) * (1-Ω_L) * pr.mu * pr.Y_r[dr]
    end
    # σ[c,dr,s] = share of T-column c in Φ[s,dr] (summed over the cells of the area)
    sig = zeros(T_COL_DIM, R_downstream, S)
    for g in 1:n_good
        s = GOOD_S[g]; l = GOOD_R[g]; c = T_GATHER[l]; w = W_RS_FLAT[g]
        for dr in 1:R_downstream
            Phi[s, dr] < 1e-300 && continue
            sig[c, dr, s] += T_ze[s, l] * (w * tau[l, dr])^(-theta) / Phi[s, dr]
        end
    end
    out = Dict{Tuple{Int,Int}, NamedTuple}()
    for s in 1:S, c in SECTOR_T_COLS[s]
        X = [sig[c, dr, s] * E[s, dr] for dr in 1:R_downstream]
        tot = sum(X); tot < 1e-300 && continue
        om = X ./ tot
        sigbar = sum(om[dr] * sig[c,dr,s]                                      for dr in 1:R_downstream)
        t1 = sum(om[dr] * sig[c,dr,s] * (1-nu)*(1-psi[s,dr])                   for dr in 1:R_downstream)
        t2 = sum(om[dr] * sig[c,dr,s] * (1-lambda)*psi[s,dr]*(1-chi[dr])       for dr in 1:R_downstream)
        t3 = sum(om[dr] * sig[c,dr,s] * chi[dr]*psi[s,dr]*(1-xi[dr])           for dr in 1:R_downstream)
        # t3 keeps ε OUT as a separate factor, so section 7 can vary it post hoc.
        G  = -(t1 + t2 + epsilon * t3) / theta
        out[(s,c)] = (sigbar = sigbar, G = G, M = 1 - sigbar + G,
                      t1 = t1, t2 = t2, t3 = t3,
                      psi = sum(om[dr]*psi[s,dr] for dr in 1:R_downstream),
                      chi = sum(om[dr]*chi[dr]   for dr in 1:R_downstream))
    end
    return out
end

# Exact FD of the diagonal: bump log T[s,c], read Δlog γ[c,s]. Includes the
# normaliser and the ref gauge, so it is the truth the formula is checked against.
function M_diag_fd(T_par, tau; eps_fd = 1e-5)
    out = Dict{Tuple{Int,Int}, Float64}()
    for (s, c) in free_coords
        Tp = copy(T_par); Tp[s,c] *= exp(eps_fd);  refnorm!(Tp)
        Tm = copy(T_par); Tm[s,c] *= exp(-eps_fd); refnorm!(Tm)
        gp = gamma_of(Tp, tau)[c,s]; gm = gamma_of(Tm, tau)[c,s]
        (gp > 0 && gm > 0) && (out[(s,c)] = (log(gp) - log(gm)) / (2 * eps_fd))
    end
    return out
end

function mechanism_table(rows, Tstars)
    println("\n" * "="^76)
    println("(6) MECHANISM: M_c = 1 − σ̄_c + G_c   (allocation contraction + GE feedback)")
    println("="^76)
    println("\n    α    max σ̄  mean σ̄ |  ψ̄     χ̄   |  η_E terms: ν      λ       ε     | maxM(form) maxM(FD)  λmax(M)  worst")
    println("   " * "-"^116)
    store = Dict{Float64, Dict{Tuple{Int,Int}, NamedTuple}}()
    for row in rows
        row.exists || continue
        tau_a = build_tau([row.a]); Tp = Tstars[row.a]
        mk = mechanism(Tp, tau_a); store[row.a] = mk
        isempty(mk) && continue
        fd  = M_diag_fd(Tp, tau_a)
        ks  = collect(keys(mk))
        Mf  = [mk[k].M for k in ks]
        kk  = ks[argmax(Mf)]
        Mfd = isempty(fd) ? NaN : maximum(values(fd))
        sb  = [mk[k].sigbar for k in ks]
        @printf("  %5.3f  %6.3f %6.3f | %5.3f %5.3f | %+7.3f %+7.3f %+8.3f | %9.3f %9.3f %8.3f  (%d,%d)\n",
                row.a, maximum(sb), mean(sb), mk[kk].psi, mk[kk].chi,
                mk[kk].t1, mk[kk].t2, epsilon * mk[kk].t3,
                maximum(Mf), Mfd, row.lmax, kk[1], kk[2])
    end
    println("   The ε column dwarfs the ν and λ terms — the feedback is a DEMAND loop, not a")
    println("   trade-cost one. maxM(form) should track maxM(FD); λmax(M) ≥ max diagonal.")
    println("   Instability at damping δ  ⟺  λ_max(M) > 2/δ   (δ=0.9 ⇒ 2.222 ; δ=0.5 ⇒ 4).")
    return store
end

mech_store = mechanism_table(rows, Tstars)

# ═══════════════════════════════════════════════════════════════════════════════
# (7) COUNTERFACTUAL ε and θ, without rebinding the consts
#     ε enters only through t3, so its counterfactual is exact to first order.
#     θ enters TWICE and in OPPOSITE directions: through the 1/θ price transmission
#     (explicit) and through σ̄, which is a function of θα. So M under θ' at α' is
#     read off the grid point a = θ'α'/θ, with the gain rescaled by θ/θ'.
# ═══════════════════════════════════════════════════════════════════════════════
function max_M_cf(store, a::Float64, eps_cf::Float64, th_cf::Float64)
    haskey(store, a) || return NaN
    mk = store[a]; isempty(mk) && return NaN
    return maximum(1 - v.sigbar - (v.t1 + v.t2 + eps_cf * v.t3) / th_cf for v in values(mk))
end

function crit_alpha(store, eps_cf::Float64, th_cf::Float64, delta::Float64)
    lim  = 2 / delta
    best = NaN
    for a in ALPHA_GRID
        m = max_M_cf(store, a, eps_cf, th_cf)
        (isnan(m) || m >= lim) && continue
        ap = a * theta / th_cf                     # the grid point a means α' = a·θ/θ'
        (isnan(best) || ap > best) && (best = ap)
    end
    return best
end

function counterfactual_table(store)
    println("\n" * "="^76)
    println("(7) COUNTERFACTUAL critical α  (largest α with max_c M_c < 2/δ, on this grid)")
    println("="^76)
    ths = (theta, 1.768)                           # 1.768 = the value load_parameters.jl:63 overrides
    for delta in (0.9, 0.5, 0.3)
        @printf("\n  damping δ = %.1f    (stability limit  max_c M_c < %.3f)\n", delta, 2/delta)
        print("       ε   ")
        for t in ths; @printf("   θ=%-8.3f", t); end
        println()
        for e in (epsilon, epsilon/2, epsilon/4, -2.0)
            @printf("   %7.2f ", e)
            for t in ths
                ca = crit_alpha(store, e, t, delta)
                lab = isnan(ca) ? "none" :
                      (ca >= maximum(ALPHA_GRID) * theta / t - 1e-9 ?
                       @sprintf(">%.2f", ca) : @sprintf("%.3f", ca))
                @printf("   %-10s", lab)
            end
            println()
        end
    end
    @printf("\n  The (ε=%.1f, θ=%.3g, δ=%.1f) cell is the production configuration — compare it\n",
            epsilon, theta, DAMP_PROD)
    println("  against the α at which the PSO log's `T non-converged` rate crosses 50%.")
    println("  `none` = unstable everywhere on the grid; `>x` = still stable at the grid top.")
end

counterfactual_table(mech_store)

# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════════════════
function verdict(rows)
    println("\n" * "="^76)
    go = (rt_err < 1e-8) && (ρ_full < 1) && (max_spread < 1e-7)
    @printf("PHASE-0 GATE (at α̂ = %.4f): %s\n", alpha_hat,
            go ? "GO ✓  (round-trip, ρ_full<1, uniqueness)" : "NO-GO ✗ — inspect above")
    lm   = [r.lmax for r in rows if r.exists && !isnan(r.lmax)]
    safe = [r.a    for r in rows if r.exists && !isnan(r.lmax) && r.lmax < 2/DAMP_PROD]
    if isempty(safe)
        @printf("STABILITY at the production δ=%.1f: UNSTABLE over the whole α grid.\n", DAMP_PROD)
    else
        @printf("STABILITY at the production δ=%.1f: stable up to α ≈ %.3f on this grid.\n",
                DAMP_PROD, maximum(safe))
    end
    if !isempty(lm)
        @printf("Largest damping stable over the WHOLE grid: δ < 2/max λ_max = %.3f\n",
                2 / maximum(lm))
    end
    nofix = [r.a for r in rows if !r.exists]
    isempty(nofix) || @printf("⚠ NO fixed point at α ∈ %s — infeasible target, damping cannot help.\n",
                              string(nofix))
    println("If the stable-δ number is well below 0.9, invert_T_ge's default damping — chosen")
    println("from a Phase-0 measurement taken INSIDE the stable region — is what caps α, and")
    println("the PSO α ceiling is a solver artefact rather than a feature of the fit.")
    println("="^76)
end

verdict(rows)
