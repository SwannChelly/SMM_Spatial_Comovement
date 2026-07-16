##### test_ge_inversion.jl #####
# Phase-0 feasibility gate for T-profiling by GE-Sinkhorn inversion (see the plan
# in the branch header + documentation/initialisation.md). STANDALONE + PRINT-ONLY:
# includes the model files + load_parameters.jl the way test_extensive_margin.jl
# does, then MEASURES whether the exact internal inversion invert_T_ge is viable
# at this calibration BEFORE any production wiring. Touches no production path.
#
#     julia test/test_ge_inversion.jl aero 4 1
#     julia test/test_ge_inversion.jl auto 1
#
# It answers the four questions the plan's go/no-go hinges on:
#
#   (1) ROUND-TRIP. From a T_true, generate γ_ls with the exact analytical map,
#       then check invert_T_ge recovers T_true to ~1e-8 (ref-normalized profile).
#       If this fails the solver is simply wrong.
#
#   (2) CONVERGENCE (plan §4). The one-step Sinkhorn map Ψ linearizes at the fixed
#       point as ∂Ψ/∂logT = J_Sink (Birkhoff contraction κ_S) + J_GE (expenditure
#       response, ‖·‖ ∝ |1−ν|,|1−λ|). At ν=0.2, λ=0.5 the GE channel is NOT small,
#       so we MEASURE, by finite differences on Ψ: ρ_full (spectral radius of the
#       full endogenous map), ρ_frozen ≈ κ_S (expenditure held fixed at T*), and
#       ‖J_GE‖ = ‖DΨ_full − DΨ_frozen‖₂. GATE: ρ_full < 1 (and, for the sufficient
#       Sinkhorn bound, κ_S + ‖J_GE‖ < 1). Also reports the observed contraction
#       from the residual history and the iteration count.
#
#   (3) UNIQUENESS. 10 random warm starts (×LogUniform[0.5,2] on active T) must all
#       land on the SAME T* (max pairwise |Δlog T| < tol). K>0 ⇒ theory predicts a
#       unique fixed point; this is the empirical check under endogenous expenditure.
#
#   (4) COST. Time one invert_T_ge vs one full_SMM loss (the outer-loop unit) and
#       one analytical moment eval, to size the per-particle overhead of profiling.
#
# GO decision (plan Phase-0 gate): round-trip OK, uniqueness OK, ρ_full < 1 at the
# real ν,λ. If the gate fails far from Cobb-Douglas, the exact internal inversion
# may need robustifying (strong damping / Newton on the augmented potential) — flag
# to the user rather than wiring production.

using Distributed
@everywhere using Random
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
include("../profiling.jl")

println("\n" * "="^72)
println("Phase-0 GATE: GE-Sinkhorn T-inversion feasibility")
@printf("  industry=%s  N_REG=%d  N_TAU=%d  |  ν=%.3g  λ=%.3g  θ=%.3g  (CD ⇔ ν=λ=1)\n",
        industry, N_REG, N_TAU, nu, lambda, theta)
println("="^72)

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
    β     = N_TAU == 1 ? [0.1] : collect(range(0.05, 0.30, length = N_TAU))
    θ = vcat(Ω_L, Ω_s, A, β, T_red)
    @printf("Using DEFAULT θ (T=T_rs_init, Ω^L=0.5, Ω^s/A=1, β=%s)\n", string(round.(β; digits=3)))
    @assert length(θ) == expected
    return θ
end

θ0                       = load_theta()
Ω_L, Ω_s, A, α, T_vec    = unpack_params(θ0)
T_true                   = reshape(T_vec, S, R)    # already ref-normalized per sector
tau0                     = build_tau(α)

# Free (active, non-reference) coordinates — the identified log-T directions.
free_coords = Tuple{Int,Int}[]
for s in 1:S, r in SECTOR_GOOD_REGIONS[s]
    r == T_REF_REGION[s] && continue
    push!(free_coords, (s, r))
end
n_free = length(free_coords)
@printf("Active T free coordinates (ref dropped): %d over %d sectors\n", n_free, S)

# ── Synthetic exact target: γ_ls generated from T_true (a fixed point exists) ──
γ_target, _, _ = gamma_ls_analytical(Ω_L, Ω_s, A, T_true, tau0)

# ═══════════════════════════════════════════════════════════════════════════════
# (1) ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "-"^72)
println("(1) ROUND-TRIP: recover T_true from its own γ_ls")
println("-"^72)
res_rt = invert_T_ge(α, Ω_L, Ω_s, A; target = γ_target, T_init = copy(T_rs_init),
                     max_iter = 2000, tol = 1e-12, damping = 0.5)
rt_err = 0.0
for (s, r) in free_coords
    global rt_err
    rt_err = max(rt_err, abs(log(res_rt.T[s, r]) - log(T_true[s, r])))
end
@printf("  iters=%d  converged=%s  final resid=%.2e\n", res_rt.iters, res_rt.converged, res_rt.resid)
@printf("  max|log T_rec − log T_true| over active free coords = %.3e  %s\n",
        rt_err, rt_err < 1e-8 ? "✓ OK" : "✗ FAIL")

# ═══════════════════════════════════════════════════════════════════════════════
# (2) CONVERGENCE: κ_S, ‖J_GE‖, ρ_full  via finite differences on Ψ
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "-"^72)
println("(2) CONVERGENCE: linearized map at the fixed point (undamped Ψ)")
println("-"^72)

T_star = res_rt.T   # exact fixed point of the synthetic target

# Endogenous expenditure at T* (frozen for the J_Sink-only map).
function exp_matrix(Omega_L, Omega_s, A, T_mat, tau)
    pr = compute_prices_analytical(Omega_L, Omega_s, A, T_mat, tau)
    E  = zeros(S, R_downstream)
    for s in 1:S, dr in 1:R_downstream
        E[s, dr] = Omega_s[s] * (pr.P_sr[s, dr] / pr.P_r[dr])^(1 - nu) *
                   (pr.P_r[dr] / pr.c_r[dr])^(1 - lambda) * (1 - Omega_L) * pr.mu * pr.Y_r[dr]
    end
    return E
end
expE = exp_matrix(Ω_L, Ω_s, A, T_star, tau0)

# γ_ls with expenditure held fixed (Φ still responds to T ⇒ pure matrix scaling).
function gamma_ls_frozen(exp_mat, T_mat, tau)
    Phi  = compute_Phi(T_mat, tau)
    X_ls = zeros(R, S)
    for g in 1:n_good
        s = GOOD_S[g]; r_p = GOOD_R[g]; T_val = T_mat[s, r_p]; w = W_RS_FLAT[g]
        for dr in 1:R_downstream
            ph = Phi[s, dr]; ph < 1e-300 && continue
            X_ls[r_p, s] += T_val * (w * tau[r_p, dr])^(-theta) / ph * exp_mat[s, dr]
        end
    end
    Xs = vec(sum(X_ls, dims=1)); g = zeros(R, S)
    for s in 1:S
        xs = Xs[s]; xs < 1e-300 && continue
        for r in 1:R; g[r, s] = X_ls[r, s] / xs * domestic_share[s]; end
    end
    return g
end

# One UNDAMPED Sinkhorn step given a model-γ (ref-normalized).
function sink_update(T_mat, gamma_model)
    Tn = copy(T_mat)
    for s in 1:S
        regs = SECTOR_GOOD_REGIONS[s]; isempty(regs) && continue
        ref  = T_REF_REGION[s] > 0 ? T_REF_REGION[s] : regs[1]
        for r in regs
            gm = gamma_model[r, s]; ge = γ_target[r, s]
            ratio = (gm > 1e-300 && ge > 0) ? ge / gm : 1.0
            Tn[s, r] = T_mat[s, r] * ratio
        end
        rv = Tn[s, ref]; rv > 0 && (Tn[s, regs] ./= rv)
    end
    return Tn
end
step_full(T_mat)   = sink_update(T_mat, gamma_ls_analytical(Ω_L, Ω_s, A, T_mat, tau0)[1])
step_frozen(T_mat) = sink_update(T_mat, gamma_ls_frozen(expE, T_mat, tau0))

# free-log helpers
function perturb(T_base, v, ε)
    Tp = copy(T_base)
    for (k, (s, r)) in enumerate(free_coords); Tp[s, r] *= exp(ε * v[k]); end
    for s in 1:S
        ref = T_REF_REGION[s]
        (ref > 0 && Tp[s, ref] > 0) && (Tp[s, :] ./= Tp[s, ref])
    end
    return Tp
end
logfree(T) = Float64[log(T[s, r]) for (s, r) in free_coords]

# Central-difference directional derivative of a step map (returns Δ in free basis).
function jvp(step_fn, T0, v; ε = 1e-6)
    (logfree(step_fn(perturb(T0, v, ε))) .- logfree(step_fn(perturb(T0, v, -ε)))) ./ (2ε)
end

# Power iteration for spectral radius of a linear operator given its matvec.
function spectral_radius(matvec, n; iters = 40, seed = 1)
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

ρ_full   = spectral_radius(v -> jvp(step_full,   T_star, v), n_free)
ρ_frozen = spectral_radius(v -> jvp(step_frozen, T_star, v), n_free)
# ‖J_GE‖: 2-norm of the difference operator DΨ_full − DΨ_frozen.
jge_norm = spectral_radius(v -> (jvp(step_full, T_star, v) .- jvp(step_frozen, T_star, v)), n_free)

# Observed contraction from the round-trip residual history (geometric tail).
obs_rate = NaN
if length(res_rt.resid_hist) >= 4
    h = res_rt.resid_hist
    tail = h[max(1, end-5):end]
    ratios = [tail[i+1] / tail[i] for i in 1:length(tail)-1 if tail[i] > 0]
    isempty(ratios) || (obs_rate = median(ratios))
end

@printf("  ρ_frozen (≈ κ_S, Birkhoff contraction of the matrix scaling) = %.4f\n", ρ_frozen)
@printf("  ρ_full   (endogenous-expenditure map, spectral radius)       = %.4f  %s\n",
        ρ_full, ρ_full < 1 ? "✓ (<1, converges)" : "✗ (≥1, DIVERGES)")
@printf("  ‖J_GE‖   (‖DΨ_full − DΨ_frozen‖₂, GE channel)                = %.4f\n", jge_norm)
@printf("  sufficient Sinkhorn bound κ_S + ‖J_GE‖                       = %.4f  %s\n",
        ρ_frozen + jge_norm, (ρ_frozen + jge_norm) < 1 ? "✓ (<1)" : "⚠ (≥1, only sufficient — see ρ_full)")
@printf("  observed contraction (median residual ratio, tail)          = %.4f\n", obs_rate)
@printf("  damped (δ=0.5) practical rate ≈ |1−δ+δ·ρ_full|               = %.4f\n", abs(0.5 + 0.5 * ρ_full))

# ═══════════════════════════════════════════════════════════════════════════════
# (3) UNIQUENESS: multi-start
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "-"^72)
println("(3) UNIQUENESS: 10 random warm starts → same fixed point?")
println("-"^72)
rng = MersenneTwister(12345)
sols = Vector{Matrix{Float64}}()
n_conv = 0
for b in 1:10
    global n_conv
    Ti = copy(T_true)
    for (s, r) in free_coords
        Ti[s, r] *= exp(log(2.0) * (2 * rand(rng) - 1))   # ×LogUniform[0.5,2]
    end
    rb = invert_T_ge(α, Ω_L, Ω_s, A; target = γ_target, T_init = Ti,
                     max_iter = 3000, tol = 1e-11, damping = 0.5)
    rb.converged && (n_conv += 1)
    push!(sols, rb.T)
end
max_spread = 0.0
for (s, r) in free_coords
    global max_spread
    vals = [log(sol[s, r]) for sol in sols]
    max_spread = max(max_spread, maximum(vals) - minimum(vals))
end
@printf("  converged starts: %d/10\n", n_conv)
@printf("  max cross-start |Δlog T| over active coords = %.3e  %s\n",
        max_spread, max_spread < 1e-7 ? "✓ UNIQUE" : "✗ NON-UNIQUE / weakly identified")

# ═══════════════════════════════════════════════════════════════════════════════
# (4) COST
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "-"^72)
println("(4) COST: invert_T_ge vs one full_SMM loss vs one analytical eval")
println("-"^72)
# warm the JIT
invert_T_ge(α, Ω_L, Ω_s, A; target = emp_gamma_ls, max_iter = 200, tol = 1e-9)
t_inv = @elapsed for _ in 1:5
    invert_T_ge(α, Ω_L, Ω_s, A; target = emp_gamma_ls, max_iter = 500, tol = 1e-9)
end
t_inv /= 5
@printf("  invert_T_ge (real emp_gamma_ls target): %.4f s / call\n", t_inv)

t_ana = NaN
try
    global t_ana
    compute_moments_analytical(θ0; n_quad = 200)
    t_ana = @elapsed compute_moments_analytical(θ0; n_quad = 200)
    @printf("  compute_moments_analytical:             %.4f s / call  (inversion = %.2f×)\n",
            t_ana, t_inv / t_ana)
catch e
    @printf("  compute_moments_analytical timing skipped (%s)\n", sprint(showerror, e))
end

t_smm = NaN
try
    global t_smm
    full_SMM(θ0; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)   # warm
    t_smm = @elapsed full_SMM(θ0; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)
    @printf("  full_SMM loss (outer-loop unit):        %.4f s / call  (inversion = %.1f%% of a loss)\n",
            t_smm, 100 * t_inv / t_smm)
catch e
    @printf("  full_SMM timing skipped (%s)\n", sprint(showerror, e))
end

# Convergence against the REAL data target (may lack an exact fixed point under
# misspecification, but Sinkhorn still converges to the best profile match).
res_emp = invert_T_ge(α, Ω_L, Ω_s, A; target = emp_gamma_ls, max_iter = 1000, tol = 1e-9)
@printf("  real-data inversion: iters=%d converged=%s resid=%.2e\n",
        res_emp.iters, res_emp.converged, res_emp.resid)

# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "="^72)
go = (rt_err < 1e-8) && (ρ_full < 1) && (max_spread < 1e-7)
@printf("PHASE-0 VERDICT: %s\n", go ? "GO ✓  (round-trip, ρ_full<1, uniqueness all pass)" :
                                       "NO-GO ✗  — inspect the failing gate above")
if !go
    println("  If ρ_full ≥ 1 or uniqueness fails at these ν,λ, the exact internal")
    println("  inversion needs robustifying (strong damping / Newton on g) before wiring.")
end
println("="^72)
