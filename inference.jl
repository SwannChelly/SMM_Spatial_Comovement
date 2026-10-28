##### inference.jl — Steps 2 and 4: efficient weighting and inference #####
#
# The inference moments are β → γ → G: [β_d ; β_κ] (BLOCK_RANGES[4]), γ_{s,a}
# (BLOCK_RANGES[5]) and Ḡ_s(0) (BLOCK_RANGES[6]), in that order everywhere (Σ file,
# W_step3, Ω, Jacobian rows).
#
# What is estimated, and how its uncertainty is obtained:
#   a = [α ; κ]  — on the moment conditions: Jacobian G_a = dm/da along the profiled
#                  manifold (T follows by Sinkhorn), sandwich variance.
#   T            — not estimated: T̂ = T*(â, γ̂). Delta method over (â, γ̂) with their
#                  correlation: the estimation error of â and the data noise of the γ
#                  target that the inversion matches.
#   N_s          — calibrated: Ḡ_s(N̂_s; q̂) = Ĝ_s. Implicit-function delta method over
#                  (Ĝ, â, γ̂).
# The head (Ω^L, Ω^s, A) is held at θ̂ (it is fixed in Step 3).

using Printf
using Dates

inference_moment_indices() = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]), collect(BLOCK_RANGES[6]))

"""Local (within β → γ → G) ranges of the three inference blocks."""
function inference_block_layout()
    n4, n5, n6 = length.(BLOCK_RANGES[4:6])
    return (1:n4, (n4 + 1):(n4 + n5), (n4 + n5 + 1):(n4 + n5 + n6))
end

"""
    gamma_block_cs() -> [(area, sector)] of the γ moments, in moment order (s-major)
"""
function gamma_block_cs()
    off = length(MOMENT_MASK) - T_COL_DIM * S - S
    return [(c, s) for s in 1:S for c in 1:T_COL_DIM if MOMENT_MASK[off + (s - 1) * T_COL_DIM + c]]
end

"""
    _sym_inv(A; floor_rel=1e-10) -> (inv(A), floored::Bool)

Inverse of a symmetric matrix by Cholesky, with an eigenvalue floor at
`floor_rel × λ_max` when it is not positive definite.
"""
function _sym_inv(A::AbstractMatrix; floor_rel::Float64 = 1e-10)
    As = Symmetric((Matrix(A) .+ Matrix(A)') ./ 2)
    M, floored = try
        (Matrix(inv(cholesky(As))), false)
    catch
        F = eigen(As)
        (F.vectors * Diagonal(1.0 ./ max.(F.values, F.values[end] * floor_rel)) * F.vectors', true)
    end
    return (M .+ M') ./ 2, floored
end

##################### Step 2: Σ_data, Σ_sim, W ###################

"""
    load_sigma_data() -> (Σ_data, "") or (nothing, reason)

The data (bootstrap) covariance of the inference moments, `SIGMA_DATA_FILE` in the
input folder, laid out β_d (N_REG) → β_κ (1) → γ (active non-reference areas, s-major)
→ Ḡ_s(0) (S). Any other size is refused.
"""
function load_sigma_data()
    path = joinpath(input_folder, SIGMA_DATA_FILE)
    isfile(path) || return (nothing, "$path does not exist")
    Σ = NPZ.npzread(path)
    n_gb = length(inference_moment_indices())
    size(Σ) == (n_gb, n_gb) || return (nothing,
        "$path is $(size(Σ)), expected ($n_gb, $n_gb) = β_d ($N_REG) + β_κ (1) + " *
        "γ ($(length(BLOCK_RANGES[5]))) + Ḡ_s(0) ($S)")
    isapprox(Σ, Σ'; atol = 1e-10) || return (nothing, "$path is not symmetric")
    return (Matrix{Float64}(Σ), "")
end

"""
    build_weight_matrix(theta_hat_1, Sigma_data; K) -> W_step3

`Σ_sim` from `K` re-simulations at θ̂_1 (i.i.d. draws, `N_RHO_INFERENCE` per cell),
`Ω = Σ_data + Σ_sim` (eigenvalue floor at λ_max/1e8 if cond > 1e10), `W_step3 = Ω⁻¹`.
Saves `step2/{M_sim, Sigma_data, Sigma_sim, Omega, W_step3}.npy` and `diagnostics.txt`.
"""
function build_weight_matrix(theta_hat_1::Vector{Float64}, Sigma_data::Matrix{Float64}; K::Int = 10_000)
    gb = inference_moment_indices()
    println("Estimating Σ_sim from K = $K simulations at θ̂_1...")
    rows = pmap(1:K) do k
        u_k, w_k = generate_draws(N_RHO_INFERENCE, n_good, :mc; randomise = true, rng = MersenneTwister(k))
        moments_to_vec(full_SMM(theta_hat_1; u_draws = u_k, sample_weights = w_k)[2])[gb]
    end
    M_sim = permutedims(reduce(hcat, rows))
    Sigma_sim = cov(M_sim; dims = 1)
    Omega = Sigma_data .+ Sigma_sim
    Omega = (Omega .+ Omega') ./ 2
    ev = eigvals(Symmetric(Omega))
    cond_Ω = maximum(ev) / max(minimum(ev), 1e-300)
    println("  Ω condition number: $(round(cond_Ω, sigdigits=4))")
    if cond_Ω > 1e10
        @warn "Ω is ill-conditioned (κ = $cond_Ω): eigenvalue floor at λ_max/1e8."
        F = eigen(Symmetric(Omega))
        Omega = F.vectors * Diagonal(max.(F.values, maximum(ev) / 1e8)) * F.vectors'
        Omega = (Omega .+ Omega') ./ 2
    end
    W = inv(Omega)
    dir = joinpath(output_folder, "step2"); mkpath(dir)
    for (f, v) in (("M_sim", M_sim), ("Sigma_data", Sigma_data), ("Sigma_sim", Sigma_sim),
                   ("Omega", Omega), ("W_step3", W))
        NPZ.npzwrite(joinpath(dir, "$f.npy"), v)
    end
    write(joinpath(dir, "diagnostics.txt"),
          "K = $K\nlambda_max = $(maximum(ev))\nlambda_min = $(minimum(ev))\ncondition_number = $cond_Ω\n")
    return W
end

##################### Profiled Jacobian ###################

"""
    profiled_jacobian(theta; K=50, base_seed, step_rel=1e-2) -> NamedTuple

Central finite differences, averaged over `K` i.i.d. draw sets (`N_RHO_INFERENCE` per
cell; the same draws for every perturbation of a replication), with `N̂_s` pinned at the
median of its free value over 5 draw sets — `N̂_s(θ)` is a step function, and its
dispersion across draw sets is simulation noise in q̂, printed here.

Columns, each moving T to the Sinkhorn image `T*` (started from T̂, deterministic):
  * α_b — log step `α·e^{±δ}` (derivative = Δm / (2δα));
  * κ   — additive step `±δ` (κ can sit at or below zero);
  * γ_j — the j-th γ target entry (area c, sector s) moved by `e^{±δ}` with the head
          fixed (derivative per unit of γ, the units of Σ_data).

Returns the moment Jacobian `G` (all masked moments × columns), its across-replication
sd, the count-distribution rows `Gc` (vec of S × |COUNT_CURVE_K|), `fT = ∂T*/∂(column)`
on the identified T entries, the pinned `N_fixed` and the column labels.
"""
function profiled_jacobian(theta::Vector{Float64}; K::Int = 50, base_seed::Int = 0,
                           step_rel::Float64 = 1e-2)
    δ = step_rel
    ΩL, Ωs, A, α, κ = normalized_head(theta)
    T_hat = unpack_T_par(theta)
    reduce_T(Tp) = vec(permutedims(Tp))[T_MASK][T_ID_REDUCED]

    # ── pin N̂_s ────────────────────────────────────────────────────────────────
    n_probe = min(K, 5)
    N_mat = reduce(hcat, pmap(1:n_probe) do k
        u_k, w_k = generate_draws(N_RHO_INFERENCE, n_good, :mc; randomise = true,
                                  rng = MersenneTwister(base_seed + k))
        full_SMM(theta; u_draws = u_k, sample_weights = w_k, return_full = true)[2].N_hat_free
    end)
    N_fixed = [round(Int, median(N_mat[s, :])) for s in 1:S]
    println("Jacobian: N̂_s pinned at $N_fixed (median over $n_probe draw sets of $N_RHO_INFERENCE draws)")
    spread = [(maximum(N_mat[s, :]) - minimum(N_mat[s, :])) / max(N_fixed[s], 1) for s in 1:S]
    for s in 1:S
        @printf("  %-8s pinned %4d  min %4d  max %4d  spread/pinned %.3f\n", SECTOR_NAMES[s],
                N_fixed[s], minimum(N_mat[s, :]), maximum(N_mat[s, :]), spread[s])
    end
    maximum(spread) > 0.10 && @warn "N̂_s varies by up to $(round(100 * maximum(spread), digits=1))% " *
        "across draw sets at a fixed θ (simulation noise in q̂): raise N_RHO_INFERENCE."

    # ── perturbed parameter vectors (T re-inverted) ──────────────────────────
    cols = NamedTuple[]
    for b in 1:N_TAU
        push!(cols, (label = N_TAU == 1 ? "alpha" : "alpha_$b", kind = :alpha, b = b,
                     denom = 2δ * α[b]))
    end
    push!(cols, (label = "kappa", kind = :kappa, b = 0, denom = 2δ))
    for (c, s) in gamma_block_cs()
        push!(cols, (label = "gamma[$(SECTOR_NAMES[s])-$(c)]", kind = :gamma, b = 0, c = c, s = s,
                     denom = 2δ * EMP_GAMMA_T[c, s]))
    end
    function perturbed(col, sgn)
        x = copy(theta); αx = copy(α); κx = κ; target = EMP_GAMMA_T
        if col.kind == :alpha
            αx[col.b] = α[col.b] * exp(sgn * δ); x[alpha_range()[col.b]] = αx[col.b]
        elseif col.kind == :kappa
            κx = κ + sgn * δ; x[kappa_index()] = κx
        else
            target = copy(EMP_GAMMA_T); target[col.c, col.s] *= exp(sgn * δ)
        end
        res = invert_T_ge(αx, ΩL, Ωs, A; kappa = κx, target = target, T_init = copy(T_hat))
        res.converged || @warn "invert_T_ge did not converge for column $(col.label) ($(sgn > 0 ? "+" : "−"))"
        return with_T(x, res.T), res.T
    end
    pert = pmap(i -> (perturbed(cols[i], +1), perturbed(cols[i], -1)), 1:length(cols))
    fT = reduce(hcat, [(reduce_T(p[1][2]) .- reduce_T(p[2][2])) ./ cols[i].denom for (i, p) in enumerate(pert)])
    θp = [p[1][1] for p in pert]; θm = [p[2][1] for p in pert]
    denoms = [c.denom for c in cols]

    # ── moment Jacobian ───────────────────────────────────────────────────────
    println("Jacobian: $K replications × $(2 * length(cols) + 1) evaluations " *
            "($(N_TAU + 1) [α ; κ] columns, $(length(cols) - N_TAU - 1) γ-target columns)")
    flush(stdout)
    reps = pmap(1:K) do k
        u_k, w_k = generate_draws(N_RHO_INFERENCE, n_good, :mc; randomise = true,
                                  rng = MersenneTwister(base_seed + 100 + k))
        ev(p) = (m = full_SMM(p; u_draws = u_k, sample_weights = w_k, N_fixed = N_fixed,
                              return_full = true)[2]; vcat(moments_to_vec(m), vec(m.G_curve)))
        reduce(hcat, [(ev(θp[i]) .- ev(θm[i])) ./ denoms[i] for i in eachindex(cols)])
    end
    J    = mean(reps)
    J_sd = K > 1 ? sqrt.(max.(mean([r .^ 2 for r in reps]) .- J .^ 2, 0.0) .* (K / (K - 1))) : zeros(size(J))
    all(isfinite, J) || error("non-finite Jacobian entries (a collapsed sector or a failed inversion)")
    nM = length(MOMENT_MASK[MOMENT_MASK])
    return (G = J[1:nM, :], G_sd = J_sd[1:nM, :], Gc = J[(nM + 1):end, :], fT = fT,
            N_fixed = N_fixed, labels = [c.label for c in cols])
end

##################### Inference at an estimate ###################

"""
    run_inference(theta, W, Omega, Sigma_data; dest, industry, K=50, base_seed, K_sim)

Inference at the profiled estimate `theta`, written under `<dest>/inference/`:
  * [α ; κ]: efficient `(G'WG)⁻¹` and sandwich `(G'WG)⁻¹ G'WΩWG (G'WG)⁻¹` variances,
    fitted-moment and residual SEs, Hansen J (`inference_summary.txt`, `J_stat.txt`);
  * T: `V_T = f_a V_a f_a' + f_γ Σ_γγ f_γ' + f_a C f_γ' + (·)'`, `C = Cov(â, γ̂) = P Σ_data[:, γ]`,
    `P = (G'WG)⁻¹ G'W` (the influence of the moments on â);
  * N_s: `dN_s = [dĜ_s − a_s dâ − b_s dγ̂] / D_s`, `D_s = Ḡ_s(N̂+1) − Ḡ_s(N̂)`,
    `a_s = ∂Ḡ_s/∂a`, `b_s = ∂Ḡ_s/∂γ` (block-6 rows of the Jacobian);
  * the count distribution Ḡ_s(K): the same substitution with loading `r_s^K = D_s^K/D_s`
    on the target and residualised structural loadings (`r = 0` for a clamped sector);
  * `inference_delta.txt`: the a / T / N_s report.
"""
function run_inference(theta::Vector{Float64}, W::Matrix{Float64}, Omega::Matrix{Float64},
                       Sigma_data::Matrix{Float64}; dest::String, industry::String = "",
                       K::Int = 50, base_seed::Int = 0, K_sim::Int = 0)
    inf_dir = joinpath(dest, "inference"); mkpath(inf_dir)
    gb = inference_moment_indices()
    rβ, rγ, rG = inference_block_layout()
    n_a = N_TAU + 1

    jac = profiled_jacobian(theta; K = K, base_seed = base_seed)
    NPZ.npzwrite(joinpath(dest, "jacobian_profiled.npy"), jac.G)
    NPZ.npzwrite(joinpath(dest, "jacobian_profiled_sd.npy"), jac.G_sd)
    write(joinpath(dest, "jacobian_profiled_columns.txt"), join(jac.labels, "\n") * "\n")
    G_a = jac.G[gb, 1:n_a];   G_γ = jac.G[gb, (n_a + 1):end]
    f_a = jac.fT[:, 1:n_a];   f_γ = jac.fT[:, (n_a + 1):end]

    gr = granular_report(theta)                      # on the estimation draws
    emp = vec(empirical_moments)[gb]
    sim = moments_to_vec(full_SMM(theta)[2])[gb]

    # ── [α ; κ] ──────────────────────────────────────────────────────────────
    a_idx = vcat(collect(alpha_range()), kappa_index())
    a_hat = theta[a_idx]
    a_lab = PARAM_LABELS[a_idx]
    V_eff, floored = _sym_inv(G_a' * W * G_a)
    V_sw = V_eff * (G_a' * W * Omega * W * G_a) * V_eff; V_sw = (V_sw .+ V_sw') ./ 2
    se_eff = sqrt.(max.(diag(V_eff), 0.0)); se_sw = sqrt.(max.(diag(V_sw), 0.0))
    Var_m = G_a * V_sw * G_a'; Var_m = (Var_m .+ Var_m') ./ 2
    se_fit = sqrt.(max.(diag(Var_m), 0.0))
    se_res = sqrt.(max.(diag(Omega) .- diag(Var_m), 0.0))
    r = emp .- sim
    J_stat = (r' * W * r)[1]
    df = length(gb) - n_a
    pval = df > 0 ? 1.0 - cdf(Chisq(df), J_stat) : NaN
    for (f, v) in (("var_theta_efficient", V_eff), ("var_theta_sandwich", V_sw), ("se_theta", se_eff),
                   ("se_theta_sandwich", se_sw), ("t_stats", a_hat ./ se_sw),
                   ("ci_95", hcat(a_hat .- 1.96 .* se_sw, a_hat .+ 1.96 .* se_sw)),
                   ("se_moments_fitted", se_fit), ("se_moment_residuals", se_res))
        NPZ.npzwrite(joinpath(inf_dir, "$f.npy"), v)
    end

    # ── T ────────────────────────────────────────────────────────────────────
    P      = V_eff * G_a' * W
    Σγγ    = Sigma_data[rγ, rγ]
    C_aγ   = P * Sigma_data[:, rγ]
    V_T_a  = f_a * V_sw * f_a'
    V_T_γ  = f_γ * Σγγ * f_γ'
    V_T    = V_T_a .+ V_T_γ .+ f_a * C_aγ * f_γ' .+ (f_a * C_aγ * f_γ')'; V_T = (V_T .+ V_T') ./ 2
    se_T   = sqrt.(max.(diag(V_T), 0.0))
    se_T_a = sqrt.(max.(diag(V_T_a), 0.0)); se_T_γ = sqrt.(max.(diag(V_T_γ), 0.0))
    T_lab  = PARAM_LABELS[t_param_start() - 1 .+ T_ID_REDUCED]
    T_hat  = theta[t_param_start() - 1 .+ T_ID_REDUCED]
    for (f, v) in (("var_theta_T_delta", V_T), ("se_theta_T_delta", se_T),
                   ("se_theta_T_delta_alpha", se_T_a), ("se_theta_T_delta_gamma", se_T_γ),
                   ("t_stats_T_delta", T_hat ./ se_T), ("ci_95_T_delta", hcat(T_hat .- 1.96 .* se_T, T_hat .+ 1.96 .* se_T)),
                   ("jacobian_T_wrt_alpha", f_a), ("jacobian_T_wrt_gamma", f_γ))
        NPZ.npzwrite(joinpath(inf_dir, "$f.npy"), v)
    end

    # ── N_s ──────────────────────────────────────────────────────────────────
    N_hat = gr.N_hat
    lg = gbar_logfact_table(gr.n_draw)
    Dfd = [gbar_sector(CELLS_OF_SECTOR[s], gr.k_counts, gr.n_draw, N_hat[s] + 1, lg) -
           gbar_sector(CELLS_OF_SECTOR[s], gr.k_counts, gr.n_draw, N_hat[s], lg) for s in 1:S]
    Dinv  = Diagonal([abs(d) > 1e-14 ? 1 / d : 0.0 for d in Dfd])
    A_mat = G_a[rG, :]; B_mat = G_γ[rG, :]
    Σ_GG  = Sigma_data[rG, rG]; C_aG = P * Sigma_data[:, rG]; Σ_γG = Sigma_data[rγ, rG]
    Joint = [Σ_GG C_aG' Σ_γG'; C_aG V_sw C_aγ; Σ_γG C_aγ' Σγγ]
    L   = hcat(Matrix(Dinv), -(Dinv * A_mat), -(Dinv * B_mat))
    V_N = L * Joint * L'; V_N = (V_N .+ V_N') ./ 2
    se_N   = sqrt.(max.(diag(V_N), 0.0))
    se_N_G = sqrt.(max.(diag(Dinv * Σ_GG * Dinv'), 0.0))
    se_N_a = sqrt.(max.(diag((Dinv * A_mat) * V_sw * (Dinv * A_mat)'), 0.0))
    se_N_γ = sqrt.(max.(diag((Dinv * B_mat) * Σγγ * (Dinv * B_mat)'), 0.0))
    for (f, v) in (("var_theta_N_s_delta", V_N), ("se_theta_N_s_delta", se_N),
                   ("jacobian_N_s_wrt_alpha", A_mat), ("jacobian_N_s_wrt_gamma", B_mat),
                   ("dGbar_dn_firstdiff", Dfd))
        NPZ.npzwrite(joinpath(inf_dir, "$f.npy"), v)
    end

    # ── the count distribution Ḡ_s(K) ─────────────────────────────────────────
    Ks = collect(COUNT_CURVE_K); nK = length(Ks)
    rows_K(i) = ((i - 1) * S + 1):(i * S)            # vec over (S, nK) is column-major
    i0 = findfirst(==(0), Ks)
    A0 = jac.Gc[rows_K(i0), 1:n_a]; B0 = jac.Gc[rows_K(i0), (n_a + 1):end]
    # Same estimator, same evaluations: the K = 0 curve rows ARE block 6.
    maximum(abs.(hcat(A0, B0) .- hcat(A_mat, B_mat))) <= 1e-8 * max(1.0, maximum(abs.(hcat(A_mat, B_mat)))) ||
        @warn "count curve: the K = 0 rows differ from the block-6 rows of the Jacobian"
    D_curve = [gbar_sector_K(CELLS_OF_SECTOR[s], gr.k_counts, gr.n_draw, N_hat[s] + 1, KK, lg) -
               gbar_sector_K(CELLS_OF_SECTOR[s], gr.k_counts, gr.n_draw, N_hat[s], KK, lg)
               for s in 1:S, KK in Ks]
    r_curve = [(gr.clamped[s] == :none && abs(Dfd[s]) > 1e-14) ? D_curve[s, i] / Dfd[s] : 0.0
               for s in 1:S, i in 1:nK]
    se_curve = zeros(S, nK); V_curve = zeros(S, S, nK)
    for i in 1:nK
        Rd = Diagonal(r_curve[:, i])
        L_K = hcat(Matrix(Rd), jac.Gc[rows_K(i), 1:n_a] .- Rd * A0,
                   jac.Gc[rows_K(i), (n_a + 1):end] .- Rd * B0)
        V = L_K * Joint * L_K'; V_curve[:, :, i] = (V .+ V') ./ 2
        se_curve[:, i] = sqrt.(max.(diag(V_curve[:, :, i]), 0.0))
    end
    for (f, v) in (("se_G_curve_delta", se_curve), ("var_G_curve_delta", V_curve),
                   ("dGbar_dn_curve", D_curve), ("r_curve", r_curve),
                   ("count_curve_K", Float64.(Ks)), ("G_curve_fitted", gr.G_curve))
        NPZ.npzwrite(joinpath(inf_dir, "$f.npy"), v)
    end

    # ── reports ──────────────────────────────────────────────────────────────
    head_idx = 1:(1 + S + R_downstream)
    ev = eigvals(Symmetric(G_a' * W * G_a))
    open(joinpath(inf_dir, "J_stat.txt"), "w") do io
        @printf(io, "Hansen J-test\n  J statistic : %.6g\n  df          : %d\n  p-value     : %s\n",
                J_stat, df, isnan(pval) ? "N/A (exactly identified)" : @sprintf("%.6f", pval))
    end
    open(joinpath(inf_dir, "inference_summary.txt"), "w") do io
        println(io, "="^88)
        println(io, "SMM INFERENCE — profiled estimator   industry: $industry   K_sim: $K_sim   ",
                Dates.format(now(), "yyyy-mm-dd HH:MM"))
        println(io, "Moments: β_d, β_κ, γ_{s,a}, Ḡ_s(0) ($(length(gb))). Parameters on the moment " *
                    "conditions: [α ; κ] ($n_a); T and N_s are calibrated (inference_delta.txt).")
        println(io, "="^88)
        @printf(io, "λ_min(G'WG) = %.4e   λ_max = %.4e   cond = %.4e%s\n", minimum(ev), maximum(ev),
                maximum(ev) / max(minimum(ev), 1e-300), floored ? "   (eigenvalue floor applied)" : "")
        @printf(io, "\n  %-12s %12s %12s %12s %8s %8s %12s %12s\n", "param", "estimate", "se_eff",
                "se_sw", "ratio", "t", "CI_lo", "CI_hi")
        for i in 1:n_a
            @printf(io, "  %-12s %12.6f %12.6f %12.6f %8.4f %8.3f %12.6f %12.6f\n", a_lab[i], a_hat[i],
                    se_eff[i], se_sw[i], se_sw[i] / se_eff[i], a_hat[i] / se_sw[i],
                    a_hat[i] - 1.96 * se_sw[i], a_hat[i] + 1.96 * se_sw[i])
        end
        println(io, "\n--- Other parameters (held at θ̂; value only) ---")
        for i in head_idx
            @printf(io, "  %-24s %12.6f\n", PARAM_LABELS[i], theta[i])
        end
        println(io, "\n--- Moment residual SEs by block (share ≈ 0: well fit; ≈ 1: weakly used) ---")
        for (name, rng) in zip(("reg_coef", "gamma_ls", "G0"), (rβ, rγ, rG))
            sd = sqrt.(max.(diag(Omega)[rng], 0.0))
            @printf(io, "  %-10s mean resid SE %.4e   mean share %.4f\n", name, mean(se_res[rng]),
                    mean(se_res[rng] ./ max.(sd, 1e-15)))
        end
        println(io, "\n--- Per moment ---")
        for (j, i) in enumerate(gb)
            @printf(io, "  %-28s emp %10.5f  sim %10.5f  se_fit %.3e  resid_SE %.3e\n",
                    MOMENT_LABELS[i], emp[j], sim[j], se_fit[j], se_res[j])
        end
        @printf(io, "\nHansen J = %.6g, df = %d, p = %s\n", J_stat, df, isnan(pval) ? "N/A" : @sprintf("%.6f", pval))
        println(io, "\nSEs are conditional on the draws; Σ_data is the bootstrap of the data moments.")
    end
    open(joinpath(inf_dir, "inference_delta.txt"), "w") do io
        println(io, "="^110)
        println(io, "PROFILED-ESTIMATOR INFERENCE — [α ; κ], T, N_s   industry: $industry")
        println(io, "="^110)
        println(io, "\n--- [α ; κ] (inference_summary.txt) ---")
        for i in 1:n_a
            @printf(io, "  %-12s %12.6f  se_sw %.6e  t %8.3f  CI [%.6f, %.6f]\n", a_lab[i], a_hat[i],
                    se_sw[i], a_hat[i] / se_sw[i], a_hat[i] - 1.96 * se_sw[i], a_hat[i] + 1.96 * se_sw[i])
        end
        println(io, "\n--- T = T*(â, γ̂): correlated delta method ---")
        println(io, "  se_T      : total;   se_T_a : [α ; κ] channel only;   se_T_γ : γ-target channel only")
        @printf(io, "  %-26s %12s %12s %12s %12s %8s %12s %12s\n", "parameter", "estimate", "se_T",
                "se_T_a", "se_T_γ", "t", "CI_lo", "CI_hi")
        for i in eachindex(T_hat)
            @printf(io, "  %-26s %12.6f %12.6e %12.6e %12.6e %8.3f %12.6f %12.6f\n", T_lab[i], T_hat[i],
                    se_T[i], se_T_a[i], se_T_γ[i], T_hat[i] / se_T[i],
                    T_hat[i] - 1.96 * se_T[i], T_hat[i] + 1.96 * se_T[i])
        end
        println(io, "\n--- N_s: calibrated, dN_s = [dĜ_s − a_s dâ − b_s dγ̂] / D_s ---")
        println(io, "  se_N_G : count-target channel; se_N_a : [α ; κ]; se_N_γ : γ target. Units: varieties.")
        @printf(io, "  %-10s %6s %6s %12s %12s %12s %12s %14s\n", "sector", "N̂_s", "clamp", "se_N",
                "se_N_G", "se_N_a", "se_N_γ", "D_s")
        for s in 1:S
            @printf(io, "  %-10s %6d %6s %12.4e %12.4e %12.4e %12.4e %14.6e\n", SECTOR_NAMES[s], N_hat[s],
                    string(gr.clamped[s]), se_N[s], se_N_G[s], se_N_a[s], se_N_γ[s], Dfd[s])
        end
        println(io, "\n--- Count distribution Ḡ_s(K) ± delta-method SE (K = 0 targeted, K ≥ 1 untargeted) ---")
        for s in 1:S
            println(io, "  ", rpad(SECTOR_NAMES[s], 10),
                    join([@sprintf(" K=%d %.3f±%.3f", Ks[i], gr.G_curve[s, i], se_curve[s, i]) for i in 1:nK]))
        end
        println(io, "\nThe head (Ω^L, Ω^s, A) is held at θ̂. The system is triangular: â → T̂ → q̂ → N̂_s.")
    end
    println("Inference at $dest: Hansen J = $(round(J_stat, sigdigits=5)), df = $df; " *
            join(["$(a_lab[i]) = $(round(a_hat[i], sigdigits=4)) (se $(round(se_sw[i], sigdigits=3)))" for i in 1:n_a], ", "))
    return (a_hat = a_hat, se_sw = se_sw, V_sw = V_sw, se_T = se_T, se_N = se_N, J_stat = J_stat)
end
