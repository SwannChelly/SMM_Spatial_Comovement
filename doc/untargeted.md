# Untargeted Moments with Three Shock Models: Technical Documentation

## 1. Objective and Economic Context

This file documents a full simulation-and-validation pipeline designed to assess **untargeted moments** in a production network model under alternative specifications of downstream demand shocks. The central objective is to verify that a calibrated network economy delivers comparable reduced-form elasticities when shocks are generated using:

1. a pooled **univariate AR(1)** process,
2. a **multivariate AR(1)** process with cross-regional correlation, and
3. a **multivariate AR(1) with time fixed effects**, where aggregate fluctuations are explicitly separated from regional innovations.

The code is explicitly constructed to avoid a common identification error: **re-using the same shock parameters across models that impose different moment restrictions**. Each model therefore relies on parameters estimated on the data transformation consistent with its own specification.

The output of interest is a set of panel regressions linking firm-level sales growth to downstream demand growth, which can be compared across models and against empirical benchmarks.

---

## 2. Overview of the Three Shock Models

### 2.1 Model 1: Univariate (No Time Fixed Effects)

**Specification**

\[
\Delta \ln x_{r t} = \rho \, \Delta \ln x_{r,t-1} + u_{r t}, \qquad u_{r t} \sim i.i.d.\, \mathcal N(0, \sigma_u^2)
\]

**Key features**
- A single persistence parameter \(\rho\) pooled across regions.
- Innovations are independent across regions.
- Parameters are estimated on *raw (non-demeaned)* data, so aggregate fluctuations are absorbed into \(u_{rt}\).

This model serves as a benchmark commonly used in reduced-form network simulations.

---

### 2.2 Model 2: Multivariate (No Time Fixed Effects)

**Specification**

\[
\Delta \ln x_{r t} = \rho_r \, \Delta \ln x_{r,t-1} + u_{r t}, \qquad \mathbf u_t \sim \mathcal N(0, \Sigma)
\]

**Key features**
- Region-specific persistence parameters \(\rho_r\).
- A full covariance matrix \(\Sigma\) allowing for contemporaneous cross-region correlation.
- Parameters are estimated on *raw data*, so \(\Sigma\) includes both aggregate and regional components.

The unconditional covariance matrix \(\Gamma\) is constructed analytically as:
\[
\Gamma_{ij} = \frac{\Sigma_{ij}}{1 - \rho_i \rho_j},
\]
subject to numerical safeguards.

---

### 2.3 Model 3: Multivariate with Time Fixed Effects (MULTIVARIATE_FE)

**Specification**

\[
\Delta \ln x_{r t} = \mu_t + z_{r t}, \qquad
z_{r t} = \rho_r z_{r,t-1} + \varepsilon_{r t}, \quad \boldsymbol\varepsilon_t \sim \mathcal N(0, \Sigma_{\text{clean}})
\]

**Key features**
- Aggregate shocks are isolated through an explicit time fixed effect \(\mu_t\).
- Regional innovations are orthogonal to aggregate fluctuations.
- Parameters are estimated on **time-demeaned data**, ensuring that \(\Sigma_{\text{clean}}\) captures only regional spillovers.

This specification is the structurally correct counterpart to the multivariate model when aggregate demand shocks are present.

---

## 3. Core Data Structures

### 3.1 `SimulationConfig`

Encapsulates all simulation-level parameters:
- time horizon \(T\),
- baseline volatility and persistence (used only as fallbacks),
- random seed,
- shock model indicator,
- method for generating time fixed effects (empirical, resampled, or parametric).

This design ensures full reproducibility and transparent control of stochastic components.

---

### 3.2 Shock Parameter Containers

#### Univariate

`UnivariateShockParams` stores:
- pooled persistence \(\rho\),
- innovation standard deviation,
- unconditional standard deviation.

The innovation variance is internally backed out from the unconditional variance when not provided.

#### Multivariate

`MultivariateShockParams` stores:
- vector \(\rho_r\),
- innovation covariance \(\Sigma\),
- unconditional covariance \(\Gamma\),
- indices of *active downstream regions*,
- optional time fixed effects and their moments.

This structure allows a single simulation code path to accommodate models with and without time FE.

---

## 4. Dimension Handling and Active Regions

A recurring issue in network simulations is that not all regions are downstream-active. The code explicitly:

1. identifies regions with positive downstream activity,
2. estimates multivariate processes only on this subset, and
3. embeds the resulting shocks back into the full regional space with zeros elsewhere.

This avoids contaminating the covariance structure with mechanically inactive regions.

---

## 5. Loading and Validating Shock Parameters

Parameters are loaded from disk using model-specific filenames. The logic enforces:
- correct dimensionality,
- positive semi-definiteness of covariance matrices (via eigenvalue truncation),
- consistency between \(\rho_r\), \(\Sigma\), and the set of active regions.

Crucially:
- **MULTIVARIATE** parameters are taken from raw data.
- **MULTIVARIATE_FE** parameters are taken from demeaned data and supplemented with empirical \(\mu_t\).

This is the key conceptual correction relative to naive implementations.

---

## 6. Exposure Distributions and Network Weights

Firm-level exposure to downstream demand, \(a_{di}^D\), is drawn either:
- from an empirical sector-specific distribution, or
- from a bounded uniform distribution when data are unavailable.

The code also computes \(a_{rdi}^D\), the share of a supplier’s downstream sales going to each region, using equilibrium network objects. These shares determine how regional shocks aggregate into firm-level demand.

---

## 7. Shock Generation

### 7.1 Downstream Shocks

Each model has its own generator:
- **Univariate**: i.i.d. AR(1) across regions.
- **Multivariate**: vector AR(1) with correlated innovations.
- **Multivariate_FE**: multivariate regional component plus an independently generated time FE.

Time fixed effects can be:
- taken directly from the data,
- resampled from the empirical distribution, or
- drawn from a parametric normal distribution matching empirical moments.

---

### 7.2 Other-Customer Shocks

Non-downstream demand shocks are generated separately, either using a global \(\sigma_{sr}\) matrix or a fallback constant variance. These shocks are common across all three downstream specifications, ensuring comparability.

---

## 8. Firm-Level Sales Simulation

Firm sales growth is simulated as:

\[
\Delta \ln x_{i t} = a_{di}^D \sum_r a_{rdi}^D \, \Delta \ln x_{r t}
+ (1 - a_{di}^D) \, \Delta \ln x_{o i t}.
\]

Aggregate downstream growth is constructed using output-weighted averages **over active regions only**, ensuring internal consistency with the shock-generating process.

---

## 9. Unified Panel Construction and Regressions

A single panel stacks outcomes from all three shock models at the firm–time level. This enables side-by-side estimation of identical regressions under different data-generating processes.

Three regressions are run for each model:

1. Firm growth on aggregate downstream growth.
2. Firm growth on exposure-weighted downstream growth.
3. Firm growth on downstream growth net of other-customer shocks.

All regressions include firm fixed effects.

---

## 10. Main Validation Routine

`run_untargeted_validation` orchestrates the entire workflow:

1. Solves the production network.
2. Draws firm exposures.
3. Loads **model-specific** shock parameters.
4. Generates shocks for all three models.
5. Simulates firm outcomes.
6. Builds the unified panel and runs regressions.
7. Compares coefficients across models and, optionally, to empirical targets.

The final output is a structured dictionary containing shocks, parameters, simulated panels, and regression results.

---

## 11. Key Takeaway

The central methodological lesson embedded in this code is:

> **Shock processes with different identifying assumptions must be calibrated on differently transformed data.**

Failing to do so mechanically biases comparisons across models. This implementation provides a clean, internally consistent benchmark for untargeted-moment validation in network economies.

