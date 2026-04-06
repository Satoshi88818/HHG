HHGv7 — Hubbard + Lindblad High-Harmonic Generation Simulator

A Python simulator for high-harmonic generation (HHG) in correlated electron systems, modelling Hubbard lattices driven by ultrafast laser pulses under open-quantum-system (Lindblad) dissipation. Version 7 introduces major tensor-network upgrades, GPU acceleration, and an extended pulse library for attosecond science.

Table of Contents

Overview

What's New in v7

Installation

Quick Start

Configuration Reference

Lattice Types

Pulse Envelopes

Tensor Network Backend

Open-System Methods

Observables and Output

Plotting Utilities

Advanced Usage

Unit System

Feature Map

Benchmarking

Overview

HHGv7 solves the time-dependent Lindblad master equation for the Fermi-Hubbard model driven by an intense laser field, then extracts the HHG spectrum from the time-dependent current. It supports:

Multiple lattice geometries (1D chain, 2D square, honeycomb, triangular, kagome)

Spin-orbit coupling (Rashba) and perpendicular magnetic fields (Hofstadter physics)

Multiple dissipation channels (dephasing, momentum relaxation, electron-phonon)

Dense (QuTiP mesolve/mcsolve) and sparse tensor-network (quimb MPS/PEPS) backends

GPU acceleration via CuPy and JAX JIT compilation

Floquet quasi-energy spectra and intensity-scan parallelism

What's New in v7

1. True Time-Dependent Sparse MPO

The Hamiltonian MPO is now constructed directly from nearest-neighbour bond tensors and on-site terms, completely bypassing the dense MPO.from_dense call used in v6.1. Only the Peierls laser phases are patched in-place at each timestep, reducing memory and compute cost from O(d^{2N}) to O(N·χ³).

Key methods: _build_mpo_local_terms(), _build_mpo_sparse(), _update_mpo_phases().

2. Hybrid TN + Monte Carlo Open-System Methods

A new tn_open_method config option selects the open-system treatment:

"trajectory" — non-Hermitian Schrödinger evolution with stochastic quantum jumps on the MPS, averaged over ntraj_tn trajectories

"purification" — thermofield doubling: the density matrix is represented as a purified MPS on a doubled chain; the Lindblad super-operator is applied as an MPO

"none" — unitary MPS evolution (v6.1 behaviour)

3. 2D Tensor Networks (PEPS / Snake-MPS)

For 2D lattices a PEPS backend is available via use_peps=True (requires quimb ≥ 1.7 and lattice_type="2d_square"). When PEPS is unavailable the code automatically falls back to a snake-MPS ordering with a warning.

4. Automatic Bond-Dimension Adaptation

Setting chi_adapt=True enables entropy-driven bond-dimension control. After each TEBD/TDVP step the half-chain von Neumann entropy S is estimated from the Schmidt spectrum. If S is below chi_adapt_S_target, χ is reduced toward chi_min; if S exceeds the target, χ is grown up to chi_max. This avoids over-allocating memory during low-entanglement phases.

Parameters: chi_min, chi_adapt_S_target, chi_adapt_growth.

5. GPU Acceleration (CuPy + JAX)

use_gpu=True routes dense numpy/scipy operations through CuPy for matrix exponentials and dense linalg; quimb TN contractions automatically use the CuPy backend.

use_jax=True applies JAX JIT to the Peierls-phase update kernel and the entanglement entropy estimator.

Both options fall back gracefully to CPU with a warning when the required library is absent.

6. Extended Pulse Library and CEP Control

Four new pulse envelopes join the existing sin2, gaussian, and few_cycle options:

pulse_envelopeDescription"trapezoidal"Linear ramp-on / flat top / ramp-off, controlled by trap_ramp_cycles and trap_flat_cycles"chirped"Gaussian envelope with a linear frequency chirp chirp_b (rad/t²)"two_colour"Fundamental ω plus n-th harmonic with amplitude ratio second_colour_ratio"attosecond_train"Comb of n_attosecond_bursts sub-cycle Gaussian bursts per optical cycle under a sin² macro-envelope 

All envelopes respect carrier-envelope phase (phi_cep) and polarisation direction.

Installation

# Core dependencies pip install numpy scipy matplotlib qutip h5py networkx tqdm pyyaml # Tensor network backend (required for use_quimb=True) pip install quimb cotengra # Optional: GPU acceleration via CuPy (adjust CUDA version) pip install cupy-cuda12x # Optional: JAX backend pip install jax jaxlib # Optional: parallel intensity scans pip install joblib # Optional: PEPS backend (quimb >= 1.7) pip install "quimb>=1.7" 

Python 3.9+ is recommended. All optional dependencies degrade gracefully to CPU/dense fallbacks if not installed.

Quick Start

Minimal run (dense QuTiP backend)

from HHGv7 import SimConfig, HubbardHHGSimulator, plot_results cfg = SimConfig(lattice_type="1d_chain", nsites=4, U=2.0, gamma=0.01) sim = HubbardHHGSimulator(cfg) res = sim.run() plot_results([res]) 

Trapezoidal pulse with CEP control

cfg = SimConfig( lattice_type="1d_chain", nsites=4, U=3.0, gamma=0.01, pulse_envelope="trapezoidal", trap_ramp_cycles=1.5, # 1.5 optical cycles rise/fall trap_flat_cycles=5.0, # 5-cycle flat top phi_cep=3.14159 / 2, # cosine-like CEP ) sim = HubbardHHGSimulator(cfg) res = sim.run() 

Sparse MPO + adaptive bond dimension (MPS backend)

cfg = SimConfig( lattice_type="1d_chain", nsites=10, U=6.0, gamma=0.01, use_quimb=True, chi_max=128, chi_adapt=True, chi_min=8, chi_adapt_S_target=0.6, track_entanglement=True, ) sim = HubbardHHGSimulator(cfg) res = sim.run() print("Final effective chi:", res["chi_history"][-1]) 

Quantum-trajectory open-system TN

cfg = SimConfig( lattice_type="1d_chain", nsites=8, U=4.0, gamma=0.05, dissipation_model="dephasing", use_quimb=True, chi_max=64, tn_open_method="trajectory", ntraj_tn=100, ) sim = HubbardHHGSimulator(cfg) res = sim.run() 

Two-colour laser on a honeycomb lattice with GPU

cfg = SimConfig( lattice_type="honeycomb", honeycomb_m=2, honeycomb_n=2, U=2.0, gamma=0.01, pulse_envelope="two_colour", second_colour_ratio=0.3, second_colour_n=2, use_gpu=True, # CuPy if available, else CPU fallback ) sim = HubbardHHGSimulator(cfg) res = sim.run() 

Configuration Reference

All parameters are set on the SimConfig dataclass. Invalid combinations raise ValueError; near-invalid combinations emit warnings.warn.

Lattice

ParameterDefaultDescriptionlattice_type"1d_chain"Geometry: "1d_chain", "2d_square", "honeycomb", "triangular", "kagome"nsites4Total number of lattice siteshoneycomb_m, honeycomb_n2, 2Honeycomb supercell dimensionsperiodicFalsePeriodic boundary conditionssublattice_delta0.0Staggered on-site potential for honeycomb (opens a gap)a1.0Lattice constant 

Hubbard Parameters

ParameterDefaultDescriptiont1.0Nearest-neighbour hopping amplitudeU2.0On-site Hubbard interactionsoc_lambda0.0Rashba spin-orbit coupling strengthB_field0.0Perpendicular magnetic field (Peierls phase) 

Dissipation

ParameterDefaultDescriptiongamma0.01Uniform dissipation rategamma_tNoneOptional callable gamma_t(t) for time-dependent rategamma_siteNoneArray of per-site rates (length nsites)dissipation_model"dephasing""dephasing", "momentum_relaxing", or "electron_phonon" 

Particle Number

ParameterDefaultDescriptionhalf_fillingFalseProject ground state onto half-filling sector (requires even nsites) 

Ground-State Preparation

ParameterDefaultDescriptionuse_gutzwillerFalseStart from a Gutzwiller-projected mean-field stategutzwiller_g1.0Gutzwiller parameter ∈ [0, 1]use_td_gutzwillerFalseSelf-consistently evolve the Gutzwiller parameter during dynamicsuse_imaginary_timeFalsePrepare ground state via imaginary-time evolution (RK45)imag_time_tau20.0Maximum imaginary timeimag_time_steps400Number of imaginary-time steps 

Driving Field

ParameterDefaultDescriptionE00.5Peak electric field amplitudeomega0.5Laser frequencypulse_cycles5Total pulse duration in optical cyclespolarization[1.0, 0.0]Polarisation unit vector (auto-normalised)pulse_envelope"sin2"Envelope shape (see Pulse Envelopes)sigma_cycles2.0Gaussian envelope width in cyclesphi_cep0.0Carrier-envelope phase (rad)chirp_b0.0Linear chirp rate (rad/t²) for "chirped" envelopesecond_colour_ratio0.1E₂/E₁ amplitude ratio for "two_colour"second_colour_n3Harmonic order of second colourn_attosecond_bursts4Bursts per optical cycle for "attosecond_train"trap_ramp_cycles1.0Rise/fall ramp length in cycles for "trapezoidal"trap_flat_cycles3.0Flat-top length in cycles for "trapezoidal" 

Tensor Network / MPS Parameters

ParameterDefaultDescriptionuse_quimbFalseUse quimb MPS backend instead of QuTiP dense solverchi_max256Maximum MPS bond dimensiontruncation_cutoff1e-10Schmidt value cutoff for MPS compressionuse_tdvpFalseUse TDVP instead of TEBD for MPS time evolutionuse_number_conservationFalseEnforce U(1) number conservation in MPS (requires half_filling=True)use_pepsFalseUse 2D PEPS backend for square lattices (quimb ≥ 1.7 required)tn_open_method"none"Open-system TN method: "none", "trajectory", or "purification"ntraj_tn50Number of trajectories for tn_open_method="trajectory"chi_adaptFalseEnable entropy-driven adaptive bond dimensionchi_min4Minimum bond dimension when adaptingchi_adapt_S_target0.5Target half-chain entropy (bits) for adaptationchi_adapt_growth1.25Multiplicative growth/shrink factor for adaptive χ 

Acceleration

ParameterDefaultDescriptionuse_gpuFalseRoute dense ops through CuPy; use CuPy backend for quimb TNuse_jaxFalseJIT-compile Peierls-phase update and entropy estimator via JAX 

Numerics and Solver

ParameterDefaultDescriptionn_time_points0Number of time steps (0 = auto: 2048 for N≤6, else 1024)zero_pad_factor2FFT zero-padding factor for HHG spectrumuse_sparseTrueUse sparse matrix representation for QuTiP operatorssolver_method"adams"ODE solver method ("adams", "bdf")solver_atol1e-8Absolute tolerancesolver_rtol1e-6Relative tolerancesolver_nsteps1_000_000Maximum ODE solver steps 

Output and Observables

ParameterDefaultDescriptionsave_hdf5FalseSave results to HDF5hdf5_path"hhg_results_v7.h5"HDF5 output pathhdf5_compressTrueEnable gzip compression in HDF5track_spin_corrFalseCompute ⟨Sᵢ·Sⱼ⟩ spin correlationstrack_entanglementFalseRecord half-chain entanglement entropy (requires use_quimb=True)benchmark_modeFalsePrint timing information 

Parallelism

ParameterDefaultDescriptionn_parallel_workers1Worker processes for floquet_intensity_scanuse_joblibFalseUse joblib instead of concurrent.futures for parallelism 

Floquet

ParameterDefaultDescriptionfloquet_n_periods10Number of driving periods for Floquet analysisfloquet_magnus_order2Magnus expansion order (1 or 2) 

Lattice Types

lattice_typeNotes"1d_chain"1D chain, open or periodic. Most efficient backend."2d_square"2D square lattice; nsites must be a perfect square. Supports PEPS backend."honeycomb"Honeycomb lattice built from honeycomb_m × honeycomb_n unit cells. Supports sublattice_delta."triangular"Triangular lattice. Frustrated geometry."kagome"Kagome lattice; nsites must be divisible by 3. 

Pulse Envelopes

All envelopes respect CEP (phi_cep) and polarisation. The vector potential A(t) and electric field E(t) = −dA/dt are computed analytically.

pulse_envelopeShapeKey parameters"sin2"sin²(πt/T)pulse_cycles"gaussian"Gaussiansigma_cycles"few_cycle"cos²(πt/T)pulse_cycles"trapezoidal"Ramp / flat / ramptrap_ramp_cycles, trap_flat_cycles"chirped"Gaussian + linear chirpsigma_cycles, chirp_b"two_colour"Fundamental + harmonicsecond_colour_ratio, second_colour_n"attosecond_train"Sub-cycle burst combn_attosecond_bursts, pulse_cycles 

Visualise pulses side-by-side with:

from HHGv7 import plot_pulse_comparison fig = plot_pulse_comparison([cfg_trap, cfg_chirp, cfg_2c, cfg_as]) 

Tensor Network Backend

Enable with use_quimb=True. Requires pip install quimb cotengra.

The v7 MPS backend builds the time-dependent Hamiltonian as a sparse MPO from pre-computed bond/on-site tensor templates. Only Peierls phases are patched each timestep — O(N·χ³) cost versus O(d^{2N}) for the dense MPO approach.

Time evolution uses TEBD (default) or TDVP (use_tdvp=True). For 2D square lattices, set use_peps=True for a true 2D PEPS contraction (quimb ≥ 1.7 required); otherwise, a snake-MPS ordering is used automatically as a fallback.

Adaptive Bond Dimension

cfg = SimConfig( use_quimb=True, chi_max=256, # hard ceiling chi_adapt=True, # enable adaptation chi_min=8, # minimum allowed chi chi_adapt_S_target=0.5, # target entropy in bits chi_adapt_growth=1.25, # grow/shrink factor per step track_entanglement=True, # record S(t) in output ) 

The history of effective χ is returned in result["chi_history"].

Open-System Methods

Dense backend (QuTiP)

By default, qt.mesolve propagates the Lindblad master equation. For N > 8–10, qt.mcsolve (Monte Carlo wave-function) is used automatically. Switch explicitly with sim.run(use_mcsolve=True, ntraj=500).

Tensor network open-system (tn_open_method)

"trajectory": Each trajectory evolves a pure MPS under the non-Hermitian effective Hamiltonian H_eff = H(t) − i/2 Σₖ Lₖ†Lₖ, then applies stochastic quantum jumps. Results are averaged over ntraj_tn trajectories.

"purification": The density matrix is represented as a purified MPS on a doubled chain (thermofield doubling). The Lindblad super-operator acts as an MPO on system + ancilla sites. More memory-intensive but trajectory-free.

Observables and Output

sim.run() returns a dictionary with the following keys:

KeyShapeDescriptiontlist(T,)Time arrayJx, Jy(T,)Total current componentsJx_para, Jy_para(T,)Paramagnetic currentJx_dia, Jy_dia(T,)Diamagnetic currentfreq_x, freq_y(F,)Frequency arrays (in units of ω₀)spectrum_x, spectrum_y(F,)HHG power spectrum |ω²J(ω)|²double_occ_t(T,)Time-resolved double occupancy D(t)total_density_t(T,)Total electron densitysite_density_t(T, N)Per-site densityspin_corr_tlist⟨SᵢSⱼ⟩ (if track_spin_corr=True)entanglement_t(T,)Half-chain entropy in bits (if track_entanglement=True)chi_historylistEffective bond dimension per step (if chi_adapt=True)k_pts(N, 2)BZ k-points (if track_momentum=True)n_k_t, J_k_t(T, N, ...)Momentum-resolved density and current 

Saving to HDF5

cfg = SimConfig(save_hdf5=True, hdf5_path="results.h5", hdf5_compress=True) 

The HDF5 file stores all numerical arrays plus the full SimConfig as metadata attributes.

Config serialisation

cfg.save("config.json") # or config.yaml if pyyaml is installed cfg2 = SimConfig.from_file("config.json") 

Plotting Utilities

FunctionDescriptionplot_results(results, titles)Current J(t), HHG spectrum, entanglement/double-occ panelsplot_pulse_comparison(cfg_list)Electric field E(t) for each pulse envelope side-by-sideplot_entanglement_and_chi(result)S(t) and adaptive χ historyplot_current_decomposition(result)Para- vs diamagnetic current split + spectrumplot_floquet_spectrum(floquet_result)Quasi-energy spectrumplot_intensity_scan(scan_result)HHG yield vs laser intensity for each harmonicplot_momentum_density(result, time_idx)n(k) scatter plot in the BZsim.plot_lattice()NetworkX graph of the lattice 

Interactive Plotly HTML export is available via plot_results(..., save_plotly=True) (requires pip install plotly).

Advanced Usage

Floquet quasi-energy spectrum

fl = sim.compute_floquet_spectrum() fig = plot_floquet_spectrum(fl) 

Uses a Trotterised Magnus expansion of order 1 or 2 (floquet_magnus_order).

Hofstadter butterfly

import numpy as np result = sim.hofstadter_butterfly(B_values=np.linspace(0, 1, 100), n_bands=20) 

Requires periodic=True.

Intensity scan (parallelised)

from HHGv7 import floquet_intensity_scan scan = floquet_intensity_scan( base_config=cfg, E0_values=np.linspace(0.1, 1.0, 20), harmonic_orders=[1, 3, 5, 7, 9], n_parallel_workers=4, ) 

Profiling

result, profile_str = sim.profile_run(track_observables=False) print(profile_str) 

Command-line interface

# Run benchmark python HHGv7.py --benchmark # Run from a JSON/YAML config file python HHGv7.py --config my_config.json 

Unit System

All quantities use natural units for lattice models:

QuantityUnitħ, e, m_eff1 (dimensionless)Energieshopping tTimeħ/tFrequencyt/ħVector potential A(t)ħ/(e·a)Electric field E(t)ħω/(e·a)Current J(t)e·t·a/ħHHG intensitye²·t²·a²·ω⁴/ħ⁶ 

Feature Map

SimConfig flagFeatureBackendpulse_envelope="trapezoidal"Linear ramp-on / flat / ramp-offAllpulse_envelope="chirped"Linear chirp in frequencyAllpulse_envelope="two_colour"Fundamental + n-th harmonicAllpulse_envelope="attosecond_train"Sub-cycle burst combAlluse_quimb=TrueMPS evolution (sparse MPO, v7)quimbchi_adapt=TrueEntropy-driven adaptive χquimbuse_peps=True2D PEPS (quimb ≥ 1.7)quimbtn_open_method="trajectory"Quantum jumps on MPSquimbtn_open_method="purification"Thermofield doubling MPOquimbuse_gpu=TrueCuPy dense linalg + quimb TNCuPyuse_jax=TrueJAX matrix exponentials + JITJAXdissipation_model="dephasing"Local density dephasingAlldissipation_model="momentum_relaxing"Bond hopping collapse opsAlldissipation_model="electron_phonon"Position-momentum couplingAlltrack_entanglement=TrueHalf-chain S(t)quimbsave_hdf5=TrueCompressed HDF5 outputAll 

Benchmarking

from HHGv7 import run_benchmark results = run_benchmark( lattice_types=["1d_chain", "2d_square"], N_values=[4, 6, 8], n_time_points=256, ) 

Or from the command line:

python HHGv7.py --benchmark 

Practical guidance on system sizes:

N ≤ 6, dense backend: fast (seconds)

N ≤ 10, dense backend: feasible (minutes to hours)

N > 10: use use_quimb=True with appropriate chi_max

N > 12, 2D: use use_peps=True or snake-MPS with large χ

GPU (use_gpu=True) gives the largest speedup for dense linalg at N = 8–12 and for TN contraction at large χ

Changelog: v6.1 → v7

True time-dependent MPO without full dense matrix per step (nearest-neighbour + on-site structure) ✅

Hybrid TN + Monte Carlo open-system methods (MPO-Lindblad via quantum trajectories / purification) ✅

2D tensor networks (PEPS / snake-MPS) — optional 2D PEPS backend for square lattices ✅

Automatic bond-dimension adaptation based on entanglement growth (not just fixed chi_max) ✅

GPU acceleration via quimb + CuPy or JAX backend for TN contraction ✅

Advanced pulse shapes & CEP control for attosecond science (trapezoidal, chirped, multi-colour, attosecond trains) ✅

