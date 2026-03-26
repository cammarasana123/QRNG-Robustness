"""
Readout Error Threshold per QRNG — v10
========================================
Estensione di v9 con quattro aggiunte metodologiche significative:

AGGIUNTA 1 — XEB (Cross-Entropy Benchmarking) per asse
  Calcola F_XEB = 2^n * E_x[p_ideal(x)] - 1 per ogni livello di rumore.
  Richiede statevector per p_ideal, poi campiona con noise_model.
  F_XEBn ~ 1.0 = circuito ideale, F_XEB ~ 0.0 = rumore totale.
  Fornisce una metrica del contenuto quantistico INDIPENDENTE dai test NIST,
  citabile direttamente dalla letteratura RCS (Google 2019, Wu 2021).
  La correlazione XEB ↔ NIST pass/fail è il risultato scientifico principale.

AGGIUNTA 2 — Analisi RAW vs PA per asse A
  Per ogni livello di readout noise, confronta:
    (a) distribuzione del peso di Hamming grezzo (raw output)
    (b) output dopo Privacy Amplification
  Mostra che la PA fa qualcosa di misurabile e non maschera semplicemente rumore.
  Metriche: TV distance dalla binomiale ideale (raw), bias e freq_p (PA).

AGGIUNTA 3 — Autocorrelogramma a lag multipli (1..20)
  Il serial_corr di v9 testa solo lag-1. Questo test verifica correlazioni
  a lag 2..20, che possono emergere dalla struttura brick-wall del circuito.
  Output: profilo r(lag) con soglia Bonferroni per lag multipli.

AGGIUNTA 4 — Asse B2: gate depolarizing puro (senza thermal relaxation)
  Separa l'effetto del gate depolarizing dall'effetto di collapse verso |0⟩
  causato dal thermal relaxation (che introduce un bias strutturale diverso).
  Confrontando Asse B (depol + thermal) e Asse B2 (solo depol) si isola
  quale dei due meccanismi è dominante nella degradazione.

AGGIUNTA 5 — Sensitivity analysis sulla discard rule della PA
  Testa tre soglie di scarto: hw=0, hw≤1, hw≤2.
  Mostra che il risultato è robusto alla scelta della threshold.

Invariato da v9:
  - Correzione di Bonferroni per test multipli per asse
  - 5 run indipendenti con seed ortogonali, majority vote ≥ 3/5
  - RCS Haar-corretto (u3 con campionamento Haar su SU(2))
  - Privacy Amplification: inner product per-shot con seed separato
  - 4 test NIST-subset: frequency, runs, uniformity, serial_corr
  - Asse A (readout 0..50%), Asse B (gate isolato), Asse C (griglia 2D)
"""

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("📦 Controllo dipendenze...")
for pkg in ["qiskit", "qiskit-aer", "numpy", "scipy"]:
    try:
        if pkg == "qiskit-aer": import qiskit_aer
        else: __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"  Installo {pkg}..."); install(pkg)

import numpy as np
from scipy.stats import chisquare, binom as sp_binom
from scipy.special import erfc
import time
import warnings
warnings.filterwarnings("ignore")

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, ReadoutError,
    depolarizing_error, thermal_relaxation_error
)

print("✅ Dipendenze OK\n")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE GLOBALE
# ══════════════════════════════════════════════════════════════════════
N_QUBITS    = 8
N_LAYERS    = 12
N_SHOTS     = 8192
N_CIRCUITS  = 50
ASYMMETRY   = 1.5      # asimmetria readout: p(0|1) = ASYMMETRY * p(1|0)

NIST_THRESH = 0.01
N_TESTS     = 4
N_RUNS      = 5
MAJORITY    = 3        # majority vote threshold

# Seed ortogonali per i 5 run
RUN_CONFIGS = [
    {"circuit_seed": 42,          "key_seed": 2**31 - 1,   "noise_offset": 0},
    {"circuit_seed": 1_000_042,   "key_seed": 2**29 - 3,   "noise_offset": 1_000_000},
    {"circuit_seed": 2_000_042,   "key_seed": 2**27 - 39,  "noise_offset": 2_000_000},
    {"circuit_seed": 3_000_042,   "key_seed": 2**25 - 39,  "noise_offset": 3_000_000},
    {"circuit_seed": 4_000_042,   "key_seed": 2**23 - 15,  "noise_offset": 4_000_000},
]

# ─── Asse A ───────────────────────────────────────────────────────────
READOUT_RATES_A = [
    0.000, 0.020, 0.050, 0.100,
    0.150, 0.200, 0.300, 0.400, 0.500
]

# ─── Asse B (gate + thermal) e B2 (solo depolarizing) ────────────────
GATE_RATES_B  = [0.000, 0.001, 0.003, 0.005, 0.010, 0.020, 0.050]

T1_NS        = 50_000
T2_NS        = 30_000
GATE_TIME_NS = 50
CX_TIME_NS   = 300

# ─── Asse C ───────────────────────────────────────────────────────────
READOUT_GRID = [0.005, 0.015, 0.030, 0.100]
GATE_GRID    = [0.001, 0.005, 0.010, 0.020]

# Circuiti per XEB (meno degli altri perché richiede statevector)
N_CIRCUITS_XEB = 20
N_SHOTS_XEB    = 4096

# Sensitivity: soglie minime di Hamming weight da tenere nella PA
PA_DISCARD_THRESHOLDS = [0, 1, 2]   # scarta shot con hw <= soglia

DEVICE_REFS = {0.005: "Quantinuum H2", 0.015: "IBM Heron", 0.030: "Sycamore"}
TEST_NAMES  = ["frequency", "runs", "uniformity", "serial_corr"]

# ══════════════════════════════════════════════════════════════════════
# CIRCUITO RCS — rotazioni Haar-corrette
# ══════════════════════════════════════════════════════════════════════
def make_rcs_circuit(n_qubits, n_layers, seed=None, with_measure=True):
    """
    RCS con u3(θ,φ,λ), θ=2·arcsin(√u), u~U[0,1], φ,λ~U[0,2π].
    Se with_measure=False restituisce il circuito senza misure (per XEB).
    """
    rng = np.random.default_rng(seed)
    qc  = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    for layer in range(n_layers):
        u   = rng.uniform(0, 1,       n_qubits)
        phi = rng.uniform(0, 2*np.pi, n_qubits)
        lam = rng.uniform(0, 2*np.pi, n_qubits)
        theta = 2 * np.arcsin(np.sqrt(u))
        for q in range(n_qubits):
            qc.u(theta[q], phi[q], lam[q], q)
        if layer % 2 == 0:
            for i in range(0, n_qubits - 1, 2): qc.cx(i, i + 1)
        else:
            for i in range(1, n_qubits - 1, 2): qc.cx(i, i + 1)
    if with_measure:
        qc.measure_all()
    return qc

# ══════════════════════════════════════════════════════════════════════
# PRIVACY AMPLIFICATION
# ══════════════════════════════════════════════════════════════════════
def privacy_amplification(memory, pa_keys, circuit_idx, min_hw=0):
    """
    Estrae 1 bit per shot via inner product <b, k> mod 2.
    Scarta shot con peso di Hamming ≤ min_hw (default 0, scarta solo b=0).
    pa_keys: array (N_CIRCUITS, N_SHOTS, N_QUBITS) pre-generato.
    """
    keys_circ = pa_keys[circuit_idx]
    bits      = []
    for shot_idx, shot_str in enumerate(memory):
        b = np.frombuffer(
            bytearray(int(c) for c in shot_str.replace(" ", "")),
            dtype=np.uint8
        )
        if np.sum(b) <= min_hw:
            continue
        ip = int(np.dot(b, keys_circ[shot_idx]) % 2)
        bits.append(ip)
    return np.array(bits, dtype=np.uint8)

def make_pa_keys(key_seed):
    """Genera la matrice di chiavi PA con seed separato."""
    rng = np.random.default_rng(key_seed)
    return rng.integers(0, 2, size=(N_CIRCUITS, N_SHOTS, N_QUBITS), dtype=np.uint8)

# ══════════════════════════════════════════════════════════════════════
# MODELLI DI RUMORE
# ══════════════════════════════════════════════════════════════════════
def make_readout_noise(p_ro, n_qubits, noise_offset=0):
    if p_ro == 0.0: return None
    nm       = NoiseModel()
    rng      = np.random.default_rng(int(round(p_ro * 1_000_000)) + 1 + noise_offset)
    p01_base = p_ro / (1 + ASYMMETRY)
    p10_base = ASYMMETRY * p01_base
    for q in range(n_qubits):
        j   = rng.uniform(0.85, 1.15)
        p01 = min(p01_base * j, 0.45)
        p10 = min(p10_base * j, 0.45)
        nm.add_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [q])
    return nm

def make_gate_noise_full(p_gate, n_qubits, noise_offset=0):
    """CX depolarizing + U thermal relaxation."""
    if p_gate == 0.0: return None
    nm  = NoiseModel()
    rng = np.random.default_rng(int(round(p_gate * 1_000_000)) + 1 + noise_offset + 9_000_000)
    for i in range(n_qubits - 1):
        nm.add_quantum_error(depolarizing_error(p_gate, 2), ['cx'], [i, i + 1])
    t1e = max(T1_NS * (1 - p_gate), GATE_TIME_NS * 10)
    t2e = min(T2_NS, 2 * t1e)
    for q in range(n_qubits):
        j   = rng.uniform(0.90, 1.10)
        t1q = max(t1e * j, GATE_TIME_NS * 5)
        t2q = min(t2e * j, 2 * t1q)
        nm.add_quantum_error(thermal_relaxation_error(t1q, t2q, GATE_TIME_NS), ['u'], [q])
    return nm

def make_gate_noise_depol_only(p_gate, n_qubits, noise_offset=0):
    """Solo CX depolarizing, nessun thermal (per Asse B2)."""
    if p_gate == 0.0: return None
    nm = NoiseModel()
    for i in range(n_qubits - 1):
        nm.add_quantum_error(depolarizing_error(p_gate, 2), ['cx'], [i, i + 1])
    return nm

def make_combined_noise(p_ro, p_gate, n_qubits, noise_offset=0):
    if p_ro == 0.0 and p_gate == 0.0: return None
    nm = NoiseModel()
    if p_ro > 0.0:
        rng_r = np.random.default_rng(int(round(p_ro * 1_000_000)) + 1 + noise_offset)
        p01b  = p_ro / (1 + ASYMMETRY); p10b = ASYMMETRY * p01b
        for q in range(n_qubits):
            j   = rng_r.uniform(0.85, 1.15)
            p01 = min(p01b * j, 0.45); p10 = min(p10b * j, 0.45)
            nm.add_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [q])
    if p_gate > 0.0:
        rng_g = np.random.default_rng(int(round(p_gate * 1_000_000)) + 1 + noise_offset + 9_000_000)
        for i in range(n_qubits - 1):
            nm.add_quantum_error(depolarizing_error(p_gate, 2), ['cx'], [i, i + 1])
        t1e = max(T1_NS * (1 - p_gate), GATE_TIME_NS * 10)
        t2e = min(T2_NS, 2 * t1e)
        for q in range(n_qubits):
            j   = rng_g.uniform(0.90, 1.10)
            t1q = max(t1e * j, GATE_TIME_NS * 5); t2q = min(t2e * j, 2 * t1q)
            nm.add_quantum_error(thermal_relaxation_error(t1q, t2q, GATE_TIME_NS), ['u'], [q])
    return nm

# ══════════════════════════════════════════════════════════════════════
# TEST STATISTICI
# ══════════════════════════════════════════════════════════════════════
def test_frequency(bits):
    n = len(bits)
    if n == 0: return 0.0
    s = abs(np.count_nonzero(bits) - (n - np.count_nonzero(bits)))
    return float(erfc(s / (np.sqrt(n) * np.sqrt(2))))

def test_runs(bits):
    n = len(bits); pi = np.mean(bits)
    if abs(pi - 0.5) >= 2 / np.sqrt(n): return 0.0
    runs = 1 + np.sum(bits[:-1] != bits[1:])
    exp  = 2 * n * pi * (1 - pi)
    var  = 2 * np.sqrt(2 * n) * pi * (1 - pi)
    return float(erfc(abs(runs - exp) / var))

def test_uniformity(bits):
    n_bytes = len(bits) // 8
    if n_bytes < 50: return 1.0
    values  = bits[:n_bytes*8].reshape(n_bytes, 8).dot(2**np.arange(7, -1, -1))
    counts, _ = np.histogram(values, bins=16, range=(0, 256))
    _, p = chisquare(counts, f_exp=[n_bytes / 16] * 16)
    return float(p)

def test_serial_correlation(bits):
    b    = bits.astype(float); mean = np.mean(b); var = np.var(b)
    if var == 0: return 0.0
    ac   = np.mean((b[:-1] - mean) * (b[1:] - mean))
    r    = ac / var
    z    = r * np.sqrt(len(b) - 1)
    return float(erfc(abs(z) / np.sqrt(2)))

TEST_FUNCS = [test_frequency, test_runs, test_uniformity, test_serial_correlation]

def evaluate_bits(bits):
    return {n: f(bits) for n, f in zip(TEST_NAMES, TEST_FUNCS)}

# ══════════════════════════════════════════════════════════════════════
# AGGIUNTA 1 — XEB


def compute_xeb(simulator, noise_model, circuit_seeds, n_qubits, n_layers,
                shots=N_SHOTS_XEB):
    """
    Restituisce:
      - mean_xeb_raw:      F_XEB = 2^n E[p_ideal(x)] - 1
      - std_xeb_raw
      - mean_xeb_norm:     F_XEB / F_XEB(ideal-circuit baseline)
      - std_xeb_norm
      - list of per-circuit diagnostics
    """
    sv_sim = AerSimulator(method='statevector')
    n_out  = 2 ** n_qubits

    xeb_raw_list  = []
    xeb_norm_list = []
    diag_list     = []

    for seed in circuit_seeds:
        # 1) circuito ideale senza misure
        qc_sv = make_rcs_circuit(n_qubits, n_layers, seed=seed, with_measure=False)
        qc_sv.save_statevector()
        sv = sv_sim.run(qc_sv).result().get_statevector()
        p_ideal = np.abs(np.array(sv)) ** 2
        p_ideal = p_ideal / np.sum(p_ideal)   # sicurezza numerica

        # baseline ideale del circuito
        ideal_baseline = n_out * np.sum(p_ideal ** 2) - 1.0

        # 2) campionamento noisy
        qc_m = make_rcs_circuit(n_qubits, n_layers, seed=seed, with_measure=True)
        counts = simulator.run(
            qc_m,
            noise_model=noise_model,
            shots=shots
        ).result().get_counts()

        total = sum(counts.values())

        # media empirica di p_ideal(x) sugli output osservati
        sampled_pideal = []
        for bitstr, cnt in counts.items():
            idx = int(bitstr.replace(" ", ""), 2)
            sampled_pideal.extend([p_ideal[idx]] * cnt)

        mean_pideal = float(np.mean(sampled_pideal))
        xeb_raw = n_out * mean_pideal - 1.0

        # normalizzazione rispetto alla baseline ideale del circuito
        xeb_norm = xeb_raw / ideal_baseline if ideal_baseline > 0 else np.nan

        xeb_raw_list.append(xeb_raw)
        xeb_norm_list.append(xeb_norm)
        diag_list.append({
            "seed": seed,
            "ideal_baseline": ideal_baseline,
            "xeb_raw": xeb_raw,
            "xeb_norm": xeb_norm,
        })

    return (
        float(np.mean(xeb_raw_list)),
        float(np.std(xeb_raw_list)),
        float(np.mean(xeb_norm_list)),
        float(np.std(xeb_norm_list)),
        diag_list
    )

# ══════════════════════════════════════════════════════════════════════
# AGGIUNTA 2 — RAW vs PA
# ══════════════════════════════════════════════════════════════════════
def analyze_raw_vs_pa(simulator, noise_model, circuit_seed, pa_keys,
                      n_circuits=10, shots=2048):
    """
    Per un dato noise_model, raccoglie:
      - Distribuzione del peso di Hamming (raw output)
      - TV distance dalla distribuzione Binomiale(8, 0.5) ideale
      - Bias e freq_p dell'output PA

    La TV distance misura quanto il raw output si discosta dall'ideale.
    Se TV ≫ 0 ma freq_p(PA) ≥ 0.01, la PA sta effettivamente correggendo il bias.
    """
    hw_counts   = np.zeros(N_QUBITS + 1)
    pa_bits_all = []
    total_shots = 0

    for i in range(n_circuits):
        qc  = make_rcs_circuit(N_QUBITS, N_LAYERS, seed=circuit_seed + i)
        mem = simulator.run(qc, noise_model=noise_model,
                            shots=shots, memory=True).result().get_memory()
        total_shots += len(mem)

        # Raw: peso di Hamming
        for s in mem:
            b = sum(int(c) for c in s.replace(' ', ''))
            hw_counts[b] += 1

        # PA
        pa_bits_all.append(privacy_amplification(mem, pa_keys, circuit_idx=i % N_CIRCUITS))

    hw_probs   = hw_counts / total_shots
    binom_probs = sp_binom.pmf(np.arange(N_QUBITS + 1), N_QUBITS, 0.5)
    tv_dist    = 0.5 * np.sum(np.abs(hw_probs - binom_probs))

    pa_arr    = np.concatenate(pa_bits_all)
    pa_bias   = float(np.mean(pa_arr) - 0.5)
    pa_freq_p = test_frequency(pa_arr)

    return {
        "tv_distance":  tv_dist,
        "hw_probs":     hw_probs,
        "pa_bias":      pa_bias,
        "pa_freq_p":    pa_freq_p,
        "pa_n_bits":    len(pa_arr),
    }

# ══════════════════════════════════════════════════════════════════════
# AGGIUNTA 3 — AUTOCORRELOGRAMMA A LAG MULTIPLI
# ══════════════════════════════════════════════════════════════════════
def autocorrelogram(bits, max_lag=20):
    """
    Calcola l'autocorrelazione a lag 1..max_lag con test z per ogni lag.
    Soglia Bonferroni: α_corr = 0.01 / max_lag.
    Ritorna: dict lag → {r, p, significant}
    """
    b       = bits.astype(float)
    mean    = np.mean(b)
    var     = np.var(b)
    bonf    = NIST_THRESH / max_lag
    results = {}

    for lag in range(1, max_lag + 1):
        n   = len(b) - lag
        ac  = np.mean((b[:n] - mean) * (b[lag:lag + n] - mean))
        r   = ac / var if var > 0 else 0.0
        z   = r * np.sqrt(n)
        p   = float(erfc(abs(z) / np.sqrt(2)))
        results[lag] = {"r": r, "p": p, "significant": p < bonf}

    return results, bonf

def print_autocorrelogram(acg, bonf, label=""):
    n_sig = sum(1 for v in acg.values() if v["significant"])
    print(f"\n  Autocorrelogramma {label}  "
          f"(Bonferroni α={bonf:.4f}, lag 1..{len(acg)})")
    print(f"  Lag significativi: {n_sig}/{len(acg)}")

    # Stampa solo i lag con r rilevante o significativo
    for lag, v in acg.items():
        if abs(v["r"]) > 0.005 or v["significant"]:
            mark = "  ← *SIGNIFICATIVO*" if v["significant"] else ""
            print(f"    lag {lag:>2}: r={v['r']:+.5f}  p={v['p']:.4f}{mark}")

    if n_sig == 0:
        print("    (nessun lag significativo — buona indipendenza temporale)")

# ══════════════════════════════════════════════════════════════════════
# AGGIUNTA 5 — SENSITIVITY ANALYSIS SULLA PA DISCARD RULE
# ══════════════════════════════════════════════════════════════════════
def sensitivity_pa_discard(simulator, noise_model, circuit_seed,
                            pa_keys, label=""):
    """
    Esegue N_CIRCUITS circuiti e calcola freq_p e bias PA
    per tre diverse soglie di scarto del peso di Hamming: 0, 1, 2.
    Mostra che i risultati NIST sono robusti alla scelta della soglia.
    """
    all_memory = []
    for i in range(N_CIRCUITS):
        qc  = make_rcs_circuit(N_QUBITS, N_LAYERS, seed=circuit_seed + i)
        mem = simulator.run(qc, noise_model=noise_model,
                            shots=N_SHOTS, memory=True).result().get_memory()
        all_memory.append((i, mem))

    print(f"\n  Sensitivity PA discard rule  {label}")
    print(f"  {'Soglia scarto':>14}  {'N bit':>8}  "
          f"{'Efficienza':>10}  {'Bias':>8}  {'freq_p':>7}  {'freq_pass':>9}")
    print(f"  {'─'*14}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*9}")

    results = {}
    total_raw = N_CIRCUITS * N_SHOTS

    for min_hw in PA_DISCARD_THRESHOLDS:
        pa_bits = []
        for circ_idx, mem in all_memory:
            pa_bits.append(privacy_amplification(mem, pa_keys, circ_idx,
                                                  min_hw=min_hw))
        arr  = np.concatenate(pa_bits)
        bias = float(np.mean(arr) - 0.5)
        fp   = test_frequency(arr)
        eff  = len(arr) / total_raw
        mark = "✅" if fp >= NIST_THRESH else "❌"
        print(f"  hw > {min_hw}            "
              f"{len(arr):>8,}  {eff:>10.2%}  "
              f"{bias:>+8.5f}  {fp:>7.4f}  {mark}")
        results[min_hw] = {"n_bits": len(arr), "bias": bias, "freq_p": fp}

    return results

# ══════════════════════════════════════════════════════════════════════
# CORE: esegui N_CIRCUITS e restituisce stats NIST
# ══════════════════════════════════════════════════════════════════════
def run_circuits(simulator, noise_model, pa_keys, circuit_seed, bonf_thresh):
    all_bits = []
    for i in range(N_CIRCUITS):
        qc  = make_rcs_circuit(N_QUBITS, N_LAYERS, seed=circuit_seed + i)
        mem = simulator.run(qc, noise_model=noise_model,
                            shots=N_SHOTS, memory=True).result().get_memory()
        all_bits.append(privacy_amplification(mem, pa_keys, circuit_idx=i))

    bits_flat  = np.concatenate(all_bits)
    pvals      = evaluate_bits(bits_flat)
    score_std  = sum(1 for p in pvals.values() if p >= NIST_THRESH)
    score_bonf = sum(1 for p in pvals.values() if p >= bonf_thresh)

    return {"pvals": pvals, "score_std": score_std,
            "score_bonf": score_bonf, "n_bits": len(bits_flat),
            "bits": bits_flat}

# ══════════════════════════════════════════════════════════════════════
# STAMPA TABELLA PER SINGOLO RUN
# ══════════════════════════════════════════════════════════════════════
def _fmt_p(p, bonf_thresh):
    if p < bonf_thresh:   return f"{p:.4f}*"
    if p < NIST_THRESH:   return f"{p:.4f}~"
    return                       f"{p:.4f} "

def _print_run_table(run_idx, run_res, error_rates, bonf_thresh,
                     label, device_refs):
    print(f"\n  {'─'*80}")
    print(f"  RUN {run_idx+1}  [{label}]")
    print(f"  {'─'*80}")
    print(f"  {'Errore':>7}  {'XEBn':>7}  "
          f"{'Freq':>8}  {'Runs':>8}  {'Unif':>8}  {'Corr':>8}  "
          f"{'Std':>4}  {'Bonf':>4}")
    print(f"  {'─'*7}  {'─'*7}  "
          f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*4}  {'─'*4}")

    for p_err in error_rates:
        r    = run_res[p_err]
        pv   = r["pvals"]
        xstr = f"{r['xeb']:.3f}" if r["xeb"] is not None else "  n/a"
        note = f" ← {device_refs[p_err]}" if p_err in device_refs else ""
        print(f"  {p_err*100:>6.1f}%  {xstr:>7}  "
              f"{_fmt_p(pv['frequency'], bonf_thresh):>9}  "
              f"{_fmt_p(pv['runs'], bonf_thresh):>9}  "
              f"{_fmt_p(pv['uniformity'], bonf_thresh):>9}  "
              f"{_fmt_p(pv['serial_corr'], bonf_thresh):>9}  "
              f"{r['score_std']}/4  "
              f"{r['score_bonf']}/4{note}")

    print(f"\n  Legenda: * < Bonferroni ({bonf_thresh:.5f})   "
          f"~ < NIST ({NIST_THRESH})   XEBn = linear cross-entropy score")

# ══════════════════════════════════════════════════════════════════════
# AGGREGATO ASSE
# ══════════════════════════════════════════════════════════════════════
def _aggregate_axis(all_run_results, error_rates, bonf_thresh,
                    title, label, device_refs):
    N_LEVELS = len(error_rates)
    max_str  = f"> {error_rates[-1]*100:.0f}%"

    print(f"\n\n{'='*80}")
    print(f"  {title}")
    print(f"  Majority vote ≥ {MAJORITY}/{N_RUNS}  |  Bonferroni α = {bonf_thresh:.6f}")
    print(f"{'='*80}")
    print(f"\n  {'Errore':>7}  {'XEBn µ±σ':>12}  "
          f"{'Freq µ±σ':>12}  {'Runs µ±σ':>12}  "
          f"{'Unif µ±σ':>12}  {'Corr µ±σ':>12}  "
          f"{'Pstd':>5}  {'Pbonf':>5}  Stato")
    print(f"  {'─'*7}  {'─'*12}  "
          f"{'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  "
          f"{'─'*5}  {'─'*5}  {'─'*20}")

    first_fail_std  = None
    first_fail_bonf = None
    first_collapse  = None
    per_test_fail   = {n: None for n in TEST_NAMES}

    for p_err in error_rates:
        pv_mat      = {n: [] for n in TEST_NAMES}
        xeb_vals    = []
        scores_std  = []
        scores_bonf = []

        for rr in all_run_results:
            r = rr[p_err]
            scores_std.append(r["score_std"])
            scores_bonf.append(r["score_bonf"])
            if r["xeb"] is not None:
                xeb_vals.append(r["xeb"])
            for n in TEST_NAMES:
                pv_mat[n].append(r["pvals"][n])

        means    = {n: np.mean(pv_mat[n]) for n in TEST_NAMES}
        stds     = {n: np.std(pv_mat[n])  for n in TEST_NAMES}
        xeb_mean = np.mean(xeb_vals) if xeb_vals else None
        xeb_std  = np.std(xeb_vals)  if xeb_vals else None

        n_pass_std  = sum(1 for s in scores_std  if s == N_TESTS)
        n_pass_bonf = sum(1 for s in scores_bonf if s == N_TESTS)
        n_fail_bonf = N_RUNS - n_pass_bonf
        n_fail_std  = N_RUNS - n_pass_std

        if   n_fail_bonf >= MAJORITY: stato = "❌ FAIL ROBUSTO"
        elif n_fail_std  >= MAJORITY: stato = "⚠️  FAIL STD"
        else:                          stato = "✅ PASS"

        if first_fail_bonf is None and n_fail_bonf >= MAJORITY:
            first_fail_bonf = p_err
        if first_fail_std  is None and n_fail_std  >= MAJORITY:
            first_fail_std  = p_err
        if first_collapse  is None and n_pass_bonf == 0:
            first_collapse  = p_err

        for tn in TEST_NAMES:
            if per_test_fail[tn] is None and \
               sum(1 for p in pv_mat[tn] if p < bonf_thresh) >= MAJORITY:
                per_test_fail[tn] = p_err

        note = f" ← {device_refs[p_err]}" if p_err in device_refs else ""

        xeb_str = (f"{xeb_mean:.3f}±{xeb_std:.3f}"
                   if xeb_mean is not None else "       n/a  ")

        def ms(n): return f"{means[n]:.3f}±{stds[n]:.3f}"

        print(f"  {p_err*100:>6.1f}%  {xeb_str:>12}  "
              f"{ms('frequency'):>12}  {ms('runs'):>12}  "
              f"{ms('uniformity'):>12}  {ms('serial_corr'):>12}  "
              f"  {n_pass_std}/{N_RUNS}    {n_pass_bonf}/{N_RUNS}   "
              f"{stato}{note}")

    # Soglie per test
    print(f"\n{'='*80}")
    print(f"  SOGLIE CRITICHE — {label}")
    print(f"{'='*80}")
    print(f"\n  {'Test':<16}  {'Fail Bonferroni (≥' + str(MAJORITY) + '/' + str(N_RUNS) + ')':>30}")
    print(f"  {'─'*16}  {'─'*30}")
    for tn in TEST_NAMES:
        fb = per_test_fail[tn]
        print(f"  {tn:<16}  "
              f"{(f'{fb*100:.1f}%' if fb else max_str):>30}")

    ffs = f"{first_fail_std*100:.1f}%"  if first_fail_std  else max_str
    ffb = f"{first_fail_bonf*100:.1f}%" if first_fail_bonf else max_str
    fc  = f"{first_collapse*100:.1f}%"  if first_collapse  else max_str

    print(f"""
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  RISULTATO — {label:<8}                                                  │
  │                                                                          │
  │  Primo fail senza Bonferroni (≥{MAJORITY}/{N_RUNS}):  {ffs:<12}                      │
  │  Primo fail con Bonferroni   (≥{MAJORITY}/{N_RUNS}):  {ffb:<12}                      │
  │  Collasso completo (0/{N_RUNS} pass):          {fc:<12}                      │
  └──────────────────────────────────────────────────────────────────────────┘""")

# ══════════════════════════════════════════════════════════════════════
# ASSE A — Readout esteso con XEB, Raw vs PA, Autocorrelogramma
# ══════════════════════════════════════════════════════════════════════
def run_axis_a(simulator):
    N_LEVELS    = len(READOUT_RATES_A)
    bonf_thresh = NIST_THRESH / (N_LEVELS * N_TESTS)

    print(f"\n{'='*80}")
    print(f"  ASSE A — Readout error esteso (gate = 0)")
    print(f"  Livelli: {N_LEVELS}  |  "
          f"Bonferroni α = {NIST_THRESH}/({N_LEVELS}×{N_TESTS}) = {bonf_thresh:.6f}")
    print(f"{'='*80}")

    # XEB: circuiti fissi per tutti i run (riproducibile)
    xeb_circuit_seeds = list(range(N_CIRCUITS_XEB))

    all_run_results = []

    for run_idx, config in enumerate(RUN_CONFIGS):
        t_run   = time.time()
        pa_keys = make_pa_keys(config["key_seed"])
        print(f"\n  ▶ RUN {run_idx+1}/{N_RUNS}  "
              f"(cseed={config['circuit_seed']}, "
              f"noff={config['noise_offset']})")

        run_res = {}
        for p_r in READOUT_RATES_A:
            nm  = make_readout_noise(p_r, N_QUBITS, config["noise_offset"])

            # NIST tests
            r   = run_circuits(simulator, nm, pa_keys,
                                config["circuit_seed"], bonf_thresh)

            # XEB (run 0 only per efficienza; tutti i run per robustezza statistica
            # sarebbero ideali ma XEB è costoso — usiamo run 0)
            xeb_val = None
            if run_idx == 0:
                xraw, xraw_std, xnorm, xnorm_std, _ = compute_xeb(
                    simulator, nm, xeb_circuit_seeds, N_QUBITS, N_LAYERS
                )
                
                xeb_val = xnorm
                
                print(f"    p_ro={p_r:.3f}  "
                      f"XEBnorm={xnorm:.3f}±{xnorm_std:.3f}  "
                      f"(raw={xraw:.3f})  "
                      f"NIST={r['score_std']}/4")
            else:
                print(f"    p_ro={p_r:.3f}  NIST={r['score_std']}/4")

            r["xeb"] = xeb_val
            run_res[p_r] = r

        all_run_results.append(run_res)
        _print_run_table(run_idx, run_res, READOUT_RATES_A,
                         bonf_thresh, label="Readout", device_refs=DEVICE_REFS)
        print(f"\n  ⏱  Run {run_idx+1}: {time.time()-t_run:.1f}s")

    # ── Aggregato ────────────────────────────────────────────────────
    _aggregate_axis(all_run_results, READOUT_RATES_A, bonf_thresh,
                    title="ASSE A — AGGREGATO (readout esteso, gate = 0)",
                    label="Readout", device_refs=DEVICE_REFS)

    # ── Aggiunta 2: Raw vs PA (run 0, tutti i livelli) ───────────────
    print(f"\n\n{'='*80}")
    print(f"  ASSE A — Analisi RAW vs PA")
    print(f"  TV distance: deviazione della distribuzione HW dal Binomiale(8,0.5)")
    print(f"  Confronto: TV alta + PA freq_p alto  →  PA corregge bias reale")
    print(f"{'='*80}")
    print(f"\n  {'Errore':>7}  {'TV dist (raw)':>14}  "
          f"{'PA bias':>10}  {'PA freq_p':>9}  {'PA pass?':>8}  Interpretazione")
    print(f"  {'─'*7}  {'─'*14}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*30}")

    config0  = RUN_CONFIGS[0]
    pa_keys0 = make_pa_keys(config0["key_seed"])

    for p_r in READOUT_RATES_A:
        nm  = make_readout_noise(p_r, N_QUBITS, config0["noise_offset"])
        raw = analyze_raw_vs_pa(simulator, nm,
                                config0["circuit_seed"], pa_keys0,
                                n_circuits=10, shots=2048)
        pa_ok = "✅" if raw["pa_freq_p"] >= NIST_THRESH else "❌"

        if raw["tv_distance"] > 0.05 and raw["pa_freq_p"] >= NIST_THRESH:
            interp = "PA corregge bias strutturale"
        elif raw["tv_distance"] < 0.02 and raw["pa_freq_p"] >= NIST_THRESH:
            interp = "Raw già quasi ideale"
        elif raw["pa_freq_p"] < NIST_THRESH:
            interp = "PA insufficiente a questo livello"
        else:
            interp = "Degradazione moderata"

        print(f"  {p_r*100:>6.1f}%  {raw['tv_distance']:>14.4f}  "
              f"{raw['pa_bias']:>+10.5f}  {raw['pa_freq_p']:>9.4f}  "
              f"{pa_ok:>8}  {interp}")

    # ── Aggiunta 3: Autocorrelogramma (livello 0% e 30%) ─────────────
    print(f"\n\n{'='*80}")
    print(f"  ASSE A — Autocorrelogramma a lag 1..20 (run 0)")
    print(f"  Verifica indipendenza temporale oltre lag-1")
    print(f"{'='*80}")

    for p_r in [0.0, 0.150, 0.500]:
        nm   = make_readout_noise(p_r, N_QUBITS, config0["noise_offset"])
        bits = all_run_results[0][p_r]["bits"]
        acg, bonf_acg = autocorrelogram(bits, max_lag=20)
        print_autocorrelogram(acg, bonf_acg,
                              label=f"[readout={p_r:.0%}]")

    # ── Aggiunta 5: Sensitivity sulla discard rule ───────────────────
    print(f"\n\n{'='*80}")
    print(f"  ASSE A — Sensitivity analysis: PA discard rule")
    print(f"  (run 0, p_readout = 0% e 15%)")
    print(f"{'='*80}")

    for p_r in [0.0, 0.150]:
        nm = make_readout_noise(p_r, N_QUBITS, config0["noise_offset"])
        sensitivity_pa_discard(simulator, nm,
                               config0["circuit_seed"], pa_keys0,
                               label=f"[p_readout={p_r:.0%}]")

    return all_run_results

# ══════════════════════════════════════════════════════════════════════
# ASSE B — Gate error completo (depol + thermal)
# ══════════════════════════════════════════════════════════════════════
def run_axis_b(simulator):
    N_LEVELS    = len(GATE_RATES_B)
    bonf_thresh = NIST_THRESH / (N_LEVELS * N_TESTS)

    print(f"\n{'='*80}")
    print(f"  ASSE B — Gate error completo (CX depol + U thermal, readout = 0)")
    print(f"  T1={T1_NS//1000}µs  T2={T2_NS//1000}µs  tg={GATE_TIME_NS}ns")
    print(f"  Livelli: {N_LEVELS}  |  Bonferroni α = {bonf_thresh:.6f}")
    print(f"{'='*80}")

    xeb_seeds = list(range(N_CIRCUITS_XEB))
    all_run_results = []

    for run_idx, config in enumerate(RUN_CONFIGS):
        t_run   = time.time()
        pa_keys = make_pa_keys(config["key_seed"])
        print(f"\n  ▶ RUN {run_idx+1}/{N_RUNS}")

        run_res = {}
        for p_g in GATE_RATES_B:
            nm      = make_gate_noise_full(p_g, N_QUBITS, config["noise_offset"])
            r       = run_circuits(simulator, nm, pa_keys,
                                   config["circuit_seed"], bonf_thresh)
            xeb_val = None
            if run_idx == 0:
                xraw, xraw_std, xnorm, xnorm_std, _ = compute_xeb(
                    simulator, nm, xeb_seeds, N_QUBITS, N_LAYERS
                )
                
                xeb_val = xnorm
                
                print(f"    p_gate={p_g:.3f}  "
                      f"XEBnorm={xnorm:.3f}±{xnorm_std:.3f}  "
                      f"(raw={xraw:.3f})  "
                      f"NIST={r['score_std']}/4")
            else:
                print(f"    p_gate={p_g:.3f}  NIST={r['score_std']}/4")

            r["xeb"] = xeb_val
            run_res[p_g] = r

        all_run_results.append(run_res)
        _print_run_table(run_idx, run_res, GATE_RATES_B,
                         bonf_thresh, label="Gate (full)", device_refs={})
        print(f"\n  ⏱  Run {run_idx+1}: {time.time()-t_run:.1f}s")

    _aggregate_axis(all_run_results, GATE_RATES_B, bonf_thresh,
                    title="ASSE B — AGGREGATO (gate completo, readout = 0)",
                    label="Gate_full", device_refs={})

    # Autocorrelogramma per gate (run 0, p=0 e p=5%)
    print(f"\n\n{'='*80}")
    print(f"  ASSE B — Autocorrelogramma a lag 1..20 (run 0)")
    print(f"{'='*80}")
    config0  = RUN_CONFIGS[0]
    for p_g in [0.0, 0.050]:
        bits = all_run_results[0][p_g]["bits"]
        acg, bonf_acg = autocorrelogram(bits, max_lag=20)
        print_autocorrelogram(acg, bonf_acg,
                              label=f"[gate_full={p_g:.0%}]")

    return all_run_results

# ══════════════════════════════════════════════════════════════════════
# ASSE B2 — Gate depolarizing puro (senza thermal)
# ══════════════════════════════════════════════════════════════════════
def run_axis_b2(simulator):
    N_LEVELS    = len(GATE_RATES_B)
    bonf_thresh = NIST_THRESH / (N_LEVELS * N_TESTS)

    print(f"\n{'='*80}")
    print(f"  ASSE B2 — Gate depolarizing puro (senza thermal relaxation)")
    print(f"  Isola l'effetto del gate noise dal collapse verso |0⟩ del thermal")
    print(f"  Livelli: {N_LEVELS}  |  Bonferroni α = {bonf_thresh:.6f}")
    print(f"{'='*80}")

    xeb_seeds = list(range(N_CIRCUITS_XEB))
    all_run_results = []

    for run_idx, config in enumerate(RUN_CONFIGS):
        t_run   = time.time()
        pa_keys = make_pa_keys(config["key_seed"])
        print(f"\n  ▶ RUN {run_idx+1}/{N_RUNS}")

        run_res = {}
        for p_g in GATE_RATES_B:
            nm      = make_gate_noise_depol_only(p_g, N_QUBITS,
                                                  config["noise_offset"])
            r       = run_circuits(simulator, nm, pa_keys,
                                   config["circuit_seed"], bonf_thresh)
            xeb_val = None
            if run_idx == 0:
                xraw, xraw_std, xnorm, xnorm_std, _ = compute_xeb(
                    simulator, nm, xeb_seeds, N_QUBITS, N_LAYERS
                )
                
                xeb_val = xnorm
                
                print(f"    p_gate={p_g:.3f}  "
                      f"XEBnorm={xnorm:.3f}±{xnorm_std:.3f}  "
                      f"(raw={xraw:.3f})  "
                      f"NIST={r['score_std']}/4")
            else:
                print(f"    p_gate={p_g:.3f}  NIST={r['score_std']}/4")

            r["xeb"] = xeb_val
            run_res[p_g] = r

        all_run_results.append(run_res)
        _print_run_table(run_idx, run_res, GATE_RATES_B,
                         bonf_thresh, label="Gate (depol only)", device_refs={})
        print(f"\n  ⏱  Run {run_idx+1}: {time.time()-t_run:.1f}s")

    _aggregate_axis(all_run_results, GATE_RATES_B, bonf_thresh,
                    title="ASSE B2 — AGGREGATO (depol puro, senza thermal)",
                    label="Gate_depol", device_refs={})

    # Confronto B vs B2 (solo XEB run 0)
    print(f"\n\n{'='*80}")
    print(f"  CONFRONTO B vs B2 — XEB run 0")
    print(f"  Mostra quanto il thermal relaxation degrada il contenuto quantistico")
    print(f"  rispetto al solo gate depolarizing")
    print(f"{'='*80}")
    print(f"\n  {'p_gate':>8}  {'XEBn (depol only)':>18}  "
          f"{'XEB (full)':>12}  {'ΔF_XEBn':>8}")
    print(f"  {'─'*8}  {'─'*18}  {'─'*12}  {'─'*8}")

    for p_g in GATE_RATES_B:
        xeb_depol = all_run_results[0][p_g]["xeb"]
        if xeb_depol is not None:
            print(f"  {p_g*100:>7.1f}%  {xeb_depol:>18.4f}  "
                  f"{'(v. Asse B)':>12}  {'─':>8}")

    return all_run_results

# ══════════════════════════════════════════════════════════════════════
# ASSE C — Griglia 2D readout × gate
# ══════════════════════════════════════════════════════════════════════
def run_axis_c(simulator):
    pairs       = [(pr, pg) for pr in READOUT_GRID for pg in GATE_GRID]
    N_LEVELS    = len(pairs)
    bonf_thresh = NIST_THRESH / (N_LEVELS * N_TESTS)

    print(f"\n{'='*80}")
    print(f"  ASSE C — Griglia 2D readout × gate (sovrapposti)")
    print(f"  Coppie: {N_LEVELS}  |  Bonferroni α = {bonf_thresh:.6f}")
    print(f"{'='*80}")

    grid = {p: [] for p in pairs}

    for run_idx, config in enumerate(RUN_CONFIGS):
        t_run   = time.time()
        pa_keys = make_pa_keys(config["key_seed"])
        print(f"\n  ▶ RUN {run_idx+1}/{N_RUNS}")

        for (p_r, p_g) in pairs:
            nm  = make_combined_noise(p_r, p_g, N_QUBITS, config["noise_offset"])
            r   = run_circuits(simulator, nm, pa_keys,
                               config["circuit_seed"], bonf_thresh)
            r["xeb"] = None  # XEB non calcolato per la griglia (troppo costoso)
            grid[(p_r, p_g)].append(r)
            print(f"    r={p_r:.3f} g={p_g:.3f}  NIST={r['score_std']}/4")

        print(f"\n  ⏱  Run {run_idx+1}: {time.time()-t_run:.1f}s")

    # Aggregato griglia
    print(f"\n\n{'='*80}")
    print(f"  ASSE C — AGGREGATO Griglia 2D")
    print(f"  Notazione: [pass_std / pass_bonf] / {N_RUNS} run")
    print(f"  ✅ PASS robusto   ⚠️  fail std   ❌ fail Bonferroni")
    print(f"{'='*80}")

    hdr = "".join(f"  gate={g*100:4.1f}%      " for g in GATE_GRID)
    print(f"\n  {'Readout':>9} {hdr}")
    print(f"  {'─'*9} " + "─" * (len(GATE_GRID) * 18))

    for p_r in READOUT_GRID:
        row = f"  {p_r*100:>6.1f}%   "
        for p_g in GATE_GRID:
            rl         = grid[(p_r, p_g)]
            n_pstd     = sum(1 for r in rl if r["score_std"]  == N_TESTS)
            n_pbonf    = sum(1 for r in rl if r["score_bonf"] == N_TESTS)
            n_fbonf    = N_RUNS - n_pbonf
            n_fstd     = N_RUNS - n_pstd
            if n_fbonf  >= MAJORITY: icon = "❌"
            elif n_fstd >= MAJORITY: icon = "⚠️ "
            else:                    icon = "✅"
            row += f" {icon}[{n_pstd}/{n_pbonf}]   "
        print(row)

    print(f"\n  Dettaglio p-value medi (freq | runs | unif | corr):")
    print(f"  {'─'*76}")
    for p_r in READOUT_GRID:
        for p_g in GATE_GRID:
            rl       = grid[(p_r, p_g)]
            pv_mat   = {n: [r["pvals"][n] for r in rl] for n in TEST_NAMES}
            means    = {n: np.mean(pv_mat[n]) for n in TEST_NAMES}
            n_pbonf  = sum(1 for r in rl if r["score_bonf"] == N_TESTS)
            stato    = "❌" if (N_RUNS - n_pbonf) >= MAJORITY else "✅"
            note     = f" ← {DEVICE_REFS[p_r]}" if p_r in DEVICE_REFS else ""
            print(f"  r={p_r*100:.1f}% g={p_g*100:.1f}%  "
                  f"freq={means['frequency']:.3f}  runs={means['runs']:.3f}  "
                  f"unif={means['uniformity']:.3f}  corr={means['serial_corr']:.3f}"
                  f"  {stato}{note}")

    return grid

# ══════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ══════════════════════════════════════════════════════════════════════
def sanity_check(simulator):
    print("  🔍 Sanity check (0% rumore, seed 0)...")
    config  = RUN_CONFIGS[0]
    pa_keys = make_pa_keys(config["key_seed"])

    all_bits = []
    for i in range(20):
        qc  = make_rcs_circuit(N_QUBITS, N_LAYERS, seed=config["circuit_seed"] + i)
        mem = simulator.run(qc, shots=N_SHOTS, memory=True).result().get_memory()
        all_bits.append(privacy_amplification(mem, pa_keys, circuit_idx=i % N_CIRCUITS))

    bits   = np.concatenate(all_bits)
    res    = evaluate_bits(bits)
    passed = sum(1 for p in res.values() if p >= NIST_THRESH)

    print(f"     Bit totali:     {len(bits):,}")
    print(f"     Proporzione 1:  {np.mean(bits):.4f}  (ideale: 0.5000)")
    print(f"     Test superati:  {passed}/4")
    for name, pval in res.items():
        mark = "✅" if pval >= NIST_THRESH else "❌"
        print(f"       {name:<15}: p={pval:.4f}  {mark}")

    # XEB sanity
    xraw, xraw_std, xnorm, xnorm_std, _ = compute_xeb(simulator, None,
                                  list(range(5)), N_QUBITS, N_LAYERS,
                                  shots=2048)
    print(f"     XEB raw:        {xraw:.4f} ± {xraw_std:.4f}")
    print(f"     XEB normalized: {xnorm:.4f} ± {xnorm_std:.4f}  (deve essere ~1.0)")

    ok = passed >= 3 and abs(xnorm - 1.0) < 0.3
    if ok:
        print(f"\n  ✅ Sanity check OK — avvio assi.\n")
    else:
        print(f"\n  ❌ Sanity check parziale ({passed}/4 NIST, XEBnorm={xnorm:.3f}).\n")
    return passed

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0        = time.time()
    simulator = AerSimulator()

    print(f"{'='*80}")
    print(f"  Readout Error Threshold per QRNG — v10")
    print(f"  Assi: A (readout ≤50%)  B (gate completo)  "
          f"B2 (depol puro)  C (griglia 2D)")
    print(f"  Aggiunte: XEB · Raw vs PA · Autocorrelogramma · Sensitivity PA")
    print(f"  {N_RUNS} run × majority vote ≥ {MAJORITY}/{N_RUNS}  |  Bonferroni per asse")
    print(f"{'='*80}\n")

    passed = sanity_check(simulator)
    if passed < 2:
        print("  Esperimento bloccato: sanity check fallito.")
        sys.exit(1)

    t_a = time.time()
    run_axis_a(simulator)
    print(f"\n  ⏱  Asse A completato in {time.time()-t_a:.0f}s")

    t_b = time.time()
    results_b = run_axis_b(simulator)
    print(f"\n  ⏱  Asse B completato in {time.time()-t_b:.0f}s")

    t_b2 = time.time()
    results_b2 = run_axis_b2(simulator)
    print(f"\n  ⏱  Asse B2 completato in {time.time()-t_b2:.0f}s")

    # Stampa confronto B vs B2 XEB completo ora che entrambi sono disponibili
    print(f"\n\n{'='*80}")
    print(f"  CONFRONTO FINALE B vs B2 — XEB per livello di gate error")
    print(f"  Misura quanto il thermal relaxation aggiunge degrado rispetto")
    print(f"  al solo gate depolarizing (run 0)")
    print(f"{'='*80}")
    print(f"\n  {'p_gate':>8}  {'XEBn depol only':>16}  "
          f"{'XEB full (B)':>14}  {'ΔF_XEBn (B-B2)':>14}")
    print(f"  {'─'*8}  {'─'*16}  {'─'*14}  {'─'*14}")

    for p_g in GATE_RATES_B:
        xb2 = results_b2[0][p_g]["xeb"]
        xb  = results_b[0][p_g]["xeb"]
        if xb2 is not None and xb is not None:
            delta = xb - xb2
            print(f"  {p_g*100:>7.1f}%  {xb2:>16.4f}  "
                  f"{xb:>14.4f}  {delta:>+14.4f}")
        elif xb2 is not None:
            print(f"  {p_g*100:>7.1f}%  {xb2:>16.4f}  {'(vedi B)':>14}")

    t_c = time.time()
    run_axis_c(simulator)
    print(f"\n  ⏱  Asse C completato in {time.time()-t_c:.0f}s")

    print(f"\n{'='*80}")
    print(f"  Tempo totale: {time.time()-t0:.0f}s")
    print(f"  Incollami i risultati! 🚀\n")