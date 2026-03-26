"""
QRNG Hardware Validation — v3
==============================
Validazione della pipeline QRNG su ibm_kingston (Heron r2, 156 qubit).

Patch rispetto a v2:
  1. find_best_qubit_chain usa backend.target invece di props.gate_error("cx")
     → legge il gate nativo disponibile sull'arco (ecr/cz su Heron r2),
       evitando lookup cx che restituisce None su Heron e rende la selezione cieca.
  2. Parsing risultato SamplerV2: sostituito vars(data).keys()[0] con
     data.meas (measure_all crea sempre il registro "meas") + fallback
     su data.keys() per robustezza.
  3. Bug XEB corretto: il mapping bitstring→stato ideale ora usa il
     layout fisico post-transpile (final_layout) per riallineare i qubit
     logici prima di indicizzare p_ideal. Senza questa correzione l'XEB
     risulta sistematicamente rumoroso dopo la transpilazione con initial_layout.
  4. Print finale arricchito con tutti i valori rilevanti in modo leggibile.

Flusso invariato rispetto a v2:
  1. DRY_RUN=True  → AerSimulator + NoiseModel da calibrazione IBM
  2. PILOT=True    → 5 XEB + 10 NIST su QPU reale, budget minimo
  3. PILOT=False   → run completo (20×4096 XEB + 50×8192 NIST)

Come impostare il token (NON metterlo nel codice):
  Windows:   set IBM_QUANTUM_TOKEN=il_tuo_token
  Linux/Mac: export IBM_QUANTUM_TOKEN=il_tuo_token

  Oppure salva una volta sola su disco:
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(
        channel='ibm_quantum_platform',
        token='il_tuo_token',
        instance='ibm-q/open/main',
        set_as_default=True
    )
"""

import os, sys, time, warnings, json
import numpy as np
from scipy.stats import chisquare
from scipy.special import erfc

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ══════════════════════════════════════════════════════════════════════
IBM_INSTANCE = None                  #"ibm-q/open/main"
IBM_BACKEND  = "ibm_kingston"        # Heron r2, 156 qubit

DRY_RUN      = False    # True = AerSimulator locale, False = QPU reale
PILOT        = False    # True = budget minimo per test flusso

# Circuiti — pilot vs full
if PILOT:
    N_CIRCUITS_XEB  = 5
    N_SHOTS_XEB     = 1024
    N_CIRCUITS_NIST = 10
    N_SHOTS_NIST    = 2048
else:
    N_CIRCUITS_XEB  = 20
    N_SHOTS_XEB     = 4096
    N_CIRCUITS_NIST = 50
    N_SHOTS_NIST    = 8192

N_QUBITS = 8
N_LAYERS = 12

# PA / NIST
PA_SEED     = 2**31 - 1
PA_MIN_HW   = 0
NIST_THRESH = 0.01
N_TESTS     = 4
TEST_NAMES  = ["frequency", "runs", "uniformity", "serial_corr"]

# Gate a due qubit candidati su Heron r2 (in ordine di preferenza)
# Il codice cerca il primo disponibile nel Target del backend.
TWO_QUBIT_GATE_CANDIDATES = ["ecr", "cz", "cx"]

# Curve v10 Asse A per confronto finale
V10_XEB_AXIS_A = {
    0.000: 0.999, 0.020: 0.930, 0.050: 0.844,
    0.100: 0.707, 0.150: 0.586, 0.200: 0.507,
    0.300: 0.321, 0.400: 0.222, 0.500: 0.150,
}

# ══════════════════════════════════════════════════════════════════════
# QISKIT IMPORTS
# ══════════════════════════════════════════════════════════════════════
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
except ImportError:
    print("Installo dipendenze...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "qiskit", "qiskit-aer", "qiskit-ibm-runtime", "-q"])
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ══════════════════════════════════════════════════════════════════════
# CIRCUITO RCS — identico a v10 (Haar-corretto)
# ══════════════════════════════════════════════════════════════════════
def make_rcs_circuit(n_qubits, n_layers, seed=None, with_measure=True):
    rng   = np.random.default_rng(seed)
    qc    = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    for layer in range(n_layers):
        u     = rng.uniform(0, 1,       n_qubits)
        phi   = rng.uniform(0, 2*np.pi, n_qubits)
        lam   = rng.uniform(0, 2*np.pi, n_qubits)
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
# PATCH 1 — SELEZIONE QUBIT VIA backend.target (non props.gate_error)
# ══════════════════════════════════════════════════════════════════════
def find_native_two_qubit_gate(target):
    """
    Restituisce il nome del gate a due qubit nativo disponibile nel Target
    del backend, cercando in ordine: ecr → cz → cx.
    Su Heron r2 il gate fisico è 'ecr'; 'cx' non è nel Target nativo.
    """
    gate_names = set(target.operation_names)
    for gate in TWO_QUBIT_GATE_CANDIDATES:
        if gate in gate_names:
            return gate
    # fallback: primo gate con qargs a 2 qubit trovato nel Target
    for name in gate_names:
        props = target[name]
        if props is not None:
            for qargs in props:
                if qargs is not None and len(qargs) == 2:
                    return name
    return None


def find_best_qubit_chain(backend, n_qubits=8):
    """
    Trova la catena lineare di n_qubits connessi con errore medio
    del gate nativo a due qubit più basso, leggendo da backend.target.

    Differenza da v2: non usa props.gate_error("cx") — su Heron r2
    cx non è nel Target nativo e restituirebbe None, rendendo la
    selezione cieca (tutti gli archi a parità → fallback casuale).
    """
    try:
        target     = backend.target
        native_2q  = find_native_two_qubit_gate(target)

        if native_2q is None:
            raise ValueError("Nessun gate a due qubit trovato nel Target")

        print(f"  Gate nativo 2-qubit rilevato: '{native_2q}'")

        # Costruisce grafo di adiacenza con errori dal Target
        gate_props = target[native_2q]   # dict: (q0,q1) → InstructionProperties
        adj = {}
        edge_error = {}

        for qargs, instr_props in gate_props.items():
            if qargs is None or len(qargs) != 2:
                continue
            a, b = qargs
            err = instr_props.error if (instr_props and instr_props.error is not None) else 1.0
            key = (min(a, b), max(a, b))
            # Prende il minore tra le due direzioni se entrambe presenti
            edge_error[key] = min(edge_error.get(key, 1.0), err)
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        def arc_error(a, b):
            return edge_error.get((min(a, b), max(a, b)), 1.0)

        # DFS per trovare catene lineari di lunghezza n_qubits
        best_chain = None
        best_err   = float("inf")

        def dfs(chain, visited):
            nonlocal best_chain, best_err
            if len(chain) == n_qubits:
                err = np.mean([arc_error(chain[i], chain[i+1])
                               for i in range(len(chain) - 1)])
                if err < best_err:
                    best_err   = err
                    best_chain = list(chain)
                return
            for nb in adj.get(chain[-1], []):
                if nb not in visited:
                    chain.append(nb)
                    visited.add(nb)
                    dfs(chain, visited)
                    chain.pop()
                    visited.remove(nb)

        # Limita il seed della DFS ai primi 40 qubit per contenere i tempi
        for start in range(min(40, backend.num_qubits)):
            dfs([start], {start})

        if best_chain:
            print(f"  ✅ Catena ottimale trovata: {best_chain}  "
                  f"(errore {native_2q} medio: {best_err:.4f})")
            return best_chain, native_2q
        else:
            raise ValueError("Nessuna catena di lunghezza richiesta trovata")

    except Exception as e:
        fallback = list(range(n_qubits))
        print(f"  ⚠️  Selezione automatica fallita ({e}). Fallback: {fallback}")
        return fallback, None

# ══════════════════════════════════════════════════════════════════════
# STIMA PARAMETRI DI RUMORE DALLA CALIBRAZIONE
# ══════════════════════════════════════════════════════════════════════
def estimate_noise_params(backend, qubit_chain, native_2q_gate):
    """
    Legge i parametri di rumore dal Target del backend invece che da
    BackendProperties, che su Heron r2 può essere parziale o assente.
    """
    target = backend.target
    result = {
        "device_name":         backend.name,
        "qubit_chain":         qubit_chain,
        "native_2q_gate":      native_2q_gate,
        "p_readout_mean":      None,
        "p_readout_per_qubit": {},
        "p_2q_mean":           None,
        "t1_mean_us":          None,
        "t2_mean_us":          None,
    }

    # Readout error (misura)
    ro_values = []
    if "measure" in target.operation_names:
        meas_props = target["measure"]
        for qargs, instr_props in meas_props.items():
            if qargs and len(qargs) == 1 and instr_props:
                q = qargs[0]
                if q in qubit_chain and instr_props.error is not None:
                    result["p_readout_per_qubit"][q] = instr_props.error
                    ro_values.append(instr_props.error)
    if ro_values:
        result["p_readout_mean"] = float(np.mean(ro_values))

    # Errore gate 2-qubit sulla catena
    if native_2q_gate and native_2q_gate in target.operation_names:
        cx_errs = []
        gate_props = target[native_2q_gate]
        for i in range(len(qubit_chain) - 1):
            a, b = qubit_chain[i], qubit_chain[i+1]
            for pair in [(a, b), (b, a)]:
                instr = gate_props.get(pair)
                if instr and instr.error is not None:
                    cx_errs.append(instr.error)
                    break
        if cx_errs:
            result["p_2q_mean"] = float(np.mean(cx_errs))

    # T1 / T2 dal Target (se disponibili come qubit_properties)
    try:
        t1_list, t2_list = [], []
        for q in qubit_chain:
            qp = target.qubit_properties[q] if target.qubit_properties else None
            if qp:
                if getattr(qp, "t1", None): t1_list.append(qp.t1 * 1e6)
                if getattr(qp, "t2", None): t2_list.append(qp.t2 * 1e6)
        if t1_list: result["t1_mean_us"] = float(np.mean(t1_list))
        if t2_list: result["t2_mean_us"] = float(np.mean(t2_list))
    except Exception:
        pass

    return result

# ══════════════════════════════════════════════════════════════════════
# PRIVACY AMPLIFICATION — identica a v10/v2
# ══════════════════════════════════════════════════════════════════════
def make_pa_keys(seed, n_circuits, n_shots, n_qubits):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_circuits, n_shots, n_qubits), dtype=np.uint8)

def privacy_amplification(memory, pa_keys, circuit_idx, min_hw=0):
    keys = pa_keys[circuit_idx]
    bits = []
    for shot_idx, shot_str in enumerate(memory):
        b = np.array([int(c) for c in shot_str.replace(" ", "")], dtype=np.uint8)
        if np.sum(b) <= min_hw:
            continue
        ip = int(np.dot(b, keys[shot_idx % len(keys)]) % 2)
        bits.append(ip)
    return np.array(bits, dtype=np.uint8)

# ══════════════════════════════════════════════════════════════════════
# TEST STATISTICI — identici a v10/v2
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
# PATCH 3 — XEB con layout post-transpile corretto
# ══════════════════════════════════════════════════════════════════════
def compute_xeb_from_counts(counts_list, circuit_seeds, n_qubits, n_layers,
                             final_layouts=None):
    """
    Calcola XEB confrontando i counts QPU con le probabilità ideali
    (statevector locale).

    Patch v3: usa final_layout per riallineare i qubit logici dopo
    transpilazione. Senza questo, dopo initial_layout il bit i-esimo
    della bitstring QPU potrebbe corrispondere al qubit fisico j ≠ i,
    introducendo errore sistematico nell'XEB.

    final_layouts: lista di dict {qubit_logico: qubit_fisico} o None.
                   Se None (dry-run), nessun riallineamento (identità).
    """
    sv_sim = AerSimulator(method='statevector')
    n_out  = 2 ** n_qubits
    xeb_raw_list  = []
    xeb_norm_list = []

    for idx, seed in enumerate(circuit_seeds):
        # Statevector ideale sul circuito logico (senza layout fisico)
        qc_sv = make_rcs_circuit(n_qubits, n_layers, seed=seed, with_measure=False)
        qc_sv.save_statevector()
        sv      = sv_sim.run(qc_sv).result().get_statevector()
        p_ideal = np.abs(np.array(sv)) ** 2
        p_ideal = p_ideal / np.sum(p_ideal)

        ideal_baseline = n_out * np.sum(p_ideal ** 2) - 1.0

        # Permutazione dal layout fisico → ordine logico originale
        layout = final_layouts[idx] if final_layouts is not None else None

        counts = counts_list[idx]
        sampled = []

        for bitstr, cnt in counts.items():
            clean = bitstr.replace(" ", "")
            # Qiskit: bitstring in little-endian (qubit 0 = char più a destra)
            bits_le = [int(c) for c in reversed(clean)]   # ora bits_le[i] = qubit i (fisico)

            if layout is not None:
                # Riordina dal qubit fisico al qubit logico
                # layout = {logico: fisico}
                # Invertiamo: fisico → logico
                phys_to_log = {v: k for k, v in layout.items()}
                try:
                    bits_logical = [bits_le[phys_to_log[i]]
                                    for i in range(n_qubits)]
                except KeyError:
                    # Se il mapping è incompleto, salta il riallineamento
                    bits_logical = bits_le[:n_qubits]
            else:
                bits_logical = bits_le[:n_qubits]

            # Indice dello stato base nell'ordinamento del statevector
            idx_state = sum(b * (2 ** i) for i, b in enumerate(bits_logical))
            if idx_state < n_out:
                sampled.extend([p_ideal[idx_state]] * cnt)

        if not sampled:
            xeb_raw_list.append(np.nan)
            xeb_norm_list.append(np.nan)
            continue

        mean_pideal = float(np.mean(sampled))
        xeb_raw     = n_out * mean_pideal - 1.0
        xeb_norm    = xeb_raw / ideal_baseline if ideal_baseline > 0 else np.nan

        xeb_raw_list.append(xeb_raw)
        xeb_norm_list.append(xeb_norm)

    return (
        float(np.nanmean(xeb_raw_list)),
        float(np.nanstd(xeb_raw_list)),
        float(np.nanmean(xeb_norm_list)),
        float(np.nanstd(xeb_norm_list)),
    )

# ══════════════════════════════════════════════════════════════════════
# CONNESSIONE IBM — invariata da v2
# ══════════════════════════════════════════════════════════════════════
def connect_ibm(backend_name, instance):
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN", None)

    try:
        if token:
            service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=token,
                instance=instance,
            )
            print(f"  Token letto da variabile d'ambiente IBM_QUANTUM_TOKEN.")
        else:
            service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                instance=instance,
            )
            print(f"  Token letto da account salvato su disco.")

        backend = service.backend(backend_name)
        print(f"  ✅ Connesso a {backend.name}  ({backend.num_qubits} qubit)")
        return backend

    except Exception as e:
        print(f"  ❌ Connessione fallita: {e}")
        print(f"\n  Come risolvere:")
        print(f"  Opzione A — variabile d'ambiente:")
        print(f"    Windows:   set IBM_QUANTUM_TOKEN=il_tuo_token")
        print(f"    Linux/Mac: export IBM_QUANTUM_TOKEN=il_tuo_token")
        print(f"  Opzione B — salva su disco (una volta sola):")
        print(f"    from qiskit_ibm_runtime import QiskitRuntimeService")
        print(f"    QiskitRuntimeService.save_account(")
        print(f"        channel='ibm_quantum_platform',")
        print(f"        token='il_tuo_token',")
        print(f"        instance='{instance}',")
        print(f"        set_as_default=True)")
        return None

# ══════════════════════════════════════════════════════════════════════
# ESECUZIONE LOCALE (dry-run)
# ══════════════════════════════════════════════════════════════════════
def run_local(circuits, noise_model, shots_list):
    sim     = AerSimulator()
    results = []
    for qc, shots in zip(circuits, shots_list):
        res    = sim.run(qc, noise_model=noise_model,
                         shots=shots, memory=True).result()
        results.append((res.get_counts(), res.get_memory()))
    # Dry-run: nessun layout fisico reale → final_layouts=None
    return results, None

# ══════════════════════════════════════════════════════════════════════
# PATCH 2 — ESECUZIONE QPU con parsing risultato robusto
# ══════════════════════════════════════════════════════════════════════
def run_qpu(circuits, backend, shots_list, qubit_chain):
    """
    Job mode (Sampler(mode=backend)), compatibile Open Plan.

    Patch v3 — parsing risultato:
      Sostituisce vars(data).keys()[0] con accesso diretto a data.meas,
      che è il nome del registro creato da measure_all(). Se per qualsiasi
      ragione data.meas non esiste, fa fallback su data.keys() (API pubblica
      dell'oggetto DataBin di SamplerV2, documentata da IBM).
    """
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    print(f"\n  Transpilazione ({len(circuits)} circuiti, "
          f"layout={qubit_chain}, opt=3)...")
    pm = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        initial_layout=qubit_chain,
    )
    t_circuits   = [pm.run(qc) for qc in circuits]
    final_layouts = []
    for tc in t_circuits:
        try:
            # final_layout: Layout Qiskit → dict {qubit_logico: qubit_fisico}
            fl = tc.layout.final_layout
            if fl is not None:
                mapping = {q._index: tc.layout.final_layout[q]
                           for q in tc.layout.final_layout.get_virtual_bits()}
            else:
                mapping = None
        except Exception:
            mapping = None
        final_layouts.append(mapping)

    depths = [qc.depth() for qc in t_circuits]
    print(f"  ✅ Profondità post-transpile: media={np.mean(depths):.1f}, "
          f"min={min(depths)}, max={max(depths)}")

    # Warning su shot totali elevati
    tot_shots = sum(shots_list)
    print(f"\n  Job summary: {len(circuits)} circuiti, {tot_shots:,} shot totali.")
    if tot_shots > 200_000:
        print(f"  ⚠️  Shot totali elevati ({tot_shots:,}): attendersi coda lunga.")

    sampler = Sampler(mode=backend)
    pubs    = [(qc, [], s) for qc, s in zip(t_circuits, shots_list)]

    print(f"  Invio job...")
    t0  = time.time()
    job = sampler.run(pubs)
    print(f"  Job ID: {job.job_id()}")
    print(f"  In attesa dei risultati...")

    result  = job.result()
    elapsed = time.time() - t0
    print(f"  ✅ Risultati ricevuti in {elapsed:.1f}s")

    output = []
    for pub_result in result:
        data = pub_result.data

        # PATCH 2: accesso robusto al registro "meas" (creato da measure_all)
        if hasattr(data, "meas"):
            bit_arr = data.meas
        else:
            # Fallback: primo registro disponibile via keys() (API pubblica DataBin)
            keys = list(data.keys()) if hasattr(data, "keys") else []
            if not keys:
                raise RuntimeError(
                    "Impossibile leggere il registro dal risultato SamplerV2. "
                    "Verifica la versione di qiskit-ibm-runtime."
                )
            bit_arr = getattr(data, keys[0])

        counts = bit_arr.get_counts()
        memory = [b.replace(" ", "") for b in bit_arr.get_bitstrings()]
        output.append((counts, memory))

    return output, final_layouts

# ══════════════════════════════════════════════════════════════════════
# STAMPA RISULTATI
# ══════════════════════════════════════════════════════════════════════
def print_noise_params(params):
    print(f"\n{'='*65}")
    print(f"  CALIBRAZIONE — {params['device_name']}")
    print(f"  Catena qubit:    {params['qubit_chain']}")
    print(f"  Gate 2Q nativo:  {params['native_2q_gate']}")
    print(f"{'='*65}")
    p_ro = params['p_readout_mean']
    p_2q = params['p_2q_mean']
    t1   = params['t1_mean_us']
    t2   = params['t2_mean_us']
    if p_ro is not None:
        print(f"  p_readout medio: {p_ro:.4f}  ({p_ro*100:.2f}%)")
    if p_2q is not None:
        print(f"  p_{params['native_2q_gate']} medio:    {p_2q:.5f}  ({p_2q*100:.3f}%)")
    if t1 is not None:
        print(f"  T1 medio:        {t1:.1f} µs")
    if t2 is not None:
        print(f"  T2 medio:        {t2:.1f} µs")
    if params['p_readout_per_qubit']:
        print(f"\n  p_readout per qubit:")
        for q, v in sorted(params['p_readout_per_qubit'].items()):
            bar = "█" * int((v or 0) * 400)
            print(f"    q{q:>3}: {(v or 0):.4f}  {bar}")

def print_xeb_results(xeb_raw_m, xeb_raw_s, xeb_norm_m, xeb_norm_s, n_circ):
    print(f"\n{'='*65}")
    print(f"  XEB — {n_circ} circuiti × {N_SHOTS_XEB} shot")
    print(f"{'='*65}")
    print(f"  XEB raw:   {xeb_raw_m:.4f} ± {xeb_raw_s:.4f}")
    print(f"  XEB norm:  {xeb_norm_m:.4f} ± {xeb_norm_s:.4f}")

    if   xeb_norm_m >= 0.85: q = "✅ alto    (device eccellente)"
    elif xeb_norm_m >= 0.60: q = "✅ buono   (rumore moderato)"
    elif xeb_norm_m >= 0.30: q = "⚠️  ridotto (rumore significativo)"
    else:                    q = "❌ basso   (circuito dominato dal rumore)"
    print(f"  Qualità:   {q}")

    print(f"\n  Collocazione sulle curve v10 Asse A:")
    print(f"  {'p_ro v10':>9}  {'XEBn v10':>9}  {'Δ':>7}")
    print(f"  {'─'*9}  {'─'*9}  {'─'*7}")
    closest = min(V10_XEB_AXIS_A.items(), key=lambda kv: abs(xeb_norm_m - kv[1]))
    for p, xv in V10_XEB_AXIS_A.items():
        mark = " ← device qui" if p == closest[0] else ""
        print(f"  {p*100:>8.1f}%  {xv:>9.3f}  {xeb_norm_m-xv:>+7.3f}{mark}")

def print_nist_results(pvals, bits_flat, n_circ, n_shots):
    bonf  = NIST_THRESH / N_TESTS
    print(f"\n{'='*65}")
    print(f"  NIST-subset + PA — {n_circ} circuiti × {n_shots} shot")
    print(f"  Bit PA totali: {len(bits_flat):,}  |  Bonferroni α: {bonf:.5f}")
    print(f"{'='*65}")
    score = 0
    for name in TEST_NAMES:
        p    = pvals[name]
        mark = "✅" if p >= NIST_THRESH else ("⚠️ " if p >= bonf else "❌")
        if p >= NIST_THRESH: score += 1
        print(f"  {name:<16}  p = {p:.4f}  {mark}")
    print(f"\n  Score: {score}/{N_TESTS}")
    prop1 = np.mean(bits_flat)
    print(f"  Proporzione 1: {prop1:.5f}  (bias: {prop1-0.5:+.5f})")
    if score == N_TESTS:
        print(f"\n  → ✅ PASS: pipeline QRNG validata su hardware reale.")
    elif score >= 3:
        print(f"\n  → ⚠️  PASS PARZIALE: {N_TESTS-score} test sotto soglia.")
    else:
        print(f"\n  → ❌ FAIL: rivedere PA o aumentare i circuiti.")

# ══════════════════════════════════════════════════════════════════════
# PRINT FINALE ARRICCHITO
# ══════════════════════════════════════════════════════════════════════
def print_final_summary(noise_params, xnm, xns, xrm, xrs,
                         pvals, bits_flat, elapsed_total,
                         mode_str, pilot_str):
    nist_score = sum(1 for p in pvals.values() if p >= NIST_THRESH)
    bonf       = NIST_THRESH / N_TESTS
    prop1      = float(np.mean(bits_flat))

    # Qualità XEB
    if   xnm >= 0.85: xeb_q = "alto    ✅"
    elif xnm >= 0.60: xeb_q = "buono   ✅"
    elif xnm >= 0.30: xeb_q = "ridotto ⚠️"
    else:             xeb_q = "basso   ❌"

    # Esito globale
    all_nist_pass  = nist_score == N_TESTS
    xeb_acceptable = xnm >= 0.30
    if all_nist_pass and xeb_acceptable:
        esito = "✅  VALIDAZIONE SUPERATA"
    elif all_nist_pass or xeb_acceptable:
        esito = "⚠️   VALIDAZIONE PARZIALE"
    else:
        esito = "❌  VALIDAZIONE FALLITA"

    W = 65
    print(f"\n{'═'*W}")
    print(f"  RIEPILOGO FINALE — {IBM_BACKEND}  [{mode_str} / {pilot_str}]")
    print(f"{'═'*W}")

    # ── Hardware ────────────────────────────────────────────────────
    print(f"\n  {'─'*28}  HARDWARE  {'─'*24}")
    if noise_params:
        print(f"  Backend:          {noise_params['device_name']}")
        print(f"  Catena qubit:     {noise_params['qubit_chain']}")
        print(f"  Gate 2Q nativo:   {noise_params['native_2q_gate']}")
        p_ro = noise_params.get('p_readout_mean')
        p_2q = noise_params.get('p_2q_mean')
        t1   = noise_params.get('t1_mean_us')
        t2   = noise_params.get('t2_mean_us')
        g    = noise_params.get('native_2q_gate', '2q')
        if p_ro is not None: print(f"  p_readout medio:  {p_ro:.4f}  ({p_ro*100:.2f}%)")
        if p_2q is not None: print(f"  p_{g} medio:    {p_2q:.5f}  ({p_2q*100:.3f}%)")
        if t1   is not None: print(f"  T1 medio:         {t1:.1f} µs")
        if t2   is not None: print(f"  T2 medio:         {t2:.1f} µs")
    else:
        print(f"  Calibrazione hardware non disponibile (dry-run puro).")

    # ── XEB ─────────────────────────────────────────────────────────
    print(f"\n  {'─'*30}  XEB  {'─'*27}")
    print(f"  Circuiti × shot:  {N_CIRCUITS_XEB} × {N_SHOTS_XEB}")
    print(f"  XEB raw:          {xrm:.4f} ± {xrs:.4f}")
    print(f"  XEB norm:         {xnm:.4f} ± {xns:.4f}")
    print(f"  Qualità fidelity: {xeb_q}")
    closest_p = min(V10_XEB_AXIS_A, key=lambda p: abs(xnm - V10_XEB_AXIS_A[p]))
    print(f"  Collocazione v10: ≈ p_ro {closest_p*100:.1f}%  "
          f"(XEBn atteso {V10_XEB_AXIS_A[closest_p]:.3f})")

    # ── NIST + PA ───────────────────────────────────────────────────
    print(f"\n  {'─'*26}  NIST + PA  {'─'*26}")
    print(f"  Circuiti × shot:  {N_CIRCUITS_NIST} × {N_SHOTS_NIST}")
    print(f"  Bit PA totali:    {len(bits_flat):,}")
    print(f"  Proporzione 1:    {prop1:.5f}  (bias {prop1-0.5:+.5f})")
    print(f"  Bonferroni α:     {bonf:.5f}")
    for name in TEST_NAMES:
        p    = pvals[name]
        mark = "✅" if p >= NIST_THRESH else ("⚠️ " if p >= bonf else "❌")
        print(f"    {name:<16}  p = {p:.6f}  {mark}")
    print(f"  Score:            {nist_score}/{N_TESTS}")

    # ── Esito e timing ──────────────────────────────────────────────
    print(f"\n  {'─'*(W - 2)}")
    mins, secs = divmod(int(elapsed_total), 60)
    print(f"  Tempo totale:     {mins}m {secs:02d}s")
    print(f"\n  {esito}")

    if PILOT and not DRY_RUN:
        print(f"\n  ✅ Pilot completato. Imposta PILOT=False per il run completo.")

    print(f"{'═'*W}\n")

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()

    mode_str  = "DRY-RUN"    if DRY_RUN else "QPU REALE"
    pilot_str = "PILOT"      if PILOT   else "FULL"

    print(f"{'='*65}")
    print(f"  QRNG Hardware Validation — v3")
    print(f"  Backend: {IBM_BACKEND}  |  Modalità: {mode_str}  |  {pilot_str}")
    print(f"  XEB:  {N_CIRCUITS_XEB} circuiti × {N_SHOTS_XEB} shot")
    print(f"  NIST: {N_CIRCUITS_NIST} circuiti × {N_SHOTS_NIST} shot")
    print(f"  Tot:  {N_CIRCUITS_XEB*N_SHOTS_XEB + N_CIRCUITS_NIST*N_SHOTS_NIST:,} shot QPU")
    print(f"{'='*65}")

    # ── Step 1: Connessione IBM ───────────────────────────────────────
    print(f"\n  Connessione a IBM Quantum Platform...")
    backend = connect_ibm(IBM_BACKEND, IBM_INSTANCE)

    if backend is None and not DRY_RUN:
        print("❌ Connessione fallita. Imposta IBM_QUANTUM_TOKEN e riprova.")
        sys.exit(1)

    # ── Step 2: Selezione qubit chain ────────────────────────────────
    qubit_chain   = None
    native_2q     = None
    noise_model   = None
    noise_params  = None

    if backend is not None:
        print(f"\n  Selezione catena qubit ottimale su {IBM_BACKEND}...")
        qubit_chain, native_2q = find_best_qubit_chain(backend, N_QUBITS)

        try:
            noise_params = estimate_noise_params(backend, qubit_chain, native_2q)
            print_noise_params(noise_params)
        except Exception as e:
            print(f"  ⚠️  Calibrazione non disponibile: {e}")

        if DRY_RUN:
            try:
                noise_model = NoiseModel.from_backend(backend)
                print(f"\n  [DRY-RUN] NoiseModel da calibrazione {IBM_BACKEND}.")
            except Exception as e:
                print(f"  ⚠️  NoiseModel fallito: {e}. Uso Aer puro.")
    else:
        qubit_chain = list(range(N_QUBITS))
        native_2q   = None
        print(f"  [DRY-RUN PURO] Nessuna calibrazione IBM. Chain: {qubit_chain}")

    # ── Step 3: Genera circuiti ───────────────────────────────────────
    print(f"\n  Generazione {N_CIRCUITS_XEB + N_CIRCUITS_NIST} circuiti RCS...")
    xeb_seeds   = list(range(N_CIRCUITS_XEB))
    nist_seeds  = list(range(N_CIRCUITS_XEB, N_CIRCUITS_XEB + N_CIRCUITS_NIST))
    xeb_circs   = [make_rcs_circuit(N_QUBITS, N_LAYERS, seed=s) for s in xeb_seeds]
    nist_circs  = [make_rcs_circuit(N_QUBITS, N_LAYERS, seed=s) for s in nist_seeds]
    all_circs   = xeb_circs + nist_circs
    all_shots   = [N_SHOTS_XEB]*N_CIRCUITS_XEB + [N_SHOTS_NIST]*N_CIRCUITS_NIST
    print(f"  ✅ Circuiti pronti.")

    # ── Step 4: Esecuzione ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    if DRY_RUN or backend is None:
        print(f"  Esecuzione DRY-RUN (AerSimulator)...")
        t_exec = time.time()
        raw, final_layouts = run_local(all_circs, noise_model, all_shots)
        print(f"  ⏱  {time.time()-t_exec:.1f}s")
    else:
        raw, final_layouts = run_qpu(all_circs, backend, all_shots, qubit_chain)

    xeb_res  = raw[:N_CIRCUITS_XEB]
    nist_res = raw[N_CIRCUITS_XEB:]

    xeb_final_layouts  = final_layouts[:N_CIRCUITS_XEB]  if final_layouts else None

    # ── Step 5: XEB ──────────────────────────────────────────────────
    print(f"\n  Calcolo XEB (statevector locale)...")
    t_xeb = time.time()
    xeb_counts = [r[0] for r in xeb_res]
    xrm, xrs, xnm, xns = compute_xeb_from_counts(
        xeb_counts, xeb_seeds, N_QUBITS, N_LAYERS,
        final_layouts=xeb_final_layouts,
    )
    print(f"  ⏱  {time.time()-t_xeb:.1f}s")
    print_xeb_results(xrm, xrs, xnm, xns, N_CIRCUITS_XEB)

    # ── Step 6: NIST + PA ────────────────────────────────────────────
    print(f"\n  Calcolo NIST + PA...")
    pa_keys  = make_pa_keys(PA_SEED, N_CIRCUITS_NIST, N_SHOTS_NIST, N_QUBITS)
    all_bits = []
    for i, (counts, memory) in enumerate(nist_res):
        all_bits.append(privacy_amplification(memory, pa_keys, i, PA_MIN_HW))
    bits_flat = np.concatenate(all_bits)
    pvals     = evaluate_bits(bits_flat)
    print_nist_results(pvals, bits_flat, N_CIRCUITS_NIST, N_SHOTS_NIST)

    # ── Riepilogo finale ─────────────────────────────────────────────
    print_final_summary(
        noise_params  = noise_params,
        xnm           = xnm,
        xns           = xns,
        xrm           = xrm,
        xrs           = xrs,
        pvals         = pvals,
        bits_flat     = bits_flat,
        elapsed_total = time.time() - t0,
        mode_str      = mode_str,
        pilot_str     = pilot_str,
    )
    
    # ── Salvataggio risultati JSON ───────────────────────────────────
    validation_result = {
        "backend": IBM_BACKEND,
        "instance": IBM_INSTANCE,
        "mode": mode_str,
        "run_type": pilot_str,
        "timestamp_unix": time.time(),
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "n_circuits_xeb": N_CIRCUITS_XEB,
        "n_shots_xeb": N_SHOTS_XEB,
        "n_circuits_nist": N_CIRCUITS_NIST,
        "n_shots_nist": N_SHOTS_NIST,
        "xeb_seeds": xeb_seeds,
        "nist_seeds": nist_seeds,
        "noise_params": noise_params,
        "xeb": {
            "raw_mean": xrm,
            "raw_std": xrs,
            "norm_mean": xnm,
            "norm_std": xns,
        },
        "nist": {
            "threshold": NIST_THRESH,
            "bonferroni_threshold": NIST_THRESH / N_TESTS,
            "score": sum(1 for p in pvals.values() if p >= NIST_THRESH),
            "pvalues": pvals,
        },
        "pa": {
            "n_bits": int(len(bits_flat)),
            "proportion_ones": float(np.mean(bits_flat)),
            "bias": float(np.mean(bits_flat) - 0.5),
            "min_hw_discard": PA_MIN_HW,
        },
        "qubit_chain": qubit_chain,
        "native_2q_gate": native_2q,
        "elapsed_seconds": float(time.time() - t0),
    }

    with open("validation_result.json", "w", encoding="utf-8") as f:
        json.dump(validation_result, f, indent=2, ensure_ascii=False)

    print("\n  💾 Risultati salvati in validation_result.json")
