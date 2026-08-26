"""
Three-factor Hebbian BCI learning model.

Implements a recurrent neural network that learns a 1D BCI cursor task
using the RPE-modulated Hebbian learning rule from Daie & Aitken (2026).

Two learning modes:
  - 'local':      Term 1 only (Eq. 12 in paper)
  - 'semi_local': Terms 1 + 2 (includes one-hop recurrent credit assignment)

Usage:
    python bci_model.py --learning_rule local
    python bci_model.py --learning_rule semi_local
"""

import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# Default hyperparameters
# ──────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    # Network
    N=100,  # number of recurrent units
    dt=0.01,  # timestep (seconds)  — 10 ms
    tau=0.05,  # membrane time constant (seconds)
    g_rec=.9,  # recurrent weight scale (slightly chaotic)
    # Activation
    activation="tanh",  # 'tanh', 'relu', or 'linear'
    # BCI task
    n_trials = 100,
    trial_duration=3.0,  # seconds
    cn_index=0,  # which neuron is the conditioned neuron
    port_distance=0.5,  # initial port distance from mouse (arbitrary units)
    speed_gain=1.0,  # gain mapping CN activity to port speed
    speed_baseline=0.0,  # minimum port speed (port doesn't move backward)
    # Learning rule
    learning_rule="local",  # 'local' or 'semi_local'
    eta=5e-3,  # learning rate
    tau_elig=0.5,  # eligibility trace time constant (seconds)
    tau_bl=1.0,  # baseline activity time constant (seconds)
    tau_rew=5.0,  # baseline reward time constant (seconds)
    n_update=1,  # weight update every n_update timesteps
    semi_local_scale=0.1,  # scaling factor for Term 2 relative to Term 1
    # Input & noise
    input_dim=1,  # dimensionality of external input
    input_strength=0.1,  # scale of go-cue input
    noise_sigma=0.02,  # ongoing noise injected into dynamics each timestep
    trial_start_strength=3,  # amplitude of trial-start pulse (fixed across trials)
    trial_start_duration=0.1,  # duration of trial-start pulse (seconds)
    # Connectivity
    sparsity=1.0,  # fraction of non-zero connections (1.0 = fully connected)
    # Misc
    seed=None,
    save_dir=r"C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\kyles model\results",
)


# ──────────────────────────────────────────────────────────────────────
# Activation helpers
# ──────────────────────────────────────────────────────────────────────
def phi(x, kind="tanh"):
    if kind == "tanh":
        return np.tanh(x)
    elif kind == "relu":
        return np.maximum(0, x)
    elif kind == "linear":
        return x
    else:
        raise ValueError(f"Unknown activation: {kind}")


def phi_prime(x, kind="tanh"):
    if kind == "tanh":
        return 1.0 - np.tanh(x) ** 2
    elif kind == "relu":
        return (x > 0).astype(float)
    elif kind == "linear":
        return np.ones_like(x)
    else:
        raise ValueError(f"Unknown activation: {kind}")


# ──────────────────────────────────────────────────────────────────────
# Network class
# ──────────────────────────────────────────────────────────────────────
class BCINetwork:
    def __init__(self, p):
        self.p = p
        if p["seed"] is not None:
            self.rng = np.random.RandomState(p["seed"])
        else:
            self.rng = np.random.RandomState()

        N = p["N"]
        alpha = p["dt"] / p["tau"]  # leak factor
        self.alpha = alpha

        # Recurrent weights (scaled by 1/sqrt(N))
        self.W = self.rng.randn(N, N) * p["g_rec"] / np.sqrt(N)
        # Zero self-connections
        np.fill_diagonal(self.W, 0)

        # Sparse connectivity: randomly zero out connections
        # W_mask is 1 where connections exist, 0 where structurally absent
        sparsity = p.get("sparsity", 1.0)
        if sparsity < 1.0:
            self.W_mask = (self.rng.rand(N, N) < sparsity).astype(float)
            np.fill_diagonal(self.W_mask, 0)  # no self-connections
            self.W *= self.W_mask
            # Rescale to preserve variance: multiply by 1/sqrt(sparsity)
            self.W *= 1.0 / np.sqrt(sparsity)
        else:
            self.W_mask = np.ones((N, N))
            np.fill_diagonal(self.W_mask, 0)

        # Input weights
        self.W_inp = self.rng.randn(N, p["input_dim"]) * p["input_strength"]

        # Neuron weights c_i: uniform (all neurons contribute equally to loss)
        self.c = np.ones(N)

        # State
        self.h = np.zeros(N)  # pre-activation (hidden state)
        self.h_bar = np.zeros(N)  # baseline activity (running avg)
        self.r_bar = 0.0  # baseline reward (running avg)

        # Eligibility trace
        self.E_bar = np.zeros((N, N))  # accumulated eligibility

    def reset_state(self):
        """Reset neural state between sessions (not weights)."""
        self.h = np.zeros(self.p["N"])

    def step(self, external_input=None):
        """Advance network by one timestep. Returns firing rates."""
        p = self.p
        alpha = self.alpha
        N = p["N"]

        # Recurrent input
        r_prev = phi(self.h, p["activation"])  # rates at t-1
        rec_input = self.W @ r_prev

        # External input
        if external_input is not None:
            ext = self.W_inp @ external_input
        else:
            ext = np.zeros(N)

        # Noise: private iid Gaussian each timestep
        noise = self.rng.randn(N) * p["noise_sigma"] * np.sqrt(alpha)

        # Leaky integration: h_t = (1 - alpha)*h_{t-1} + alpha*(rec + ext) + noise
        h_new = (1 - alpha) * self.h + alpha * (rec_input + ext) + noise
        self.h = h_new

        # Firing rates
        r = phi(self.h, p["activation"])
        return r

    def update_baselines(self, r_reward):
        """Update running averages for activity baselines and reward baseline."""
        p = self.p
        dt_tau_bl = p["dt"] / p["tau_bl"]
        dt_tau_rew = p["dt"] / p["tau_rew"]

        # Activity baselines
        self.h_bar = (1 - dt_tau_bl) * self.h_bar + dt_tau_bl * self.h

        # Reward baseline
        self.r_bar = (1 - dt_tau_rew) * self.r_bar + dt_tau_rew * r_reward

    def compute_eligibility(self, h_prev):
        """Compute instantaneous eligibility trace E_{ij,t} and accumulate."""
        p = self.p
        dt_tau_elig = p["dt"] / p["tau_elig"]

        # Deviation from baseline
        delta_h = self.h - self.h_bar  # (N,)
        # Activation derivative at current pre-activation
        dphi = phi_prime(self.h, p["activation"])  # (N,)
        # Pre-synaptic rates at t-1
        r_prev = phi(h_prev, p["activation"])  # (N,)

        # ── Term 1 (local): E_{ij} = (h_i - h_bar_i) * phi'(h_i) * r_j(t-1)
        E_local = (delta_h * dphi)[:, None] * r_prev[None, :]  # (N, N)

        E = E_local

        # ── Term 2 (semi-local): sum_k c_k*(h_k - h_bar_k)*phi'(h_k)*W_ki * phi'(h_i(t-1))*r_j(t-2)
        if p["learning_rule"] == "semi_local" and hasattr(self, "_r_prev_prev"):
            dphi_prev = phi_prime(h_prev, p["activation"])  # phi'(h_i,t-1)
            r_prev_prev = self._r_prev_prev  # r_j(t-2)

            # sum_k c_k * (h_k - h_bar_k) * phi'(h_k) * W_ki  →  shape (N,)
            # For each i, this is: sum over k of [c_k * delta_h_k * dphi_k * W_{ki}]
            weighted_dev = self.c * delta_h * dphi  # (N,) = c_k * (h_k - h_bar_k) * phi'(h_k)
            backprop_signal = weighted_dev @ self.W  # (N,): for each i, sum_k weighted_dev_k * W_ki

            # E^(2)_{ij} = backprop_signal_i * phi'(h_i,t-1) * r_j(t-1)
            # Using r_j(t-1) rather than r_j(t-2): the one-step lag is a
            # discretization artifact; both approximate the same continuous-time
            # presynaptic rate, and this avoids the term vanishing after trial resets.
            E_semi = (backprop_signal * dphi_prev)[:, None] * r_prev[None, :]
            E = E + p["semi_local_scale"] * E_semi

            # Diagnostic: track relative magnitude of Term 2 vs Term 1
            if not hasattr(self, "_diag_count"):
                self._diag_count = 0
                self._diag_ratio_sum = 0.0
            norm_local = np.linalg.norm(E_local)
            norm_semi = np.linalg.norm(E_semi)
            if norm_local > 1e-12:
                self._diag_ratio_sum += norm_semi / norm_local
                self._diag_count += 1

        # Accumulate eligibility (total and per-term)
        self.E_bar = (1 - dt_tau_elig) * self.E_bar + dt_tau_elig * E

        # Track per-term eligibility traces for analysis
        if not hasattr(self, "E_bar_local"):
            self.E_bar_local = np.zeros_like(self.E_bar)
        self.E_bar_local = (1 - dt_tau_elig) * self.E_bar_local + dt_tau_elig * E_local

        if p["learning_rule"] == "semi_local" and hasattr(self, "_r_prev_prev"):
            if not hasattr(self, "E_bar_semi"):
                self.E_bar_semi = np.zeros_like(self.E_bar)
            self.E_bar_semi = (1 - dt_tau_elig) * self.E_bar_semi + dt_tau_elig * E_semi

        # Store for next timestep's semi-local term
        self._r_prev_prev = r_prev.copy()

    def apply_weight_update(self, RPE):
        """Apply weight update: Delta W = eta * RPE * E_bar."""
        p = self.p
        dW = p["eta"] * RPE * self.E_bar
        # Enforce structural connectivity mask (zero connections stay zero)
        dW *= self.W_mask
        self.W += dW
        # Maintain zero self-connections
        np.fill_diagonal(self.W, 0)

        # Accumulate per-term contributions to total ΔW
        if not hasattr(self, "dW_from_local"):
            self.dW_from_local = np.zeros_like(self.W)
        self.dW_from_local += p["eta"] * RPE * self.E_bar_local * self.W_mask

        if p["learning_rule"] == "semi_local" and hasattr(self, "E_bar_semi"):
            if not hasattr(self, "dW_from_semi"):
                self.dW_from_semi = np.zeros_like(self.W)
            self.dW_from_semi += p["eta"] * RPE * p["semi_local_scale"] * self.E_bar_semi * self.W_mask

        return dW


# ──────────────────────────────────────────────────────────────────────
# BCI Task — reward port mechanics
# ──────────────────────────────────────────────────────────────────────
class BCITask:
    def __init__(self, p):
        self.p = p
        self.n_steps = int(p["trial_duration"] / p["dt"])

    def cn_to_speed(self, cn_activity):
        """Transfer function: CN activity → port speed.
        Maps tanh output [-1, 1] → [0, speed_gain] so port always moves
        and RPE is informative for any neuron regardless of baseline activity."""
        speed = self.p["speed_gain"] * (cn_activity + 1.0) / 2.0
        return max(0.0, speed)

    def run_trial(self, net, trial_idx, learning=True):
        """Run a single trial.
        Port starts at port_distance, moves toward mouse.
        Speed = f(CN activity). Mouse feels port speed continuously.
        RPE = port_speed - baseline_speed.
        Trial ends (hit) when port reaches mouse (position <= 0).
        If learning=False, eligibility and weight updates are skipped."""
        p = self.p
        n_steps = self.n_steps
        cn = p["cn_index"]
        dt = p["dt"]

        # Port state
        port_pos = p["port_distance"]  # starts far from mouse
        hit = False
        hit_time = None

        # Trial-start pulse: fixed pattern (same every trial, like a go cue)
        n_start = int(p["trial_start_duration"] / dt)
        if not hasattr(net, "_start_pulse"):
            net._start_pulse = net.rng.randn(p["N"]) * p["trial_start_strength"]
        start_pulse = net._start_pulse

        # Storage
        N = p["N"]
        rates_cn = np.zeros(n_steps)
        rates_all = np.zeros((n_steps, N))
        h_all = np.zeros((n_steps, N))
        h_bar_all = np.zeros((n_steps, N))
        port_speed_trace = np.zeros(n_steps)
        port_pos_trace = np.zeros(n_steps)
        rpe_trace = np.zeros(n_steps)

        # Coactivity accumulators
        coact_sum = np.zeros((N, N))       # C_{kj} = sum_t (r_k - r_bar_k) * r_j
        coact_rpe_sum = np.zeros((N, N))   # C^RPE_{kj} = sum_t RPE_t * (r_k - r_bar_k) * r_j
        r_sum = np.zeros(N)  # for computing running mean of rates

        # Initialize semi-local storage if needed
        if p["learning_rule"] == "semi_local" and not hasattr(net, "_r_prev_prev"):
            net._r_prev_prev = phi(net.h, p["activation"]).copy()

        for t in range(n_steps):
            h_prev = net.h.copy()

            # Go-cue input (constant) + trial-start pulse (brief)
            inp = np.array([1.0]) * p["input_strength"]
            r = net.step(external_input=inp)
            # Inject trial-start pulse directly into h (bypasses W_inp)
            if t < n_start:
                net.h += start_pulse * (dt / p["tau"])

            # Recompute rates after pulse injection
            r = phi(net.h, p["activation"])

            # CN activity → port speed
            cn_act = r[cn]
            port_speed = self.cn_to_speed(cn_act)

            # Move port toward mouse
            port_pos -= port_speed * dt
            port_pos = max(port_pos, 0.0)  # clamp at mouse position

            # The reward signal is the port speed (mouse feels it continuously)
            reward_signal = port_speed

            # Update baselines (activity + reward)
            net.update_baselines(reward_signal)

            # RPE = current port speed - baseline speed
            RPE = reward_signal - net.r_bar

            # Compute and accumulate eligibility, apply weight updates
            if learning:
                net.compute_eligibility(h_prev)
                if (t + 1) % p["n_update"] == 0:
                    net.apply_weight_update(RPE)

            # Check for hit
            if port_pos <= 0.0 and not hit:
                hit = True
                hit_time = t * dt

            # Accumulate coactivity (with and without RPE weighting)
            r_sum += r
            r_bar_running = r_sum / (t + 1)
            dev = r - r_bar_running  # (N,) deviation from running mean
            coact_sum += dev[:, None] * r[None, :]            # (N, N)
            coact_rpe_sum += RPE * dev[:, None] * r[None, :]  # (N, N) RPE-weighted

            # Log
            rates_cn[t] = cn_act
            rates_all[t] = r
            h_all[t] = net.h.copy()
            h_bar_all[t] = net.h_bar.copy()
            port_speed_trace[t] = port_speed
            port_pos_trace[t] = port_pos
            rpe_trace[t] = RPE

        trial_log = {
            "trial": trial_idx,
            "cn_mean": float(np.mean(rates_cn)),
            "cn_max": float(np.max(rates_cn)),
            "mean_speed": float(np.mean(port_speed_trace)),
            "hit": hit,
            "hit_time": hit_time,  # None if miss
            "mean_rpe": float(np.mean(rpe_trace)),
            "mean_rate": float(np.mean(rates_all)),
            "neuron_means": np.mean(rates_all, axis=0),  # (N,) mean rate per neuron
            "neuron_maxs": np.max(rates_all, axis=0),   # (N,) max rate per neuron
            "rates_cn": rates_cn,
            "rates_all": rates_all,        # (n_steps, N) full activity traces
            "h_all": h_all,                # (n_steps, N) pre-activation traces
            "h_bar_all": h_bar_all,        # (n_steps, N) baseline traces
            "port_speed_trace": port_speed_trace,
            "port_pos_trace": port_pos_trace,
            "rpe_trace": rpe_trace,
            "coactivity": coact_sum / n_steps,      # time-averaged C_{kj}
            "coactivity_rpe": coact_rpe_sum / n_steps,  # RPE-weighted C_{kj}
        }
        return trial_log


# ──────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────
def train(p):
    net = BCINetwork(p)
    task = BCITask(p)

    # ── Probe trial (no learning) to select CN ──
    # Run one trial to measure each neuron's baseline response
    probe_log = task.run_trial(net, -1, learning=False)
    neuron_maxs_probe = probe_log["neuron_maxs"]

    # Pick CN: neuron with weakest max response (most room to learn)
    cn_idx = int(np.argmin(np.abs(neuron_maxs_probe)))
    p["cn_index"] = cn_idx
    print(f"Selected CN = neuron {cn_idx} "
          f"(max response = {neuron_maxs_probe[cn_idx]:.4f})")

    # Reset state after probe
    net.h[:] = 0.0
    net.E_bar[:] = 0.0

    # Estimate initial baselines with selected CN
    for _ in range(200):
        inp = np.array([1.0]) * p["input_strength"]
        r = net.step(external_input=inp)
        cn_act = r[cn_idx]
        port_speed = task.cn_to_speed(cn_act)
        net.update_baselines(port_speed)

    # Reset state but keep baselines
    net.h = np.zeros(p["N"])
    net.E_bar = np.zeros((p["N"], p["N"]))

    # Storage
    trial_logs = []
    neuron_means_all = []  # list of (N,) arrays, one per trial
    neuron_maxs_all = []
    rates_all_trials = []  # list of (n_steps, N) arrays
    rpe_all_trials = []    # list of (n_steps,) arrays
    h_all_trials = []      # list of (n_steps, N) arrays
    h_bar_all_trials = []  # list of (n_steps, N) arrays
    coact_session = np.zeros((p["N"], p["N"]))      # session-total coactivity
    coact_rpe_session = np.zeros((p["N"], p["N"]))  # RPE-weighted coactivity
    W_initial = net.W.copy()
    W_snapshots = {}

    print(f"Training {p['n_trials']} trials | rule={p['learning_rule']} | "
          f"N={p['N']} | eta={p['eta']}")
    print("-" * 60)

    t0 = time.time()
    for trial in range(p["n_trials"]):
        log = task.run_trial(net, trial)

        # Summary to console
        if (trial + 1) % 20 == 0 or trial == 0:
            ht = f"{log['hit_time']:.2f}s" if log['hit'] else "miss"
            print(f"Trial {trial+1:4d} | CN mean={log['cn_mean']:.4f} | "
                  f"speed={log['mean_speed']:.4f} | {ht}")

        # Store per-neuron means and maxs
        neuron_means_all.append(log["neuron_means"])
        neuron_maxs_all.append(log["neuron_maxs"])
        rates_all_trials.append(log["rates_all"])
        rpe_all_trials.append(log["rpe_trace"])
        h_all_trials.append(log["h_all"])
        h_bar_all_trials.append(log["h_bar_all"])
        coact_session += log["coactivity"]
        coact_rpe_session += log["coactivity_rpe"]

        # Keep only summary scalars for JSON serialization
        trial_logs.append({
            "trial": log["trial"],
            "cn_mean": log["cn_mean"],
            "cn_max": log["cn_max"],
            "mean_speed": log["mean_speed"],
            "hit": log["hit"],
            "hit_time": log["hit_time"],
            "mean_rpe": log["mean_rpe"],
            "mean_rate": log["mean_rate"],
        })

        # Snapshot weights at select trials
        if trial in [0, 9, 49, 99, p["n_trials"] - 1]:
            W_snapshots[trial] = net.W.copy()

        # Reset state between trials (CN must return to baseline)
        net.h[:] = 0.0
        net.E_bar[:] = 0.0

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Diagnostic: report Term 2 / Term 1 ratio
    if p["learning_rule"] == "semi_local" and hasattr(net, "_diag_count"):
        if net._diag_count > 0:
            avg_ratio = net._diag_ratio_sum / net._diag_count
            print(f"Semi-local diagnostic: avg ||Term2||/||Term1|| = {avg_ratio:.6f} "
                  f"({net._diag_count} samples)")
        else:
            print("Semi-local diagnostic: Term 1 was always ~zero, no ratio computed")

    # (n_trials, N) arrays
    neuron_means_all = np.array(neuron_means_all)
    neuron_maxs_all = np.array(neuron_maxs_all)
    coact_session /= p["n_trials"]
    coact_rpe_session /= p["n_trials"]
    return (net, trial_logs, W_initial, W_snapshots,
            neuron_means_all, neuron_maxs_all, coact_session, coact_rpe_session,
            rates_all_trials, rpe_all_trials, h_all_trials, h_bar_all_trials)


# ──────────────────────────────────────────────────────────────────────
# Plotting & saving
# ──────────────────────────────────────────────────────────────────────
def save_results(p, trial_logs, W_initial, W_final, W_snapshots,
                  neuron_means_all, neuron_maxs_all, coact_session,
                  coact_rpe_session):
    save_dir = p["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    rule = p["learning_rule"]

    # ── 1. Summary variables (JSON) ──
    summary = {
        "params": {k: v for k, v in p.items()},
        "trial_logs": trial_logs,
    }
    summary_path = os.path.join(save_dir, f"summary_{rule}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary → {summary_path}")

    # ── 2. Weight matrices (npz) ──
    weight_path = os.path.join(save_dir, f"weights_{rule}.npz")
    save_dict = {"W_initial": W_initial, "W_final": W_final,
                  "neuron_means_all": neuron_means_all,
                  "neuron_maxs_all": neuron_maxs_all}
    for trial_idx, W in W_snapshots.items():
        save_dict[f"W_trial{trial_idx}"] = W
    np.savez(weight_path, **save_dict)
    print(f"Saved weights → {weight_path}")

    # ── 3. Plots ──
    trials = [log["trial"] for log in trial_logs]
    cn_means = [log["cn_mean"] for log in trial_logs]
    cn_maxs = [log["cn_max"] for log in trial_logs]
    mean_speeds = [log["mean_speed"] for log in trial_logs]
    hit_times = [log["hit_time"] if log["hit"] else p["trial_duration"]
                 for log in trial_logs]
    hits = [log["hit"] for log in trial_logs]
    mean_rpes = [log["mean_rpe"] for log in trial_logs]

    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    fig.suptitle(f"BCI Learning — {rule} rule", fontsize=14)

    # (a) CN mean activity
    axes[0, 0].plot(trials, cn_means, "k-", lw=0.8)
    axes[0, 0].set_ylabel("CN mean rate")
    axes[0, 0].set_title("CN mean activity per trial")

    # (b) CN max activity
    axes[0, 1].plot(trials, cn_maxs, "k-", lw=0.8)
    axes[0, 1].set_ylabel("CN max rate")
    axes[0, 1].set_title("CN max activity per trial")

    # (c) Mean port speed
    axes[1, 0].plot(trials, mean_speeds, "k-", lw=0.8)
    axes[1, 0].set_ylabel("Mean port speed")
    axes[1, 0].set_title("Mean port speed per trial")

    # (d) Hit time (reward time)
    axes[1, 1].plot(trials, hit_times, "k-", lw=0.8)
    axes[1, 1].axhline(p["trial_duration"], color="r", ls="--", lw=0.8,
                        label="trial limit (miss)")
    axes[1, 1].set_ylabel("Time to reward (s)")
    axes[1, 1].set_title("Reward time per trial")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].invert_yaxis()

    # (e) Mean RPE
    axes[2, 0].plot(trials, mean_rpes, "k-", lw=0.8)
    axes[2, 0].axhline(0, color="gray", ls=":", lw=0.5)
    axes[2, 0].set_xlabel("Trial")
    axes[2, 0].set_ylabel("Mean RPE")
    axes[2, 0].set_title("Mean RPE per trial")

    # (f) Weight change distribution
    dW = W_snapshots.get(p["n_trials"] - 1, np.zeros_like(W_initial)) - W_initial
    axes[2, 1].hist(dW.ravel(), bins=80, color="k", alpha=0.7, density=True)
    axes[2, 1].set_xlabel("ΔW")
    axes[2, 1].set_ylabel("Density")
    axes[2, 1].set_title("Weight change distribution (final − initial)")

    # (g) Per-neuron activity change vs CN
    # Change in max rate: last 10 trials minus first 10 trials
    early = neuron_maxs_all[:10].mean(axis=0)   # (N,)
    late = neuron_maxs_all[-10:].mean(axis=0)   # (N,)
    delta_rate = late - early
    cn_idx = p["cn_index"]
    non_cn = np.ones(p["N"], dtype=bool)
    non_cn[cn_idx] = False

    axes[3, 0].bar(0, delta_rate[cn_idx], color="r", width=0.8, label="CN")
    sorted_idx = np.argsort(delta_rate[non_cn])[::-1]
    axes[3, 0].bar(np.arange(1, p["N"]), delta_rate[non_cn][sorted_idx],
                   color="0.5", width=1.0, label="other neurons")
    axes[3, 0].set_xlabel("Neuron (sorted)")
    axes[3, 0].set_ylabel("Δ max rate (last 10 − first 10)")
    axes[3, 0].set_title("Activity change per neuron (max)")
    axes[3, 0].legend(fontsize=8)
    axes[3, 0].axhline(0, color="gray", ls=":", lw=0.5)

    # (h) Per-neuron max rate across trials, CN highlighted
    axes[3, 1].plot(trials, neuron_maxs_all[:, non_cn], color="0.8", lw=0.3)
    axes[3, 1].plot(trials, neuron_maxs_all[:, cn_idx], color="r", lw=1.5,
                    label="CN")
    axes[3, 1].set_xlabel("Trial")
    axes[3, 1].set_ylabel("Max rate")
    axes[3, 1].set_title("Per-neuron max rate across trials")
    axes[3, 1].legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"learning_curves_{rule}.png")
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"Saved figure → {fig_path}")

    # ── 4. Weight matrix comparison ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    vmax = max(abs(W_initial).max(), abs(W_snapshots[p["n_trials"] - 1]).max())
    im0 = axes2[0].imshow(W_initial, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes2[0].set_title("W initial")
    plt.colorbar(im0, ax=axes2[0], fraction=0.046)

    W_final_snap = W_snapshots[p["n_trials"] - 1]
    im1 = axes2[1].imshow(W_final_snap, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes2[1].set_title("W final")
    plt.colorbar(im1, ax=axes2[1], fraction=0.046)

    dW_max = max(abs(dW).max(), 1e-8)
    im2 = axes2[2].imshow(dW, cmap="RdBu_r", vmin=-dW_max, vmax=dW_max, aspect="auto")
    axes2[2].set_title("ΔW (final − initial)")
    plt.colorbar(im2, ax=axes2[2], fraction=0.046)

    fig2.suptitle(f"Weight matrices — {rule} rule", fontsize=14)
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, f"weight_matrices_{rule}.png")
    plt.savefig(fig2_path, dpi=150)
    plt.show()
    print(f"Saved figure → {fig2_path}")

    # ── 5. Non-local plasticity analysis ──
    # RPE-weighted predictors (matches the learning rule: ΔW = η * RPE * E)
    # Local predictor:     P^L_{ij}  = <RPE * (r_i - r_bar_i) * r_j>
    # Non-local predictor: P^NL_{ij} = sum_k W_{ki} * <RPE * (r_k - r_bar_k) * r_j>
    N = p["N"]
    C = coact_rpe_session  # (N, N): RPE-weighted C_{kj}

    P_local = C  # P_local[i,j] = <RPE * (r_i - r_bar_i) * r_j>

    # P_NL[i,j] = sum_k W_initial[k,i] * C[k,j] = (W_initial.T @ C)[i,j]
    P_nonlocal = W_initial.T @ C  # (N, N)

    # Flatten, excluding diagonal (no self-connections)
    mask = ~np.eye(N, dtype=bool)
    dw_flat = dW[mask]
    pl_flat = P_local[mask]
    pnl_flat = P_nonlocal[mask]

    # Correlations
    r_local = np.corrcoef(dw_flat, pl_flat)[0, 1]
    r_nonlocal = np.corrcoef(dw_flat, pnl_flat)[0, 1]

    # ── Multiple regression: ΔW ~ β1*P_local + β2*P_nonlocal ──
    # Build design matrix [P_local, P_nonlocal, intercept]
    X = np.column_stack([pl_flat, pnl_flat, np.ones(len(pl_flat))])
    beta, _, _, _ = np.linalg.lstsq(X, dw_flat, rcond=None)

    # Predictions from each model
    pred_local_only = pl_flat * np.linalg.lstsq(
        np.column_stack([pl_flat, np.ones(len(pl_flat))]), dw_flat, rcond=None)[0][0] + \
        np.linalg.lstsq(
        np.column_stack([pl_flat, np.ones(len(pl_flat))]), dw_flat, rcond=None)[0][1]
    pred_full = X @ beta

    # R² for local-only and full model
    ss_tot = np.sum((dw_flat - dw_flat.mean()) ** 2)
    r2_local = 1 - np.sum((dw_flat - pred_local_only) ** 2) / ss_tot
    r2_full = 1 - np.sum((dw_flat - pred_full) ** 2) / ss_tot
    r2_nonlocal_added = r2_full - r2_local

    # Partial correlation: correlate residuals of ΔW ~ P_local with residuals of P_nonlocal ~ P_local
    # Regress out P_local from both ΔW and P_nonlocal
    coef_dw = np.polyfit(pl_flat, dw_flat, 1)
    resid_dw = dw_flat - np.polyval(coef_dw, pl_flat)
    coef_pnl = np.polyfit(pl_flat, pnl_flat, 1)
    resid_pnl = pnl_flat - np.polyval(coef_pnl, pl_flat)
    r_partial = np.corrcoef(resid_dw, resid_pnl)[0, 1]

    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    fig3.suptitle(f"Non-local plasticity analysis — {rule} rule", fontsize=14)

    # (a) Local: ΔW vs C_{ij}
    axes3[0].scatter(pl_flat, dw_flat, s=1, alpha=0.3, color="k")
    m, b = np.polyfit(pl_flat, dw_flat, 1)
    x_range = np.array([pl_flat.min(), pl_flat.max()])
    axes3[0].plot(x_range, m * x_range + b, "r-", lw=1.5)
    axes3[0].set_xlabel(r"Local: $\langle RPE \cdot (r_i - \bar{r}_i) \cdot r_j\rangle$")
    axes3[0].set_ylabel(r"$\Delta W_{ij}$")
    axes3[0].set_title(f"Local: r = {r_local:.4f}, R² = {r2_local:.4f}")

    # (b) Non-local: ΔW vs sum_k W_{ki} C_{kj}
    axes3[1].scatter(pnl_flat, dw_flat, s=1, alpha=0.3, color="k")
    m, b = np.polyfit(pnl_flat, dw_flat, 1)
    x_range = np.array([pnl_flat.min(), pnl_flat.max()])
    axes3[1].plot(x_range, m * x_range + b, "r-", lw=1.5)
    axes3[1].set_xlabel(r"Non-local: $\sum_k W_{ki} \langle RPE \cdot (r_k - \bar{r}_k) \cdot r_j\rangle$")
    axes3[1].set_ylabel(r"$\Delta W_{ij}$")
    axes3[1].set_title(f"Non-local: r = {r_nonlocal:.4f}")

    # (c) Partial correlation: residuals after regressing out local predictor
    axes3[2].scatter(resid_pnl, resid_dw, s=1, alpha=0.3, color="k")
    m, b = np.polyfit(resid_pnl, resid_dw, 1)
    x_range = np.array([resid_pnl.min(), resid_pnl.max()])
    axes3[2].plot(x_range, m * x_range + b, "r-", lw=1.5)
    axes3[2].set_xlabel("Non-local predictor\n(residual after removing local)")
    axes3[2].set_ylabel(r"$\Delta W_{ij}$ residual")
    axes3[2].set_title(f"Partial r = {r_partial:.4f}, ΔR² = {r2_nonlocal_added:.4f}")

    plt.tight_layout()
    fig3_path = os.path.join(save_dir, f"nonlocal_analysis_{rule}.png")
    plt.savefig(fig3_path, dpi=150)
    plt.show()
    print(f"Saved figure → {fig3_path}")
    print(f"Local R²:                {r2_local:.4f}")
    print(f"Full (local+nonlocal) R²: {r2_full:.4f}")
    print(f"ΔR² from non-local:       {r2_nonlocal_added:.4f}")
    print(f"Partial r (non-local | local): {r_partial:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Main — edit parameters here, then run the whole file in Spyder (F5)
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── User parameters (edit these) ──
    p = dict(DEFAULTS)
    p["learning_rule"] = "semi_local"   # "local" or "semi_local"
    # p["eta"] = 1e-3
    # p["n_trials"] = 200

    # ── Run ──
    (net, trial_logs, W_initial, W_snapshots,
     neuron_means_all, neuron_maxs_all, coact_session, coact_rpe_session,
     rates_all_trials, rpe_all_trials, h_all_trials, h_bar_all_trials) = train(p)
    save_results(p, trial_logs, W_initial, net.W, W_snapshots,
                 neuron_means_all, neuron_maxs_all, coact_session, coact_rpe_session)
    print("\nDone. Check results/ for saved files.")
