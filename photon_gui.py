"""
Interactive GUI for the photon-counting histogram simulation.

Run:  python photon_gui.py      (NumPy, Matplotlib, and Tkinter required)

* Runs start automatically and accumulate in geometrically spaced batches; Apply & Restart starts a
    fresh simulation with the current settings.
* Physics / detector parameters are applied on "Apply & Restart" (the accumulated histogram would
  be meaningless if they changed mid-run).
* Display options (components, error bars, band, y-scale, convergence panel) change live.
* Click on the histogram (toolbar in zoom/pan off) to pick the bin shown in the convergence panel.
* The "Scattering" tab reuses the Mie/Rayleigh scattering-to-lens model from
  scattering_to_lens_v2.py (needs `pip install miepython` for the fog_mie medium) to estimate how
  many photons/s a laser beam scatters into the lens. Tick "Compute signal rate from scattering
  model" on the Physics tab to drive the simulation's signal rate from that estimate (photons/s
  into the lens x SPDE, the single-photon detection efficiency); untick it to set the signal rate
  manually as before.
"""
import time
import heapq
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass, field

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection

import scattering_to_lens_v2 as scat

# Experiment and detector defaults
DEFAULT_T_RUN = 50e-9
DEFAULT_BIN_WIDTH = 25e-12
DEFAULT_SIGNAL_RATE_HZ = 2e6
DEFAULT_NOISE_RATE = 2e4
DEFAULT_DARK_COUNT_RATE = 200.0
DEFAULT_AFTERPULSE_PROBABILITY = 0.005
DEFAULT_AFTERPULSE_TIME_CONSTANT = 20e-9
DEFAULT_TIMING_FWHM = 40e-12
DEFAULT_T_CENTER = 20e-9
DEFAULT_A = 1e9
DEFAULT_GAUSSIAN_SIGMA = 5e-9
DEFAULT_SIGNAL_RATE_MODE = "Peak"

# Mie/Rayleigh scattering-to-lens defaults (mirrors scattering_to_lens_v2.py SETTINGS)
DEFAULT_SPDE = 0.3  # single-photon detection efficiency, multiplied into the computed signal rate
DEFAULT_USE_SCATTERING_RATE = False
DEFAULT_SCATTER_WAVELENGTH_NM = scat.WAVELENGTH_NM
DEFAULT_SCATTER_POWER_W = scat.POWER_W
DEFAULT_SCATTER_POWER_MODE = scat.POWER_MODE
DEFAULT_SCATTER_PULSE_LENGTH_S = scat.PULSE_LENGTH_S
DEFAULT_SCATTER_REP_RATE_HZ = scat.REP_RATE_HZ
DEFAULT_SCATTER_LENS_DIAMETER_M = scat.LENS_DIAMETER_M
DEFAULT_SCATTER_DISTANCE_M = scat.DISTANCE_M
DEFAULT_SCATTER_SOLID_ANGLE_SR = scat.SOLID_ANGLE_SR
DEFAULT_SCATTER_LASER_DISTANCE_M = scat.LASER_DISTANCE_M
DEFAULT_SCATTER_FOV_DEG = scat.FOV_DEG
DEFAULT_SCATTER_HORIZONTAL_RESOLUTION = scat.HORIZONTAL_RESOLUTION
DEFAULT_SCATTER_VIEWED_BEAM_LENGTH_M = scat.VIEWED_BEAM_LENGTH_M
DEFAULT_SCATTER_POLARIZATION = scat.POLARIZATION
DEFAULT_SCATTER_MEDIUM = scat.MEDIUM
DEFAULT_SCATTER_TEMPERATURE_K = scat.TEMPERATURE_K
DEFAULT_SCATTER_PRESSURE_PA = scat.PRESSURE_PA
DEFAULT_SCATTER_FOG_VISIBILITY_M = scat.FOG_VISIBILITY_M
DEFAULT_SCATTER_FOG_G = scat.FOG_G
DEFAULT_SCATTER_FOG_MODE_RADIUS_UM = scat.FOG_MODE_RADIUS_UM
DEFAULT_SCATTER_FOG_ALPHA = scat.FOG_ALPHA
DEFAULT_SCATTER_WATER_N = scat.WATER_REFRACTIVE_INDEX.real
DEFAULT_SCATTER_WATER_K = -scat.WATER_REFRACTIVE_INDEX.imag
DEFAULT_SCATTER_MIE_N_RADII = scat.MIE_N_RADII
DEFAULT_SCATTER_MIE_N_ANGLES = scat.MIE_N_ANGLES

SPDE = DEFAULT_SPDE
USE_SCATTERING_RATE = DEFAULT_USE_SCATTERING_RATE
T_RUN = DEFAULT_T_RUN
BIN_WIDTH = DEFAULT_BIN_WIDTH
MAX_RUNS = 1_000_000
SIGNAL_RATE_HZ = DEFAULT_SIGNAL_RATE_HZ
SIGNAL_RATE_MODE = DEFAULT_SIGNAL_RATE_MODE
SIGNAL_MEAN_PER_RUN = 0.0
T_CENTER = DEFAULT_T_CENTER
A = DEFAULT_A
PULSE_SHAPE = "Logistic"
GAUSSIAN_SIGMA = DEFAULT_GAUSSIAN_SIGMA
DARK_COUNT_RATE = DEFAULT_DARK_COUNT_RATE
NOISE_RATE = DEFAULT_NOISE_RATE
AFTERPULSE_PROBABILITY = DEFAULT_AFTERPULSE_PROBABILITY
AFTERPULSE_TIME_CONSTANT = DEFAULT_AFTERPULSE_TIME_CONSTANT
DEAD_TIME = 45e-9
RUN_PERIOD = T_RUN
TIMING_FWHM = DEFAULT_TIMING_FWHM
ENABLE_DARK_COUNTS = True
ENABLE_AFTERPULSING = True
ENABLE_DEADTIME = True
ENABLE_TIMING_JITTER = True
ENABLE_HPTDC_QUANTIZATION = True
ALLOW_AFTERPULSE_RECURSION = True
AFTERPULSE_FROM_DARK = True
AFTERPULSE_FROM_NOISE = True
SHOW_SIGNAL_COMPONENT = True
SHOW_DARK_COMPONENT = True
SHOW_AP_COMPONENT = True
SHOW_NOISE_COMPONENT = True
SHOW_EXPECTED_BAND = False
SHOW_DEADTIME_ESTIMATE = True
N_SNAPSHOTS = 120
SUBSAMPLE = 10
CHUNK_RUNS = 1_000_000
RANDOM_SEED = 42
N_BINS = int(round(T_RUN / BIN_WIDTH))
SIGMA_DETECTOR = TIMING_FWHM / 2.355
SIGMA_HPTDC = BIN_WIDTH / np.sqrt(12)
SIGMA_TOTAL = np.hypot(SIGMA_DETECTOR, SIGMA_HPTDC)
KIND_NAMES = ("signal", "dark", "afterpulse", "noise")
SIG, DARK, AP, NOISE = 0, 1, 2, 3


@dataclass
class SimulationState:
    next_run: int = 0
    last_accepted_time: float = -np.inf
    event_serial: int = 0
    pending_events: list = field(default_factory=list)
    accepted_total: int = 0


@dataclass(frozen=True)
class Detector:
    dark_rate: float
    noise_rate: float
    afterpulse_probability: float
    tau_ap: float
    dead_time: float
    dark: bool
    ap: bool
    deadtime: bool
    jitter: bool
    recursion: bool
    ap_from_dark: bool
    ap_from_noise: bool

    @property
    def p_afterpulse(self):
        return self.afterpulse_probability if self.ap else 0.0


def _softplus_integral(t):
    return np.logaddexp(0.0, A * (t - T_CENTER)) / A


def pulse_cell_counts(n_cells):
    edges = np.linspace(0.0, T_RUN, n_cells + 1)
    dt = T_RUN / n_cells
    if PULSE_SHAPE == "Gaussian":
        centers = (edges[:-1] + edges[1:]) / 2
        intensity = np.exp(-0.5 * ((centers - T_CENTER) / GAUSSIAN_SIGMA) ** 2)
        peak_time = np.clip(T_CENTER, 0.0, T_RUN)
        peak_intensity = np.exp(-0.5 * ((peak_time - T_CENTER) / GAUSSIAN_SIGMA) ** 2)
        mass = intensity * dt
    else:
        mass = np.diff(_softplus_integral(edges))
        peak_intensity = 1.0 / (1.0 + np.exp(-A * (T_RUN - T_CENTER)))
    if SIGNAL_RATE_MODE == "Peak":
        return mass * SIGNAL_RATE_HZ / peak_intensity
    return mass / mass.sum() * SIGNAL_RATE_HZ * T_RUN


def signal_mean_per_run():
    return float(pulse_cell_counts(N_BINS * SUBSAMPLE).sum())


SIGNAL_MEAN_PER_RUN = signal_mean_per_run()


def _smear(pmf):
    dt = BIN_WIDTH / SUBSAMPLE
    half = int(np.ceil(5 * SIGMA_TOTAL / dt))
    x = np.arange(-half, half + 1) * dt
    kernel = np.exp(-0.5 * (x / SIGMA_TOTAL) ** 2)
    return np.convolve(pmf, kernel / kernel.sum(), mode="same")


def expected_bin_probabilities(cfg):
    n = N_BINS * SUBSAMPLE
    dt = BIN_WIDTH / SUBSAMPLE
    sig = pulse_cell_counts(n)
    dark = np.full(n, cfg.dark_rate * T_RUN / n if cfg.dark else 0.0)
    noise = np.full(n, cfg.noise_rate * T_RUN / n)
    period_cells = int(round(RUN_PERIOD / dt))
    j = np.arange(period_cells)
    start = int(np.ceil(cfg.dead_time / dt)) if cfg.deadtime else 0
    min_periods = np.where(j >= start, 0, np.ceil((start - j) / period_cells)).astype(int)
    q = np.exp(-dt / cfg.tau_ap)
    delay_pmf = q ** (j + min_periods * period_cells) * (1 - q) / (1 - q ** period_cells)
    parents = np.zeros(period_cells)
    parents[:n] = sig
    if cfg.ap_from_dark:
        parents[:n] += dark
    if cfg.ap_from_noise:
        parents[:n] += noise
    circ = np.fft.irfft(np.fft.rfft(parents) * np.fft.rfft(delay_pmf), period_cells)
    ap = np.clip(cfg.p_afterpulse * circ[:n], 0.0, None)
    out = []
    for component in (sig, dark, ap, noise):
        if cfg.jitter:
            component = _smear(component)
        out.append(component.reshape(N_BINS, SUBSAMPLE).sum(axis=1))
    return np.array(out)


def simulate_chunk(n_runs, cdf_fine, cfg, rngs, state):
    r_sig, r_dark, r_ap, r_jit, r_noise = rngs
    dt_fine = BIN_WIDTH / SUBSAMPLE
    start_time = state.next_run * RUN_PERIOD
    end_time = start_time + n_runs * RUN_PERIOD
    events = state.pending_events
    state.pending_events = []

    signal_counts = r_sig.poisson(SIGNAL_MEAN_PER_RUN, n_runs)
    sig_runs = np.repeat(np.arange(n_runs, dtype=np.int64), signal_counts)
    cell = np.minimum(np.searchsorted(cdf_fine, r_sig.random(sig_runs.size), side="right"),
                      cdf_fine.size - 1)
    t_sig = (cell + r_sig.random(sig_runs.size)) * dt_fine
    dark_counts = r_dark.poisson(cfg.dark_rate * T_RUN, n_runs) if cfg.dark else np.zeros(n_runs, int)
    dark_runs = np.repeat(np.arange(n_runs, dtype=np.int64), dark_counts)
    t_dark = r_dark.random(dark_runs.size) * T_RUN
    noise_counts = r_noise.poisson(cfg.noise_rate * T_RUN, n_runs)
    noise_runs = np.repeat(np.arange(n_runs, dtype=np.int64), noise_counts)
    t_noise = r_noise.random(noise_runs.size) * T_RUN

    serial = state.event_serial
    for runs, times, kind in ((sig_runs, t_sig, SIG), (dark_runs, t_dark, DARK),
                              (noise_runs, t_noise, NOISE)):
        for run, time_in_run in zip(runs, times):
            events.append((start_time + int(run) * RUN_PERIOD + float(time_in_run), serial, kind))
            serial += 1
    heapq.heapify(events)

    rejected = np.zeros(len(KIND_NAMES), dtype=np.int64)
    accepted = [[] for _ in KIND_NAMES]
    while events and events[0][0] < end_time:
        event_time, _, kind = heapq.heappop(events)
        if cfg.deadtime and event_time - state.last_accepted_time < cfg.dead_time:
            rejected[kind] += 1
            continue
        state.last_accepted_time = event_time
        accepted[kind].append(event_time - start_time)

        may_spawn = (kind == SIG or (kind == DARK and cfg.ap_from_dark) or
                 (kind == NOISE and cfg.ap_from_noise) or
                     (kind == AP and cfg.recursion))
        if may_spawn and cfg.p_afterpulse > 0 and r_ap.random() < cfg.p_afterpulse:
            delay = r_ap.exponential(cfg.tau_ap)
            heapq.heappush(events, (event_time + delay, serial, AP))
            serial += 1

    state.pending_events = events
    state.event_serial = serial
    state.next_run += n_runs
    state.accepted_total += sum(len(component) for component in accepted)

    hist = np.zeros((len(KIND_NAMES), N_BINS), dtype=np.int64)
    for kind, event_times in enumerate(accepted):
        times = np.asarray(event_times, dtype=float)
        if cfg.jitter:
            times += r_jit.normal(0.0, SIGMA_TOTAL, times.size)
        idx = np.floor(times / BIN_WIDTH).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < N_BINS * n_runs)]
        hist[kind] = np.bincount(idx % N_BINS, minlength=N_BINS)
    return hist, rejected


def info_text(nr, h3, cfg):
    n = h3.sum(axis=1)
    total = h3.sum(axis=0)
    nz = total[total > 0]
    f_ap = n[AP] / n.sum() if n.sum() else 0.0
    rel = 1 / np.sqrt(n.sum() / N_BINS) if n.sum() else np.nan
    return (f"Runs:                  {nr:>12,}\n"
            f"Total detections:      {int(n.sum()):>12,}\n"
            f"Signal / noise: {n[SIG]:>9,} / {n[NOISE]:>9,}\n"
            f"Dark / afterpulse: {n[DARK]:>7,} / {n[AP]:>7,}\n"
            f"Afterpulse fraction:   {f_ap:>11.2%}\n"
            f"Signal rate:           {SIGNAL_RATE_HZ / 1e6:.3g} MHz ({SIGNAL_RATE_MODE.lower()})\n"
            f"Signal mean/run:       {SIGNAL_MEAN_PER_RUN:.4g}\n"
            f"Rates MHz (S/N):       {SIGNAL_RATE_HZ / 1e6:.3g} / {cfg.noise_rate / 1e6:.3g}\n"
            f"Dark count rate:       {cfg.dark_rate:.3g} Hz\n"
            f"Afterpulse chance:     {cfg.p_afterpulse:.2%}\n"
            f"Mean relative uncertainty: {rel:.3f}\n"
            f"Peak / smallest occupied bin: {int(total.max()):,} / {int(nz.min()) if nz.size else 0:,}")


def deadtime_corrected_expected(n_runs, cfg, probs):
    expected = n_runs * probs.sum(axis=0)
    if not cfg.deadtime:
        return expected
    rate_per_bin = probs.sum(axis=0) / BIN_WIDTH
    return expected / (1.0 + rate_per_bin * cfg.dead_time)


class HistogramView:
    def __init__(self, ax, cfg, probs, show_components=None):
        self.ax, self.cfg, self.probs = ax, cfg, probs
        self.show_components = tuple(show_components or (
            SHOW_SIGNAL_COMPONENT, SHOW_DARK_COMPONENT, SHOW_AP_COMPONENT, SHOW_NOISE_COMPONENT))
        self.centers = (np.arange(N_BINS) + 0.5) * BIN_WIDTH * 1e9
        self.edges = np.arange(N_BINS + 1) * BIN_WIDTH * 1e9
        z = np.zeros(N_BINS)
        self.total = ax.stairs(z, self.edges, fill=True, color="tab:blue", alpha=0.35,
                               label="Measured total (25 ps bins)")
        self.err = LineCollection([], colors="navy", linewidths=0.5,
                                  label=r"$\pm\sqrt{N_i}$ (measured)")
        ax.add_collection(self.err)
        self.theory, = ax.plot(self.centers, z, color="black", lw=1.2, label="Expected total")
        self.deadtime_theory, = ax.plot(
            self.centers, z, color="tab:purple", ls="--", lw=1.6,
            label="Expected total with deadtime")
        self.band = None
        self.comp, self.comp_th = {}, {}
        for kind, color in enumerate(("tab:green", "tab:orange", "tab:red", "tab:purple")):
            if self.show_components[kind]:
                self.comp[kind] = ax.stairs(z, self.edges, color=color, lw=0.8,
                                            label=f"{KIND_NAMES[kind]} (measured)")
                self.comp_th[kind], = ax.plot(self.centers, z, color=color, ls="--", lw=1.0)
        self.txt = ax.text(1.01, 1.0, "", transform=ax.transAxes, va="top", family="monospace",
                           fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="gray"))
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlim(0, T_RUN * 1e9)
        ax.set_xlabel("Time in run (ns)")
        ax.set_ylabel("Accumulated detections per 25 ps bin")
        ax.legend(loc="upper left", fontsize=7)

    def update(self, nr, h3):
        p = self.probs
        total = h3.sum(axis=0)
        sigma = np.sqrt(total)
        expected = nr * p.sum(axis=0)
        self.deadtime_theory.set_ydata(deadtime_corrected_expected(nr, self.cfg, p))
        self.deadtime_theory.set_visible(SHOW_DEADTIME_ESTIMATE and self.cfg.deadtime)
        self.total.set_data(total, self.edges)
        self.err.set_segments([[(x, max(y - s, 0)), (x, y + s)]
                               for x, y, s in zip(self.centers, total, sigma) if y > 0])
        self.theory.set_ydata(expected)
        if SHOW_EXPECTED_BAND:
            if self.band is not None:
                self.band.remove()
            s = np.sqrt(expected)
            self.band = self.ax.fill_between(self.centers, np.maximum(expected - s, 0),
                                             expected + s, color="black", alpha=0.15, lw=0)
        for k in self.comp:
            self.comp[k].set_data(h3[k], self.edges)
            self.comp_th[k].set_ydata(nr * p[k])
        top = max(2.0, 1.5 * max(expected.max(), (total + sigma).max()))
        self.ax.set_ylim(0, top)
        self.ax.set_title(f"Histogram after {nr:,} runs ({nr * T_RUN:.1e} seconds)")
        self.txt.set_text(info_text(nr, h3, self.cfg))


def default_detector():
    return Detector(DARK_COUNT_RATE, NOISE_RATE, AFTERPULSE_PROBABILITY, AFTERPULSE_TIME_CONSTANT, DEAD_TIME,
                    ENABLE_DARK_COUNTS, ENABLE_AFTERPULSING, ENABLE_DEADTIME, ENABLE_TIMING_JITTER,
                    ALLOW_AFTERPULSE_RECURSION, AFTERPULSE_FROM_DARK, AFTERPULSE_FROM_NOISE)


# Preserve the existing qualified references without importing another file.
sim = sys.modules[__name__]

FRAME_DELAY_MS = 25


class App:
    def __init__(self, root):
        self.root = root
        root.title("Photon-counting histogram simulator")
        self.vars = {}
        self.running = False
        self.state_ready = False
        self.pix_pos = 0.0
        self.pixel_after_id = None
        self.pix_win = None
        self.snaps, self.totals = [], []
        self.sel_bin = int(sim.T_CENTER / sim.BIN_WIDTH)
        self._build_ui()
        self.restart()
        self.root.after(FRAME_DELAY_MS, self._tick)

    # ------------------------------------------------------------------ UI
    def _var(self, name, value, kind=tk.StringVar):
        v = kind(value=value)
        self.vars[name] = v
        return v

    def _entry_row(self, parent, label, name, default, width=11):
        r = ttk.Frame(parent)
        r.pack(fill="x", pady=1)
        ttk.Label(r, text=label, width=26).pack(side="left")
        entry = ttk.Entry(r, textvariable=self._var(name, str(default)), width=width)
        entry.pack(side="right")
        return entry

    def _optional_entry_row(self, parent, label, name, default):
        """Entry row for a setting that may be None (blank = auto / override off)."""
        return self._entry_row(parent, label, name, "" if default is None else str(default))

    def _combo_row(self, parent, label, name, default, values, command=None):
        r = ttk.Frame(parent)
        r.pack(fill="x", pady=1)
        ttk.Label(r, text=label, width=26).pack(side="left")
        self._var(name, default)
        cb = ttk.Combobox(r, textvariable=self.vars[name], values=values, state="readonly", width=16)
        cb.pack(side="right")
        if command is not None:
            cb.bind("<<ComboboxSelected>>", lambda e: command())
        return cb

    def _check(self, parent, label, name, default, command=None):
        v = self._var(name, default, tk.BooleanVar)
        ttk.Checkbutton(parent, text=label, variable=v, command=command).pack(anchor="w")

    def _on_toggle_scattering_rate(self):
        """Grey out the manual signal-rate entry while the scattering model drives it."""
        use_scattering = self.vars["use_scattering_rate"].get()
        self.signal_rate_entry.config(state="disabled" if use_scattering else "normal")

    def _build_scrollable_sidebar(self):
        """Wrap the left control sidebar in a vertically scrollable canvas so a tall tab
        (e.g. Scattering) cannot push the pixel-preview/run-control sections off screen."""
        outer = ttk.Frame(self.root)
        outer.pack(side="left", fill="y")
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="y", expand=True)
        vsb.pack(side="right", fill="y")

        left = ttk.Frame(canvas, padding=6)
        window_id = canvas.create_window((0, 0), window=left, anchor="nw")

        def _sync_scroll_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"), width=left.winfo_reqwidth())
        left.bind("<Configure>", _sync_scroll_region)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        return left

    def _scrollable_tab(self, notebook, title):
        """Add a notebook tab whose content scrolls internally if it doesn't fit the
        notebook's (fixed) height, instead of growing the notebook itself."""
        page = ttk.Frame(notebook)
        notebook.add(page, text=title)
        canvas = tk.Canvas(page, highlightthickness=0)
        vsb = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        inner = ttk.Frame(canvas, padding=4)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        return inner

    def _build_ui(self):
        left = self._build_scrollable_sidebar()
        center = ttk.Frame(self.root)
        center.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(self.root, padding=6)
        right.pack(side="left", fill="y")

        nb = ttk.Notebook(left)
        nb.pack(fill="x")
        phys, scatter, det, disp = (self._scrollable_tab(nb, title) for title in
                                    ("Physics", "Scattering", "Detector", "Display"))

        # ---- physics ----
        signal_box = ttk.LabelFrame(phys, text="Signal detection", padding=4)
        signal_box.pack(fill="x")
        self._check(signal_box, "Compute signal rate from scattering model (Scattering tab)",
                    "use_scattering_rate", sim.USE_SCATTERING_RATE, self._on_toggle_scattering_rate)
        self.signal_rate_entry = self._entry_row(
            signal_box, "Signal rate [MHz]", "signal_rate_mhz", sim.SIGNAL_RATE_HZ / 1e6)
        row = ttk.Frame(signal_box)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="Rate interpretation", width=26).pack(side="left")
        self._var("signal_rate_mode", sim.SIGNAL_RATE_MODE)
        ttk.Combobox(row, textvariable=self.vars["signal_rate_mode"],
                 values=("Peak", "Average"), state="readonly", width=18).pack(side="right")
        noise_box = ttk.LabelFrame(phys, text="Noise background", padding=4)
        noise_box.pack(fill="x", pady=(4, 0))
        self._entry_row(noise_box, "Noise rate [MHz]", "noise_rate_mhz", sim.NOISE_RATE / 1e6)
        laser = ttk.LabelFrame(phys, text="Laser pulse", padding=4)
        laser.pack(fill="x", pady=(4, 0))
        self._var("pulse_shape", sim.PULSE_SHAPE)
        row = ttk.Frame(laser)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="Pulse shape", width=26).pack(side="left")
        pulse_menu = ttk.Combobox(row, textvariable=self.vars["pulse_shape"],
                      values=("Logistic", "Gaussian"), state="readonly", width=18)
        pulse_menu.pack(side="right")
        self._entry_row(laser, "Steepness a [1/ns]", "a", sim.A / 1e9)
        self._entry_row(laser, "Transition centre [ns]", "tc", sim.T_CENTER * 1e9)
        self._entry_row(laser, "Gaussian sigma [ns]", "gaussian_sigma", sim.GAUSSIAN_SIGMA * 1e9)

        general = ttk.LabelFrame(phys, text="General settings", padding=4)
        general.pack(fill="x", pady=(4, 0))
        self._entry_row(general, "Maximum runs", "max_runs", sim.MAX_RUNS)
        self._entry_row(general, "Random seed", "seed", sim.RANDOM_SEED)
        self._entry_row(general, "Window size [ns]", "window_ns", sim.T_RUN * 1e9)
        self._entry_row(general, "Bin size [ps]", "bin_ps", sim.BIN_WIDTH * 1e12)

        # ---- scattering (Mie/Rayleigh scattering-to-lens model, from scattering_to_lens_v2.py) ----
        spde_box = ttk.LabelFrame(scatter, text="Detection efficiency", padding=4)
        spde_box.pack(fill="x")
        self._entry_row(spde_box, "SPDE (0-1)", "spde", sim.SPDE)

        laser_box = ttk.LabelFrame(scatter, text="Laser / beam", padding=4)
        laser_box.pack(fill="x", pady=(4, 0))
        self._entry_row(laser_box, "Wavelength [nm]", "sc_wavelength_nm", sim.DEFAULT_SCATTER_WAVELENGTH_NM)
        self._entry_row(laser_box, "Power [mW]", "sc_power_mw", sim.DEFAULT_SCATTER_POWER_W * 1e3)
        self._combo_row(laser_box, "Power mode", "sc_power_mode", sim.DEFAULT_SCATTER_POWER_MODE.capitalize(),
                        ("Peak", "Average"))
        self._entry_row(laser_box, "Pulse length [ns] (peak)", "sc_pulse_length_ns",
                        sim.DEFAULT_SCATTER_PULSE_LENGTH_S * 1e9)
        self._entry_row(laser_box, "Rep rate [MHz] (average)", "sc_rep_rate_mhz",
                        sim.DEFAULT_SCATTER_REP_RATE_HZ / 1e6)
        self._entry_row(laser_box, "Laser distance [m]", "sc_laser_distance_m", sim.DEFAULT_SCATTER_LASER_DISTANCE_M)

        lens_box = ttk.LabelFrame(scatter, text="Lens / geometry", padding=4)
        lens_box.pack(fill="x", pady=(4, 0))
        self._entry_row(lens_box, "Lens diameter [mm]", "sc_lens_diameter_mm", sim.DEFAULT_SCATTER_LENS_DIAMETER_M * 1e3)
        self._entry_row(lens_box, "Distance to beam [m]", "sc_distance_m", sim.DEFAULT_SCATTER_DISTANCE_M)
        self._optional_entry_row(lens_box, "Solid angle override [sr]", "sc_solid_angle_sr",
                                 sim.DEFAULT_SCATTER_SOLID_ANGLE_SR)

        fov_box = ttk.LabelFrame(scatter, text="Field of view", padding=4)
        fov_box.pack(fill="x", pady=(4, 0))
        self._entry_row(fov_box, "FOV [deg]", "sc_fov_deg", sim.DEFAULT_SCATTER_FOV_DEG)
        self._entry_row(fov_box, "Horizontal resolution", "sc_horizontal_resolution",
                        sim.DEFAULT_SCATTER_HORIZONTAL_RESOLUTION)
        self._optional_entry_row(fov_box, "Viewed beam length override [mm]", "sc_viewed_beam_length_mm",
                                 sim.DEFAULT_SCATTER_VIEWED_BEAM_LENGTH_M)

        medium_box = ttk.LabelFrame(scatter, text="Medium & polarization", padding=4)
        medium_box.pack(fill="x", pady=(4, 0))
        self._combo_row(medium_box, "Medium", "sc_medium", sim.DEFAULT_SCATTER_MEDIUM,
                        ("air", "fog_mie", "fog_hg"))
        self._combo_row(medium_box, "Polarization", "sc_polarization", sim.DEFAULT_SCATTER_POLARIZATION,
                        ("perpendicular", "in_plane", "unpolarized"))

        air_box = ttk.LabelFrame(scatter, text="Air (medium = air)", padding=4)
        air_box.pack(fill="x", pady=(4, 0))
        self._entry_row(air_box, "Temperature [K]", "sc_temperature_k", sim.DEFAULT_SCATTER_TEMPERATURE_K)
        self._entry_row(air_box, "Pressure [Pa]", "sc_pressure_pa", sim.DEFAULT_SCATTER_PRESSURE_PA)

        fog_box = ttk.LabelFrame(scatter, text="Fog (both fog models)", padding=4)
        fog_box.pack(fill="x", pady=(4, 0))
        self._entry_row(fog_box, "Visibility [m]", "sc_fog_visibility_m", sim.DEFAULT_SCATTER_FOG_VISIBILITY_M)
        self._entry_row(fog_box, "Henyey-Greenstein g (fog_hg)", "sc_fog_g", sim.DEFAULT_SCATTER_FOG_G)

        mie_box = ttk.LabelFrame(scatter, text="Mie droplets (medium = fog_mie)", padding=4)
        mie_box.pack(fill="x", pady=(4, 0))
        self._entry_row(mie_box, "Mode radius [um]", "sc_fog_mode_radius_um", sim.DEFAULT_SCATTER_FOG_MODE_RADIUS_UM)
        self._entry_row(mie_box, "Alpha (distribution shape)", "sc_fog_alpha", sim.DEFAULT_SCATTER_FOG_ALPHA)
        self._entry_row(mie_box, "Water refractive index n", "sc_water_n", sim.DEFAULT_SCATTER_WATER_N)
        self._entry_row(mie_box, "Water refractive index k (absorption)", "sc_water_k", sim.DEFAULT_SCATTER_WATER_K)
        self._entry_row(mie_box, "Mie radii samples", "sc_mie_n_radii", sim.DEFAULT_SCATTER_MIE_N_RADII)
        self._entry_row(mie_box, "Mie angle samples", "sc_mie_n_angles", sim.DEFAULT_SCATTER_MIE_N_ANGLES)

        result_box = ttk.LabelFrame(scatter, text="Last computed result", padding=4)
        result_box.pack(fill="x", pady=(4, 0))
        self.scatter_info = ttk.Label(result_box, text="(enable the checkbox on the Physics tab and "
                                      "Apply & Restart to compute)", font=("Consolas", 8), justify="left",
                                      wraplength=240)
        self.scatter_info.pack(anchor="w")

        self._on_toggle_scattering_rate()

        # ---- detector ----
        self._check(det, "Dark counts", "dark", sim.ENABLE_DARK_COUNTS)
        dark_box = ttk.LabelFrame(det, text="Dark-count intensity", padding=4)
        dark_box.pack(fill="x")
        self._entry_row(dark_box, "Dark-count rate [Hz]", "dark_rate_hz", sim.DARK_COUNT_RATE)

        ap_box = ttk.LabelFrame(det, text="Afterpulsing", padding=4)
        ap_box.pack(fill="x", pady=(4, 0))
        self._check(ap_box, "Enable afterpulsing", "ap", sim.ENABLE_AFTERPULSING)
        self._entry_row(ap_box, "Chance per pulse [%]", "afterpulse_percent",
                sim.AFTERPULSE_PROBABILITY * 100)
        self._entry_row(ap_box, "Delay tau [ns]", "tau", sim.AFTERPULSE_TIME_CONSTANT * 1e9)
        self._check(ap_box, "Dark counts can trigger afterpulses", "ap_dark", sim.AFTERPULSE_FROM_DARK)
        self._check(ap_box, "Noise can trigger afterpulses", "ap_noise", sim.AFTERPULSE_FROM_NOISE)
        self._check(ap_box, "Afterpulse recursion", "recursion", sim.ALLOW_AFTERPULSE_RECURSION)

        timing = ttk.LabelFrame(det, text="Timing", padding=4)
        timing.pack(fill="x", pady=(4, 0))
        self._entry_row(timing, "Timing FWHM [ps]", "fwhm", sim.TIMING_FWHM * 1e12)
        self._check(timing, "Timing jitter", "jitter", sim.ENABLE_TIMING_JITTER)
        self._check(timing, "HPTDC quantization sigma", "hptdc", sim.ENABLE_HPTDC_QUANTIZATION)
        self._check(timing, "Deadtime", "deadtime", sim.ENABLE_DEADTIME)
        self._entry_row(timing, "Deadtime [ns]", "dead", sim.DEAD_TIME * 1e9)

        # ---- display (live) ----
        self._check(disp, "Show signal", "show_signal", sim.SHOW_SIGNAL_COMPONENT, self.refresh_view)
        self._check(disp, "Show noise", "show_noise", sim.SHOW_NOISE_COMPONENT, self.refresh_view)
        self._check(disp, "Show dark counts", "show_dark", sim.SHOW_DARK_COMPONENT, self.refresh_view)
        self._check(disp, "Show afterpulses", "show_afterpulse", sim.SHOW_AP_COMPONENT, self.refresh_view)
        self._check(disp, "Show error bars", "errbars", False, self.redraw)
        self._check(disp, "Show expected +-sqrt(N) band", "band", sim.SHOW_EXPECTED_BAND, self.refresh_view)
        self._check(disp, "Show expected deadtime estimate", "deadtime_estimate",
                sim.SHOW_DEADTIME_ESTIMATE, self.redraw)
        self._check(disp, "Show convergence panel", "conv", False, self.refresh_view)
        r = ttk.Frame(disp)
        r.pack(fill="x", pady=3)
        ttk.Label(r, text="Y scale").pack(side="left")
        self._var("yscale", "linear")
        cb = ttk.Combobox(r, textvariable=self.vars["yscale"], values=("symlog", "linear"),
                          width=8, state="readonly")
        cb.pack(side="right")
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_view())
        r = ttk.Frame(disp)
        r.pack(fill="x", pady=3)
        ttk.Label(r, text="Convergence bin [ns]").pack(side="left")
        ttk.Entry(r, textvariable=self._var("bin_ns", f"{self.sel_bin * sim.BIN_WIDTH * 1e9:.3f}"),
                  width=9).pack(side="right")
        ttk.Button(disp, text="Set bin", command=self._set_bin_from_entry).pack(anchor="e")

        # Fix the notebook's height to the typical (non-Scattering) tab size: without this,
        # ttk.Notebook sizes itself to the tallest tab, leaving a big gap below short tabs.
        # The Scattering tab's own scrollbar (added in _scrollable_tab) handles its overflow.
        self.root.update_idletasks()
        short_tab_heights = [frame.winfo_reqheight() for frame in (phys, det, disp)]
        nb.configure(height=max(short_tab_heights, default=400))

        # Shared pixel settings sit outside the tab pages so they remain accessible.
        pv = ttk.LabelFrame(left, text="Pixel view settings", padding=6)
        pv.pack(fill="x", pady=8)
        self._entry_row(pv, "Render FPS", "pix_fps", 30)
        self._entry_row(pv, "Playback [bins/s]", "pix_bps", 200)
        r = ttk.Frame(pv)
        r.pack(fill="x", pady=2)
        ttk.Label(r, text="Intensity source", width=26).pack(side="left")
        self._var("pix_src", "Measured histogram")
        ttk.Combobox(r, textvariable=self.vars["pix_src"], width=18, state="readonly",
                     values=("Measured histogram", "Expected (theory)")).pack(side="right")
        self.pixel_open_button = ttk.Button(pv, text="Show pixel window",
                            command=self._show_pixel_view, state="disabled")
        self.pixel_open_button.pack(fill="x", pady=2)

        # ---- run controls ----
        ctl = ttk.LabelFrame(left, text="Run control", padding=6)
        ctl.pack(fill="x", pady=8)
        ttk.Button(ctl, text="Apply & Restart", command=self.restart).pack(fill="x", pady=3)
        ttk.Button(ctl, text="Reset parameters to defaults", command=self.reset_defaults).pack(fill="x")
        ttk.Label(ctl, text="Speed (growth of N_r per frame)").pack(anchor="w", pady=(6, 0))
        self.speed = tk.DoubleVar(value=1.6)
        ttk.Scale(ctl, from_=1.01, to=1.6, variable=self.speed).pack(fill="x")
        self.progress = ttk.Progressbar(ctl, maximum=1.0)
        self.progress.pack(fill="x", pady=4)
        row = ttk.Frame(ctl)
        row.pack(fill="x")
        ttk.Button(row, text="Save PNG", command=self.save_png).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Export CSV", command=self.export_csv).pack(side="left", expand=True, fill="x")

        # ---- figure ----
        self.fig = Figure(figsize=(10, 7), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, center)
        self.toolbar.update()
        self.canvas.mpl_connect("button_press_event", self._on_click)

        # ---- info panel ----
        info_box = ttk.LabelFrame(right, text="Run summary", padding=6)
        info_box.pack(fill="y")
        self.info = ttk.Label(info_box, text="", font=("Consolas", 9), justify="left")
        self.info.pack(anchor="nw")
        self.status = ttk.Label(right, text="", wraplength=280, foreground="gray")
        self.status.pack(anchor="w", pady=4)

    # ------------------------------------------------------- parameters
    def _num(self, name, lo=None, hi=None, integer=False):
        txt = self.vars[name].get().strip()
        try:
            x = int(float(txt)) if integer else float(txt)
        except ValueError:
            raise ValueError(f"'{txt}' is not a valid number for {name}")
        if not np.isfinite(x):
            raise ValueError(f"{name} must be finite")
        if (lo is not None and x < lo) or (hi is not None and x > hi):
            raise ValueError(f"{name} must be in [{lo}, {hi}]")
        return x

    def _optional_num(self, name, lo=None, hi=None):
        """Like _num, but a blank entry means 'no override' (None)."""
        txt = self.vars[name].get().strip()
        if txt == "":
            return None
        try:
            x = float(txt)
        except ValueError:
            raise ValueError(f"'{txt}' is not a valid number for {name}")
        if not np.isfinite(x):
            raise ValueError(f"{name} must be finite")
        if (lo is not None and x < lo) or (hi is not None and x > hi):
            raise ValueError(f"{name} must be in [{lo}, {hi}]")
        return x

    def _compute_scattering_rate(self):
        """Apply the Scattering-tab settings to scattering_to_lens_v2.py's model and
        return the detected signal rate [Hz] = (photons/s into the lens) x SPDE."""
        spde = self._num("spde", 0.0, 1.0)
        power_mode = self.vars["sc_power_mode"].get().lower()
        if power_mode not in ("peak", "average"):
            raise ValueError("Power mode must be Peak or Average")
        medium = self.vars["sc_medium"].get()
        polarization = self.vars["sc_polarization"].get()
        if medium not in scat.VALID_MEDIA:
            raise ValueError(f"Medium must be one of {scat.VALID_MEDIA}")
        if polarization not in scat.VALID_POLARIZATIONS:
            raise ValueError(f"Polarization must be one of {scat.VALID_POLARIZATIONS}")

        scat.WAVELENGTH_NM = self._num("sc_wavelength_nm", 1.0)
        scat.POWER_W = self._num("sc_power_mw", 0.0) * 1e-3
        scat.POWER_MODE = power_mode
        scat.PULSE_LENGTH_S = self._num("sc_pulse_length_ns", 1e-6) * 1e-9
        scat.REP_RATE_HZ = self._num("sc_rep_rate_mhz", 1e-6) * 1e6
        scat.LENS_DIAMETER_M = self._num("sc_lens_diameter_mm", 1e-6) * 1e-3
        scat.DISTANCE_M = self._num("sc_distance_m", 1e-6)
        scat.SOLID_ANGLE_SR = self._optional_num("sc_solid_angle_sr", 0.0)
        scat.LASER_DISTANCE_M = self._num("sc_laser_distance_m", 0.0)
        scat.FOV_DEG = self._num("sc_fov_deg", 1e-6, 360.0)
        scat.HORIZONTAL_RESOLUTION = self._num("sc_horizontal_resolution", 1, None, integer=True)
        viewed_len_mm = self._optional_num("sc_viewed_beam_length_mm", 0.0)
        scat.VIEWED_BEAM_LENGTH_M = None if viewed_len_mm is None else viewed_len_mm * 1e-3
        scat.POLARIZATION = polarization
        scat.MEDIUM = medium
        scat.TEMPERATURE_K = self._num("sc_temperature_k", 1.0)
        scat.PRESSURE_PA = self._num("sc_pressure_pa", 0.0)
        scat.FOG_VISIBILITY_M = self._num("sc_fog_visibility_m", 1e-6)
        scat.FOG_G = self._num("sc_fog_g", -0.999, 0.999)
        scat.FOG_MODE_RADIUS_UM = self._num("sc_fog_mode_radius_um", 1e-6)
        scat.FOG_ALPHA = self._num("sc_fog_alpha", 1e-6)
        water_n = self._num("sc_water_n", 1.0)
        water_k = self._num("sc_water_k", 0.0)
        scat.WATER_REFRACTIVE_INDEX = complex(water_n, -water_k)
        scat.MIE_N_RADII = self._num("sc_mie_n_radii", 2, 100000, integer=True)
        scat.MIE_N_ANGLES = self._num("sc_mie_n_angles", 3, 1_000_000, integer=True)

        try:
            scat.validate_settings()
        except ValueError as e:
            raise ValueError(f"Scattering settings: {e}")

        lam = scat.WAVELENGTH_NM * 1e-9
        n_dot = scat.photon_rate(scat.POWER_W, lam)
        try:
            mie = scat.MieFog() if scat.MEDIUM == "fog_mie" else None
        except ImportError as e:
            raise ValueError(str(e))
        res = scat.fraction_into_lens(mie)
        frac = res["fraction"]
        photons_into_lens_per_s = n_dot * frac
        detected_rate_hz = photons_into_lens_per_s * spde

        sim.SPDE = spde
        info_lines = [
            f"Medium: {scat.MEDIUM}   polarization: {scat.POLARIZATION}",
            f"beta_ext: {res['beta']:.3e} 1/m   tau: {res['tau_path']:.3g}",
            f"Beam photons: {n_dot:.3e} /s",
            f"Fraction into lens: {frac:.3e}",
            f"Photons into lens (pre-SPDE): {photons_into_lens_per_s:.3e} /s",
            f"SPDE: {spde:.3g}",
            f"Detected signal rate: {detected_rate_hz / 1e6:.4g} MHz",
        ]
        if res["tau_path"] > 0.3:
            info_lines.append("WARNING: tau > 0.3, multiple scattering not negligible.")
        self.scatter_info.config(text="\n".join(info_lines))
        return detected_rate_hz

    def reset_defaults(self):
        defaults = {
            "signal_rate_mhz": sim.DEFAULT_SIGNAL_RATE_HZ / 1e6,
            "signal_rate_mode": sim.DEFAULT_SIGNAL_RATE_MODE,
            "noise_rate_mhz": sim.DEFAULT_NOISE_RATE / 1e6,
            "a": sim.DEFAULT_A / 1e9,
            "tc": sim.DEFAULT_T_CENTER * 1e9,
            "max_runs": sim.MAX_RUNS,
            "seed": sim.RANDOM_SEED,
            "fwhm": sim.DEFAULT_TIMING_FWHM * 1e12,
            "dark_rate_hz": sim.DEFAULT_DARK_COUNT_RATE,
            "afterpulse_percent": sim.DEFAULT_AFTERPULSE_PROBABILITY * 100,
            "tau": sim.DEFAULT_AFTERPULSE_TIME_CONSTANT * 1e9,
            "dead": sim.DEAD_TIME * 1e9,
            "gaussian_sigma": sim.DEFAULT_GAUSSIAN_SIGMA * 1e9,
            "window_ns": sim.DEFAULT_T_RUN * 1e9,
            "bin_ps": sim.DEFAULT_BIN_WIDTH * 1e12,
            "pix_fps": 30,
            "pix_bps": 200,
            "spde": sim.DEFAULT_SPDE,
            "sc_wavelength_nm": sim.DEFAULT_SCATTER_WAVELENGTH_NM,
            "sc_power_mw": sim.DEFAULT_SCATTER_POWER_W * 1e3,
            "sc_power_mode": sim.DEFAULT_SCATTER_POWER_MODE.capitalize(),
            "sc_pulse_length_ns": sim.DEFAULT_SCATTER_PULSE_LENGTH_S * 1e9,
            "sc_rep_rate_mhz": sim.DEFAULT_SCATTER_REP_RATE_HZ / 1e6,
            "sc_laser_distance_m": sim.DEFAULT_SCATTER_LASER_DISTANCE_M,
            "sc_lens_diameter_mm": sim.DEFAULT_SCATTER_LENS_DIAMETER_M * 1e3,
            "sc_distance_m": sim.DEFAULT_SCATTER_DISTANCE_M,
            "sc_solid_angle_sr": "" if sim.DEFAULT_SCATTER_SOLID_ANGLE_SR is None else sim.DEFAULT_SCATTER_SOLID_ANGLE_SR,
            "sc_fov_deg": sim.DEFAULT_SCATTER_FOV_DEG,
            "sc_horizontal_resolution": sim.DEFAULT_SCATTER_HORIZONTAL_RESOLUTION,
            "sc_viewed_beam_length_mm": ("" if sim.DEFAULT_SCATTER_VIEWED_BEAM_LENGTH_M is None
                                        else sim.DEFAULT_SCATTER_VIEWED_BEAM_LENGTH_M * 1e3),
            "sc_medium": sim.DEFAULT_SCATTER_MEDIUM,
            "sc_polarization": sim.DEFAULT_SCATTER_POLARIZATION,
            "sc_temperature_k": sim.DEFAULT_SCATTER_TEMPERATURE_K,
            "sc_pressure_pa": sim.DEFAULT_SCATTER_PRESSURE_PA,
            "sc_fog_visibility_m": sim.DEFAULT_SCATTER_FOG_VISIBILITY_M,
            "sc_fog_g": sim.DEFAULT_SCATTER_FOG_G,
            "sc_fog_mode_radius_um": sim.DEFAULT_SCATTER_FOG_MODE_RADIUS_UM,
            "sc_fog_alpha": sim.DEFAULT_SCATTER_FOG_ALPHA,
            "sc_water_n": sim.DEFAULT_SCATTER_WATER_N,
            "sc_water_k": sim.DEFAULT_SCATTER_WATER_K,
            "sc_mie_n_radii": sim.DEFAULT_SCATTER_MIE_N_RADII,
            "sc_mie_n_angles": sim.DEFAULT_SCATTER_MIE_N_ANGLES,
        }
        for key, value in defaults.items():
            self.vars[key].set(str(value))
        switches = {
            "jitter": sim.ENABLE_TIMING_JITTER,
            "hptdc": sim.ENABLE_HPTDC_QUANTIZATION,
            "dark": sim.ENABLE_DARK_COUNTS,
            "ap": sim.ENABLE_AFTERPULSING,
            "ap_dark": sim.AFTERPULSE_FROM_DARK,
            "ap_noise": sim.AFTERPULSE_FROM_NOISE,
            "recursion": sim.ALLOW_AFTERPULSE_RECURSION,
            "deadtime": sim.ENABLE_DEADTIME,
            "show_signal": sim.SHOW_SIGNAL_COMPONENT,
            "show_noise": sim.SHOW_NOISE_COMPONENT,
            "show_dark": sim.SHOW_DARK_COMPONENT,
            "show_afterpulse": sim.SHOW_AP_COMPONENT,
            "use_scattering_rate": sim.DEFAULT_USE_SCATTERING_RATE,
        }
        for key, value in switches.items():
            self.vars[key].set(value)
        self.vars["pulse_shape"].set("Logistic")
        self.vars["deadtime_estimate"].set(sim.SHOW_DEADTIME_ESTIMATE)
        self.vars["conv"].set(False)
        self.speed.set(1.6)
        self._on_toggle_scattering_rate()

    def _apply_parameters(self):
        signal_rate = self._num("signal_rate_mhz", 0.0) * 1e6
        signal_rate_mode = self.vars["signal_rate_mode"].get()
        if signal_rate_mode not in ("Peak", "Average"):
            raise ValueError("Signal rate interpretation must be Peak or Average")
        use_scattering_rate = self.vars["use_scattering_rate"].get()
        if use_scattering_rate:
            signal_rate = self._compute_scattering_rate()
            self.vars["signal_rate_mhz"].set(f"{signal_rate / 1e6:.6g}")
        sim.USE_SCATTERING_RATE = use_scattering_rate
        noise_rate = self._num("noise_rate_mhz", 0.0) * 1e6
        window_seconds = self._num("window_ns", 1e-6, 1e9) * 1e-9
        bin_width = self._num("bin_ps", 1e-3, 1e9) * 1e-12
        bin_count_float = window_seconds / bin_width
        bin_count = int(round(bin_count_float))
        if bin_count < 1 or not np.isclose(bin_count_float, bin_count, rtol=1e-9, atol=1e-9):
            raise ValueError("Window size must be an integer multiple of bin size")
        a = self._num("a", 1e-6, 1e3)
        tc = self._num("tc", 0.0, window_seconds * 1e9)
        gaussian_sigma = self._num("gaussian_sigma", 1e-6, 1e6) * 1e-9
        pulse_shape = self.vars["pulse_shape"].get()
        max_runs = self._num("max_runs", 1, 1e9, integer=True)
        seed = self._num("seed", 0, 2 ** 32, integer=True)
        fwhm = self._num("fwhm", 0.0, 1e4)
        dark_rate = self._num("dark_rate_hz", 0.0)
        afterpulse_probability = self._num("afterpulse_percent", 0.0, 100.0) / 100.0
        tau = self._num("tau", 1e-3, 1e6)
        dead = self._num("dead", 0.0, 1e6)
        if self.vars["recursion"].get() and afterpulse_probability >= 1.0:
            raise ValueError("Afterpulse recursion requires a chance below 100% per pulse")
        sim.SIGNAL_RATE_HZ = signal_rate
        sim.SIGNAL_RATE_MODE = signal_rate_mode
        sim.NOISE_RATE = noise_rate
        sim.DARK_COUNT_RATE = dark_rate
        sim.AFTERPULSE_PROBABILITY = afterpulse_probability
        sim.T_RUN = window_seconds
        sim.RUN_PERIOD = window_seconds
        sim.BIN_WIDTH = bin_width
        sim.N_BINS = bin_count
        sim.SIGMA_HPTDC = bin_width / np.sqrt(12)
        sim.PULSE_SHAPE = pulse_shape
        sim.GAUSSIAN_SIGMA = gaussian_sigma
        sim.A = a * 1e9
        sim.T_CENTER = tc * 1e-9
        sim.SIGNAL_MEAN_PER_RUN = sim.signal_mean_per_run()
        sim.TIMING_FWHM = fwhm * 1e-12
        sim.SIGMA_DETECTOR = sim.TIMING_FWHM / 2.355
        sim.SIGMA_TOTAL = float(np.hypot(sim.SIGMA_DETECTOR,
                                         sim.SIGMA_HPTDC if self.vars["hptdc"].get() else 0.0))
        cfg = sim.Detector(dark_rate, noise_rate, afterpulse_probability, tau * 1e-9, dead * 1e-9,
                           self.vars["dark"].get(), self.vars["ap"].get(), self.vars["deadtime"].get(),
                           self.vars["jitter"].get() and sim.SIGMA_TOTAL > 0,
                           self.vars["recursion"].get(), self.vars["ap_dark"].get(),
                           self.vars["ap_noise"].get())
        return cfg, max_runs, seed

    # ---------------------------------------------------------- simulation
    def restart(self):
        if (self.vars.get("use_scattering_rate") is not None and self.vars["use_scattering_rate"].get()
                and self.vars["sc_medium"].get() == "fog_mie"):
            self.status.config(text="Computing Mie scattering model (can take up to a minute)...")
            self.root.update_idletasks()
        try:
            cfg, max_runs, seed = self._apply_parameters()
        except ValueError as e:
            messagebox.showerror("Invalid parameter", str(e))
            self.status.config(text="")
            return
        self.cfg, self.max_runs = cfg, max_runs
        self.probs = sim.expected_bin_probabilities(cfg)
        self.p_tot = self.probs.sum(axis=0)
        self.rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
        self.sim_state = sim.SimulationState()
        cdf = np.cumsum(sim.pulse_cell_counts(sim.N_BINS * sim.SUBSAMPLE))
        self.cdf = cdf / cdf[-1] if cdf[-1] > 0 else np.linspace(0.0, 1.0, cdf.size)
        self.hist = np.zeros((len(sim.KIND_NAMES), sim.N_BINS), dtype=np.int64)
        self.rej = np.zeros(len(sim.KIND_NAMES), dtype=np.int64)
        self.done = 0
        self.snaps, self.totals = [], []
        self.sel_bin = min(max(int(self.sel_bin), 0), sim.N_BINS - 1)
        self.pix_pos = 0.0
        if self.pixel_after_id is not None:
            self.root.after_cancel(self.pixel_after_id)
            self.pixel_after_id = None
        if self.pix_win is not None and self.pix_win.winfo_exists():
            self.pix_win.withdraw()
        sim.SHOW_EXPECTED_BAND = self.vars["band"].get()
        self.state_ready = True
        self.refresh_view()
        self.set_running(True)

    def advance(self, target):
        target = min(int(target), self.max_runs)
        while self.done < target:
            n = min(target - self.done, sim.CHUNK_RUNS)
            h, r = sim.simulate_chunk(n, self.cdf, self.cfg, self.rngs, self.sim_state)
            self.hist += h
            self.rej += r
            self.done += n
        self.snaps.append(self.done)
        self.totals.append(self.hist.sum(axis=0).astype(np.int32))

    def _next_target(self):
        # Geometric frame spacing, at most one simulation chunk per frame.
        nxt = max(self.done + 1, int(self.done * self.speed.get()))
        return min(nxt, self.done + sim.CHUNK_RUNS, self.max_runs)

    def _tick(self):
        if self.running and self.state_ready:
            if self.done >= self.max_runs:
                self.set_running(False)
            else:
                self.advance(self._next_target())
                self.redraw()
                if self.done >= self.max_runs:
                    self.set_running(False)
        self.root.after(FRAME_DELAY_MS, self._tick)

    def set_running(self, flag):
        self.running = flag

    # ------------------------------------------------------------- plotting
    def refresh_view(self):
        """Rebuild axes/artists (needed when layout or artist set changes) and redraw."""
        if not self.state_ready:
            return
        sim.SHOW_EXPECTED_BAND = self.vars["band"].get()
        self.fig.clear()
        if self.vars["conv"].get():
            ax, self.conv_ax = self.fig.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1.4]})
        else:
            ax, self.conv_ax = self.fig.subplots(1, 1), None
        self.ax = ax
        components = tuple(self.vars[name].get() for name in
                   ("show_signal", "show_dark", "show_afterpulse", "show_noise"))
        self.view = sim.HistogramView(ax, self.cfg, self.probs, show_components=components)
        self.view.txt.set_visible(False)
        ax.set_yscale(self.vars["yscale"].get(), **({"linthresh": 1} if self.vars["yscale"].get() == "symlog" else {}))
        self.marker = ax.axvline(self.sel_bin * sim.BIN_WIDTH * 1e9, color="purple", lw=0.8, alpha=0.6)
        if self.conv_ax is not None:
            c = self.conv_ax
            c.set_xscale("log")
            c.set_yscale("log")
            c.set_title("Statistical convergence")
            c.set_xlabel("Number of runs $N_r$")
            c.set_ylabel(r"$\sigma_{N_b}/N_b$")
            c.grid(True, which="both", alpha=0.3)
            self.l_meas, = c.plot([], [], "o", ms=3, color="tab:blue", label=r"measured $1/\sqrt{N_b}$")
            self.l_th, = c.plot([], [], "-", color="crimson", label=r"$1/\sqrt{N_r p_b}\ \propto N_r^{-1/2}$")
            c.legend(fontsize=7, loc="lower left")
        self.redraw()

    def redraw(self):
        if not self.state_ready or not hasattr(self, "view"):
            return
        sim.SHOW_DEADTIME_ESTIMATE = self.vars["deadtime_estimate"].get()
        self.view.update(self.done, self.hist)
        self.view.err.set_visible(self.vars["errbars"].get())
        self._update_convergence()
        text = sim.info_text(self.done, self.hist, self.cfg)
        rejected_total = int(self.rej.sum())
        attempted_total = self.sim_state.accepted_total + rejected_total
        deadtime_loss = rejected_total / attempted_total if attempted_total else 0.0
        text += (f"\nDeadtime loss: {deadtime_loss:.2%}\n"
             f"Rejected by deadtime: {rejected_total:,}\n"
             f"Signal {self.rej[0]:,}  Dark {self.rej[1]:,}  AP {self.rej[2]:,}  Noise {self.rej[3]:,}")
        self.info.config(text=text)
        self.progress["value"] = np.log10(max(self.done, 1)) / max(np.log10(self.max_runs), 1e-9)
        self.status.config(text="Finished." if self.done >= self.max_runs else "")
        self.pixel_open_button.config(state="normal" if self.done >= self.max_runs else "disabled")
        self.canvas.draw_idle()

    def _update_convergence(self):
        if self.conv_ax is None or not self.snaps:
            return
        b = self.sel_bin
        runs = np.array(self.snaps, dtype=float)
        counts = np.array([t[b] for t in self.totals], dtype=float)
        ok = counts > 0
        measured = 1 / np.sqrt(counts[ok])
        p = self.p_tot[b]
        theoretical = 1 / np.sqrt(runs * p) if p > 0 else np.full_like(runs, np.nan)
        self.l_meas.set_data(runs[ok], measured)
        if np.any(ok):
            self.l_th.set_data(runs[ok], theoretical[ok])
        else:
            self.l_th.set_data(runs, theoretical)
        self.conv_ax.set_autoscalex_on(True)
        self.conv_ax.set_autoscaley_on(True)
        self.conv_ax.relim(visible_only=True)
        self.conv_ax.margins(x=0.08, y=0.12)
        self.conv_ax.autoscale_view(scalex=True, scaley=True)
        self.conv_ax.set_title(f"Statistical convergence: bin {b} "
                       f"(t = {(b + 0.5) * sim.BIN_WIDTH * 1e9:.2f} ns)", fontsize=8)

    # ---------------------------------------------------------- bin choice
    def _select_bin(self, ns):
        self.sel_bin = int(np.clip(ns * 1e-9 / sim.BIN_WIDTH, 0, sim.N_BINS - 1))
        self.vars["bin_ns"].set(f"{(self.sel_bin + 0.5) * sim.BIN_WIDTH * 1e9:.3f}")
        if hasattr(self, "marker"):
            self.marker.set_xdata([(self.sel_bin + 0.5) * sim.BIN_WIDTH * 1e9] * 2)
        self.redraw()

    def _set_bin_from_entry(self):
        try:
            self._select_bin(float(self.vars["bin_ns"].get()))
        except ValueError:
            messagebox.showerror("Invalid value", "Enter the bin time in ns.")

    def _on_click(self, event):
        if event.inaxes is self.ax and event.xdata is not None and not self.toolbar.mode:
            self._select_bin(event.xdata)

    # -------------------------------------------------------------- export
    def _show_pixel_view(self):
        if self.pix_win is None or not self.pix_win.winfo_exists():
            self._create_pixel_window()
        else:
            self.pix_win.deiconify()
            self.pix_win.lift()
        if self.pixel_after_id is None:
            self.pix_last = time.perf_counter()
            self.pixel_after_id = self.root.after(50, self._pixel_tick)

    def _create_pixel_window(self):
        self.pix_win = tk.Toplevel(self.root)
        self.pix_win.title("Pixel intensity playback")
        self.pix_win.protocol("WM_DELETE_WINDOW", self._close_pixel_window)
        self.pixel_canvas = tk.Canvas(self.pix_win, width=380, height=380,
                                      bg="#202020", highlightthickness=0)
        self.pixel_canvas.pack(padx=10, pady=10)
        self.pix_rect = self.pixel_canvas.create_rectangle(20, 20, 360, 360,
                                                           fill="#000000", outline="")
        self.pix_label = ttk.Label(self.pix_win, text="", font=("Consolas", 10), justify="left")
        self.pix_label.pack(padx=10, pady=(0, 6), anchor="w")
        self.pix_bar = ttk.Progressbar(self.pix_win, maximum=1.0)
        self.pix_bar.pack(fill="x", padx=10, pady=(0, 10))

    def _close_pixel_window(self):
        if self.pixel_after_id is not None:
            self.root.after_cancel(self.pixel_after_id)
            self.pixel_after_id = None
        if self.pix_win is not None:
            self.pix_win.destroy()
            self.pix_win = None

    def _pixel_tick(self):
        self.pixel_after_id = None
        if self.done < self.max_runs or self.pix_win is None or not self.pix_win.winfo_exists():
            return
        if not self.pix_win.winfo_viewable():
            self.pixel_after_id = self.root.after(100, self._pixel_tick)
            return
        now = time.perf_counter()
        dt, self.pix_last = now - self.pix_last, now
        try:
            fps = min(max(float(self.vars["pix_fps"].get()), 1.0), 240.0)
            bps = min(max(float(self.vars["pix_bps"].get()), 0.0), 1e7)
        except ValueError:
            fps, bps = 30.0, 200.0
        n = sim.N_BINS
        data = (self.p_tot if self.vars["pix_src"].get().startswith("Expected")
            else self.hist.sum(axis=0).astype(float))
        new = self.pix_pos + bps * dt
        span = int(new) - int(self.pix_pos)
        idx = np.arange(n) if span >= n else np.arange(int(self.pix_pos), int(new) + 1) % n
        peak = data.max()
        level = float(data[idx].mean() / peak) if peak > 0 else 0.0
        self.pix_pos = new % n
        rgb = tuple(int(round(c * min(max(level, 0.0), 1.0))) for c in (255, 255, 255))
        self.pixel_canvas.itemconfig(self.pix_rect, fill="#%02x%02x%02x" % rgb)
        t_ns = self.pix_pos * sim.BIN_WIDTH * 1e9
        self.pix_label.config(text=f"t = {t_ns:6.2f} ns   intensity = {level:5.2f}\n"
                       f"runs = {self.done:,}   {bps:g} bins/s @ {fps:g} fps")
        self.pix_bar["value"] = self.pix_pos / n
        self.pixel_after_id = self.root.after(int(1000 / fps), self._pixel_tick)

    # -------------------------------------------------------------- export
    def save_png(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=150)

    def export_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            t = (np.arange(sim.N_BINS) + 0.5) * sim.BIN_WIDTH * 1e9
            data = np.column_stack([t, self.hist.T, self.hist.sum(axis=0)])
            np.savetxt(path, data, delimiter=",", header="t_ns,signal,dark,afterpulse,noise,total",
                       comments="", fmt=["%.4f", "%d", "%d", "%d", "%d", "%d"])


def main():
    root = tk.Tk()
    App(root)
    root.geometry("1500x820")
    root.mainloop()


if __name__ == "__main__":
    main()
