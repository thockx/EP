"""
scattering_to_lens_v2.py

How many photons from a laser beam are scattered sideways by the air (or fog)
and end up in my lens?

Changes compared to v1
----------------------
  * MEDIUM = "fog_mie": full Mie scattering by a droplet size distribution
    (needs `pip install miepython`). Includes polarization (S1 / S2).
  * MEDIUM = "fog_hg": the old Henyey-Greenstein model, kept for comparison.
  * Attenuation exp(-beta * (path laser->scatterer + path scatterer->lens)).
  * Settings are validated (typos give a clear error).
  * Trapezoidal integration instead of a plain sum.
  * Pulse photons can be computed from peak power OR average power + rep rate.
  * The printout respects the SOLID_ANGLE_SR override and uses 1 - exp(-beta).
  * A comparison Mie vs Henyey-Greenstein is printed for fog.

Geometry (seen from above):

                       lens  (diameter D)
                        ||
                        ||   <- distance r (perpendicular to the beam)
                        ||
   laser ---- L0 ----[segment]===========>   beam travels along z
                    z = 0 is the point on the beam closest to the lens

For one piece of beam dz at position z:

    photons into lens = N_dot                       photons/s in the beam
                      x exp(-beta_ext * (s + d))    survive the way in and out
                      x b(theta) * dz               b = scattering per metre per
                                                    steradian [1/(m sr)]
                      x Omega                       solid angle of the lens

where b(theta) = beta_sca * p(theta), with p the phase function [1/sr].

Single-scattering approximation: a photon scatters at most once. Fine in air,
and acceptable in fog as long as the optical depth tau = beta*(path) is well
below 1. The script prints tau so you can see where you are.

Only change the SETTINGS block, then run:
    python scattering_to_lens_v2.py
"""

import numpy as np

# =============================================================================
# SETTINGS - change these
# =============================================================================

WAVELENGTH_NM   = 680.0
POWER_W         = 2.2e-3     # see POWER_MODE
POWER_MODE      = "peak"     # "peak"    : POWER_W is the power during a pulse
                             # "average" : POWER_W is the average power; needs REP_RATE_HZ
PULSE_LENGTH_S  = 5e-9       # only used when POWER_MODE = "peak"
REP_RATE_HZ     = 1e6        # only used when POWER_MODE = "average"

# --- lens / detector ---------------------------------------------------------
LENS_DIAMETER_M = 0.008      # Thorlabs F240FC: f = 8 mm, NA = 0.5 -> D = 8 mm
DISTANCE_M      = 0.5        # perpendicular distance from the beam to the lens [m]
SOLID_ANGLE_SR  = None       # number = override, None = compute from diameter/distance

# --- where does the laser sit? ---------------------------------------------------
LASER_DISTANCE_M = 0.5       # distance along the beam from the laser to the point
                             # closest to the lens [m]. Only matters for attenuation.

# --- which part of the beam does the lens see? --------------------------------------
FOV_DEG               = 100.0
HORIZONTAL_RESOLUTION = 480
VIEWED_BEAM_LENGTH_M  = None  # number [m] = override, None = use FOV / resolution

# --- polarization -----------------------------------------------------------
#   "perpendicular" : E-field perpendicular to the plane (beam, lens)
#   "in_plane"      : E-field in that plane
#   "unpolarized"   : average of the two
POLARIZATION = "perpendicular"

# --- medium -----------------------------------------------------------------
#   "air"     : Rayleigh scattering by molecules
#   "fog_mie" : Mie scattering by a droplet size distribution (needs miepython)
#   "fog_hg"  : Henyey-Greenstein approximation with asymmetry parameter FOG_G
MEDIUM = "fog_mie"

# air (MEDIUM = "air")
TEMPERATURE_K = 293.15
PRESSURE_PA   = 101325.0

# fog (both fog models)
FOG_VISIBILITY_M = 10.0      # meteorological visibility V; beta_ext = 3.912 / V

# fog_hg only
FOG_G = 0.8

# fog_mie only: modified-gamma droplet size distribution
#       n(r)  ~  r^alpha * exp(-alpha * r / r_mode)
# r_mode is the most common radius. The total droplet number is NOT set here;
# it is scaled so that the extinction matches FOG_VISIBILITY_M.
#   alpha = 6, r_mode = 4 um   ~ typical radiation fog / haze (Deirmendjian-like)
#   r_mode = 8-10 um           ~ thicker advection fog
FOG_MODE_RADIUS_UM  = 4.0
FOG_ALPHA           = 6.0
WATER_REFRACTIVE_INDEX = 1.331 - 1e-8j    # water at 680 nm (miepython convention n - ik)
MIE_N_RADII  = 150           # number of radii used for the size distribution
MIE_N_ANGLES = 3000          # angular resolution of the phase function table

# =============================================================================
# Physical constants
# =============================================================================
h   = 6.62607015e-34
c   = 2.99792458e8
k_B = 1.380649e-23

_trapz = getattr(np, "trapezoid", None) or np.trapz

VALID_MEDIA = ("air", "fog_mie", "fog_hg")
VALID_POLARIZATIONS = ("perpendicular", "in_plane", "unpolarized")


def validate_settings():
    if MEDIUM not in VALID_MEDIA:
        raise ValueError(f"MEDIUM must be one of {VALID_MEDIA}, got {MEDIUM!r}")
    if POLARIZATION not in VALID_POLARIZATIONS:
        raise ValueError(f"POLARIZATION must be one of {VALID_POLARIZATIONS}, got {POLARIZATION!r}")
    if POWER_MODE not in ("peak", "average"):
        raise ValueError(f"POWER_MODE must be 'peak' or 'average', got {POWER_MODE!r}")
    if VIEWED_BEAM_LENGTH_M is None:
        half = viewed_beam_length(DISTANCE_M) / 2
        if half >= LASER_DISTANCE_M:
            raise ValueError("The viewed stretch of beam reaches the laser: "
                             "increase LASER_DISTANCE_M or reduce the view angle.")


# =============================================================================
# Step 1: photons per second
# =============================================================================
def photon_rate(power_W, wavelength_m):
    """photons/s = P / (h c / lambda)"""
    return power_W / (h * c / wavelength_m)


def photons_per_pulse(N_dot):
    if POWER_MODE == "peak":
        return N_dot * PULSE_LENGTH_S
    return N_dot / REP_RATE_HZ


# =============================================================================
# Step 2a: clean air (Rayleigh)
# =============================================================================
def refractive_index_air_standard(wavelength_m):
    """Peck & Reeder (1972) fit for dry standard air (15 C, 1 atm)."""
    s2 = (1.0 / (wavelength_m * 1e6)) ** 2
    n_minus_1 = (8060.51 + 2480990.0 / (132.274 - s2)
                 + 17455.7 / (39.32957 - s2)) * 1e-8
    return 1.0 + n_minus_1


def rayleigh_cross_section(wavelength_m):
    """
    Rayleigh cross-section of ONE air molecule [m^2]:
        sigma = 24 pi^3 / (lambda^4 N_s^2) * ((n^2-1)/(n^2+2))^2 * F_K
    N_s = molecule density of standard air, F_K = King factor (1.049).
    """
    N_s = 2.54743e25
    n = refractive_index_air_standard(wavelength_m)
    F_K = 1.049
    return (24 * np.pi**3 / (wavelength_m**4 * N_s**2)
            * ((n**2 - 1) / (n**2 + 2))**2 * F_K)


def beta_air(wavelength_m, T_K, p_Pa):
    """beta [1/m] = molecules per m^3 x cross-section. N from ideal gas law."""
    return p_Pa / (k_B * T_K) * rayleigh_cross_section(wavelength_m)


# =============================================================================
# Step 2b: fog extinction from visibility (Koschmieder)
# =============================================================================
def beta_fog(visibility_m):
    """exp(-beta V) = 0.02  ->  beta = ln(50) / V = 3.912 / V"""
    return 3.912 / visibility_m


# =============================================================================
# Step 3: phase functions
# =============================================================================
def rayleigh_phase(cos_theta, sin2_psi, polarization):
    """Polarized: 3/(8 pi) sin^2(psi). Unpolarized: 3/(16 pi)(1 + cos^2 theta)."""
    if polarization == "unpolarized":
        return 3.0 / (16 * np.pi) * (1 + cos_theta**2)
    return 3.0 / (8 * np.pi) * sin2_psi


def henyey_greenstein_phase(cos_theta, g):
    """p = 1/(4 pi) (1 - g^2) / (1 + g^2 - 2 g cos theta)^(3/2)"""
    return (1.0 / (4 * np.pi)) * (1 - g**2) / (1 + g**2 - 2 * g * cos_theta)**1.5


# =============================================================================
# Step 3b: Mie scattering by a droplet size distribution
# =============================================================================
class MieFog:
    """
    Pre-computes, for the droplet size distribution above:

        beta_ext        extinction coefficient [1/m] (= FOG_VISIBILITY-based value)
        theta           angle grid [rad]
        b_perp(theta)   angular scattering coefficient [1/(m sr)] for E perpendicular
                        to the scattering plane   (Mie S1)
        b_par(theta)    same for E in the scattering plane (Mie S2)

    b is "scattering per metre of beam per steradian". Averaged over polarization
    it integrates over 4 pi sr to beta_sca. For water droplets at 680 nm there is
    practically no absorption, so beta_sca = beta_ext.

    How it works, for one droplet of radius r (size parameter x = 2 pi r / lambda):
      - Mie theory gives the complex amplitudes S1(theta), S2(theta).
      - |S1|^2 and |S2|^2 are the angular patterns for the two polarizations.
        We normalize them so that their average integrates to the scattering
        cross-section C_sca = Q_sca * pi r^2.
      - We average over the size distribution n(r) and multiply by the droplet
        number density N, chosen such that N * <C_ext> = beta_ext (visibility).
    """

    def __init__(self):
        try:
            import miepython
        except ImportError as e:
            raise ImportError("MEDIUM = 'fog_mie' needs miepython: pip install miepython") from e
        self.mp = miepython

        lam = WAVELENGTH_NM * 1e-9
        r_mode = FOG_MODE_RADIUS_UM * 1e-6
        alpha = FOG_ALPHA

        # radius grid: from tiny droplets to far into the tail
        r = np.linspace(0.05e-6, 10 * r_mode, MIE_N_RADII)
        n_r = r**alpha * np.exp(-alpha * r / r_mode)
        n_r /= _trapz(n_r, r)                      # normalize: integral n(r) dr = 1

        # angle grid, denser near 0 where the forward peak is
        n1 = MIE_N_ANGLES // 3
        th = np.concatenate([np.linspace(0, np.radians(5), n1, endpoint=False),
                             np.linspace(np.radians(5), np.pi, MIE_N_ANGLES - n1)])
        mu = np.cos(th)
        sin_th = np.sin(th)

        d_perp = np.zeros((len(r), len(th)))       # dC/dOmega for each radius [m^2/sr]
        d_par  = np.zeros_like(d_perp)
        c_ext  = np.zeros(len(r))
        for i, ri in enumerate(r):
            x = 2 * np.pi * ri / lam
            qext, qsca = self._efficiencies(WATER_REFRACTIVE_INDEX, x)
            S1, S2 = self._amplitudes(WATER_REFRACTIVE_INDEX, x, mu)
            I1, I2 = np.abs(S1)**2, np.abs(S2)**2
            A = 2 * np.pi * _trapz(0.5 * (I1 + I2) * sin_th, th)   # integral over 4 pi
            csca = qsca * np.pi * ri**2
            d_perp[i] = csca * I1 / A
            d_par[i]  = csca * I2 / A
            c_ext[i]  = qext * np.pi * ri**2

        # number density N [1/m^3] such that extinction matches the visibility
        mean_cext = _trapz(n_r * c_ext, r)
        self.beta_ext = beta_fog(FOG_VISIBILITY_M)
        N = self.beta_ext / mean_cext
        self.droplets_per_cm3 = N * 1e-6

        self.theta = th
        self.b_perp = N * _trapz(n_r[:, None] * d_perp, r, axis=0)
        self.b_par  = N * _trapz(n_r[:, None] * d_par,  r, axis=0)

        b_avg = 0.5 * (self.b_perp + self.b_par)
        self.beta_sca = 2 * np.pi * _trapz(b_avg * sin_th, th)
        self.g = 2 * np.pi * _trapz(b_avg * mu * sin_th, th) / self.beta_sca
        self.mean_radius_um = _trapz(n_r * r, r) * 1e6
        self.eff_radius_um = _trapz(n_r * r**3, r) / _trapz(n_r * r**2, r) * 1e6

    def _amplitudes(self, m, x, mu):
        """Mie amplitudes S1, S2 (miepython 3 calls it S1_S2, version 2 mie_S1_S2).
        Their overall normalization does not matter: we renormalize numerically."""
        mp = self.mp
        fn = getattr(mp, "S1_S2", None) or getattr(mp, "mie_S1_S2")
        S1, S2 = fn(m, x, mu)
        return np.asarray(S1), np.asarray(S2)

    def _efficiencies(self, m, x):
        """Q_ext and Q_sca for one sphere (copes with different miepython versions)."""
        mp = self.mp
        if hasattr(mp, "efficiencies_mx"):
            out = mp.efficiencies_mx(m, x)
        else:
            out = mp.single_sphere(m, x)
        return float(out[0]), float(out[1])

    def b(self, cos_theta, polarization):
        """Angular scattering coefficient [1/(m sr)] at the given cos(theta)."""
        theta = np.arccos(np.clip(cos_theta, -1, 1))
        bp = np.interp(theta, self.theta, self.b_perp)
        bl = np.interp(theta, self.theta, self.b_par)
        if polarization == "perpendicular":
            return bp
        if polarization == "in_plane":
            return bl
        return 0.5 * (bp + bl)

    def phase(self, theta_rad, polarization="unpolarized"):
        """Phase function p [1/sr] = b / beta_sca."""
        cos_t = np.cos(theta_rad)
        return self.b(cos_t, polarization) / self.beta_sca


# =============================================================================
# Step 4: add up all pieces of the beam the lens can see
# =============================================================================
def view_angle_rad():
    return np.radians(FOV_DEG) / HORIZONTAL_RESOLUTION


def viewed_beam_length(r):
    """Exact: 2 r tan(angle/2); approx r * angle."""
    if VIEWED_BEAM_LENGTH_M is not None:
        return VIEWED_BEAM_LENGTH_M
    return 2 * r * np.tan(view_angle_rad() / 2)


def lens_solid_angle():
    """Solid angle seen from the closest point of the beam."""
    if SOLID_ANGLE_SR is not None:
        return SOLID_ANGLE_SR
    return np.pi * (LENS_DIAMETER_M / 2)**2 / DISTANCE_M**2


def fraction_into_lens(mie=None):
    """
    Returns a dict with beta_ext, the fraction of beam photons ending up in the lens
    (with and without attenuation) and the optical depth of the longest path.
    """
    lam = WAVELENGTH_NM * 1e-9
    r = DISTANCE_M
    half_length = viewed_beam_length(r) / 2

    n_pieces = 2001
    z = np.linspace(-half_length, half_length, n_pieces)

    # beam along +z, laser at z = -LASER_DISTANCE_M, lens at x = r, z = 0
    d = np.sqrt(r**2 + z**2)                 # piece -> lens distance
    cos_theta = -z / d                       # angle between beam and direction to lens
    s = LASER_DISTANCE_M + z                 # laser -> piece distance

    # solid angle of the lens from each piece (lens faces the beam: tilt cosine r/d)
    if SOLID_ANGLE_SR is not None:
        Omega = np.full_like(z, SOLID_ANGLE_SR)
    else:
        Omega = np.pi * (LENS_DIAMETER_M / 2)**2 / d**2 * (r / d)

    # b(theta) = beta_sca * p(theta)   [1/(m sr)], and beta_ext for attenuation
    if MEDIUM == "air":
        beta_ext = beta_air(lam, TEMPERATURE_K, PRESSURE_PA)
        if POLARIZATION == "perpendicular":
            sin2_psi = np.ones_like(z)
        elif POLARIZATION == "in_plane":
            sin2_psi = (z / d)**2
        else:
            sin2_psi = None
        b = beta_ext * rayleigh_phase(cos_theta, sin2_psi, POLARIZATION)
    elif MEDIUM == "fog_hg":
        beta_ext = beta_fog(FOG_VISIBILITY_M)
        b = beta_ext * henyey_greenstein_phase(cos_theta, FOG_G)
    else:  # fog_mie
        beta_ext = mie.beta_ext
        b = mie.b(cos_theta, POLARIZATION)

    # single scattering with attenuation on the way in and on the way out
    atten = np.exp(-beta_ext * (s + d))
    integrand = b * Omega
    return {
        "beta": beta_ext,
        "fraction_no_atten": _trapz(integrand, z),
        "fraction": _trapz(integrand * atten, z),
        "tau_path": beta_ext * (LASER_DISTANCE_M + d[0]),
        "b_90": float(np.interp(0.0, z, b)),
    }


# =============================================================================
# Run and print
# =============================================================================
if __name__ == "__main__":
    validate_settings()
    lam = WAVELENGTH_NM * 1e-9
    N_dot = photon_rate(POWER_W, lam)
    mie = MieFog() if MEDIUM == "fog_mie" else None
    res = fraction_into_lens(mie)
    beta, frac = res["beta"], res["fraction"]

    print(f"Wavelength            : {WAVELENGTH_NM:.0f} nm,  power {POWER_W*1e3:.3g} mW ({POWER_MODE})")
    if MEDIUM == "air":
        print(f"Medium                : air (Rayleigh), {POLARIZATION}")
        print(f"Cross-section/molecule: {rayleigh_cross_section(lam):.3e} m^2")
    elif MEDIUM == "fog_hg":
        print(f"Medium                : fog, Henyey-Greenstein g = {FOG_G}, visibility {FOG_VISIBILITY_M} m, {POLARIZATION}")
    else:
        print(f"Medium                : fog, Mie, visibility {FOG_VISIBILITY_M} m, {POLARIZATION}")
        print(f"  droplets            : mode radius {FOG_MODE_RADIUS_UM} um (alpha {FOG_ALPHA}), "
              f"mean {mie.mean_radius_um:.2f} um, effective {mie.eff_radius_um:.2f} um")
        print(f"  droplet density     : {mie.droplets_per_cm3:.1f} per cm^3")
        print(f"  asymmetry g         : {mie.g:.3f}   (single-scattering albedo "
              f"{mie.beta_sca/mie.beta_ext:.4f})")
    print(f"Extinction coefficient: beta = {beta:.3e} 1/m  (mean free path {1/beta:.3g} m)")
    print(f"Scattered per metre   : {(1-np.exp(-beta))*100:.3e} % of the photons")
    print()

    L = viewed_beam_length(DISTANCE_M)
    if VIEWED_BEAM_LENGTH_M is None:
        print(f"View angle            : {FOV_DEG} deg / {HORIZONTAL_RESOLUTION} = "
              f"{view_angle_rad()*1e3:.4g} mrad")
    print(f"Beam length in view   : {L*1e3:.4g} mm" + (" (set manually)" if VIEWED_BEAM_LENGTH_M else ""))
    print(f"Lens solid angle      : {lens_solid_angle():.3e} sr"
          + (" (override)" if SOLID_ANGLE_SR is not None else ""))
    print()

    N_pulse = photons_per_pulse(N_dot)
    print(f"Photons in beam       : {N_dot:.3e} /s,  {N_pulse:.3e} per pulse")
    print(f"Fraction into lens    : {frac:.3e}   (without attenuation {res['fraction_no_atten']:.3e})")
    print(f"Photons into lens     : {N_dot*frac:.3e} /s")
    print(f"                        {N_pulse*frac:.3e} per pulse  (before detector efficiency)")
    print()
    print(f"Optical depth laser -> scatterer -> lens: tau = {res['tau_path']:.3g}")
    if res["tau_path"] > 0.3:
        print("  WARNING: tau > 0.3, multiple scattering is no longer negligible; "
              "treat the result as a lower bound / order of magnitude.")

    if MEDIUM == "fog_mie":
        print()
        print("Phase function p(theta) [1/sr], unpolarized, Mie vs Henyey-Greenstein "
              f"(g = {mie.g:.2f} from the Mie result):")
        print("  theta [deg]     Mie          HG")
        for a in (10, 30, 60, 90, 120, 150, 170):
            t = np.radians(a)
            print(f"  {a:6d}      {mie.phase(t):.3e}   {henyey_greenstein_phase(np.cos(t), mie.g):.3e}")
        print(f"  polarization ratio at 90 deg: perpendicular / in_plane = "
              f"{mie.phase(np.pi/2,'perpendicular')/mie.phase(np.pi/2,'in_plane'):.2f}")
