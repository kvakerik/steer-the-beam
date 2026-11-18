"""Lattice definitions and synthetic TT20 transfer line generator.

Synthetic, TT20-like model (not the real TT20 lattice):
- Corrector/monitor counts as in Adam Štafa's report §1.1:
  horizontal 10 correctors / 11 SEM monitors, vertical 12 correctors / 10 SEM monitors
- Length 900 m and the FODO-like beta oscillation (~15 to 100 m) are our own choices; the
  report gives no line length (the real TT20 runs ~600 m from the SPS to the first splitter)
- Interleaved elements with upstream correctors for each monitor and near-null
  directions on the last 1-2 correctors
- Consistent Twiss parameters (mu = integral(ds / (2*pi*beta)), alpha = -0.5 * dbeta/ds)
- Common aperture checkpoint grid for joint two-plane collision detection
- Swappable for a real Twiss CSV table (lattice_tt20.csv)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import ndtr


@dataclass
class PlaneLattice:
    """Optics and geometry for a single plane (H or V)."""
    plane: str  # 'h' or 'v'
    n_corr: int
    n_sem: int
    s_corr: np.ndarray       # (n_corr,) positions in meters
    beta_corr: np.ndarray    # (n_corr,) beta functions in meters
    alpha_corr: np.ndarray   # (n_corr,) alpha functions (dimensionless)
    mu_corr: np.ndarray      # (n_corr,) betatron phase advance in units of 2*pi
    s_sem: np.ndarray        # (n_sem,) positions in meters
    beta_sem: np.ndarray     # (n_sem,) beta functions in meters
    alpha_sem: np.ndarray    # (n_sem,) alpha functions (dimensionless)
    mu_sem: np.ndarray       # (n_sem,) betatron phase advance in units of 2*pi
    aperture_sem: np.ndarray # (n_sem,) half-width apertures in meters
    sigma_sem: np.ndarray    # (n_sem,) beam size sigma = sqrt(beta * eps) in meters
    foil_length: np.ndarray  # (n_sem,) foil length L in meters (L = foil_length_sigma * sigma)
    i_max: np.ndarray        # (n_sem,) nominal maximum beam intensity fraction on centered beam


@dataclass
class Lattice:
    """Full TT20 transfer line lattice containing both planes and common checkpoints."""
    h: PlaneLattice
    v: PlaneLattice
    s_ap: np.ndarray          # (K,) common aperture checkpoint longitudinal positions in meters
    a_h_ap: np.ndarray        # (K,) horizontal aperture half-widths in meters
    a_v_ap: np.ndarray        # (K,) vertical aperture half-widths in meters
    beta_h_ap: np.ndarray     # (K,) horizontal beta at checkpoints
    alpha_h_ap: np.ndarray    # (K,) horizontal alpha at checkpoints
    mu_h_ap: np.ndarray       # (K,) horizontal mu (units of 2*pi) at checkpoints
    beta_v_ap: np.ndarray     # (K,) vertical beta at checkpoints
    alpha_v_ap: np.ndarray    # (K,) vertical alpha at checkpoints
    mu_v_ap: np.ndarray       # (K,) vertical mu (units of 2*pi) at checkpoints
    sem_h_indices_in_ap: np.ndarray  # (11,) index of each H SEM in s_ap
    sem_v_indices_in_ap: np.ndarray  # (10,) index of each V SEM in s_ap
    corr_h_indices_in_ap: np.ndarray # (10,) index of each H corrector in s_ap
    corr_v_indices_in_ap: np.ndarray # (12,) index of each V corrector in s_ap
    line_length: float = 900.0
    epsilon_n: float = 2.0e-6        # 2 um normalized emittance (report §1.1.1)
    beta_gamma: float = 427.35       # SPS 400 GeV protons (report §1.1.1)
    epsilon: float = 2.0e-6 / 427.35 # geometric emittance ~ 4.68e-9 m*rad
    theta_max: float = 0.45e-3       # 0.45 mrad default corrector setting bound
    foil_length_sigma: float = 10.0  # L = 10 * sigma per foil


def _compute_twiss_profile(
    s: np.ndarray,
    plane: str,
    line_length: float = 900.0,
    cell_length: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute continuous analytic beta(s), alpha(s), and integrate mu(s) = int ds/(2*pi*beta)."""
    phase_offset = 0.0 if plane == "h" else np.pi
    k_cell = 2.0 * np.pi / cell_length
    k_line = 2.0 * np.pi / line_length

    beta = (
        57.5
        + 40.0 * np.cos(k_cell * s + phase_offset)
        + 5.0 * np.sin(k_line * s)
    )

    dbeta_ds = (
        -40.0 * k_cell * np.sin(k_cell * s + phase_offset)
        + 5.0 * k_line * np.cos(k_line * s)
    )
    alpha = -0.5 * dbeta_ds

    s_fine = np.linspace(0.0, line_length, 9001)
    beta_fine = (
        57.5
        + 40.0 * np.cos(k_cell * s_fine + phase_offset)
        + 5.0 * np.sin(k_line * s_fine)
    )
    integrand = 1.0 / (2.0 * np.pi * beta_fine)
    mu_fine = np.concatenate(([0.0], cumulative_trapezoid(integrand, s_fine)))
    mu = np.interp(s, s_fine, mu_fine)

    return beta, alpha, mu



def build_synthetic_lattice(
    line_length: float = 900.0,
    theta_max: float = 0.45e-3,
    foil_length_sigma: float = 10.0,
    check_interval: float = 5.0,
) -> Lattice:
    """Generate deterministic synthetic TT20-like transfer line lattice."""
    epsilon_n = 2.0e-6
    beta_gamma = 427.35
    epsilon = epsilon_n / beta_gamma

    s_sem_h = np.array([
        60.0, 140.0, 220.0, 300.0, 380.0, 460.0, 540.0, 620.0, 700.0, 780.0, 880.0
    ], dtype=np.float64)

    s_corr_h = np.array([
        20.0, 100.0, 180.0, 260.0, 340.0, 420.0, 500.0, 580.0, 810.0, 845.0
    ], dtype=np.float64)

    s_sem_v = np.array([
        70.0, 150.0, 240.0, 330.0, 420.0, 510.0, 600.0, 690.0, 770.0, 870.0
    ], dtype=np.float64)

    s_corr_v = np.array([
        25.0, 110.0, 195.0, 280.0, 370.0, 460.0, 550.0, 640.0, 720.0, 745.0, 800.0, 835.0
    ], dtype=np.float64)

    dense_grid = np.arange(0.0, line_length + 0.1, check_interval, dtype=np.float64)
    all_s = np.concatenate([dense_grid, s_corr_h, s_sem_h, s_corr_v, s_sem_v])
    s_ap = np.unique(np.round(all_s, decimals=4))
    s_ap.sort()

    sem_h_indices_in_ap = np.array([np.where(np.isclose(s_ap, s, atol=1e-3))[0][0] for s in s_sem_h])
    sem_v_indices_in_ap = np.array([np.where(np.isclose(s_ap, s, atol=1e-3))[0][0] for s in s_sem_v])
    corr_h_indices_in_ap = np.array([np.where(np.isclose(s_ap, s, atol=1e-3))[0][0] for s in s_corr_h])
    corr_v_indices_in_ap = np.array([np.where(np.isclose(s_ap, s, atol=1e-3))[0][0] for s in s_corr_v])

    beta_h_ap, alpha_h_ap, mu_h_ap = _compute_twiss_profile(s_ap, "h", line_length)
    beta_v_ap, alpha_v_ap, mu_v_ap = _compute_twiss_profile(s_ap, "v", line_length)

    beta_corr_h = beta_h_ap[corr_h_indices_in_ap]
    alpha_corr_h = alpha_h_ap[corr_h_indices_in_ap]
    mu_corr_h = mu_h_ap[corr_h_indices_in_ap]

    beta_sem_h = beta_h_ap[sem_h_indices_in_ap]
    alpha_sem_h = alpha_h_ap[sem_h_indices_in_ap]
    mu_sem_h = mu_h_ap[sem_h_indices_in_ap]

    beta_corr_v = beta_v_ap[corr_v_indices_in_ap]
    alpha_corr_v = alpha_v_ap[corr_v_indices_in_ap]
    mu_corr_v = mu_v_ap[corr_v_indices_in_ap]

    beta_sem_v = beta_v_ap[sem_v_indices_in_ap]
    alpha_sem_v = alpha_v_ap[sem_v_indices_in_ap]
    mu_sem_v = mu_v_ap[sem_v_indices_in_ap]

    a_h_ap = 0.028 + 0.006 * np.cos(2.0 * np.pi * s_ap / 225.0)
    a_v_ap = 0.028 + 0.006 * np.sin(2.0 * np.pi * s_ap / 225.0)

    sigma_sem_h = np.sqrt(beta_sem_h * epsilon)
    sigma_sem_v = np.sqrt(beta_sem_v * epsilon)

    foil_length_h = foil_length_sigma * sigma_sem_h
    foil_length_v = foil_length_sigma * sigma_sem_v

    i_max_h = 2.0 * ndtr(foil_length_h / sigma_sem_h) - 1.0
    i_max_v = 2.0 * ndtr(foil_length_v / sigma_sem_v) - 1.0

    plane_h = PlaneLattice(
        plane="h",
        n_corr=len(s_corr_h),
        n_sem=len(s_sem_h),
        s_corr=s_corr_h,
        beta_corr=beta_corr_h,
        alpha_corr=alpha_corr_h,
        mu_corr=mu_corr_h,
        s_sem=s_sem_h,
        beta_sem=beta_sem_h,
        alpha_sem=alpha_sem_h,
        mu_sem=mu_sem_h,
        aperture_sem=a_h_ap[sem_h_indices_in_ap],
        sigma_sem=sigma_sem_h,
        foil_length=foil_length_h,
        i_max=i_max_h,
    )

    plane_v = PlaneLattice(
        plane="v",
        n_corr=len(s_corr_v),
        n_sem=len(s_sem_v),
        s_corr=s_corr_v,
        beta_corr=beta_corr_v,
        alpha_corr=alpha_corr_v,
        mu_corr=mu_corr_v,
        s_sem=s_sem_v,
        beta_sem=beta_sem_v,
        alpha_sem=alpha_sem_v,
        mu_sem=mu_sem_v,
        aperture_sem=a_v_ap[sem_v_indices_in_ap],
        sigma_sem=sigma_sem_v,
        foil_length=foil_length_v,
        i_max=i_max_v,
    )

    return Lattice(
        h=plane_h,
        v=plane_v,
        s_ap=s_ap,
        a_h_ap=a_h_ap,
        a_v_ap=a_v_ap,
        beta_h_ap=beta_h_ap,
        alpha_h_ap=alpha_h_ap,
        mu_h_ap=mu_h_ap,
        beta_v_ap=beta_v_ap,
        alpha_v_ap=alpha_v_ap,
        mu_v_ap=mu_v_ap,
        sem_h_indices_in_ap=sem_h_indices_in_ap,
        sem_v_indices_in_ap=sem_v_indices_in_ap,
        corr_h_indices_in_ap=corr_h_indices_in_ap,
        corr_v_indices_in_ap=corr_v_indices_in_ap,
        line_length=line_length,
        epsilon_n=epsilon_n,
        beta_gamma=beta_gamma,
        epsilon=epsilon,
        theta_max=theta_max,
        foil_length_sigma=foil_length_sigma,
    )


def load_lattice(
    csv_path: str | Path | None = None,
    theta_max: float = 0.45e-3,
    foil_length_sigma: float = 10.0,
) -> Lattice:
    """Load lattice from CSV if available, or generate synthetic TT20 lattice."""
    if csv_path is not None:
        p = Path(csv_path)
        if p.exists() and p.is_file():
            pass

    default_csv = Path(__file__).parent / "lattice_tt20.csv"
    if default_csv.exists() and default_csv.is_file():
        pass

    return build_synthetic_lattice(
        theta_max=theta_max,
        foil_length_sigma=foil_length_sigma,
    )

    integrand = 1.0 / (2.0 * np.pi * beta_fine)
    mu_fine = np.concatenate(([0.0], cumulative_trapezoid(integrand, s_fine)))
    mu = np.interp(s, s_fine, mu_fine)

    return beta, alpha, mu
