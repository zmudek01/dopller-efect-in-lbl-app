# core.py
from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# Geometry + trajectory
# ============================================================

def make_beacons_preset(preset: str, radius: float, z: float = 0.0) -> np.ndarray:
    """
    Returns beacons positions (N,3) for preset shapes centered at (0,0,z).
    """
    preset = preset.lower()
    if preset == "trójkąt":
        angles = np.deg2rad([90, 210, 330])
    elif preset == "kwadrat":
        angles = np.deg2rad([45, 135, 225, 315])
    elif preset == "pięciokąt":
        angles = np.deg2rad([90, 162, 234, 306, 18])
    else:
        raise ValueError("Unknown preset")

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    zarr = np.full_like(x, z, dtype=float)
    return np.column_stack([x, y, zarr]).astype(float)


def make_trajectory(
    kind: str,
    T: float,
    dt: float,
    speed: float,
    heading_deg: float,
    start: np.ndarray,
    z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns time vector t, positions p(t) (K,3), velocities v(t) (K,3)
    """
    kind = kind.lower()
    t = np.arange(0.0, T + 1e-12, dt)
    K = len(t)

    heading = np.deg2rad(heading_deg)
    vxy = np.array([speed * np.cos(heading), speed * np.sin(heading)], dtype=float)

    p = np.zeros((K, 3), dtype=float)
    v = np.zeros((K, 3), dtype=float)

    if kind == "linia":
        for k in range(K):
            p[k, 0:2] = start[0:2] + vxy * t[k]
            p[k, 2] = z
            v[k, 0:2] = vxy
            v[k, 2] = 0.0

    elif kind == "racetrack":
        # prosta + łuk, prosty model demonstracyjny
        L = 0.6 * speed * T
        R = max(10.0, 0.15 * L)
        s = speed * t
        per = 2 * (L + np.pi * R)
        s_mod = np.mod(s, per)

        for k in range(K):
            sk = s_mod[k]
            if sk < L:
                p[k] = np.array([start[0] - L/2 + sk, start[1] - R, z])
                v[k] = np.array([speed, 0.0, 0.0])
            elif sk < L + np.pi * R:
                phi = (sk - L) / R  # 0..pi
                cx, cy = start[0] + L/2, start[1]
                p[k] = np.array([cx + R*np.cos(-np.pi/2 + phi), cy + R*np.sin(-np.pi/2 + phi), z])
                v[k] = speed * np.array([-np.sin(-np.pi/2 + phi), np.cos(-np.pi/2 + phi), 0.0])
            elif sk < 2*L + np.pi * R:
                s2 = sk - (L + np.pi*R)
                p[k] = np.array([start[0] + L/2 - s2, start[1] + R, z])
                v[k] = np.array([-speed, 0.0, 0.0])
            else:
                phi = (sk - (2*L + np.pi*R)) / R
                cx, cy = start[0] - L/2, start[1]
                p[k] = np.array([cx + R*np.cos(np.pi/2 + phi), cy + R*np.sin(np.pi/2 + phi), z])
                v[k] = speed * np.array([-np.sin(np.pi/2 + phi), np.cos(np.pi/2 + phi), 0.0])
    else:
        raise ValueError("Unknown trajectory kind")

    return t, p, v


# ============================================================
# Measurement models: TDOA + Doppler-as-vr + gross errors
# ============================================================

def _ranges_and_u(p: np.ndarray, beacons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      R (N,) distances
      u (N,3) unit LOS vectors from beacon i to object (direction of increasing range)
          u_i = (p - b_i)/||p-b_i||
    """
    diff = p[None, :] - beacons
    R = np.linalg.norm(diff, axis=1)
    R = np.maximum(R, 1e-9)
    u = diff / R[:, None]
    return R, u


def _dvr_dp_full(u: np.ndarray, R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Full Jacobian for vr_i = u_i^T v wrt position p.
    For each i:
      ∂vr_i/∂p = (1/R_i) * (I - u_i u_i^T) v
    Returns Jp (N,3): each row is the gradient wrt p.
    """
    N = u.shape[0]
    I = np.eye(3, dtype=float)
    Jp = np.zeros((N, 3), dtype=float)
    for i in range(N):
        ui = u[i][:, None]  # (3,1)
        P = I - ui @ ui.T   # (3,3)
        Jp[i] = (P @ v) / max(R[i], 1e-9)
    return Jp


def simulate_tdoa_and_vr(
    p: np.ndarray,
    v: np.ndarray,
    beacons: np.ndarray,
    c: float,
    sigma_t: float,
    sigma_vr: float,
    rng: np.random.Generator,
    ref_idx: int = 0,
    gross_enable: bool = False,
    gross_p_tdoa: float = 0.0,
    gross_scale_tdoa: float = 10.0,
    gross_p_vr: float = 0.0,
    gross_scale_vr: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulate one epoch:
      - One-way TOA for each beacon with a common bias b (cancels in TDOA)
      - TDOA formed as dt_i = toa_i - toa_ref
      - drho_hat = c*dt_hat (in meters)
      - Doppler provided as radial velocity observation vr_hat (m/s)

    Gross errors:
      - applied to TOA samples before differencing (more realistic)
      - applied to vr samples

    Returns:
      drho_hat (N-1,)   for all i != ref_idx
      vr_hat   (N,)
      meta dict with gross error counters
    """
    N = beacons.shape[0]
    R_true, u = _ranges_and_u(p, beacons)
    vr_true = u @ v

    # Common clock bias (cancels in differences)
    b = rng.normal(0.0, 1e-3)

    toa = R_true / c + b
    toa_hat = toa + rng.normal(0.0, sigma_t, size=N)

    gross_toa_ct = 0
    if gross_enable and gross_p_tdoa > 0.0:
        mask = rng.random(N) < gross_p_tdoa
        if np.any(mask):
            toa_hat[mask] += rng.normal(0.0, gross_scale_tdoa * sigma_t, size=int(np.sum(mask)))
            gross_toa_ct = int(np.sum(mask))

    idx = np.arange(N)
    idx_oth = idx[idx != ref_idx]
    dt_hat = toa_hat[idx_oth] - toa_hat[ref_idx]
    drho_hat = c * dt_hat

    vr_hat = vr_true + rng.normal(0.0, sigma_vr, size=N)

    gross_vr_ct = 0
    if gross_enable and gross_p_vr > 0.0:
        mask = rng.random(N) < gross_p_vr
        if np.any(mask):
            vr_hat[mask] += rng.normal(0.0, gross_scale_vr * sigma_vr, size=int(np.sum(mask)))
            gross_vr_ct = int(np.sum(mask))

    meta = {"gross_toa_ct": gross_toa_ct, "gross_vr_ct": gross_vr_ct}
    return drho_hat, vr_hat, meta


# ============================================================
# VLS / WLS (Gauss-Newton)
#   - position only from TDOA
#   - position+velocity from TDOA + vr (Doppler)
# ============================================================

def vls_position_from_tdoa(
    drho_hat: np.ndarray,     # (N-1,)
    beacons: np.ndarray,      # (N,3)
    ref_idx: int,
    p0: np.ndarray,           # (3,)
    sigma_drho: float,
    iters: int = 20
) -> np.ndarray:
    """
    Minimize sum ((drho_hat - (||p-bi|| - ||p-br||))/sigma_drho)^2
    """
    p = p0.astype(float).copy()
    N = beacons.shape[0]
    idx = np.arange(N)
    idx_oth = idx[idx != ref_idx]

    w = 1.0 / (sigma_drho**2)

    for _ in range(iters):
        R, u = _ranges_and_u(p, beacons)
        Rr = R[ref_idx]
        ur = u[ref_idx]

        h = R[idx_oth] - Rr
        r = drho_hat - h

        H = (u[idx_oth] - ur)  # (N-1,3)

        # Weighted normal equations
        W = w * np.eye(len(idx_oth), dtype=float)
        A = H.T @ W @ H
        b = H.T @ W @ r
        dp = np.linalg.solve(A + 1e-9*np.eye(3), b)

        p = p + dp
        if np.linalg.norm(dp) < 1e-6:
            break

    return p


def vls_pv_from_tdoa_vr(
    drho_hat: np.ndarray,     # (N-1,)
    vr_hat: np.ndarray,       # (N,)
    beacons: np.ndarray,      # (N,3)
    ref_idx: int,
    x0: np.ndarray,           # (6,) [p(3), v(3)]
    sigma_drho: float,
    sigma_vr: float,
    iters: int = 25,
    full_doppler_jacobian: bool = True
) -> np.ndarray:
    """
    Joint WLS for x=[p,v] using:
      - TDOA (N-1): drho = ||p-bi|| - ||p-br||
      - vr (N):    vr = u_i(p)^T v
    If full_doppler_jacobian=False => ∂vr/∂p ≈ 0 (uproszczenie).
    """
    x = x0.astype(float).copy()
    N = beacons.shape[0]
    idx = np.arange(N)
    idx_oth = idx[idx != ref_idx]

    w_drho = 1.0 / (sigma_drho**2)
    w_vr = 1.0 / (sigma_vr**2)

    for _ in range(iters):
        p = x[0:3]
        v = x[3:6]

        R, u = _ranges_and_u(p, beacons)
        Rr = R[ref_idx]
        ur = u[ref_idx]

        # residuals
        h_drho = R[idx_oth] - Rr
        r_drho = drho_hat - h_drho

        h_vr = u @ v
        r_vr = vr_hat - h_vr

        r = np.concatenate([r_drho, r_vr])

        # Jacobian
        H = np.zeros((len(idx_oth) + N, 6), dtype=float)

        # TDOA: d/drho wrt p = u_i - u_r ; wrt v = 0
        H[0:len(idx_oth), 0:3] = (u[idx_oth] - ur)
        H[0:len(idx_oth), 3:6] = 0.0

        # vr: d(vr)/dv = u ; d(vr)/dp = full or 0
        H[len(idx_oth):, 3:6] = u
        if full_doppler_jacobian:
            Jp = _dvr_dp_full(u, R, v)  # (N,3)
            H[len(idx_oth):, 0:3] = Jp
        else:
            H[len(idx_oth):, 0:3] = 0.0

        # weights
        W = np.diag(
            np.concatenate([
                np.full(len(idx_oth), w_drho),
                np.full(N, w_vr)
            ])
        )

        A = H.T @ W @ H
        b = H.T @ W @ r
        dx = np.linalg.solve(A + 1e-9*np.eye(6), b)

        x = x + dx
        if np.linalg.norm(dx) < 1e-6:
            break

    return x


def run_vls_tdoa_series(
    t: np.ndarray,
    beacons: np.ndarray,
    drho_hats: np.ndarray,    # (K,N-1)
    ref_idx: int,
    p_init: np.ndarray,
    sigma_drho: float
) -> np.ndarray:
    K = len(t)
    p_est = np.zeros((K, 3), dtype=float)
    p_prev = p_init.astype(float).copy()

    for k in range(K):
        p_prev = vls_position_from_tdoa(drho_hats[k], beacons, ref_idx, p_prev, sigma_drho)
        p_est[k] = p_prev

    return p_est


def run_vls_tdoa_vr_series(
    t: np.ndarray,
    beacons: np.ndarray,
    drho_hats: np.ndarray,    # (K,N-1)
    vr_hats: np.ndarray,      # (K,N)
    ref_idx: int,
    x_init: np.ndarray,       # (6,)
    sigma_drho: float,
    sigma_vr: float,
    full_doppler_jacobian: bool = True
) -> np.ndarray:
    K = len(t)
    xs = np.zeros((K, 6), dtype=float)
    x_prev = x_init.astype(float).copy()

    for k in range(K):
        x_prev = vls_pv_from_tdoa_vr(
            drho_hats[k],
            vr_hats[k],
            beacons,
            ref_idx,
            x_prev,
            sigma_drho=sigma_drho,
            sigma_vr=sigma_vr,
            full_doppler_jacobian=full_doppler_jacobian,
        )
        xs[k] = x_prev

    return xs


# ============================================================
# EKF: x=[p(3), v(3)]
#   - update with TDOA only
#   - update with TDOA + vr
# ============================================================

def _motion_matrices(dt: float, q_acc: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Constant velocity model with accel random walk.
    Returns (F,Q).
    """
    F = np.eye(6, dtype=float)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    G = np.zeros((6, 3), dtype=float)
    G[0:3, :] = 0.5 * dt**2 * np.eye(3)
    G[3:6, :] = dt * np.eye(3)

    Q = (q_acc**2) * (G @ G.T)
    return F, Q


def ekf_run_tdoa(
    t: np.ndarray,
    dt: float,
    beacons: np.ndarray,
    drho_hats: np.ndarray,    # (K,N-1)
    ref_idx: int,
    sigma_drho: float,
    x0: np.ndarray,
    P0: np.ndarray,
    q_acc: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    EKF with z = drho (range-difference in meters) for i != ref:
      z_i = ||p - b_i|| - ||p - b_r||
    """
    K = len(t)
    N = beacons.shape[0]
    idx = np.arange(N)
    idx_oth = idx[idx != ref_idx]
    m = len(idx_oth)

    x = x0.astype(float).copy()
    P = P0.astype(float).copy()

    xs = np.zeros((K, 6), dtype=float)
    Ps = np.zeros((K, 6, 6), dtype=float)

    F, Q = _motion_matrices(dt, q_acc)
    Rm = (sigma_drho**2) * np.eye(m, dtype=float)

    for k in range(K):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q

        p = x[0:3]
        R, u = _ranges_and_u(p, beacons)
        Rr = R[ref_idx]
        ur = u[ref_idx]

        z_pred = R[idx_oth] - Rr
        z = drho_hats[k]
        y = z - z_pred

        H = np.zeros((m, 6), dtype=float)
        H[:, 0:3] = (u[idx_oth] - ur)
        H[:, 3:6] = 0.0

        S = H @ P @ H.T + Rm
        Kk = P @ H.T @ np.linalg.inv(S)
        x = x + Kk @ y
        P = (np.eye(6) - Kk @ H) @ P

        xs[k] = x
        Ps[k] = P

    return xs, Ps


def ekf_run_tdoa_vr(
    t: np.ndarray,
    dt: float,
    beacons: np.ndarray,
    drho_hats: np.ndarray,    # (K,N-1)
    vr_hats: np.ndarray,      # (K,N)
    ref_idx: int,
    sigma_drho: float,
    sigma_vr: float,
    x0: np.ndarray,
    P0: np.ndarray,
    q_acc: float = 0.05,
    full_doppler_jacobian: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    EKF with z = [drho (N-1), vr (N)].
    vr model: vr_i = u_i(p)^T v

    If full_doppler_jacobian=False => ∂vr/∂p ≈ 0 (uproszczenie).
    """
    K = len(t)
    N = beacons.shape[0]
    idx = np.arange(N)
    idx_oth = idx[idx != ref_idx]
    m = len(idx_oth) + N

    x = x0.astype(float).copy()
    P = P0.astype(float).copy()

    xs = np.zeros((K, 6), dtype=float)
    Ps = np.zeros((K, 6, 6), dtype=float)

    F, Q = _motion_matrices(dt, q_acc)

    Rm = np.diag(
        np.concatenate([
            np.full(len(idx_oth), sigma_drho**2),
            np.full(N, sigma_vr**2)
        ])
    ).astype(float)

    for k in range(K):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q

        p = x[0:3]
        v = x[3:6]

        R, u = _ranges_and_u(p, beacons)
        Rr = R[ref_idx]
        ur = u[ref_idx]

        drho_pred = R[idx_oth] - Rr
        vr_pred = u @ v

        z_pred = np.concatenate([drho_pred, vr_pred])
        z = np.concatenate([drho_hats[k], vr_hats[k]])
        y = z - z_pred

        H = np.zeros((m, 6), dtype=float)

        # drho part
        H[0:len(idx_oth), 0:3] = (u[idx_oth] - ur)
        H[0:len(idx_oth), 3:6] = 0.0

        # vr part
        H[len(idx_oth):, 3:6] = u
        if full_doppler_jacobian:
            Jp = _dvr_dp_full(u, R, v)  # (N,3)
            H[len(idx_oth):, 0:3] = Jp
        else:
            H[len(idx_oth):, 0:3] = 0.0

        S = H @ P @ H.T + Rm
        Kk = P @ H.T @ np.linalg.inv(S)
        x = x + Kk @ y
        P = (np.eye(6) - Kk @ H) @ P

        xs[k] = x
        Ps[k] = P

    return xs, Ps


# ============================================================
# Metrics + runners
# ============================================================

def position_errors(p_est: np.ndarray, p_true: np.ndarray) -> np.ndarray:
    return np.linalg.norm(p_est - p_true, axis=1)


def summarize_errors(e: np.ndarray) -> dict:
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "MED": float(np.median(e)),
        "P95": float(np.quantile(e, 0.95)),
        "MAX": float(np.max(e)),
    }


def run_single_experiment(config: dict, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)

    # ---- Unpack config ----
    beacons = np.array(config["beacons"], dtype=float)
    c = float(config["acoustics"]["c"])
    sigma_t = float(config["noise"]["sigma_t"])
    sigma_vr = float(config["noise"]["sigma_vr"])

    ref_idx = int(config.get("tdoa", {}).get("ref_idx", 0))

    gross = config.get("gross", {})
    gross_enable = bool(gross.get("enable", False))
    gross_p_tdoa = float(gross.get("p_tdoa", 0.0))
    gross_scale_tdoa = float(gross.get("scale_tdoa", 10.0))
    gross_p_vr = float(gross.get("p_vr", 0.0))
    gross_scale_vr = float(gross.get("scale_vr", 10.0))

    dop = config.get("doppler", {})
    full_jac = bool(dop.get("full_jacobian", True))

    T = float(config["trajectory"]["T"])
    dt = float(config["trajectory"]["dt"])
    speed = float(config["trajectory"]["speed"])
    heading_deg = float(config["trajectory"]["heading_deg"])
    start_xy = np.array(config["trajectory"]["start_xy"], dtype=float)
    z = float(config["trajectory"]["z"])
    start = np.array([start_xy[0], start_xy[1], z], dtype=float)
    traj_kind = config["trajectory"]["kind"]

    q_acc = float(config["filter"]["q_acc"])

    # ---- Trajectory ----
    t, p_true, v_true = make_trajectory(traj_kind, T, dt, speed, heading_deg, start, z)

    K = len(t)
    N = beacons.shape[0]
    if N < 4:
        # Technicznie do TDOA w 3D potrzebujesz >= 4 beaconów, a do 2D >= 3.
        # Nie blokuję, ale ostrzegam przez meta.
        pass

    drho_hats = np.zeros((K, N-1), dtype=float)
    vr_hats = np.zeros((K, N), dtype=float)
    gross_meta = []

    for k in range(K):
        drho_hats[k], vr_hats[k], meta = simulate_tdoa_and_vr(
            p_true[k], v_true[k], beacons, c,
            sigma_t=sigma_t, sigma_vr=sigma_vr, rng=rng,
            ref_idx=ref_idx,
            gross_enable=gross_enable,
            gross_p_tdoa=gross_p_tdoa,
            gross_scale_tdoa=gross_scale_tdoa,
            gross_p_vr=gross_p_vr,
            gross_scale_vr=gross_scale_vr
        )
        gross_meta.append(meta)

    # ---- Noise scaling for TDOA ----
    # TOA noise sigma_t -> TDOA difference noise sigma_dt = sqrt(2)*sigma_t
    sigma_dt = np.sqrt(2.0) * sigma_t
    sigma_drho = c * sigma_dt  # [m]

    # ---- Initial guesses ----
    p_init = p_true[0] + np.array([5.0, -5.0, 2.0])  # small offset
    x_init = np.zeros(6, dtype=float)
    x_init[0:3] = p_init
    x_init[3:6] = np.array([0.0, 0.0, 0.0])

    P0 = np.diag([25.0, 25.0, 25.0, 4.0, 4.0, 4.0]).astype(float)

    # ============================================================
    # VLS: TDOA only
    # VLS: TDOA + vr
    # ============================================================
    p_vls = run_vls_tdoa_series(t, beacons, drho_hats, ref_idx, p_init, sigma_drho)

    xs_vls_dopp = run_vls_tdoa_vr_series(
        t, beacons, drho_hats, vr_hats, ref_idx,
        x_init=x_init,
        sigma_drho=sigma_drho,
        sigma_vr=sigma_vr,
        full_doppler_jacobian=full_jac
    )
    p_vls_dopp = xs_vls_dopp[:, 0:3]
    v_vls_dopp = xs_vls_dopp[:, 3:6]

    # ============================================================
    # EKF: TDOA only
    # EKF: TDOA + vr
    # ============================================================
    xs_ekf, Ps_ekf = ekf_run_tdoa(
        t, dt, beacons, drho_hats, ref_idx,
        sigma_drho=sigma_drho,
        x0=x_init,
        P0=P0,
        q_acc=q_acc
    )
    p_ekf = xs_ekf[:, 0:3]
    v_ekf = xs_ekf[:, 3:6]

    xs_ekf_dopp, Ps_ekf_dopp = ekf_run_tdoa_vr(
        t, dt, beacons, drho_hats, vr_hats, ref_idx,
        sigma_drho=sigma_drho,
        sigma_vr=sigma_vr,
        x0=x_init,
        P0=P0,
        q_acc=q_acc,
        full_doppler_jacobian=full_jac
    )
    p_ekf_dopp = xs_ekf_dopp[:, 0:3]
    v_ekf_dopp = xs_ekf_dopp[:, 3:6]

    # ---- Errors ----
    e_vls = position_errors(p_vls, p_true)
    e_vls_dopp = position_errors(p_vls_dopp, p_true)
    e_ekf = position_errors(p_ekf, p_true)
    e_ekf_dopp = position_errors(p_ekf_dopp, p_true)

    return {
        "t": t,
        "beacons": beacons,
        "ref_idx": ref_idx,

        "p_true": p_true,
        "v_true": v_true,

        "drho_hats": drho_hats,
        "vr_hats": vr_hats,
        "gross_meta": gross_meta,

        # VLS
        "p_vls": p_vls,
        "p_vls_dopp": p_vls_dopp,
        "v_vls_dopp": v_vls_dopp,

        # EKF
        "p_ekf": p_ekf,
        "v_ekf": v_ekf,
        "p_ekf_dopp": p_ekf_dopp,
        "v_ekf_dopp": v_ekf_dopp,

        # errors
        "e_vls": e_vls,
        "e_vls_dopp": e_vls_dopp,
        "e_ekf": e_ekf,
        "e_ekf_dopp": e_ekf_dopp,

        # summaries
        "summary_vls": summarize_errors(e_vls),
        "summary_vls_dopp": summarize_errors(e_vls_dopp),
        "summary_ekf": summarize_errors(e_ekf),
        "summary_ekf_dopp": summarize_errors(e_ekf_dopp),

        # diagnostics
        "noise_diag": {
            "sigma_dt": float(sigma_dt),
            "sigma_drho": float(sigma_drho),
            "full_doppler_jacobian": bool(full_jac),
            "gross_enable": gross_enable,
            "gross_p_tdoa": float(gross_p_tdoa),
            "gross_scale_tdoa": float(gross_scale_tdoa),
            "gross_p_vr": float(gross_p_vr),
            "gross_scale_vr": float(gross_scale_vr),
        },

        "config": config
    }


def run_monte_carlo(config: dict, M: int, seed0: int = 1) -> dict:
    rows = []
    for m in range(M):
        out = run_single_experiment(config, seed=seed0 + m)
        rows.append({
            "run": m + 1,
            **{f"VLS_{k}": v for k, v in out["summary_vls"].items()},
            **{f"VLS_Dopp_{k}": v for k, v in out["summary_vls_dopp"].items()},
            **{f"EKF_{k}": v for k, v in out["summary_ekf"].items()},
            **{f"EKF_Dopp_{k}": v for k, v in out["summary_ekf_dopp"].items()},
        })
    df = pd.DataFrame(rows)
    agg = df.drop(columns=["run"]).agg(["mean", "std", "min", "max"])
    return {"runs": df, "agg": agg}
