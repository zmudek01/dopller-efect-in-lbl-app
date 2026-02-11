# core.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ============================================================
# Geometry + trajectory
# ============================================================

def make_beacons_preset(preset: str, radius: float, z: float = 0.0) -> np.ndarray:
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

def make_trajectory(kind: str, T: float, dt: float,
                    speed: float, heading_deg: float,
                    start: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                phi = (sk - L) / R
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
    diff = p[None, :] - beacons  # (N,3)
    R = np.linalg.norm(diff, axis=1)  # (N,)
    R = np.maximum(R, 1e-9)
    u = diff / R[:, None]
    return R, u

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
    Returns:
      drho_hat (N-1,) where drho = c*(t_i - t_ref)  [m]
      vr_hat   (N,)   radial velocity obs           [m/s]
      meta: counts of gross errors in this epoch
    """
    N = beacons.shape[0]
    R_true, u = _ranges_and_u(p, beacons)
    vr_true = (u @ v)  # (N,)

    # one-way TOA with common clock bias b (cancels in TDOA)
    b = rng.normal(0.0, 1e-3)  # bias in seconds (arbitrary, cancels in TDOA)
    toa = R_true / c + b
    toa_hat = toa + rng.normal(0.0, sigma_t, size=N)

    # gross errors on TOA before differencing (realistic "bad timestamp")
    gross_tdoa_ct = 0
    if gross_enable and gross_p_tdoa > 0:
        mask = rng.random(N) < gross_p_tdoa
        if np.any(mask):
            toa_hat[mask] += rng.normal(0.0, gross_scale_tdoa * sigma_t, size=np.sum(mask))
            gross_tdoa_ct = int(np.sum(mask))

    # form TDOA to ref
    idx = np.arange(N)
    idx_oth = idx[idx != ref_idx]
    dt_hat = toa_hat[idx_oth] - toa_hat[ref_idx]
    drho_hat = c * dt_hat  # [m]

    # Doppler observation as vr with noise
    vr_hat = vr_true + rng.normal(0.0, sigma_vr, size=N)

    gross_vr_ct = 0
    if gross_enable and gross_p_vr > 0:
        mask = rng.random(N) < gross_p_vr
        if np.any(mask):
            vr_hat[mask] += rng.normal(0.0, gross_scale_vr * sigma_vr, size=np.sum(mask))
            gross_vr_ct = int(np.sum(mask))

    meta = {"gross_toa_ct": gross_tdoa_ct, "gross_vr_ct": gross_vr_ct}
    return drho_hat, vr_hat, meta

# ============================================================
# VLS / WLS (Gauss-Newton) for:
#   - position only from TDOA
#   - position+velocity from (TDOA + vr)
# ============================================================

def vls_position_from_tdoa(
    drho_hat: np.ndarray,
    beacons: np.ndarray,
    ref_idx: int,
    p0: np.ndarray,
    sigma_drho: float,
    iters: int = 20
) -> np.ndarray:
    """
    Minimize sum_i ((drho_hat_i - (||p-bi||-||p-br||))/sigma_drho)^2
    drho_hat has length N-1 (all except ref_idx).
    """
    p = p0.astype(float).copy()
    N = beacons.shape[0]
    idx = np.arange(N)
    idx_oth = idx[idx != ref_idx]
    br = beacons[ref_idx]

    w = 1.0 / (sigma_drho**2)

    for _ in range(iters):
        # predicted
        Ri, ui = _ranges_and_u(p, beacons)  # (N,), (N,3)
        Rr = Ri[ref_idx]
        ur = ui[ref_idx]

        h = Ri[idx_oth] - Rr  # (N-1,)
        r = drho_hat - h

        # Jacobian: d(Ri-Rr)/dp = ui - ur
        H = (ui[idx_oth] - ur)  # (N-1,3)

        # Weighted LS step
        # Solve (H^T W H) dp = H^T W r
        W = w * np.eye(len(idx_oth))
        A = H.T @ W @ H
        b = H.T @ W @ r
        dp = np.linalg.solve(A + 1e-9*np.eye(3), b)
        p = p + dp

        if np.linalg.norm(dp) < 1e-6:
            break

    return p

def vls_pv_from_tdoa_vr(
    drho_hat: np.ndarray,
    vr_hat: np.ndarray,
    beacons: np.ndarray,
    ref_idx: int,
    x0: np.ndarray,           # [p(3), v(3)]
    sigma_drho: float,
    sigma_vr: float,
    iters: int = 25
) -> np.ndarray:
    """
    Joint WLS for x=[p,v] using:
      - TDOA (N-1): drho = ||p-bi|| - ||p-br||
      - vr (N):    vr = u_i^T v
    Jacobian uses practical simplification: d(vr)/dp ~ 0 (ignoring du/dp).
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
        Ri, ui = _ranges_and_u(p, beacons)
        Rr = Ri[ref_idx]
        ur = ui[ref_idx]

        # residuals
        h_drho = Ri[idx_oth] - Rr
        r_drho = drho_hat - h_drho

        h_vr = ui @ v
        r_vr = vr_hat - h_vr

        r = np.concatenate([r_drho, r_vr])  # ( (N-1)+N, )

        # Jacobian
        H = np.zeros((len(idx_oth) + N, 6), dtype=float)

        # TDOA rows: d/drho wrt p: ui-ur; wrt v: 0
        H[0:len(idx_oth), 0:3] = (ui[idx_oth] - ur)
        H[0:len(idx_oth), 3:6] = 0.0

        # vr rows: d(vr)/dv = ui ; d(vr)/dp ~ 0
        H[len(idx_oth):, 0:3] = 0.0
        H[len(idx_oth):, 3:6] = ui

        # weights
        W = np.diag(
            np.concatenate([
                np.full(len(idx_oth), w_drho),
                np.full(N, w_vr),
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
    drho_hats: np.ndarray,   # (K,N-1)
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
    drho_hats: np.ndarray,   # (K,N-1)
    vr_hats: np.ndarray,     # (K,N)
    ref_idx: int,
    x_init: np.ndarray,      # (6,)
    sigma_drho: float,
    sigma_vr: float
) -> np.ndarray:
    K = len(t)
    xs = np.zeros((K, 6), dtype=float)
    x_prev = x_init.astype(float).copy()
    for k in range(K):
        x_prev = vls_pv_from_tdoa_vr(drho_hats[k], vr_hats[k], beacons, ref_idx, x_prev, sigma_drho, sigma_vr)
        xs[k] = x_prev
    return xs

# ============================================================
# EKF: state x=[p(3), v(3)] with:
#   - TDOA only (N-1)
#   - TDOA + vr (N-1 + N)
# ============================================================

def ekf_run_tdoa(
    t: np.ndarray,
    dt: float,
    beacons: np.ndarray,
    drho_hats: np.ndarray,    # (K,N-1)
    ref_idx: int,
    sigma_drho: float,
    x0: np.ndarray,
    P0: np.ndarray,
    q_acc: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    EKF with z = TDOA range-difference (meters): drho_i = ||p-bi|| - ||p-br||
    """
    K = len(t)
    N = beacons.shape[0]
    idx = np.arange(N)
    idx_oth = idx[idx != ref_idx]

    x = x0.astype(float).copy()
    P = P0.astype(float).copy()
    xs = np.zeros((K, 6), dtype=float)
    Ps = np.zeros((K, 6, 6), dtype=float)

    # motion model
    F = np.eye(6, dtype=float)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    G = np.zeros((6, 3), dtype=float)
    G[0:3, :] = 0.5 * dt**2 * np.eye(3)
    G[3:6, :] = dt * np.eye(3)
    Q = (q_acc**2) * (G @ G.T)

    Rm = (sigma_drho**2) * np.eye(len(idx_oth), dtype=float)

    for k in range(K):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q

        p = x[0:3]
        Ri, ui = _ranges_and_u(p, beacons)
        Rr = Ri[ref_idx]
        ur = ui[ref_idx]

        z_pred = Ri[idx_oth] - Rr
        z = drho_hats[k]
        y = z - z_pred

        # H: d(drho)/dp = ui - ur ; d/drho wrt v = 0
        H = np.zeros((len(idx_oth), 6), dtype=float)
        H[:, 0:3] = (ui[idx_oth] - ur)
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    EKF with z=[drho (N-1), vr (N)]
    Jacobian for vr uses simplification: d(vr)/dp ~ 0, d(vr)/dv = u
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

    # motion model
    F = np.eye(6, dtype=float)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    G = np.zeros((6, 3), dtype=float)
    G[0:3, :] = 0.5 * dt**2 * np.eye(3)
    G[3:6, :] = dt * np.eye(3)
    Q = (q_acc**2) * (G @ G.T)

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

        Ri, ui = _ranges_and_u(p, beacons)
        Rr = Ri[ref_idx]
        ur = ui[ref_idx]

        drho_pred = Ri[idx_oth] - Rr
        vr_pred = ui @ v

        z_pred = np.concatenate([drho_pred, vr_pred])
        z = np.concatenate([drho_hats[k], vr_hats[k]])
        y = z - z_pred

        H = np.zeros((m, 6), dtype=float)

        # drho part
        H[0:len(idx_oth), 0:3] = (ui[idx_oth] - ur)
        H[0:len(idx_oth), 3:6] = 0.0

        # vr part
        H[len(idx_oth):, 0:3] = 0.0
        H[len(idx_oth):, 3:6] = ui

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
    ref_idx = int(config["tdoa"]["ref_idx"])

    gross = config.get("gross", {})
    gross_enable = bool(gross.get("enable", False))
    gross_p_tdoa = float(gross.get("p_tdoa", 0.0))
    gross_scale_tdoa = float(gross.get("scale_tdoa", 10.0))
    gross_p_vr = float(gross.get("p_vr", 0.0))
    gross_scale_vr = float(gross.get("scale_vr", 10.0))

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
    drho_hats = np.zeros((K, N-1), dtype=float)
    vr_hats = np.zeros((K, N), dtype=float)
    gross_meta = []

    for k in range(K):
        drho_hats[k], vr_hats[k], meta = simulate_tdoa_and_vr(
            p_true[k], v_true[k], beacons, c,
            sigma_t=sigma_t, sigma_vr=sigma_vr, rng=rng,
            ref_idx=ref_idx,
            gross_enable=gross_enable,
            gross_p_tdoa=gross_p_tdoa, gross_scale_tdoa=gross_scale_tdoa,
            gross_p_vr=gross_p_vr, gross_scale_vr=gross_scale_vr
        )
        gross_meta.append(meta)

    # ---- Noise scaling for TDOA ----
    sigma_dt = np.sqrt(2.0) * sigma_t
    sigma_drho = c * sigma_dt  # [m]

    # ---- Initial guesses ----
    p_init = p_true[0] + np.array([5.0, -5.0, 2.0])
    x_init = np.zeros(6, dtype=float)
    x_init[0:3] = p_init
    x_init[3:6] = np.array([0.0, 0.0, 0.0])

    P0 = np.diag([25.0, 25.0, 25.0, 4.0, 4.0, 4.0]).astype(float)

    # ============================================================
    # VLS (TDOA only)  -> p_vls
    # VLS (TDOA + vr)  -> x_vls_dopp -> p, v
    # ============================================================
    p_vls = run_vls_tdoa_series(t, beacons, drho_hats, ref_idx, p_init, sigma_drho)

    xs_vls_dopp = run_vls_tdoa_vr_series(
        t, beacons, drho_hats, vr_hats, ref_idx, x_init, sigma_drho, sigma_vr
    )
    p_vls_dopp = xs_vls_dopp[:, 0:3]
    v_vls_dopp = xs_vls_dopp[:, 3:6]

    # ============================================================
    # EKF (TDOA only)  -> xs_ekf
    # EKF (TDOA + vr)  -> xs_ekf_dopp
    # ============================================================
    xs_ekf, Ps_ekf = ekf_run_tdoa(
        t, dt, beacons, drho_hats, ref_idx,
        sigma_drho=sigma_drho, x0=x_init, P0=P0, q_acc=q_acc
    )
    p_ekf = xs_ekf[:, 0:3]
    v_ekf = xs_ekf[:, 3:6]

    xs_ekf_dopp, Ps_ekf_dopp = ekf_run_tdoa_vr(
        t, dt, beacons, drho_hats, vr_hats, ref_idx,
        sigma_drho=sigma_drho, sigma_vr=sigma_vr,
        x0=x_init, P0=P0, q_acc=q_acc
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

        "summary_vls": summarize_errors(e_vls),
        "summary_vls_dopp": summarize_errors(e_vls_dopp),
        "summary_ekf": summarize_errors(e_ekf),
        "summary_ekf_dopp": summarize_errors(e_ekf_dopp),

        "config": config,
        "noise_diag": {
            "sigma_dt": float(sigma_dt),
            "sigma_drho": float(sigma_drho),
            "gross_enable": gross_enable,
            "gross_p_tdoa": gross_p_tdoa,
            "gross_scale_tdoa": gross_scale_tdoa,
            "gross_p_vr": gross_p_vr,
            "gross_scale_vr": gross_scale_vr,
        }
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
