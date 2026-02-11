# core.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ===========================================================
# Helpers: geometry
# ============================================================

def _ranges_and_u(p: np.ndarray, beacons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    p: (3,)
    beacons: (N,3)
    Returns:
      R: (N,)      ranges
      u: (N,3)     unit vectors from beacon -> receiver (p - b)/||p-b||
    """
    diff = p[None, :] - beacons
    R = np.linalg.norm(diff, axis=1)
    R = np.maximum(R, 1e-9)
    u = diff / R[:, None]
    return R, u


# ============================================================
# Scenario helpers
# ============================================================

def make_beacons_preset(preset: str, radius: float, z: float = 0.0) -> np.ndarray:
    """Returns beacons positions (N,3) for preset shapes centered at (0,0,z)."""
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
    """
    Returns:
      t: (K,)
      p: (K,3)
      v: (K,3)
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
                p[k] = np.array([cx + R*np.cos(-np.pi/2 + phi),
                                 cy + R*np.sin(-np.pi/2 + phi), z])
                v[k] = speed * np.array([-np.sin(-np.pi/2 + phi),
                                         np.cos(-np.pi/2 + phi), 0.0])
            elif sk < 2*L + np.pi * R:
                s2 = sk - (L + np.pi*R)
                p[k] = np.array([start[0] + L/2 - s2, start[1] + R, z])
                v[k] = np.array([-speed, 0.0, 0.0])
            else:
                phi = (sk - (2*L + np.pi*R)) / R
                cx, cy = start[0] - L/2, start[1]
                p[k] = np.array([cx + R*np.cos(np.pi/2 + phi),
                                 cy + R*np.sin(np.pi/2 + phi), z])
                v[k] = speed * np.array([-np.sin(np.pi/2 + phi),
                                         np.cos(np.pi/2 + phi), 0.0])
    else:
        raise ValueError("Unknown trajectory kind")

    return t, p, v


# ============================================================
# Measurement model: TDOA (meters) + Doppler as vr (m/s) + outliers
# ============================================================

def simulate_measurements_tdoa(
    p: np.ndarray, v: np.ndarray,
    beacons: np.ndarray,
    c: float,
    sigma_tdoa: float,
    sigma_vr: float,
    rng: np.random.Generator,
    ref_idx: int = 0,
    enabled_gross: bool = False,
    p_gross: float = 0.0,
    gross_R_m: float = 10.0,
    gross_vr_mps: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      dR_hat: (N-1,)  where dR_i = (R_i - R_ref) + noise , in meters
      vr_hat: (N,)    radial velocity observation per beacon
    """
    R_true, u = _ranges_and_u(p, beacons)      # (N,), (N,3)
    vr_true = (u @ v)                          # (N,)

    N = beacons.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[ref_idx] = False

    dR_true = (R_true - R_true[ref_idx])[mask]  # (N-1,)
    dR_hat = dR_true + c * rng.normal(0.0, sigma_tdoa, size=dR_true.shape)

    vr_hat = vr_true + rng.normal(0.0, sigma_vr, size=vr_true.shape)

    if enabled_gross and p_gross > 0.0:
        out_dR = rng.random(size=dR_hat.shape) < p_gross
        if np.any(out_dR):
            dR_hat[out_dR] += rng.normal(0.0, gross_R_m, size=int(np.sum(out_dR)))

        out_vr = rng.random(size=vr_hat.shape) < p_gross
        if np.any(out_vr):
            vr_hat[out_vr] += rng.normal(0.0, gross_vr_mps, size=int(np.sum(out_vr)))

    return dR_hat, vr_hat


# ============================================================
# VLS 1: WLS position from TDOA (per epoch)
# ============================================================

def wls_position_from_tdoa(
    dR_hat: np.ndarray,
    beacons: np.ndarray,
    p0: np.ndarray,
    ref_idx: int = 0,
    iters: int = 15
) -> np.ndarray:
    """
    Solve p from dR measurements:
      dR_i = R_i - R_ref
    Gauss-Newton:
      minimize sum (dR_hat_i - (R_i - R_ref))^2
    """
    p = p0.astype(float).copy()
    N = beacons.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[ref_idx] = False

    for _ in range(iters):
        R, u = _ranges_and_u(p, beacons)
        dR_pred = (R - R[ref_idx])[mask]     # (N-1,)
        r = dR_hat - dR_pred                 # (N-1,)

        # Jacobian: d(R_i - R_ref)/dp = u_i - u_ref
        H = (u[mask] - u[ref_idx])           # (N-1,3)

        dp, *_ = np.linalg.lstsq(H, r, rcond=None)
        p = p + dp
        if np.linalg.norm(dp) < 1e-6:
            break

    return p


def run_vls_tdoa_series(
    t: np.ndarray,
    beacons: np.ndarray,
    dR_hats: np.ndarray,
    p_init: np.ndarray,
    ref_idx: int = 0
) -> np.ndarray:
    """Per-epoch VLS from TDOA only."""
    K = len(t)
    p_est = np.zeros((K, 3), dtype=float)
    p_prev = p_init.astype(float).copy()
    for k in range(K):
        p_prev = wls_position_from_tdoa(dR_hats[k], beacons, p_prev, ref_idx=ref_idx)
        p_est[k] = p_prev
    return p_est


# ============================================================
# VLS 2: Weighted LS on state [p,v] from TDOA + vr (per epoch)
# ============================================================

def wls_pv_from_tdoa_vr_weighted(
    dR_hat: np.ndarray,
    vr_hat: np.ndarray,
    beacons: np.ndarray,
    x0: np.ndarray,
    ref_idx: int = 0,
    iters: int = 15,
    sigma_dR: float = 0.15,
    sigma_vr: float = 0.05
) -> np.ndarray:
    """
    Gauss-Newton on x=[p(3), v(3)] with weights:
      sum ((dR_hat - dR_pred)/sigma_dR)^2 + sum((vr_hat - vr_pred)/sigma_vr)^2

    vr_pred = u(p)^T v
    Includes d(vr)/dp term: (I - uu^T) v / R
    """
    x = x0.astype(float).copy()
    N = beacons.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[ref_idx] = False

    inv_sdR = 1.0 / float(max(sigma_dR, 1e-12))
    inv_svr = 1.0 / float(max(sigma_vr, 1e-12))

    for _ in range(iters):
        p = x[0:3]
        v = x[3:6]

        R, u = _ranges_and_u(p, beacons)

        dR_pred = (R - R[ref_idx])[mask]    # (N-1,)
        vr_pred = (u @ v)                   # (N,)

        r_dR = (dR_hat - dR_pred) * inv_sdR
        r_vr = (vr_hat - vr_pred) * inv_svr
        r = np.concatenate([r_dR, r_vr])    # (2N-1,)

        H = np.zeros(((N - 1) + N, 6), dtype=float)

        # d(dR)/dp = u_i - u_ref
        H[0:(N-1), 0:3] = (u[mask] - u[ref_idx]) * inv_sdR

        # d(vr)/dv = u
        H[(N-1):, 3:6] = u * inv_svr

        # d(vr)/dp = v_perp / R
        for i in range(N):
            ui = u[i]
            Ri = R[i]
            v_perp = v - ui * float(ui @ v)
            H[(N-1) + i, 0:3] = (v_perp / Ri) * inv_svr

        dx, *_ = np.linalg.lstsq(H, r, rcond=None)
        x = x + dx
        if np.linalg.norm(dx) < 1e-6:
            break

    return x


def run_vls_tdoa_vr_series(
    t: np.ndarray,
    beacons: np.ndarray,
    dR_hats: np.ndarray,
    vr_hats: np.ndarray,
    x_init: np.ndarray,
    ref_idx: int = 0,
    sigma_dR: float = 0.15,
    sigma_vr: float = 0.05
) -> np.ndarray:
    """Per-epoch weighted VLS on [p,v] using TDOA + vr."""
    K = len(t)
    xs = np.zeros((K, 6), dtype=float)
    x_prev = x_init.astype(float).copy()

    for k in range(K):
        x_prev = wls_pv_from_tdoa_vr_weighted(
            dR_hats[k], vr_hats[k], beacons, x_prev,
            ref_idx=ref_idx, sigma_dR=sigma_dR, sigma_vr=sigma_vr
        )
        xs[k] = x_prev

    return xs


# ============================================================
# EKF: state [p,v] with TDOA, optional vr, robust update
# ============================================================

def ekf_run_tdoa(
    t: np.ndarray,
    dt: float,
    beacons: np.ndarray,
    dR_hats: np.ndarray,
    vr_hats: np.ndarray,
    sigma_dR: float,
    sigma_vr: float,
    x0: np.ndarray,
    P0: np.ndarray,
    q_acc: float = 0.05,
    ref_idx: int = 0,
    use_doppler: bool = True,
    robust_k: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    EKF with CV model.
    Measurements:
      - always TDOA: dR (N-1)
      - optionally Doppler: vr (N)

    robust_k: if |innovation| > robust_k*sigma -> inflate measurement variance.
    """
    K = len(t)
    N = beacons.shape[0]

    mask = np.ones(N, dtype=bool)
    mask[ref_idx] = False

    M = (N - 1) + (N if use_doppler else 0)

    x = x0.astype(float).copy()
    P = P0.astype(float).copy()

    xs = np.zeros((K, 6), dtype=float)
    Ps = np.zeros((K, 6, 6), dtype=float)

    # state transition (CV)
    F = np.eye(6, dtype=float)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    # process noise from accel RW
    G = np.zeros((6, 3), dtype=float)
    G[0:3, :] = 0.5 * dt**2 * np.eye(3)
    G[3:6, :] = dt * np.eye(3)
    Q = (q_acc**2) * (G @ G.T)

    base_var = np.concatenate([
        np.full(N - 1, sigma_dR**2),
        np.full(N, sigma_vr**2) if use_doppler else np.array([], dtype=float),
    ])
    base_sigma = np.sqrt(base_var)

    for k in range(K):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q

        p = x[0:3]
        v = x[3:6]

        R, u = _ranges_and_u(p, beacons)

        dR_pred = (R - R[ref_idx])[mask]  # (N-1,)
        vr_pred = (u @ v)                 # (N,)

        if use_doppler:
            z = np.concatenate([dR_hats[k], vr_hats[k]])
            z_pred = np.concatenate([dR_pred, vr_pred])
        else:
            z = dR_hats[k].copy()
            z_pred = dR_pred.copy()

        y = z - z_pred  # innovation

        # Jacobian
        H = np.zeros((M, 6), dtype=float)

        # d(dR)/dp = u_i - u_ref
        H[0:(N-1), 0:3] = (u[mask] - u[ref_idx])

        if use_doppler:
            # d(vr)/dv = u
            H[(N-1):, 3:6] = u

            # d(vr)/dp = v_perp / R
            for i in range(N):
                ui = u[i]
                Ri = R[i]
                v_perp = v - ui * float(ui @ v)
                H[(N-1) + i, 0:3] = (v_perp / Ri)

        # robust scaling -> inflate variance for large innovations
        scale = np.ones_like(base_sigma)
        big = np.abs(y) > (robust_k * base_sigma)
        if np.any(big):
            scale[big] = (np.abs(y[big]) / (robust_k * base_sigma[big]))**2

        Rm = np.diag(base_var * scale)

        S = H @ P @ H.T + Rm
        Kk = P @ H.T @ np.linalg.inv(S)

        x = x + Kk @ y
        P = (np.eye(6) - Kk @ H) @ P

        xs[k] = x
        Ps[k] = P

    return xs, Ps


# ============================================================
# Metrics
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


# ============================================================
# Main runner
# ============================================================

def run_single_experiment(config: dict, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)

    beacons = np.array(config["beacons"], dtype=float)
    c = float(config["acoustics"]["c"])

    # TDOA + Doppler noises
    sigma_tdoa = float(config["noise"]["sigma_tdoa"])
    sigma_vr = float(config["noise"]["sigma_vr"])

    # ref for TDOA
    ref_idx = int(config.get("tdoa", {}).get("ref_idx", 0))

    # outliers
    gross = config.get("gross", {})
    enabled_gross = bool(gross.get("enabled", False))
    p_gross = float(gross.get("p_gross", 0.0))
    gross_R_m = float(gross.get("gross_R_m", 10.0))
    gross_vr_mps = float(gross.get("gross_vr_mps", 1.0))

    # trajectory
    T = float(config["trajectory"]["T"])
    dt = float(config["trajectory"]["dt"])
    speed = float(config["trajectory"]["speed"])
    heading_deg = float(config["trajectory"]["heading_deg"])
    start_xy = np.array(config["trajectory"]["start_xy"], dtype=float)
    z = float(config["trajectory"]["z"])
    start = np.array([start_xy[0], start_xy[1], z], dtype=float)
    traj_kind = config["trajectory"]["kind"]

    t, p_true, v_true = make_trajectory(traj_kind, T, dt, speed, heading_deg, start, z)

    K = len(t)
    N = beacons.shape[0]

    dR_hats = np.zeros((K, N - 1), dtype=float)
    vr_hats = np.zeros((K, N), dtype=float)

    for k in range(K):
        dR_hats[k], vr_hats[k] = simulate_measurements_tdoa(
            p_true[k], v_true[k],
            beacons=beacons,
            c=c,
            sigma_tdoa=sigma_tdoa,
            sigma_vr=sigma_vr,
            rng=rng,
            ref_idx=ref_idx,
            enabled_gross=enabled_gross,
            p_gross=p_gross,
            gross_R_m=gross_R_m,
            gross_vr_mps=gross_vr_mps
        )

    # initial guesses
    p_init = p_true[0] + np.array([5.0, -5.0, 2.0])
    x_init = np.zeros(6, dtype=float)
    x_init[0:3] = p_init
    x_init[3:6] = np.array([0.0, 0.0, 0.0])

    sigma_dR = c * sigma_tdoa

    # ---------- VLS ----------
    p_vls_noD = run_vls_tdoa_series(t, beacons, dR_hats, p_init, ref_idx=ref_idx)

    xs_vls_D = run_vls_tdoa_vr_series(
        t, beacons,
        dR_hats=dR_hats,
        vr_hats=vr_hats,
        x_init=x_init,
        ref_idx=ref_idx,
        sigma_dR=sigma_dR,
        sigma_vr=sigma_vr
    )
    p_vls_D = xs_vls_D[:, 0:3]
    v_vls_D = xs_vls_D[:, 3:6]

    # ---------- EKF ----------
    q_acc = float(config.get("filter", {}).get("q_acc", 0.05))
    robust_k = float(config.get("filter", {}).get("robust_k", 3.0))

    P0 = np.diag([25.0, 25.0, 25.0, 4.0, 4.0, 4.0]).astype(float)

    xs_ekf_noD, _ = ekf_run_tdoa(
        t, dt, beacons,
        dR_hats=dR_hats,
        vr_hats=vr_hats,
        sigma_dR=sigma_dR,
        sigma_vr=sigma_vr,
        x0=x_init,
        P0=P0,
        q_acc=q_acc,
        ref_idx=ref_idx,
        use_doppler=False,
        robust_k=robust_k
    )
    p_ekf_noD = xs_ekf_noD[:, 0:3]

    xs_ekf_D, _ = ekf_run_tdoa(
        t, dt, beacons,
        dR_hats=dR_hats,
        vr_hats=vr_hats,
        sigma_dR=sigma_dR,
        sigma_vr=sigma_vr,
        x0=x_init,
        P0=P0,
        q_acc=q_acc,
        ref_idx=ref_idx,
        use_doppler=True,
        robust_k=robust_k
    )
    p_ekf_D = xs_ekf_D[:, 0:3]
    v_ekf_D = xs_ekf_D[:, 3:6]

    # ---------- errors ----------
    e_vls_noD = position_errors(p_vls_noD, p_true)
    e_vls_D = position_errors(p_vls_D, p_true)
    e_ekf_noD = position_errors(p_ekf_noD, p_true)
    e_ekf_D = position_errors(p_ekf_D, p_true)

    return {
        "t": t,
        "beacons": beacons,
        "p_true": p_true,
        "v_true": v_true,

        "dR_hats": dR_hats,
        "vr_hats": vr_hats,

        "p_vls_noD": p_vls_noD,
        "p_vls_D": p_vls_D,
        "v_vls_D": v_vls_D,

        "p_ekf_noD": p_ekf_noD,
        "p_ekf_D": p_ekf_D,
        "v_ekf_D": v_ekf_D,

        "e_vls_noD": e_vls_noD,
        "e_vls_D": e_vls_D,
        "e_ekf_noD": e_ekf_noD,
        "e_ekf_D": e_ekf_D,

        "summary_vls_noD": summarize_errors(e_vls_noD),
        "summary_vls_D": summarize_errors(e_vls_D),
        "summary_ekf_noD": summarize_errors(e_ekf_noD),
        "summary_ekf_D": summarize_errors(e_ekf_D),

        "config": config,
    }


def run_monte_carlo(config: dict, M: int, seed0: int = 1) -> dict:
    rows = []
    for m in range(M):
        out = run_single_experiment(config, seed=seed0 + m)
        rows.append({
            "run": m + 1,
            **{f"VLS0_{k}": v for k, v in out["summary_vls_noD"].items()},
            **{f"VLSD_{k}": v for k, v in out["summary_vls_D"].items()},
            **{f"EKF0_{k}": v for k, v in out["summary_ekf_noD"].items()},
            **{f"EKFD_{k}": v for k, v in out["summary_ekf_D"].items()},
        })

    df = pd.DataFrame(rows)
    agg = df.drop(columns=["run"]).agg(["mean", "std", "min", "max"])
    return {"runs": df, "agg": agg}
