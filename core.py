# core.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# Geometry helpers
# ============================================================

def _ranges_u_3d(p3: np.ndarray, beacons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    p3: (3,)
    beacons: (N,3)
    Returns:
      R: (N,) ranges
      u: (N,3) unit vectors from beacon -> receiver: (p-b)/||p-b||
    """
    diff = p3[None, :] - beacons
    R = np.linalg.norm(diff, axis=1)
    R = np.maximum(R, 1e-9)
    u = diff / R[:, None]
    return R, u


# ============================================================
# Scenario helpers
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
        raise ValueError("Unknown preset shape.")

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
    Returns:
      t: (K,)
      p_true: (K,3)
      v_true: (K,3)
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
        Rr = max(10.0, 0.15 * L)

        s = speed * t
        per = 2 * (L + np.pi * Rr)
        s_mod = np.mod(s, per)

        for k in range(K):
            sk = s_mod[k]
            if sk < L:
                p[k] = np.array([start[0] - L / 2 + sk, start[1] - Rr, z])
                v[k] = np.array([speed, 0.0, 0.0])
            elif sk < L + np.pi * Rr:
                phi = (sk - L) / Rr
                cx, cy = start[0] + L / 2, start[1]
                p[k] = np.array(
                    [cx + Rr * np.cos(-np.pi / 2 + phi),
                     cy + Rr * np.sin(-np.pi / 2 + phi), z]
                )
                v[k] = speed * np.array(
                    [-np.sin(-np.pi / 2 + phi), np.cos(-np.pi / 2 + phi), 0.0]
                )
            elif sk < 2 * L + np.pi * Rr:
                s2 = sk - (L + np.pi * Rr)
                p[k] = np.array([start[0] + L / 2 - s2, start[1] + Rr, z])
                v[k] = np.array([-speed, 0.0, 0.0])
            else:
                phi = (sk - (2 * L + np.pi * Rr)) / Rr
                cx, cy = start[0] - L / 2, start[1]
                p[k] = np.array(
                    [cx + Rr * np.cos(np.pi / 2 + phi),
                     cy + Rr * np.sin(np.pi / 2 + phi), z]
                )
                v[k] = speed * np.array(
                    [-np.sin(np.pi / 2 + phi), np.cos(np.pi / 2 + phi), 0.0]
                )
    else:
        raise ValueError("Unknown trajectory kind.")

    return t, p, v


# ============================================================
# Measurements: TDOA (meters) + Doppler (vr OR df) + outliers
# ============================================================

def simulate_measurements_tdoa(
    p3: np.ndarray,
    v3: np.ndarray,
    beacons: np.ndarray,
    c: float,
    sigma_tdoa: float,
    rng: np.random.Generator,
    ref_idx: int = 0,
    gross_enabled: bool = False,
    p_gross: float = 0.0,
    gross_R_m: float = 10.0,
    # Doppler handling:
    doppler_mode: str = "vr",     # "vr" or "df"
    f0: float = 25000.0,          # Hz (used when doppler_mode="df")
    sigma_vr: float = 0.05,       # m/s  (used when doppler_mode="vr" OR as fallback)
    sigma_df_hz: float | None = None,  # Hz (used when doppler_mode="df"; if None -> derived)
    gross_vr_mps: float = 1.0,
    gross_df_hz: float = 30.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Returns:
      dR_true: (N-1,)   true TDOA in meters: dR_i = R_i - R_ref
      dR_hat:  (N-1,)   noisy + (optional) gross

      vr_true: (N,)     true radial velocity u_i^T v  [m/s]
      vr_hat:  (N,)     observed radial velocity      [m/s] (always provided for estimators)

      df_true: (N,)     true doppler shift            [Hz] (zeros if mode="vr")
      df_hat:  (N,)     observed doppler shift        [Hz] (zeros if mode="vr")

      sigma_vr_eff: effective std of vr_hat           [m/s] (depends on f0 in mode="df")
      sigma_df_used: std used for df noise            [Hz]  (0 in mode="vr")
    """
    doppler_mode = (doppler_mode or "vr").lower().strip()
    if doppler_mode not in ("vr", "df"):
        doppler_mode = "vr"

    R_true, u = _ranges_u_3d(p3, beacons)
    vr_true = (u @ v3)

    N = beacons.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[ref_idx] = False

    dR_true = (R_true - R_true[ref_idx])[mask]
    dR_hat = dR_true + c * rng.normal(0.0, sigma_tdoa, size=dR_true.shape)

    # Defaults for df outputs
    df_true = np.zeros_like(vr_true)
    df_hat = np.zeros_like(vr_true)
    sigma_df_used = 0.0
    sigma_vr_eff = float(sigma_vr)

    if doppler_mode == "vr":
        # direct vr observation
        vr_hat = vr_true + rng.normal(0.0, sigma_vr, size=vr_true.shape)
        sigma_vr_eff = float(sigma_vr)
        sigma_df_used = 0.0

        if gross_enabled and p_gross > 0.0:
            out_vr = rng.random(size=vr_hat.shape) < p_gross
            if np.any(out_vr):
                vr_hat[out_vr] += rng.normal(0.0, gross_vr_mps, size=int(np.sum(out_vr)))

    else:
        # doppler measured as df (Hz), then converted to vr
        f0 = float(max(f0, 1e-9))
        df_true = (f0 / c) * vr_true

        if sigma_df_hz is None:
            # derive sigma_df from sigma_vr (so user can keep old slider and still get df model)
            sigma_df_used = (f0 / c) * float(sigma_vr)
        else:
            sigma_df_used = float(sigma_df_hz)

        df_hat = df_true + rng.normal(0.0, sigma_df_used, size=df_true.shape)

        if gross_enabled and p_gross > 0.0:
            out_df = rng.random(size=df_hat.shape) < p_gross
            if np.any(out_df):
                df_hat[out_df] += rng.normal(0.0, gross_df_hz, size=int(np.sum(out_df)))

        # convert to vr observation used by solvers
        vr_hat = (c / f0) * df_hat
        sigma_vr_eff = (c / f0) * sigma_df_used

    # TDOA gross errors (common for both modes)
    if gross_enabled and p_gross > 0.0:
        out_dR = rng.random(size=dR_hat.shape) < p_gross
        if np.any(out_dR):
            dR_hat[out_dR] += rng.normal(0.0, gross_R_m, size=int(np.sum(out_dR)))

    return dR_true, dR_hat, vr_true, vr_hat, df_true, df_hat, float(sigma_vr_eff), float(sigma_df_used)


# ============================================================
# VLS (2D): TDOA only (stable: LM + step limit)
# ============================================================

def wls_xy_from_tdoa(
    dR_hat: np.ndarray,
    beacons: np.ndarray,
    xy0: np.ndarray,
    z_known: float,
    ref_idx: int = 0,
    iters: int = 15,
    step_max: float = 20.0,
    lm_lambda: float = 1e-2
) -> np.ndarray:
    xy = xy0.astype(float).copy()

    N = beacons.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[ref_idx] = False

    for _ in range(iters):
        p3 = np.array([xy[0], xy[1], z_known], dtype=float)
        R, u = _ranges_u_3d(p3, beacons)

        dR_pred = (R - R[ref_idx])[mask]
        r = dR_hat - dR_pred

        Hxy = (u[mask] - u[ref_idx])[:, 0:2]

        A = Hxy.T @ Hxy + lm_lambda * np.eye(2)
        b = Hxy.T @ r
        dxy = np.linalg.solve(A, b)

        n = float(np.linalg.norm(dxy))
        if n > step_max:
            dxy *= (step_max / n)

        xy = xy + dxy
        if np.linalg.norm(dxy) < 1e-6:
            break

    return xy


def run_vls_tdoa_series(
    t: np.ndarray,
    beacons: np.ndarray,
    dR_hats: np.ndarray,
    p_init: np.ndarray,
    z_known: float,
    ref_idx: int = 0
) -> np.ndarray:
    K = len(t)
    p_est = np.zeros((K, 3), dtype=float)

    xy_prev = p_init[0:2].astype(float).copy()
    for k in range(K):
        xy_prev = wls_xy_from_tdoa(dR_hats[k], beacons, xy_prev, z_known=z_known, ref_idx=ref_idx)
        p_est[k] = np.array([xy_prev[0], xy_prev[1], z_known], dtype=float)

    return p_est


# ============================================================
# VLS (2D): TDOA + vr on state [x,y,vx,vy]
# ============================================================

def wls_xv_from_tdoa_vr(
    dR_hat: np.ndarray,
    vr_hat: np.ndarray,
    beacons: np.ndarray,
    x0: np.ndarray,       # [x,y,vx,vy]
    z_known: float,
    ref_idx: int = 0,
    iters: int = 15,
    sigma_dR: float = 0.15,
    sigma_vr: float = 0.05,
    step_pos_max: float = 20.0,
    step_vel_max: float = 2.0,
    lm_lambda: float = 1e-2
) -> np.ndarray:
    x = x0.astype(float).copy()
    N = beacons.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[ref_idx] = False

    inv_sdR = 1.0 / float(max(sigma_dR, 1e-12))
    inv_svr = 1.0 / float(max(sigma_vr, 1e-12))

    for _ in range(iters):
        xy = x[0:2]
        vxy = x[2:4]

        p3 = np.array([xy[0], xy[1], z_known], dtype=float)
        v3 = np.array([vxy[0], vxy[1], 0.0], dtype=float)

        R, u = _ranges_u_3d(p3, beacons)

        dR_pred = (R - R[ref_idx])[mask]
        vr_pred = (u @ v3)

        r_dR = (dR_hat - dR_pred) * inv_sdR
        r_vr = (vr_hat - vr_pred) * inv_svr
        r = np.concatenate([r_dR, r_vr])

        H = np.zeros(((N - 1) + N, 4), dtype=float)

        H[0:(N-1), 0:2] = (u[mask] - u[ref_idx])[:, 0:2] * inv_sdR
        H[(N-1):, 2:4] = u[:, 0:2] * inv_svr

        for i in range(N):
            ui = u[i]
            Ri = R[i]
            v_perp = v3 - ui * float(ui @ v3)
            dvr_dp = (v_perp / Ri)
            H[(N-1) + i, 0:2] = dvr_dp[0:2] * inv_svr

        A = H.T @ H + lm_lambda * np.eye(4)
        b = H.T @ r
        dx = np.linalg.solve(A, b)

        dpos = dx[0:2]
        dvel = dx[2:4]

        npn = float(np.linalg.norm(dpos))
        if npn > step_pos_max:
            dpos *= (step_pos_max / npn)

        nvn = float(np.linalg.norm(dvel))
        if nvn > step_vel_max:
            dvel *= (step_vel_max / nvn)

        dx_lim = np.concatenate([dpos, dvel])
        x = x + dx_lim

        if np.linalg.norm(dx_lim) < 1e-6:
            break

    return x


def run_vls_tdoa_vr_series(
    t: np.ndarray,
    beacons: np.ndarray,
    dR_hats: np.ndarray,
    vr_hats: np.ndarray,
    x_init: np.ndarray,       # [x,y,vx,vy]
    z_known: float,
    ref_idx: int = 0,
    sigma_dR: float = 0.15,
    sigma_vr: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    K = len(t)
    p_est = np.zeros((K, 3), dtype=float)
    v_est = np.zeros((K, 3), dtype=float)

    x_prev = x_init.astype(float).copy()
    for k in range(K):
        x_prev = wls_xv_from_tdoa_vr(
            dR_hats[k], vr_hats[k], beacons,
            x_prev, z_known=z_known, ref_idx=ref_idx,
            sigma_dR=sigma_dR, sigma_vr=sigma_vr
        )
        p_est[k] = np.array([x_prev[0], x_prev[1], z_known], dtype=float)
        v_est[k] = np.array([x_prev[2], x_prev[3], 0.0], dtype=float)

    return p_est, v_est


# ============================================================
# EKF (2D): state [x,y,vx,vy] with TDOA + optional vr, robust update
# ============================================================

def ekf_run_tdoa_2d(
    t: np.ndarray,
    dt: float,
    beacons: np.ndarray,
    dR_hats: np.ndarray,
    vr_hats: np.ndarray,
    sigma_dR: float,
    sigma_vr: float,
    x0: np.ndarray,
    P0: np.ndarray,
    z_known: float,
    ref_idx: int = 0,
    use_doppler: bool = True,
    q_acc: float = 0.05,
    robust_k: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    K = len(t)
    N = beacons.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[ref_idx] = False

    M = (N - 1) + (N if use_doppler else 0)

    x = x0.astype(float).copy()
    P = P0.astype(float).copy()

    xs = np.zeros((K, 4), dtype=float)
    Ps = np.zeros((K, 4, 4), dtype=float)

    F = np.eye(4, dtype=float)
    F[0, 2] = dt
    F[1, 3] = dt

    G = np.zeros((4, 2), dtype=float)
    G[0:2, :] = 0.5 * dt**2 * np.eye(2)
    G[2:4, :] = dt * np.eye(2)
    Q = (q_acc**2) * (G @ G.T)

    base_var = np.concatenate([
        np.full(N - 1, sigma_dR**2),
        np.full(N, sigma_vr**2) if use_doppler else np.array([], dtype=float),
    ])
    base_sigma = np.sqrt(base_var)

    for k in range(K):
        x = F @ x
        P = F @ P @ F.T + Q

        xy = x[0:2]
        vxy = x[2:4]
        p3 = np.array([xy[0], xy[1], z_known], dtype=float)
        v3 = np.array([vxy[0], vxy[1], 0.0], dtype=float)

        R, u = _ranges_u_3d(p3, beacons)
        dR_pred = (R - R[ref_idx])[mask]
        vr_pred = (u @ v3)

        if use_doppler:
            z = np.concatenate([dR_hats[k], vr_hats[k]])
            z_pred = np.concatenate([dR_pred, vr_pred])
        else:
            z = dR_hats[k].copy()
            z_pred = dR_pred.copy()

        y = z - z_pred

        H = np.zeros((M, 4), dtype=float)
        H[0:(N-1), 0:2] = (u[mask] - u[ref_idx])[:, 0:2]

        if use_doppler:
            H[(N-1):, 2:4] = u[:, 0:2]
            for i in range(N):
                ui = u[i]
                Ri = R[i]
                v_perp = v3 - ui * float(ui @ v3)
                dvr_dp = (v_perp / Ri)
                H[(N-1) + i, 0:2] = dvr_dp[0:2]

        scale = np.ones_like(base_sigma)
        big = np.abs(y) > (robust_k * base_sigma)
        if np.any(big):
            scale[big] = (np.abs(y[big]) / (robust_k * base_sigma[big]))**2

        Rm = np.diag(base_var * scale)

        S = H @ P @ H.T + Rm
        Kk = P @ H.T @ np.linalg.inv(S)

        x = x + Kk @ y
        P = (np.eye(4) - Kk @ H) @ P

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
# Runner
# ============================================================

def run_single_experiment(config: dict, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)

    beacons = np.array(config["beacons"], dtype=float)

    # acoustics
    c = float(config["acoustics"]["c"])
    f0 = float(config["acoustics"].get("f0", 25000.0))

    # doppler mode
    doppler_cfg = config.get("doppler", {}) or {}
    doppler_mode = str(doppler_cfg.get("mode", "vr")).lower().strip()  # "vr" or "df"

    # noises
    sigma_tdoa = float(config["noise"]["sigma_tdoa"])
    sigma_vr = float(config["noise"]["sigma_vr"])
    sigma_df_hz = config["noise"].get("sigma_df_hz", None)
    if sigma_df_hz is not None:
        sigma_df_hz = float(sigma_df_hz)

    gross = config.get("gross", {}) or {}
    gross_enabled = bool(gross.get("enabled", False))
    p_gross = float(gross.get("p_gross", 0.0))
    gross_R_m = float(gross.get("gross_R_m", 10.0))
    gross_vr_mps = float(gross.get("gross_vr_mps", 1.0))
    gross_df_hz = float(gross.get("gross_df_hz", 30.0))

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
    if N < 4:
        raise ValueError("Dla stabilnych wyników TDOA zalecane jest N>=4 (np. kwadrat/pieciokat).")

    # fixed hidden reference index
    ref_idx = 0

    dR_true = np.zeros((K, N - 1), dtype=float)
    dR_hats = np.zeros((K, N - 1), dtype=float)

    vr_true = np.zeros((K, N), dtype=float)
    vr_hats = np.zeros((K, N), dtype=float)

    df_true = np.zeros((K, N), dtype=float)
    df_hats = np.zeros((K, N), dtype=float)

    sigma_vr_eff = np.zeros(K, dtype=float)
    sigma_df_used = np.zeros(K, dtype=float)

    for k in range(K):
        (
            dR_true[k], dR_hats[k],
            vr_true[k], vr_hats[k],
            df_true[k], df_hats[k],
            sigma_vr_eff[k], sigma_df_used[k]
        ) = simulate_measurements_tdoa(
            p_true[k], v_true[k],
            beacons=beacons,
            c=c,
            sigma_tdoa=sigma_tdoa,
            rng=rng,
            ref_idx=ref_idx,
            gross_enabled=gross_enabled,
            p_gross=p_gross,
            gross_R_m=gross_R_m,
            doppler_mode=doppler_mode,
            f0=f0,
            sigma_vr=sigma_vr,
            sigma_df_hz=sigma_df_hz,
            gross_vr_mps=gross_vr_mps,
            gross_df_hz=gross_df_hz
        )

    sigma_dR = c * sigma_tdoa  # meters

    # IMPORTANT:
    # estimators use vr_hats, so in mode="df" we should use sigma_vr_eff (depends on f0)
    # use average across time (they're constant unless you do something dynamic)
    sigma_vr_for_solvers = float(np.mean(sigma_vr_eff)) if doppler_mode == "df" else float(sigma_vr)

    # initial guess
    p_init = p_true[0].copy()
    p_init[0:2] += np.array([5.0, -5.0], dtype=float)
    p_init[2] = z

    x_init_vls = np.array([p_init[0], p_init[1], 0.0, 0.0], dtype=float)  # [x,y,vx,vy]

    # -------- VLS --------
    p_vls_noD = run_vls_tdoa_series(t, beacons, dR_hats, p_init, z_known=z, ref_idx=ref_idx)
    p_vls_D, v_vls_D = run_vls_tdoa_vr_series(
        t, beacons, dR_hats, vr_hats,
        x_init=x_init_vls,
        z_known=z,
        ref_idx=ref_idx,
        sigma_dR=sigma_dR,
        sigma_vr=sigma_vr_for_solvers
    )

    # vr predicted from VLS+D
    vr_pred_vlsD = np.zeros_like(vr_hats)
    for k in range(K):
        xy = p_vls_D[k, 0:2]
        vxy = v_vls_D[k, 0:2]
        p3 = np.array([xy[0], xy[1], z], dtype=float)
        v3 = np.array([vxy[0], vxy[1], 0.0], dtype=float)
        _, u = _ranges_u_3d(p3, beacons)
        vr_pred_vlsD[k] = (u @ v3)

    # -------- EKF --------
    q_acc = float(config.get("filter", {}).get("q_acc", 0.05))
    robust_k = float(config.get("filter", {}).get("robust_k", 3.0))

    x0 = x_init_vls.copy()
    P0 = np.diag([25.0, 25.0, 4.0, 4.0]).astype(float)

    xs_ekf_noD, _ = ekf_run_tdoa_2d(
        t, dt, beacons, dR_hats, vr_hats,
        sigma_dR=sigma_dR,
        sigma_vr=sigma_vr_for_solvers,
        x0=x0, P0=P0,
        z_known=z,
        ref_idx=ref_idx,
        use_doppler=False,
        q_acc=q_acc,
        robust_k=robust_k
    )

    xs_ekf_D, _ = ekf_run_tdoa_2d(
        t, dt, beacons, dR_hats, vr_hats,
        sigma_dR=sigma_dR,
        sigma_vr=sigma_vr_for_solvers,
        x0=x0, P0=P0,
        z_known=z,
        ref_idx=ref_idx,
        use_doppler=True,
        q_acc=q_acc,
        robust_k=robust_k
    )

    p_ekf_noD = np.column_stack([xs_ekf_noD[:, 0], xs_ekf_noD[:, 1], np.full(K, z)])
    p_ekf_D = np.column_stack([xs_ekf_D[:, 0], xs_ekf_D[:, 1], np.full(K, z)])

    v_ekf_D = np.column_stack([xs_ekf_D[:, 2], xs_ekf_D[:, 3], np.zeros(K)])

    # vr predicted from EKF+D
    vr_pred_ekfD = np.zeros_like(vr_hats)
    for k in range(K):
        xy = p_ekf_D[k, 0:2]
        vxy = v_ekf_D[k, 0:2]
        p3 = np.array([xy[0], xy[1], z], dtype=float)
        v3 = np.array([vxy[0], vxy[1], 0.0], dtype=float)
        _, u = _ranges_u_3d(p3, beacons)
        vr_pred_ekfD[k] = (u @ v3)

    # -------- errors --------
    e_vls_noD = position_errors(p_vls_noD, p_true)
    e_vls_D = position_errors(p_vls_D, p_true)
    e_ekf_noD = position_errors(p_ekf_noD, p_true)
    e_ekf_D = position_errors(p_ekf_D, p_true)

    return {
        "t": t,
        "beacons": beacons,
        "p_true": p_true,
        "v_true": v_true,

        "dR_true": dR_true,
        "dR_hats": dR_hats,

        # always provide vr (solvers + app)
        "vr_true": vr_true,
        "vr_hats": vr_hats,

        # optional extra diagnostics (helps show why f0 matters)
        "df_true": df_true,
        "df_hats": df_hats,
        "sigma_vr_eff": sigma_vr_eff,     # [m/s]
        "sigma_df_used": sigma_df_used,   # [Hz]
        "doppler_mode": doppler_mode,
        "f0": f0,

        "p_vls_noD": p_vls_noD,
        "p_vls_D": p_vls_D,
        "v_vls_D": v_vls_D,
        "vr_pred_vlsD": vr_pred_vlsD,

        "p_ekf_noD": p_ekf_noD,
        "p_ekf_D": p_ekf_D,
        "v_ekf_D": v_ekf_D,
        "vr_pred_ekfD": vr_pred_ekfD,

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
