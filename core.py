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
      R: (N,)
      u: (N,3)  u_i = (p - b_i)/||p-b_i||
    """
    diff = p3[None, :] - beacons
    R = np.linalg.norm(diff, axis=1)
    R = np.maximum(R, 1e-9)
    u = diff / R[:, None]
    return R, u


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
        Rr = max(10.0, 0.15 * L)

        s = speed * t
        per = 2 * (L + np.pi * Rr)
        s_mod = np.mod(s, per)

        for k in range(K):
            sk = s_mod[k]
            if sk < L:
                p[k] = np.array([start[0] - L/2 + sk, start[1] - Rr, z])
                v[k] = np.array([speed, 0.0, 0.0])
            elif sk < L + np.pi * Rr:
                phi = (sk - L) / Rr
                cx, cy = start[0] + L/2, start[1]
                p[k] = np.array([cx + Rr*np.cos(-np.pi/2 + phi),
                                 cy + Rr*np.sin(-np.pi/2 + phi), z])
                v[k] = speed * np.array([-np.sin(-np.pi/2 + phi),
                                         np.cos(-np.pi/2 + phi), 0.0])
            elif sk < 2*L + np.pi * Rr:
                s2 = sk - (L + np.pi*Rr)
                p[k] = np.array([start[0] + L/2 - s2, start[1] + Rr, z])
                v[k] = np.array([-speed, 0.0, 0.0])
            else:
                phi = (sk - (2*L + np.pi*Rr)) / Rr
                cx, cy = start[0] - L/2, start[1]
                p[k] = np.array([cx + Rr*np.cos(np.pi/2 + phi),
                                 cy + Rr*np.sin(np.pi/2 + phi), z])
                v[k] = speed * np.array([-np.sin(np.pi/2 + phi),
                                         np.cos(np.pi/2 + phi), 0.0])
    else:
        raise ValueError("Unknown trajectory kind")

    return t, p, v


# ============================================================
# Measurements: TDOA (meters) + Doppler as vr (m/s) + gross errors
# ============================================================

def simulate_measurements_tdoa(
    p3: np.ndarray, v3: np.ndarray,
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
    dR_hat: (N-1,)  dR_i = (R_i - R_ref) + noise, in meters
    vr_hat: (N,)    vr_i = u_i^T v + noise
    """
    R_true, u = _ranges_u_3d(p3, beacons)
    vr_true = (u @ v3)

    N = beacons.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[ref_idx] = False

    dR_true = (R_true - R_true[ref_idx])[mask]
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
# VLS (2D): TDOA only, per epoch (stable: LM + step limit)
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
    """
    Estimate xy using TDOA only:
      dR_i = R_i - R_ref
    Use LM damping and step limit for stability.
    """
    xy = xy0.astype(float).copy()
    N = beacons.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[ref_idx] = False

    for _ in range(iters):
        p3 = np.array([xy[0], xy[1], z_known], dtype=float)
        R, u = _ranges_u_3d(p3, beacons)

        dR_pred = (R - R[ref_idx])[mask]
        r = dR_hat - dR_pred  # (N-1,)

        # Jacobian wrt xy: (u_i - u_ref) projected to x,y
        Hxy = (u[mask] - u[ref_idx])[:, 0:2]  # (N-1, 2)

        # LM: (H^T H + λI) dx = H^T r
        A = Hxy.T @ Hxy + lm_lambda * np.eye(2)
        b = Hxy.T @ r
        dxy = np.linalg.solve(A, b)

        # step limit
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
        xy_prev = wls_xy_from_tdoa(
            dR_hats[k], beacons, xy_prev, z_known=z_known, ref_idx=ref_idx
        )
        p_est[k] = np.array([xy_prev[0], xy_prev[1], z_known], dtype=float)

    return p_est


# ============================================================
# VLS (2D): TDOA + vr, per epoch (state [x,y,vx,vy], stable)
# ============================================================

def wls_xv_from_tdoa_vr(
    dR_hat: np.ndarray,
    vr_hat: np.ndarray,
    beacons: np.ndarray,
    x0: np.ndarray,          # [x,y,vx,vy]
    z_known: float,
    ref_idx: int = 0,
    iters: int = 15,
    sigma_dR: float = 0.15,
    sigma_vr: float = 0.05,
    step_pos_max: float = 20.0,
    step_vel_max: float = 2.0,
    lm_lambda: float = 1e-2
) -> np.ndarray:
    """
    Weighted LS on x=[x,y,vx,vy] using:
      - TDOA (N-1)
      - vr   (N)

    Includes d(vr)/d(xy) term (projected to xy).
    Uses LM + step limits.
    """
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

        dR_pred = (R - R[ref_idx])[mask]      # (N-1,)
        vr_pred = (u @ v3)                    # (N,)

        r_dR = (dR_hat - dR_pred) * inv_sdR
        r_vr = (vr_hat - vr_pred) * inv_svr
        r = np.concatenate([r_dR, r_vr])      # (2N-1,)

        H = np.zeros(((N - 1) + N, 4), dtype=float)

        # ---- TDOA: d(dR)/d(xy) = (u_i - u_ref)_{xy}
        H[0:(N-1), 0:2] = (u[mask] - u[ref_idx])[:, 0:2] * inv_sdR

        # ---- vr: d(vr)/d(vxy) = u_{xy}
        H[(N-1):, 2:4] = u[:, 0:2] * inv_svr

        # ---- vr: d(vr)/d(xy) = ((I - u u^T) v)/R projected to xy
        for i in range(N):
            ui = u[i]
            Ri = R[i]
            v_perp = v3 - ui * float(ui @ v3)             # (I - u u^T) v
            dvr_dp = (v_perp / Ri)                        # (3,)
            H[(N-1) + i, 0:2] = dvr_dp[0:2] * inv_svr

        # LM solve: (H^T H + λI) dx = H^T r
        A = H.T @ H + lm_lambda * np.eye(4)
        b = H.T @ r
        dx = np.linalg.solve(A, b)

        # step limits
        dpos = dx[0:2]
        dvel = dx[2:4]

        npn = float(np.linalg.norm(dpos))
        if npn > step_pos_max:
            dpos *= (step_pos_max / npn)

        nvn = float(np.linalg.norm(dvel))
        if nvn > step_vel_max:
            dvel *= (step_vel_max / nvn)

        dx_limited = np.concatenate([dpos, dvel])
        x = x + dx_limited

        if np.linalg.norm(dx_limited) < 1e-6:
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
    """
    Returns:
      p_est: (K,3)
      v_est: (K,3)
    """
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
# EKF (2D): state [x,y,vx,vy], measurements TDOA + optional vr
# ============================================================

def ekf_run_tdoa_2d(
    t: np.ndarray,
    dt: float,
    beacons: np.ndarray,
    dR_hats: np.ndarray,
    vr_hats: np.ndarray,
    sigma_dR: float,
    sigma_vr: float,
    x0: np.ndarray,      # (4,)
    P0: np.ndarray,      # (4,4)
    z_known: float,
    q_acc: float = 0.05,
    ref_idx: int = 0,
    use_doppler: bool = True,
    robust_k: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    EKF in 2D with CV model.
    State: [x, y, vx, vy]

    Measurements:
      TDOA: dR (N-1)
      optional vr: (N,)
    """
    K = len(t)
    N = beacons.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[ref_idx] = False

    M = (N - 1) + (N if use_doppler else 0)

    x = x0.astype(float).copy()
    P = P0.astype(float).copy()

    xs = np.zeros((K, 4), dtype=float)
    Ps = np.zeros((K, 4, 4), dtype=float)

    # F (CV)
    F = np.eye(4, dtype=float)
    F[0, 2] = dt
    F[1, 3] = dt

    # Q from accel RW in 2D
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
        # predict
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

        y = z - z_pred  # innovation

        # H
        H = np.zeros((M, 4), dtype=float)

        # TDOA: d(dR)/d(xy) = (u_i - u_ref)_{xy}
        H[0:(N-1), 0:2] = (u[mask] - u[ref_idx])[:, 0:2]

        if use_doppler:
            # d(vr)/d(vxy) = u_{xy}
            H[(N-1):, 2:4] = u[:, 0:2]

            # d(vr)/d(xy) = ((I - u u^T) v)/R projected to xy
            for i in range(N):
                ui = u[i]
                Ri = R[i]
                v_perp = v3 - ui * float(ui @ v3)
                dvr_dp = v_perp / Ri
                H[(N-1) + i, 0:2] = dvr_dp[0:2]

        # robust variance inflation
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
# Main runner
# ============================================================

def run_single_experiment(config: dict, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)

    beacons = np.array(config["beacons"], dtype=float)
    c = float(config["acoustics"]["c"])

    # noises
    sigma_tdoa = float(config["noise"]["sigma_tdoa"])
    sigma_vr = float(config["noise"]["sigma_vr"])

    # internal reference index (can be hidden from UI)
    ref_idx = int(config.get("tdoa", {}).get("ref_idx", 0))

    # gross errors
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
    z = float(config["trajectory"]["z"])  # known depth
    start = np.array([start_xy[0], start_xy[1], z], dtype=float)
    traj_kind = config["trajectory"]["kind"]

    t, p_true, v_true = make_trajectory(traj_kind, T, dt, speed, heading_deg, start, z)

    K = len(t)
    N = beacons.shape[0]
    if N < 4:
        raise ValueError("Dla TDOA w 2D zalecane jest N>=4 (np. kwadrat/pieciokat).")

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

    sigma_dR = c * sigma_tdoa  # meters

    # initial guesses (keep z fixed!)
    p_init = p_true[0].copy()
    p_init[0:2] += np.array([5.0, -5.0], dtype=float)
    p_init[2] = z

    x_init_vls = np.array([p_init[0], p_init[1], 0.0, 0.0], dtype=float)  # [x,y,vx,vy]

    # ---------- VLS ----------
    p_vls_noD = run_vls_tdoa_series(
        t, beacons, dR_hats, p_init, z_known=z, ref_idx=ref_idx
    )

    p_vls_D, v_vls_D = run_vls_tdoa_vr_series(
        t, beacons,
        dR_hats=dR_hats,
        vr_hats=vr_hats,
        x_init=x_init_vls,
        z_known=z,
        ref_idx=ref_idx,
        sigma_dR=sigma_dR,
        sigma_vr=sigma_vr
    )

    # ---------- EKF ----------
    q_acc = float(config.get("filter", {}).get("q_acc", 0.05))
    robust_k = float(config.get("filter", {}).get("robust_k", 3.0))

    x0 = x_init_vls.copy()
    P0 = np.diag([25.0, 25.0, 4.0, 4.0]).astype(float)

    xs_ekf_noD, _ = ekf_run_tdoa_2d(
        t, dt, beacons,
        dR_hats=dR_hats,
        vr_hats=vr_hats,
        sigma_dR=sigma_dR,
        sigma_vr=sigma_vr,
        x0=x0, P0=P0,
        z_known=z,
        q_acc=q_acc,
        ref_idx=ref_idx,
        use_doppler=False,
        robust_k=robust_k
    )
    xs_ekf_D, _ = ekf_run_tdoa_2d(
        t, dt, beacons,
        dR_hats=dR_hats,
        vr_hats=vr_hats,
        sigma_dR=sigma_dR,
        sigma_vr=sigma_vr,
        x0=x0, P0=P0,
        z_known=z,
        q_acc=q_acc,
        ref_idx=ref_idx,
        use_doppler=True,
        robust_k=robust_k
    )

    p_ekf_noD = np.column_stack([xs_ekf_noD[:, 0], xs_ekf_noD[:, 1], np.full(K, z)])
    p_ekf_D   = np.column_stack([xs_ekf_D[:, 0],   xs_ekf_D[:, 1],   np.full(K, z)])

    v_ekf_D = np.column_stack([xs_ekf_D[:, 2], xs_ekf_D[:, 3], np.zeros(K)])

    # ---------- errors ----------
    e_vls_noD = position_errors(p_vls_noD, p_true)
    e_vls_D   = position_errors(p_vls_D, p_true)
    e_ekf_noD = position_errors(p_ekf_noD, p_true)
    e_ekf_D   = position_errors(p_ekf_D, p_true)

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
