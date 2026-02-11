# core.py
from __future__ import annotations
import json
import numpy as np
import pandas as pd

# -----------------------------
# Scenario helpers
# -----------------------------

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


def make_trajectory(kind: str, T: float, dt: float,
                    speed: float, heading_deg: float,
                    start: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        # parametry toru:
        L = 0.6 * speed * T  # długość "długiej" osi (heurystyka)
        R = max(10.0, 0.15 * L)
        # ruch po torze z prędkością stałą - parametryzacja po długości łuku
        s = speed * t
        per = 2 * (L + np.pi * R)
        s_mod = np.mod(s, per)

        # odcinki: prosta1 (L), półokrąg (pi R), prosta2 (L), półokrąg (pi R)
        for k in range(K):
            sk = s_mod[k]
            if sk < L:
                # prosta1: wzdłuż osi x
                p[k] = np.array([start[0] - L/2 + sk, start[1] - R, z])
                v[k] = np.array([speed, 0.0, 0.0])
            elif sk < L + np.pi * R:
                # półokrąg prawy: od dołu do góry
                phi = (sk - L) / R  # 0..pi
                cx, cy = start[0] + L/2, start[1]
                p[k] = np.array([cx + R*np.cos(-np.pi/2 + phi), cy + R*np.sin(-np.pi/2 + phi), z])
                v[k] = speed * np.array([-np.sin(-np.pi/2 + phi), np.cos(-np.pi/2 + phi), 0.0])
            elif sk < 2*L + np.pi * R:
                # prosta2: wzdłuż -x
                s2 = sk - (L + np.pi*R)
                p[k] = np.array([start[0] + L/2 - s2, start[1] + R, z])
                v[k] = np.array([-speed, 0.0, 0.0])
            else:
                # półokrąg lewy: od góry do dołu
                phi = (sk - (2*L + np.pi*R)) / R  # 0..pi
                cx, cy = start[0] - L/2, start[1]
                p[k] = np.array([cx + R*np.cos(np.pi/2 + phi), cy + R*np.sin(np.pi/2 + phi), z])
                v[k] = speed * np.array([-np.sin(np.pi/2 + phi), np.cos(np.pi/2 + phi), 0.0])
    else:
        raise ValueError("Unknown trajectory kind")

    return t, p, v


# -----------------------------
# Measurement models
# -----------------------------

def _ranges_and_u(p: np.ndarray, beacons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = p[None, :] - beacons  # (N,3)
    R = np.linalg.norm(diff, axis=1)  # (N,)
    u = diff / R[:, None]
    return R, u


def simulate_measurements_tdoa(p: np.ndarray, v: np.ndarray, beacons: np.ndarray,
                               c: float,
                               sigma_tdoa: float, sigma_vr: float,
                               rng: np.random.Generator,
                               ref_idx: int = 0,
                               p_gross: float = 0.0,
                               gross_R_m: float = 10.0,
                               gross_vr_mps: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Zwraca obserwacje:
      dR_hat (N-1,)   - pomiar TDOA przeskalowany na metry: dR_i = R_i - R_ref
      vr_hat (N,)     - obserwacja prędkości radialnej dla każdego beacona
    """
    R_true, u = _ranges_and_u(p, beacons)     # (N,), (N,3)
    vr_true = (u @ v)                         # (N,)

    # TDOA: dR = (R_i - R_ref) + szum
    dR_true = R_true - R_true[ref_idx]
    # generujemy tylko N-1 (pomijamy ref)
    mask = np.ones(len(R_true), dtype=bool)
    mask[ref_idx] = False
    dR_true = dR_true[mask]

    # szum TDOA w sekundach -> metry
    dR_hat = dR_true + c * rng.normal(0.0, sigma_tdoa, size=dR_true.shape)

    # Doppler jako vr
    vr_hat = vr_true + rng.normal(0.0, sigma_vr, size=vr_true.shape)

    # Błędy grube (outliery)
    if p_gross > 0.0:
        # dla dR (N-1)
        out_dR = rng.random(size=dR_hat.shape) < p_gross
        dR_hat[out_dR] += rng.normal(0.0, gross_R_m, size=np.sum(out_dR))

        # dla vr (N)
        out_vr = rng.random(size=vr_hat.shape) < p_gross
        vr_hat[out_vr] += rng.normal(0.0, gross_vr_mps, size=np.sum(out_vr))

    return dR_hat, vr_hat


def wls_position_from_ranges(R_hat: np.ndarray, beacons: np.ndarray,
                            p0: np.ndarray,
                            iters: int = 15) -> np.ndarray:
    """
    Gauss-Newton on: minimize sum (R_hat_i - ||p - p_i||)^2
    """
    p = p0.astype(float).copy()
    for _ in range(iters):
        diff = p[None, :] - beacons
        R = np.linalg.norm(diff, axis=1)
        # Avoid division by 0
        R = np.maximum(R, 1e-9)
        H = diff / R[:, None]     # dR/dp
        r = (R_hat - R)

        # Solve H dp = r (least squares)
        dp, *_ = np.linalg.lstsq(H, r, rcond=None)
        p = p + dp
        if np.linalg.norm(dp) < 1e-6:
            break
    return p


def run_lbl_wls_series(t: np.ndarray, beacons: np.ndarray,
                       R_hats: np.ndarray,
                       p_init: np.ndarray) -> np.ndarray:
    """
    Estimates position at each epoch independently using WLS on ranges.
    """
    K = len(t)
    p_est = np.zeros((K, 3), dtype=float)
    p_prev = p_init.astype(float).copy()
    for k in range(K):
        p_prev = wls_position_from_ranges(R_hats[k], beacons, p_prev)
        p_est[k] = p_prev
    return p_est


# -----------------------------
# Estimator B: EKF with ranges + radial velocity
# State: x = [p(3), v(3)]^T
# -----------------------------

def ekf_run(t: np.ndarray, dt: float,
            beacons: np.ndarray,
            R_hats: np.ndarray,
            vr_hats: np.ndarray,
            sigma_R: float, sigma_vr: float,
            x0: np.ndarray, P0: np.ndarray,
            q_acc: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    EKF with constant velocity model + process noise as accel random walk.
    q_acc: accel noise std [m/s^2] (tuning)
    """
    K = len(t)
    N = beacons.shape[0]
    x = x0.astype(float).copy()  # (6,)
    P = P0.astype(float).copy()  # (6,6)

    xs = np.zeros((K, 6), dtype=float)
    Ps = np.zeros((K, 6, 6), dtype=float)

    # State transition
    F = np.eye(6, dtype=float)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    # Process noise (discretized constant-velocity with accel noise)
    # p = p + v dt + 0.5 a dt^2
    # v = v + a dt
    G = np.zeros((6, 3), dtype=float)
    G[0:3, :] = 0.5 * dt**2 * np.eye(3)
    G[3:6, :] = dt * np.eye(3)
    Q = (q_acc**2) * (G @ G.T)

    # Measurement noise
    Rm = np.diag(np.concatenate([np.full(N, sigma_R**2), np.full(N, sigma_vr**2)])).astype(float)

    for k in range(K):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update with measurement z = [R_1..R_N, vr_1..vr_N]
        p = x[0:3]
        v = x[3:6]

        R_pred, u = _ranges_and_u(p, beacons)  # (N,), (N,3)
        vr_pred = (u @ v)                      # (N,)

        z_pred = np.concatenate([R_pred, vr_pred])  # (2N,)
        z = np.concatenate([R_hats[k], vr_hats[k]])

        # Jacobian H (2N x 6)
        # dR_i/dp = u_i^T ; dR_i/dv = 0
        # dvr_i/dv = u_i^T
        # dvr_i/dp is non-linear via u_i(p). We approximate it by ignoring du/dp (common practical simplification),
        # or you can include it later for higher fidelity.
        H = np.zeros((2*N, 6), dtype=float)
        H[0:N, 0:3] = u
        H[0:N, 3:6] = 0.0
        H[N:2*N, 0:3] = 0.0
        H[N:2*N, 3:6] = u

        y = z - z_pred
        S = H @ P @ H.T + Rm
        Kk = P @ H.T @ np.linalg.inv(S)

        x = x + Kk @ y
        P = (np.eye(6) - Kk @ H) @ P

        xs[k] = x
        Ps[k] = P

    return xs, Ps


# -----------------------------
# Metrics + runners
# -----------------------------

def position_errors(p_est: np.ndarray, p_true: np.ndarray) -> np.ndarray:
    e = np.linalg.norm(p_est - p_true, axis=1)
    return e


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

    # Unpack config
    beacons = np.array(config["beacons"], dtype=float)
    c = float(config["acoustics"]["c"])
    f0 = float(config["acoustics"]["f0"])
    sigma_t = float(config["noise"]["sigma_t"])
    sigma_vr = float(config["noise"]["sigma_vr"])

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
    R_hats = np.zeros((K, N), dtype=float)
    vr_hats = np.zeros((K, N), dtype=float)

    for k in range(K):
        R_hats[k], vr_hats[k] = simulate_measurements(
            p_true[k], v_true[k], beacons, c, f0, sigma_t, sigma_vr, rng
        )

    # Estimator A: WLS on ranges
    p_init = p_true[0] + np.array([5.0, -5.0, 2.0])  # small offset for realism
    p_wls = run_lbl_wls_series(t, beacons, R_hats, p_init)

    # Estimator B: EKF (ranges + vr)
    sigma_R = c * sigma_t
    x0 = np.zeros(6, dtype=float)
    x0[0:3] = p_init
    x0[3:6] = np.array([0.0, 0.0, 0.0])
    P0 = np.diag([25.0, 25.0, 25.0, 4.0, 4.0, 4.0]).astype(float)  # initial covariance

    xs, Ps = ekf_run(t, dt, beacons, R_hats, vr_hats,
                     sigma_R=sigma_R, sigma_vr=sigma_vr,
                     x0=x0, P0=P0,
                     q_acc=float(config["filter"]["q_acc"]))

    p_ekf = xs[:, 0:3]

    # Metrics
    e_wls = position_errors(p_wls, p_true)
    e_ekf = position_errors(p_ekf, p_true)

    return {
        "t": t,
        "beacons": beacons,
        "p_true": p_true,
        "v_true": v_true,
        "R_hats": R_hats,
        "vr_hats": vr_hats,
        "p_wls": p_wls,
        "p_ekf": p_ekf,
        "e_wls": e_wls,
        "e_ekf": e_ekf,
        "summary_wls": summarize_errors(e_wls),
        "summary_ekf": summarize_errors(e_ekf),
        "config": config
    }


def run_monte_carlo(config: dict, M: int, seed0: int = 1) -> dict:
    summaries = []
    for m in range(M):
        out = run_single_experiment(config, seed=seed0 + m)
        summaries.append({
            "run": m + 1,
            **{f"WLS_{k}": v for k, v in out["summary_wls"].items()},
            **{f"EKF_{k}": v for k, v in out["summary_ekf"].items()},
        })
    df = pd.DataFrame(summaries)
    # aggregate
    agg = df.drop(columns=["run"]).agg(["mean", "std", "min", "max"])
    return {"runs": df, "agg": agg}
