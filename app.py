# app.py
from __future__ import annotations
import io
import json
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core import make_beacons_preset, run_single_experiment, run_monte_carlo

st.set_page_config(page_title="LBL (TDOA) + Doppler – środowisko testowe", layout="wide")

# ============================================================
# Helpers
# ============================================================

def _grid(ax):
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)

def _set_equal(ax):
    ax.set_aspect("equal", adjustable="box")

def _fig(ax_w=7.2, ax_h=4.8):
    return plt.figure(figsize=(ax_w, ax_h), dpi=120)

def vr_from_pv(p: np.ndarray, v: np.ndarray, beacons: np.ndarray) -> np.ndarray:
    """
    p: (K,3)
    v: (K,3)
    beacons: (N,3)
    Returns vr: (K,N), vr_i = u_i^T v
    """
    diff = p[:, None, :] - beacons[None, :, :]          # (K,N,3)
    R = np.linalg.norm(diff, axis=2)
    R = np.maximum(R, 1e-9)
    u = diff / R[:, :, None]                            # (K,N,3)
    vr = np.sum(u * v[:, None, :], axis=2)              # (K,N)
    return vr

def config_table(config: dict) -> pd.DataFrame:
    rows = []
    b = np.array(config["beacons"], dtype=float)

    rows.append(("Liczba beaconów N", b.shape[0], "-"))
    rows.append(("Beacon referencyjny TDOA", "B1 (stały)", "-"))

    rows.append(("c (prędkość dźwięku)", config["acoustics"]["c"], "m/s"))
    rows.append(("f0 (częstotliwość nośna)", config["acoustics"]["f0"], "Hz"))

    rows.append(("σ_tdoa (TDOA)", config["noise"]["sigma_tdoa"], "s"))
    rows.append(("σ_dR = c·σ_tdoa", config["acoustics"]["c"] * config["noise"]["sigma_tdoa"], "m"))
    rows.append(("σ_vr (Doppler jako v_r)", config["noise"]["sigma_vr"], "m/s"))

    rows.append(("Trajektoria", config["trajectory"]["kind"], "-"))
    rows.append(("T", config["trajectory"]["T"], "s"))
    rows.append(("dt", config["trajectory"]["dt"], "s"))
    rows.append(("Prędkość", config["trajectory"]["speed"], "m/s"))
    rows.append(("Heading", config["trajectory"]["heading_deg"], "deg"))
    rows.append(("Start x", config["trajectory"]["start_xy"][0], "m"))
    rows.append(("Start y", config["trajectory"]["start_xy"][1], "m"))
    rows.append(("Głębokość obiektu z", config["trajectory"]["z"], "m"))

    rows.append(("q_acc (strojenie EKF)", config["filter"]["q_acc"], "m/s²"))
    rows.append(("robust_k (próg)", config["filter"]["robust_k"], "σ"))

    rows.append(("Błędy grube: włączone", "TAK" if config["gross"]["enabled"] else "NIE", "-"))
    rows.append(("p_gross (prawdopodobieństwo)", config["gross"]["p_gross"], "-"))
    rows.append(("gross_R_m (skok w dR)", config["gross"]["gross_R_m"], "m"))
    rows.append(("gross_vr_mps (skok w vr)", config["gross"]["gross_vr_mps"], "m/s"))

    return pd.DataFrame(rows, columns=["Parametr", "Wartość", "Jednostka"])

def metric_cards(summary: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RMSE [m]", f"{summary['RMSE']:.3f}")
    c2.metric("MED [m]", f"{summary['MED']:.3f}")
    c3.metric("P95 [m]", f"{summary['P95']:.3f}")
    c4.metric("MAE [m]", f"{summary['MAE']:.3f}")
    c5.metric("MAX [m]", f"{summary['MAX']:.3f}")

def plot_xy(p_true, p_est, beacons_xy, title, label_est):
    fig = _fig(6.4, 5.0)
    ax = fig.add_subplot(111)
    ax.plot(p_true[:, 0], p_true[:, 1], label="Trajektoria rzeczywista", linestyle="--", linewidth=2)
    ax.plot(p_est[:, 0], p_est[:, 1], label=label_est, linewidth=2)
    ax.scatter(beacons_xy[:, 0], beacons_xy[:, 1], marker="^", s=80, label="Beacony")
    ax.scatter(p_true[0, 0], p_true[0, 1], marker="o", s=70, label="Start")
    ax.scatter(p_true[-1, 0], p_true[-1, 1], marker="s", s=70, label="Koniec")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _grid(ax)
    _set_equal(ax)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)

def plot_error_t(t, e, title, label):
    fig = _fig(6.4, 5.0)
    ax = fig.add_subplot(111)
    ax.plot(t, e, label=label)
    ax.set_title(title)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e(t) [m]")
    _grid(ax)
    ax.legend(loc="upper right")
    st.pyplot(fig, clear_figure=True)

def plot_vr_panel(out: dict, method: str):
    """
    method: "VLS" or "EKF"
    uses out["v_vls_D"] or out["v_ekf_D"] for predicted vr from estimate
    """
    N = out["beacons"].shape[0]
    b_idx = st.selectbox("Wybierz beacon do podglądu", list(range(1, N + 1)), index=0, key=f"b_{method}")
    i = int(b_idx) - 1

    t = out["t"]
    vr_hat = out["vr_hats"][:, i]
    vr_true = vr_from_pv(out["p_true"], out["v_true"], out["beacons"])[:, i]

    if method == "VLS":
        vr_pred = vr_from_pv(out["p_vls_D"], out["v_vls_D"], out["beacons"])[:, i]
        label_est = "v_r z estymaty VLS"
    else:
        vr_pred = vr_from_pv(out["p_ekf_D"], out["v_ekf_D"], out["beacons"])[:, i]
        label_est = "v_r z estymaty EKF"

    fig = _fig(6.8, 4.8)
    ax = fig.add_subplot(111)
    ax.plot(t, vr_hat, label="pomiar v_r (Doppler)")
    ax.plot(t, vr_pred, label=label_est)
    ax.plot(t, vr_true, label="v_r rzeczywiste (symulacja)", linestyle="--")
    ax.set_title("Prędkość radialna v_r(t)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("v_r [m/s]")
    _grid(ax)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)


# ============================================================
# Header
# ============================================================

st.title("Środowisko testowe: LBL (TDOA) – VLS vs EKF, bez Dopplera i z Dopplerem")
st.caption("Porównanie dokładności estymacji pozycji w tych samych scenariuszach (TDOA). Doppler wnosi dodatkową informację o prędkości radialnej v_r.")

# ============================================================
# Sidebar
# ============================================================

sb = st.sidebar
sb.header("Konfiguracja eksperymentu")

geom_mode = sb.selectbox("Geometria beaconów", ["Preset", "Ręcznie (tabela)"])

if geom_mode == "Preset":
    preset = sb.selectbox("Kształt", ["trójkąt", "kwadrat", "pięciokąt"])
    radius = sb.number_input("Promień układu [m]", min_value=10.0, value=200.0, step=10.0)
    bz = sb.number_input("Głębokość beaconów [m]", value=0.0, step=1.0)
    beacons = make_beacons_preset(preset, radius=radius, z=bz)
else:
    sb.write("Wklej/edytuj współrzędne beaconów [m].")
    df_b = pd.DataFrame({"x": [0.0, 200.0, 0.0, 200.0],
                         "y": [0.0, 0.0, 200.0, 200.0],
                         "z": [0.0, 0.0, 0.0, 0.0]})
    edited = sb.data_editor(df_b, num_rows="dynamic", use_container_width=True)
    beacons = edited[["x", "y", "z"]].to_numpy(dtype=float)

sb.divider()

with sb.expander("Trajektoria", expanded=True):
    traj_kind = sb.selectbox("Typ", ["linia", "racetrack"], key="traj_kind")
    T = sb.number_input("Czas symulacji T [s]", min_value=5.0, value=120.0, step=5.0, key="T")
    dt = sb.number_input("Krok dt [s]", min_value=0.1, value=1.0, step=0.1, key="dt")
    speed = sb.number_input("Prędkość [m/s]", min_value=0.0, value=1.5, step=0.1, key="speed")
    heading = sb.number_input("Heading [deg] (kurs)", value=30.0, step=5.0, key="heading")
    start_x = sb.number_input("Start x [m]", value=0.0, step=5.0, key="start_x")
    start_y = sb.number_input("Start y [m]", value=0.0, step=5.0, key="start_y")
    obj_z = sb.number_input("Głębokość obiektu z [m]", value=30.0, step=1.0, key="obj_z")

with sb.expander("Akustyka", expanded=False):
    c = sb.number_input("Prędkość dźwięku c [m/s]", value=1500.0, step=10.0, key="c")
    f0 = sb.number_input("Częstotliwość nośna f0 [Hz]", value=20000.0, step=1000.0, key="f0")

with sb.expander("Szumy pomiarów (TDOA + Doppler)", expanded=True):
    sigma_tdoa = sb.number_input("σ_tdoa (TDOA) [s]", value=1e-4, step=1e-5, format="%.6f", key="sigma_tdoa")
    sigma_vr = sb.number_input("σ_vr (Doppler jako v_r) [m/s]", value=0.05, step=0.01, key="sigma_vr")
    sb.caption("TDOA liczone zawsze względem B1. Szum TDOA w metrach: σ_dR = c·σ_tdoa.")

with sb.expander("Błędy grube (symulacja zakłóceń)", expanded=False):
    enabled_gross = sb.checkbox("Włącz błędy grube", value=False)

    sb.caption(
        "Błąd gruby to rzadki, duży skok w pomiarze (np. multipath, błędna detekcja pingu). "
        "EKF z odpornością powinien go tłumić lepiej niż metody per-epoka (VLS)."
    )

    every_k = sb.number_input("Średnio co ile epok ma wystąpić zakłócenie?", min_value=1, value=30, step=1)
    p_gross = (1.0 / float(every_k)) if enabled_gross else 0.0

    gross_R_m = sb.number_input("Skok w dR [m] (TDOA)", min_value=0.0, value=10.0, step=1.0)
    gross_vr_mps = sb.number_input("Skok w v_r [m/s] (Doppler)", min_value=0.0, value=1.0, step=0.1)

with sb.expander("Filtr EKF (strojenie + odporność)", expanded=False):
    q_acc = sb.number_input("q_acc [m/s²] (strojenie)", value=0.05, step=0.01, key="q_acc")
    robust_k = sb.number_input("robust_k [σ] (próg tłumienia outlierów)", value=3.0, step=0.5)

with sb.expander("Monte-Carlo", expanded=False):
    do_mc = sb.checkbox("Wykonaj Monte-Carlo", value=False, key="do_mc")
    M = sb.number_input("Liczba prób M", min_value=5, value=100, step=10, key="M")
    seed0 = sb.number_input("Seed startowy", min_value=1, value=1, step=1, key="seed0")

run_btn = sb.button("Uruchom", type="primary")

# TDOA always relative to B1
ref_idx = 0

config = {
    "beacons": beacons.tolist(),
    "tdoa": {"ref_idx": int(ref_idx)},
    "trajectory": {
        "kind": traj_kind,
        "T": float(T),
        "dt": float(dt),
        "speed": float(speed),
        "heading_deg": float(heading),
        "start_xy": [float(start_x), float(start_y)],
        "z": float(obj_z),
    },
    "acoustics": {"c": float(c), "f0": float(f0)},
    "noise": {"sigma_tdoa": float(sigma_tdoa), "sigma_vr": float(sigma_vr)},
    "gross": {
        "enabled": bool(enabled_gross),
        "p_gross": float(p_gross),
        "gross_R_m": float(gross_R_m),
        "gross_vr_mps": float(gross_vr_mps),
    },
    "filter": {"q_acc": float(q_acc), "robust_k": float(robust_k)},
}

# ============================================================
# Tabs
# ============================================================

tab_geom, tab_vls0, tab_vlsD, tab_ekf0, tab_ekfD, tab_cmp, tab_mc, tab_export = st.tabs([
    "Geometria i parametry",
    "VLS: TDOA (bez Dopplera)",
    "VLS: TDOA + Doppler",
    "EKF: TDOA (bez Dopplera)",
    "EKF: TDOA + Doppler (robust)",
    "Porównania",
    "Monte-Carlo",
    "Eksport"
])

# ============================================================
# Run
# ============================================================

if "out" not in st.session_state:
    st.session_state["out"] = None
if "mc_out" not in st.session_state:
    st.session_state["mc_out"] = None

if run_btn:
    st.session_state["out"] = run_single_experiment(config, seed=int(seed0))
    st.session_state["mc_out"] = None

out = st.session_state["out"]

def require_out():
    if out is None:
        st.warning("Najpierw kliknij **Uruchom** w panelu bocznym.")
        st.stop()

# ============================================================
# Geometry tab
# ============================================================

with tab_geom:
    st.subheader("Podgląd beaconów i parametrów eksperymentu")

    col1, col2 = st.columns([1.0, 1.0])

    with col1:
        fig = _fig(6.6, 5.2)
        ax = fig.add_subplot(111)
        ax.scatter(beacons[:, 0], beacons[:, 1], marker="^", s=80, label="Beacony")
        for i, (x, y, _) in enumerate(beacons):
            tag = f"B{i+1}"
            if i == ref_idx:
                tag += " (ref)"
            ax.text(x, y, tag, fontsize=10, ha="left", va="bottom")
        ax.set_title("Geometria beaconów (rzut XY)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        _grid(ax)
        _set_equal(ax)
        ax.legend(loc="upper right")
        st.pyplot(fig, clear_figure=True)

        st.write("Tabela beaconów [m]")
        st.dataframe(pd.DataFrame(beacons, columns=["x", "y", "z"]), use_container_width=True)

    with col2:
        st.write("Tabela parametrów")
        st.dataframe(config_table(config), use_container_width=True)

        with st.expander("Pokaż konfigurację JSON (reprodukowalność)"):
            st.code(json.dumps(config, indent=2), language="json")

    st.info("Wyniki pojawią się w zakładkach po kliknięciu **Uruchom**.")

# ============================================================
# VLS no Doppler
# ============================================================

with tab_vls0:
    require_out()
    st.subheader("VLS: TDOA (bez Dopplera)")

    metric_cards(out["summary_vls_noD"])

    col1, col2 = st.columns(2)
    with col1:
        plot_xy(out["p_true"], out["p_vls_noD"], out["beacons"], "Trajektoria (XY)", "Estymata VLS (TDOA)")
    with col2:
        plot_error_t(out["t"], out["e_vls_noD"], "Błąd pozycji w czasie", "e(t) – VLS (TDOA)")

# ============================================================
# VLS Doppler
# ============================================================

with tab_vlsD:
    require_out()
    st.subheader("VLS: TDOA + Doppler (ważone LS)")

    metric_cards(out["summary_vls_D"])

    col1, col2 = st.columns(2)
    with col1:
        plot_xy(out["p_true"], out["p_vls_D"], out["beacons"], "Trajektoria (XY)", "Estymata VLS (TDOA + Doppler)")
    with col2:
        plot_error_t(out["t"], out["e_vls_D"], "Błąd pozycji w czasie", "e(t) – VLS (TDOA + Doppler)")

    st.divider()
    show_vr = st.checkbox("Pokaż prędkość radialną v_r(t) (Doppler)", value=False, key="show_vr_vls")
    if show_vr:
        plot_vr_panel(out, method="VLS")

# ============================================================
# EKF no Doppler
# ============================================================

with tab_ekf0:
    require_out()
    st.subheader("EKF: TDOA (bez Dopplera)")

    metric_cards(out["summary_ekf_noD"])

    col1, col2 = st.columns(2)
    with col1:
        plot_xy(out["p_true"], out["p_ekf_noD"], out["beacons"], "Trajektoria (XY)", "Estymata EKF (TDOA)")
    with col2:
        plot_error_t(out["t"], out["e_ekf_noD"], "Błąd pozycji w czasie", "e(t) – EKF (TDOA)")

# ============================================================
# EKF Doppler
# ============================================================

with tab_ekfD:
    require_out()
    st.subheader("EKF: TDOA + Doppler (robust)")

    metric_cards(out["summary_ekf_D"])

    col1, col2 = st.columns(2)
    with col1:
        plot_xy(out["p_true"], out["p_ekf_D"], out["beacons"], "Trajektoria (XY)", "Estymata EKF (TDOA + Doppler)")
    with col2:
        plot_error_t(out["t"], out["e_ekf_D"], "Błąd pozycji w czasie", "e(t) – EKF (TDOA + Doppler)")

    st.divider()
    show_vr = st.checkbox("Pokaż prędkość radialną v_r(t) (Doppler)", value=False, key="show_vr_ekf")
    if show_vr:
        plot_vr_panel(out, method="EKF")

# ============================================================
# Comparisons
# ============================================================

with tab_cmp:
    require_out()
    t = out["t"]

    e_vls0 = out["e_vls_noD"]
    e_vlsD = out["e_vls_D"]
    e_ekf0 = out["e_ekf_noD"]
    e_ekfD = out["e_ekf_D"]

    st.subheader("Porównania")

    st.markdown("### 1) VLS: bez Dopplera vs z Dopplerem")
    df1 = pd.DataFrame([
        {"Wariant": "VLS: TDOA", **out["summary_vls_noD"]},
        {"Wariant": "VLS: TDOA + Doppler", **out["summary_vls_D"]},
    ])
    st.dataframe(df1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = _fig(6.4, 5.0)
        ax = fig.add_subplot(111)
        ax.plot(t, e_vls0, label="VLS TDOA")
        ax.plot(t, e_vlsD, label="VLS TDOA + Doppler")
        ax.set_title("e(t) – VLS: porównanie")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("e(t) [m]")
        _grid(ax); ax.legend(loc="upper right")
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig = _fig(6.4, 5.0)
        ax = fig.add_subplot(111)
        ax.plot(t, e_vls0 - e_vlsD, label="Δe(t)=e_VLS−e_VLS+D")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title("Δe(t) – VLS: różnica")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Δe(t) [m]")
        _grid(ax); ax.legend(loc="upper right")
        st.pyplot(fig, clear_figure=True)

    st.markdown("### 2) EKF: bez Dopplera vs z Dopplerem")
    df2 = pd.DataFrame([
        {"Wariant": "EKF: TDOA", **out["summary_ekf_noD"]},
        {"Wariant": "EKF: TDOA + Doppler", **out["summary_ekf_D"]},
    ])
    st.dataframe(df2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = _fig(6.4, 5.0)
        ax = fig.add_subplot(111)
        ax.plot(t, e_ekf0, label="EKF TDOA")
        ax.plot(t, e_ekfD, label="EKF TDOA + Doppler")
        ax.set_title("e(t) – EKF: porównanie")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("e(t) [m]")
        _grid(ax); ax.legend(loc="upper right")
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig = _fig(6.4, 5.0)
        ax = fig.add_subplot(111)
        ax.plot(t, e_ekf0 - e_ekfD, label="Δe(t)=e_EKF−e_EKF+D")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title("Δe(t) – EKF: różnica")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Δe(t) [m]")
        _grid(ax); ax.legend(loc="upper right")
        st.pyplot(fig, clear_figure=True)

    st.markdown("### 3) VLS z Dopplerem vs EKF z Dopplerem")
    df3 = pd.DataFrame([
        {"Wariant": "VLS: TDOA + Doppler", **out["summary_vls_D"]},
        {"Wariant": "EKF: TDOA + Doppler", **out["summary_ekf_D"]},
    ])
    st.dataframe(df3, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = _fig(6.4, 5.0)
        ax = fig.add_subplot(111)
        ax.plot(t, e_vlsD, label="VLS TDOA + Doppler")
        ax.plot(t, e_ekfD, label="EKF TDOA + Doppler")
        ax.set_title("e(t) – VLS vs EKF (z Dopplerem)")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("e(t) [m]")
        _grid(ax); ax.legend(loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with col2:
        st.write("Histogram błędu e")
        fig = _fig(10.5, 4.5)
        ax = fig.add_subplot(111)
        ax.hist(e_vlsD, bins=25, alpha=0.65, label="VLS + Doppler")
        ax.hist(e_ekfD, bins=25, alpha=0.65, label="EKF + Doppler")
        ax.set_title("Histogram błędu pozycji e")
        ax.set_xlabel("e [m]")
        ax.set_ylabel("liczność")
        _grid(ax); ax.legend(loc="upper right")
        st.pyplot(fig, clear_figure=True)

# ============================================================
# Monte Carlo
# ============================================================

with tab_mc:
    st.subheader("Monte-Carlo (statystyka)")
    st.caption("Tryb Monte-Carlo uruchamia wiele prób z różnymi seedami i agreguje metryki.")

    if not do_mc:
        st.info("Zaznacz **Wykonaj Monte-Carlo** w panelu bocznym, a potem kliknij **Uruchom**.")
    else:
        require_out()
        if st.session_state["mc_out"] is None:
            st.session_state["mc_out"] = run_monte_carlo(config, M=int(M), seed0=int(seed0))
        mc = st.session_state["mc_out"]

        st.write("Wyniki per próba")
        st.dataframe(mc["runs"], use_container_width=True)
        st.write("Agregacja (mean/std/min/max)")
        st.dataframe(mc["agg"], use_container_width=True)

# ============================================================
# Export
# ============================================================

with tab_export:
    require_out()
    st.subheader("Eksport eksperymentu (reprodukowalność)")
    st.caption("Paczka zawiera konfigurację, przebiegi czasowe oraz tabelę metryk.")

    t = out["t"]
    p_true = out["p_true"]

    df_sum = pd.DataFrame([
        {"Wariant": "VLS: TDOA", **out["summary_vls_noD"]},
        {"Wariant": "VLS: TDOA + Doppler", **out["summary_vls_D"]},
        {"Wariant": "EKF: TDOA", **out["summary_ekf_noD"]},
        {"Wariant": "EKF: TDOA + Doppler", **out["summary_ekf_D"]},
    ])

    df_ts = pd.DataFrame({
        "t": t,
        "true_x": p_true[:, 0], "true_y": p_true[:, 1], "true_z": p_true[:, 2],

        "vls_noD_x": out["p_vls_noD"][:, 0], "vls_noD_y": out["p_vls_noD"][:, 1], "vls_noD_z": out["p_vls_noD"][:, 2],
        "vls_D_x": out["p_vls_D"][:, 0], "vls_D_y": out["p_vls_D"][:, 1], "vls_D_z": out["p_vls_D"][:, 2],

        "ekf_noD_x": out["p_ekf_noD"][:, 0], "ekf_noD_y": out["p_ekf_noD"][:, 1], "ekf_noD_z": out["p_ekf_noD"][:, 2],
        "ekf_D_x": out["p_ekf_D"][:, 0], "ekf_D_y": out["p_ekf_D"][:, 1], "ekf_D_z": out["p_ekf_D"][:, 2],

        "e_vls_noD": out["e_vls_noD"],
        "e_vls_D": out["e_vls_D"],
        "e_ekf_noD": out["e_ekf_noD"],
        "e_ekf_D": out["e_ekf_D"],
    })

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("config.json", json.dumps(config, indent=2))
        z.writestr("timeseries.csv", df_ts.to_csv(index=False))
        z.writestr("summary.csv", df_sum.to_csv(index=False))

    st.download_button(
        label="Pobierz paczkę ZIP (config + CSV)",
        data=mem.getvalue(),
        file_name="experiment_bundle.zip",
        mime="application/zip"
    )
    
