# app.py
from __future__ import annotations

import io
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core import (
    make_beacons_preset,
    run_single_experiment,
    run_monte_carlo,
)

# ============================================================
# Streamlit setup
# ============================================================

st.set_page_config(page_title="LBL (TDOA) + Doppler – środowisko testowe", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def _grid(ax):
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)

def _set_equal(ax):
    ax.set_aspect("equal", adjustable="box")

def _fig(w=7.2, h=4.8):
    return plt.figure(figsize=(w, h), dpi=120)

def _metric_cards(summary: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RMSE [m]", f"{summary['RMSE']:.3f}")
    c2.metric("MED [m]", f"{summary['MED']:.3f}")
    c3.metric("P95 [m]", f"{summary['P95']:.3f}")
    c4.metric("MAE [m]", f"{summary['MAE']:.3f}")
    c5.metric("MAX [m]", f"{summary['MAX']:.3f}")

def _config_table(config: dict) -> pd.DataFrame:
    rows = []
    b = np.array(config["beacons"], dtype=float)
    rows.append(("Liczba beaconów N", b.shape[0], "-"))
    rows.append(("Beacon referencyjny (TDOA)", config["tdoa"]["ref_idx"], "index"))
    rows.append(("c (prędkość dźwięku)", config["acoustics"]["c"], "m/s"))
    rows.append(("σ_t (TOA)", config["noise"]["sigma_t"], "s"))
    rows.append(("σ_vr (Doppler jako v_r)", config["noise"]["sigma_vr"], "m/s"))
    rows.append(("Trajektoria", config["trajectory"]["kind"], "-"))
    rows.append(("T", config["trajectory"]["T"], "s"))
    rows.append(("dt", config["trajectory"]["dt"], "s"))
    rows.append(("Prędkość", config["trajectory"]["speed"], "m/s"))
    rows.append(("Heading", config["trajectory"]["heading_deg"], "deg"))
    rows.append(("Start x", config["trajectory"]["start_xy"][0], "m"))
    rows.append(("Start y", config["trajectory"]["start_xy"][1], "m"))
    rows.append(("Głębokość obiektu z", config["trajectory"]["z"], "m"))
    rows.append(("q_acc (strojenie filtru)", config["filter"]["q_acc"], "m/s²"))

    g = config.get("gross", {})
    rows.append(("Błąd gruby (włączony)", bool(g.get("enable", False)), "-"))
    rows.append(("P(outlier) TOA→TDOA", g.get("p_tdoa", 0.0), "-"))
    rows.append(("Skala outlier TOA (×σ_t)", g.get("scale_tdoa", 10.0), "-"))
    rows.append(("P(outlier) vr", g.get("p_vr", 0.0), "-"))
    rows.append(("Skala outlier vr (×σ_vr)", g.get("scale_vr", 10.0), "-"))

    return pd.DataFrame(rows, columns=["Parametr", "Wartość", "Jednostka"])

def _plot_xy_with_beacons(beacons, p_true, p_est, title, show_vel=False, v_est=None, vel_label="Kierunek ruchu (v est.)"):
    fig = _fig(6.6, 5.2)
    ax = fig.add_subplot(111)

    ax.plot(p_est[:, 0], p_est[:, 1], label="Estymata", linewidth=2, alpha=0.9, zorder=2)
    ax.plot(p_true[:, 0], p_true[:, 1], label="Trajektoria rzeczywista", linestyle="--", linewidth=3, zorder=3)

    ax.scatter(beacons[:, 0], beacons[:, 1], marker="^", s=80, label="Beacony", zorder=4)
    for i, (x, y, _) in enumerate(beacons):
        ax.text(x, y, f"B{i}", fontsize=10, ha="left", va="bottom")

    ax.scatter(p_true[0, 0], p_true[0, 1], marker="o", s=70, label="Start", zorder=5)
    ax.scatter(p_true[-1, 0], p_true[-1, 1], marker="s", s=70, label="Koniec", zorder=5)

    if show_vel and (v_est is not None) and (len(v_est) == len(p_est)):
        step = max(1, len(p_est) // 18)  # ~18 strzałek
        ax.quiver(
            p_est[::step, 0], p_est[::step, 1],
            v_est[::step, 0], v_est[::step, 1],
            scale=20,
            width=0.004,
            zorder=6,
            label=vel_label,
        )

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _grid(ax)
    _set_equal(ax)
    ax.legend(loc="best")
    return fig

def _plot_error(t, e, title, label):
    fig = _fig(6.6, 5.2)
    ax = fig.add_subplot(111)
    ax.plot(t, e, label=label)
    ax.set_title(title)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e(t) [m]")
    _grid(ax)
    ax.legend(loc="upper right")
    return fig

def _plot_four_errors(t, e1, e2, e3, e4, labels, title):
    fig = _fig(10.5, 4.8)
    ax = fig.add_subplot(111)
    ax.plot(t, e1, label=labels[0])
    ax.plot(t, e2, label=labels[1])
    ax.plot(t, e3, label=labels[2])
    ax.plot(t, e4, label=labels[3])
    ax.set_title(title)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e(t) [m]")
    _grid(ax)
    ax.legend(loc="upper right")
    return fig

# ============================================================
# Header
# ============================================================

st.title("Środowisko testowe: LBL (TDOA) – VLS i EKF, z i bez Dopplera")
st.caption(
    "Aplikacja porównuje estymację pozycji w systemie LBL opartym o TDOA. "
    "Dostępne są cztery warianty: VLS i EKF, każdy w wersji bez Dopplera oraz z Dopplerem (obserwacja v_r). "
    "Dodatkowo można symulować błąd gruby (outlier) pomiarów."
)

# ============================================================
# Sidebar config
# ============================================================

st.sidebar.header("Konfiguracja eksperymentu")

geom_mode = st.sidebar.selectbox("Geometria beaconów", ["Preset", "Ręcznie (tabela)"])

if geom_mode == "Preset":
    preset = st.sidebar.selectbox("Kształt", ["trójkąt", "kwadrat", "pięciokąt"])
    radius = st.sidebar.number_input("Promień układu [m]", min_value=10.0, value=200.0, step=10.0)
    bz = st.sidebar.number_input("Głębokość beaconów [m]", value=0.0, step=1.0)
    beacons = make_beacons_preset(preset, radius=radius, z=bz)
else:
    st.sidebar.write("Wklej/edytuj współrzędne beaconów [m].")
    df_b = pd.DataFrame({"x": [0.0, 200.0, 0.0, 200.0],
                         "y": [0.0, 0.0, 200.0, 200.0],
                         "z": [0.0, 0.0, 0.0, 0.0]})
    edited = st.sidebar.data_editor(df_b, num_rows="dynamic", use_container_width=True)
    beacons = edited[["x", "y", "z"]].to_numpy(dtype=float)

st.sidebar.divider()

with st.sidebar.expander("TDOA", expanded=True):
    ref_idx = st.number_input(
        "Beacon referencyjny (index od 0)",
        min_value=0,
        max_value=max(0, int(len(beacons) - 1)),
        value=0,
        step=1
    )
    st.caption("TDOA liczone jako różnica czasu przyjścia sygnału względem beacona referencyjnego.")

with st.sidebar.expander("Trajektoria", expanded=True):
    traj_kind = st.sidebar.selectbox("Typ", ["linia", "racetrack"], key="traj_kind")
    T = st.sidebar.number_input("Czas symulacji T [s]", min_value=5.0, value=120.0, step=5.0, key="T")
    dt = st.sidebar.number_input("Krok dt [s]", min_value=0.1, value=1.0, step=0.1, key="dt")
    speed = st.sidebar.number_input("Prędkość [m/s]", min_value=0.0, value=1.5, step=0.1, key="speed")
    heading = st.sidebar.number_input("Heading [deg] (kurs)", value=30.0, step=5.0, key="heading")
    start_x = st.sidebar.number_input("Start x [m]", value=0.0, step=5.0, key="start_x")
    start_y = st.sidebar.number_input("Start y [m]", value=0.0, step=5.0, key="start_y")
    obj_z = st.sidebar.number_input("Głębokość obiektu z [m]", value=30.0, step=1.0, key="obj_z")

with st.sidebar.expander("Akustyka", expanded=False):
    c = st.sidebar.number_input("Prędkość dźwięku c [m/s]", value=1500.0, step=10.0, key="c")

with st.sidebar.expander("Szumy pomiarów", expanded=True):
    sigma_t = st.sidebar.number_input("σ_t (TOA) [s]", value=1e-4, step=1e-5, format="%.6f", key="sigma_t")
    sigma_vr = st.sidebar.number_input("σ_vr (Doppler jako v_r) [m/s]", value=0.05, step=0.01, key="sigma_vr")
    st.caption("Uwaga: dla TDOA wariancja rośnie: σ_Δt = √2·σ_t.")

with st.sidebar.expander("Filtr EKF", expanded=False):
    q_acc = st.sidebar.number_input("q_acc [m/s²] (strojenie)", value=0.05, step=0.01, key="q_acc")

with st.sidebar.expander("Błąd gruby (outlier)", expanded=False):
    gross_enable = st.sidebar.checkbox("Włącz błąd gruby", value=False)
    gross_p_tdoa = st.sidebar.number_input("P(outlier) TOA→TDOA", min_value=0.0, max_value=1.0, value=0.00, step=0.01)
    gross_scale_tdoa = st.sidebar.number_input("Skala outlier TOA (×σ_t)", min_value=1.0, value=10.0, step=1.0)
    gross_p_vr = st.sidebar.number_input("P(outlier) v_r", min_value=0.0, max_value=1.0, value=0.00, step=0.01)
    gross_scale_vr = st.sidebar.number_input("Skala outlier v_r (×σ_vr)", min_value=1.0, value=10.0, step=1.0)

with st.sidebar.expander("Monte-Carlo", expanded=False):
    do_mc = st.sidebar.checkbox("Wykonaj Monte-Carlo", value=False, key="do_mc")
    M = st.sidebar.number_input("Liczba prób M", min_value=5, value=100, step=10, key="M")
    seed0 = st.sidebar.number_input("Seed startowy", min_value=1, value=1, step=1, key="seed0")

run_btn = st.sidebar.button("Uruchom", type="primary")

# ============================================================
# Build config
# ============================================================

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
    "acoustics": {"c": float(c)},
    "noise": {"sigma_t": float(sigma_t), "sigma_vr": float(sigma_vr)},
    "filter": {"q_acc": float(q_acc)},
    "gross": {
        "enable": bool(gross_enable),
        "p_tdoa": float(gross_p_tdoa),
        "scale_tdoa": float(gross_scale_tdoa),
        "p_vr": float(gross_p_vr),
        "scale_vr": float(gross_scale_vr),
    },
}

# ============================================================
# Tabs
# ============================================================

tab_geom, tab_vls, tab_vlsd, tab_ekf, tab_ekfd, tab_cmp, tab_mc, tab_export = st.tabs([
    "Geometria i parametry",
    "VLS (TDOA) – bez Dopplera",
    "VLS (TDOA + Doppler)",
    "EKF (TDOA) – bez Dopplera",
    "EKF (TDOA + Doppler)",
    "Porównania",
    "Monte-Carlo",
    "Eksport"
])

# ============================================================
# Session state (results)
# ============================================================

if "out" not in st.session_state:
    st.session_state["out"] = None

if run_btn:
    st.session_state["out"] = run_single_experiment(config, seed=int(seed0))

out = st.session_state["out"]

def require_out():
    if out is None:
        st.warning("Najpierw kliknij **Uruchom** w panelu bocznym.")
        st.stop()

# ============================================================
# Tab: Geometry & params
# ============================================================

with tab_geom:
    st.subheader("Podgląd beaconów i parametrów eksperymentu")

    col1, col2 = st.columns([1.0, 1.0])
    with col1:
        fig = _fig(6.6, 5.2)
        ax = fig.add_subplot(111)
        ax.scatter(beacons[:, 0], beacons[:, 1], marker="^", s=80, label="Beacony")
        for i, (x, y, _) in enumerate(beacons):
            ax.text(x, y, f"B{i}", fontsize=10, ha="left", va="bottom")
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
        st.dataframe(_config_table(config), use_container_width=True)

        with st.expander("Pokaż konfigurację JSON (reprodukowalność)"):
            st.code(json.dumps(config, indent=2), language="json")

    st.info("Wyniki pojawią się w zakładkach po kliknięciu **Uruchom**.")

# ============================================================
# Tab: VLS (TDOA) no Doppler
# ============================================================

with tab_vls:
    require_out()

    st.subheader("VLS (ważona MNK) – tylko TDOA (bez Dopplera)")
    st.caption("Estymacja pozycji w każdej epoce niezależnie: minimalizacja błędu modelu TDOA.")

    t = out["t"]
    b = out["beacons"]
    p_true = out["p_true"]
    p_est = out["p_vls"]
    e = out["e_vls"]

    _metric_cards(out["summary_vls"])

    col1, col2 = st.columns(2)
    with col1:
        fig = _plot_xy_with_beacons(b, p_true, p_est, "Trajektoria (XY) – VLS (TDOA)")
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig = _plot_error(t, e, "Błąd pozycji w czasie – VLS (TDOA)", "e(t) – VLS")
        st.pyplot(fig, clear_figure=True)

# ============================================================
# Tab: VLS with Doppler
# ============================================================

with tab_vlsd:
    require_out()

    st.subheader("VLS (ważona MNK) – TDOA + Doppler (obserwacja v_r)")
    st.caption(
        "Estymacja łączna stanu [p, v] w każdej epoce: TDOA dostarcza informacji geometrycznej, "
        "a Doppler (v_r) dostarcza informacji o ruchu i kierunku."
    )

    t = out["t"]
    b = out["beacons"]
    p_true = out["p_true"]
    p_est = out["p_vls_dopp"]
    v_est = out["v_vls_dopp"]
    e = out["e_vls_dopp"]

    _metric_cards(out["summary_vls_dopp"])

    show_dir = st.checkbox("Pokaż kierunek ruchu (z estymowanej prędkości)", value=True, key="show_dir_vls")

    col1, col2 = st.columns(2)
    with col1:
        fig = _plot_xy_with_beacons(
            b, p_true, p_est,
            "Trajektoria (XY) – VLS (TDOA + Doppler)",
            show_vel=show_dir, v_est=v_est,
            vel_label="Kierunek ruchu (VLS: v est.)"
        )
        st.pyplot(fig, clear_figure=True)

    with col2:
        fig = _plot_error(t, e, "Błąd pozycji w czasie – VLS (TDOA + Doppler)", "e(t) – VLS+Doppler")
        st.pyplot(fig, clear_figure=True)

    with st.expander("Diagnostyka Dopplera: obserwacje v_r i estymata", expanded=False):
        vr_hat = out["vr_hats"]
        # pokaż prostą diagnostykę: mean|vr| oraz korelację z estymowaną
        st.write(f"Średnia |v_r| (pomiar): {float(np.mean(np.abs(vr_hat))):.3f} m/s")
        st.write(f"Średnia |v| (estymata): {float(np.mean(np.linalg.norm(v_est, axis=1))):.3f} m/s")

# ============================================================
# Tab: EKF (TDOA) no Doppler
# ============================================================

with tab_ekf:
    require_out()

    st.subheader("EKF – tylko TDOA (bez Dopplera)")
    st.caption("Stan [p, v] jest propagowany modelem ruchu, a aktualizacja wykorzystuje pomiar TDOA.")

    t = out["t"]
    b = out["beacons"]
    p_true = out["p_true"]
    p_est = out["p_ekf"]
    v_est = out["v_ekf"]
    e = out["e_ekf"]

    _metric_cards(out["summary_ekf"])

    col1, col2 = st.columns(2)
    with col1:
        fig = _plot_xy_with_beacons(b, p_true, p_est, "Trajektoria (XY) – EKF (TDOA)")
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig = _plot_error(t, e, "Błąd pozycji w czasie – EKF (TDOA)", "e(t) – EKF")
        st.pyplot(fig, clear_figure=True)

# ============================================================
# Tab: EKF with Doppler
# ============================================================

with tab_ekfd:
    require_out()

    st.subheader("EKF – TDOA + Doppler (obserwacja v_r)")
    st.caption("Aktualizacja EKF wykorzystuje jednocześnie TDOA i obserwacje prędkości radialnych v_r dla beaconów.")

    t = out["t"]
    b = out["beacons"]
    p_true = out["p_true"]
    p_est = out["p_ekf_dopp"]
    v_est = out["v_ekf_dopp"]
    e = out["e_ekf_dopp"]

    _metric_cards(out["summary_ekf_dopp"])

    show_dir = st.checkbox("Pokaż kierunek ruchu (z estymowanej prędkości)", value=True, key="show_dir_ekf")

    col1, col2 = st.columns(2)
    with col1:
        fig = _plot_xy_with_beacons(
            b, p_true, p_est,
            "Trajektoria (XY) – EKF (TDOA + Doppler)",
            show_vel=show_dir, v_est=v_est,
            vel_label="Kierunek ruchu (EKF: v est.)"
        )
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig = _plot_error(t, e, "Błąd pozycji w czasie – EKF (TDOA + Doppler)", "e(t) – EKF+Doppler")
        st.pyplot(fig, clear_figure=True)

# ============================================================
# Tab: Comparisons
# ============================================================

with tab_cmp:
    require_out()

    st.subheader("Porównania metod: VLS vs EKF, z i bez Dopplera")
    st.caption("Poniżej tabela metryk oraz przebiegi błędów dla czterech wariantów.")

    t = out["t"]

    df_sum = pd.DataFrame([
        {"Wariant": "VLS (TDOA)", **out["summary_vls"]},
        {"Wariant": "VLS (TDOA + Doppler)", **out["summary_vls_dopp"]},
        {"Wariant": "EKF (TDOA)", **out["summary_ekf"]},
        {"Wariant": "EKF (TDOA + Doppler)", **out["summary_ekf_dopp"]},
    ])
    st.dataframe(df_sum, use_container_width=True)

    # "zysk" Dopplera dla VLS i EKF
    gain_vls = out["summary_vls"]["RMSE"] - out["summary_vls_dopp"]["RMSE"]
    gain_ekf = out["summary_ekf"]["RMSE"] - out["summary_ekf_dopp"]["RMSE"]

    c1, c2 = st.columns(2)
    c1.metric("Zysk Dopplera (VLS): ΔRMSE [m]", f"{gain_vls:.3f}")
    c2.metric("Zysk Dopplera (EKF): ΔRMSE [m]", f"{gain_ekf:.3f}")

    fig = _plot_four_errors(
        t,
        out["e_vls"],
        out["e_vls_dopp"],
        out["e_ekf"],
        out["e_ekf_dopp"],
        labels=["VLS (TDOA)", "VLS (TDOA + Doppler)", "EKF (TDOA)", "EKF (TDOA + Doppler)"],
        title="Błąd pozycji w czasie – porównanie (4 warianty)"
    )
    st.pyplot(fig, clear_figure=True)

    st.write("Histogram błędu pozycji (e)")
    fig = _fig(10.5, 4.5)
    ax = fig.add_subplot(111)
    ax.hist(out["e_vls"], bins=25, alpha=0.55, label="VLS (TDOA)")
    ax.hist(out["e_vls_dopp"], bins=25, alpha=0.55, label="VLS (TDOA + Doppler)")
    ax.hist(out["e_ekf"], bins=25, alpha=0.55, label="EKF (TDOA)")
    ax.hist(out["e_ekf_dopp"], bins=25, alpha=0.55, label="EKF (TDOA + Doppler)")
    ax.set_title("Histogram błędu pozycji")
    ax.set_xlabel("e [m]")
    ax.set_ylabel("liczność")
    _grid(ax)
    ax.legend(loc="upper right")
    st.pyplot(fig, clear_figure=True)

    with st.expander("Diagnostyka: błąd gruby", expanded=False):
        meta = out.get("gross_meta", [])
        if not meta:
            st.info("Brak metadanych błędu grubego.")
        else:
            toa_ct = sum(m.get("gross_toa_ct", 0) for m in meta)
            vr_ct = sum(m.get("gross_vr_ct", 0) for m in meta)
            st.write(f"Liczba wstrzykniętych outlierów TOA (przed TDOA): {toa_ct}")
            st.write(f"Liczba wstrzykniętych outlierów v_r: {vr_ct}")

# ============================================================
# Tab: Monte-Carlo
# ============================================================

with tab_mc:
    st.subheader("Monte-Carlo (statystyka)")
    st.caption("Tryb Monte-Carlo uruchamia wiele prób z różnymi seedami i agreguje metryki.")

    if not do_mc:
        st.info("Zaznacz **Wykonaj Monte-Carlo** w panelu bocznym, a potem kliknij **Uruchom**.")
    else:
        # MC nie musi korzystać z "out", ale wymagamy spójności działania
        mc = run_monte_carlo(config, M=int(M), seed0=int(seed0))
        st.write("Wyniki per próba")
        st.dataframe(mc["runs"], use_container_width=True)

        st.write("Agregacja (mean/std/min/max)")
        st.dataframe(mc["agg"], use_container_width=True)

# ============================================================
# Tab: Export
# ============================================================

with tab_export:
    require_out()

    st.subheader("Eksport eksperymentu (reprodukowalność)")
    st.caption("Paczka zawiera konfigurację oraz przebiegi czasowe i tabelę metryk (4 warianty).")

    t = out["t"]
    p_true = out["p_true"]

    df_sum = pd.DataFrame([
        {"Wariant": "VLS (TDOA)", **out["summary_vls"]},
        {"Wariant": "VLS (TDOA + Doppler)", **out["summary_vls_dopp"]},
        {"Wariant": "EKF (TDOA)", **out["summary_ekf"]},
        {"Wariant": "EKF (TDOA + Doppler)", **out["summary_ekf_dopp"]},
    ])

    df_ts = pd.DataFrame({
        "t": t,
        "true_x": p_true[:, 0], "true_y": p_true[:, 1], "true_z": p_true[:, 2],

        "vls_x": out["p_vls"][:, 0], "vls_y": out["p_vls"][:, 1], "vls_z": out["p_vls"][:, 2],
        "vls_d_x": out["p_vls_dopp"][:, 0], "vls_d_y": out["p_vls_dopp"][:, 1], "vls_d_z": out["p_vls_dopp"][:, 2],

        "ekf_x": out["p_ekf"][:, 0], "ekf_y": out["p_ekf"][:, 1], "ekf_z": out["p_ekf"][:, 2],
        "ekf_d_x": out["p_ekf_dopp"][:, 0], "ekf_d_y": out["p_ekf_dopp"][:, 1], "ekf_d_z": out["p_ekf_dopp"][:, 2],

        "e_vls": out["e_vls"],
        "e_vls_dopp": out["e_vls_dopp"],
        "e_ekf": out["e_ekf"],
        "e_ekf_dopp": out["e_ekf_dopp"],
    })

    # velocity exports (useful for "direction of motion" analysis)
    if "v_vls_dopp" in out:
        v = out["v_vls_dopp"]
        df_ts["vls_d_vx"] = v[:, 0]
        df_ts["vls_d_vy"] = v[:, 1]
        df_ts["vls_d_vz"] = v[:, 2]

    if "v_ekf_dopp" in out:
        v = out["v_ekf_dopp"]
        df_ts["ekf_d_vx"] = v[:, 0]
        df_ts["ekf_d_vy"] = v[:, 1]
        df_ts["ekf_d_vz"] = v[:, 2]

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
