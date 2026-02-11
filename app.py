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

def _fig(w=7.2, h=4.8):
    return plt.figure(figsize=(w, h), dpi=120)
    
def legend_outside_right(fig, ax, ncol: int = 1, pad: float = 0.02, shrink: float = 0.78):
    """
    Legenda na zewnątrz po prawej stronie.
    - shrink: ile miejsca zostawić na wykres (0.78 = 78% szerokości figury na osie, reszta na legendę)
    - pad: mały odstęp między osią a legendą
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    # zmniejsz obszar osi, żeby zrobić miejsce na legendę
    fig.subplots_adjust(right=shrink)

    # legenda poza osią (po prawej)
    ax.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.0 + pad, 0.5),
        ncol=ncol,
        frameon=True
    )





def config_table(config: dict) -> pd.DataFrame:
    rows = []
    b = np.array(config["beacons"], dtype=float)
    rows.append(("Liczba beaconów N", b.shape[0], "-"))
    rows.append(("c (prędkość dźwięku)", config["acoustics"]["c"], "m/s"))
    rows.append(("σ_tdoa (szum TDOA)", config["noise"]["sigma_tdoa"], "s"))
    rows.append(("σ_vr (szum Doppler jako v_r)", config["noise"]["sigma_vr"], "m/s"))

    rows.append(("Trajektoria", config["trajectory"]["kind"], "-"))
    rows.append(("T", config["trajectory"]["T"], "s"))
    rows.append(("dt", config["trajectory"]["dt"], "s"))
    rows.append(("Prędkość", config["trajectory"]["speed"], "m/s"))
    rows.append(("Heading", config["trajectory"]["heading_deg"], "deg"))
    rows.append(("Start x", config["trajectory"]["start_xy"][0], "m"))
    rows.append(("Start y", config["trajectory"]["start_xy"][1], "m"))
    rows.append(("Głębokość obiektu z (znana)", config["trajectory"]["z"], "m"))

    g = config.get("gross", {})
    rows.append(("Błędy grube – enabled", bool(g.get("enabled", False)), "-"))
    rows.append(("p_gross", g.get("p_gross", 0.0), "-"))
    rows.append(("gross_R_m", g.get("gross_R_m", 10.0), "m"))
    rows.append(("gross_vr_mps", g.get("gross_vr_mps", 1.0), "m/s"))

    f = config.get("filter", {})
    rows.append(("q_acc (strojenie EKF)", f.get("q_acc", 0.05), "m/s²"))
    rows.append(("robust_k (EKF)", f.get("robust_k", 3.0), "-"))

    return pd.DataFrame(rows, columns=["Parametr", "Wartość", "Jednostka"])


def plot_xy(p_true: np.ndarray, p_est: np.ndarray, beacons: np.ndarray,
            title: str, label_est: str):
    bxy = np.asarray(beacons, dtype=float)[:, 0:2]
    txy = np.asarray(p_true, dtype=float)[:, 0:2]
    exy = np.asarray(p_est, dtype=float)[:, 0:2]

    fig = _fig(7.6, 5.4)  # trochę szersza, żeby legenda miała miejsce
    ax = fig.add_subplot(111)

    ax.scatter(bxy[:, 0], bxy[:, 1], marker="^", s=90, label="Beacony", zorder=6)
    for i, (x, y) in enumerate(bxy):
        ax.text(x, y, f"B{i+1}", fontsize=9, ha="left", va="bottom", zorder=7)

    ax.plot(exy[:, 0], exy[:, 1], label=label_est, linewidth=2, alpha=0.95, zorder=4)
    ax.plot(txy[:, 0], txy[:, 1], label="Pozycja rzeczywista (symulacja)",
            linestyle="--", linewidth=3, zorder=5)

    ax.scatter(txy[0, 0], txy[0, 1], marker="o", s=80, label="Start", zorder=8)
    ax.scatter(txy[-1, 0], txy[-1, 1], marker="s", s=80, label="Koniec", zorder=8)

    allx = np.concatenate([bxy[:, 0], txy[:, 0], exy[:, 0]])
    ally = np.concatenate([bxy[:, 1], txy[:, 1], exy[:, 1]])
    xmin, xmax = float(allx.min()), float(allx.max())
    ymin, ymax = float(ally.min()), float(ally.max())
    pad = 0.08 * max(xmax - xmin, ymax - ymin, 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _grid(ax)
    _set_equal(ax)

    legend_outside_right(fig, ax, ncol=1, shrink=0.78, pad=0.02)
    st.pyplot(fig, clear_figure=True)


def plot_error(t: np.ndarray, e: np.ndarray, title: str, label: str):
    fig = _fig(7.6, 5.0)
    ax = fig.add_subplot(111)

    ax.plot(t, e, label=label, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e(t) [m]")
    _grid(ax)

    legend_outside_right(fig, ax, ncol=1, shrink=0.80, pad=0.02)
    st.pyplot(fig, clear_figure=True)


def plot_vr_timeseries(t: np.ndarray, vr_true: np.ndarray, vr_hat: np.ndarray, vr_pred: np.ndarray | None,
                       title: str, max_beacons: int = 5):
    N = vr_true.shape[1]
    nb = min(N, max_beacons)

    fig = _fig(12.0, 5.0)
    ax = fig.add_subplot(111)

    for i in range(nb):
        ax.plot(t, vr_true[:, i], linewidth=2, label=f"vr_true B{i+1}", alpha=0.9)
        ax.plot(t, vr_hat[:, i], linewidth=1, linestyle="--", label=f"vr_hat B{i+1}", alpha=0.8)
        if vr_pred is not None:
            ax.plot(t, vr_pred[:, i], linewidth=1.5, linestyle=":", label=f"vr_pred B{i+1}", alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("v_r [m/s]")
    _grid(ax)

    # dużo serii -> więcej miejsca na legendę
    legend_outside_right(fig, ax, ncol=2, shrink=0.70, pad=0.02)
    st.pyplot(fig, clear_figure=True)

def show_metrics(summary: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RMSE [m]", f"{summary['RMSE']:.3f}")
    c2.metric("MED [m]",  f"{summary['MED']:.3f}")
    c3.metric("P95 [m]",  f"{summary['P95']:.3f}")
    c4.metric("MAE [m]",  f"{summary['MAE']:.3f}")
    c5.metric("MAX [m]",  f"{summary['MAX']:.3f}")


# ============================================================
# UI
# ============================================================

st.title("Środowisko testowe: LBL (TDOA) – VLS vs EKF, bez Dopplera i z Dopplerem")
st.caption("Porównanie metod estymacji pozycji w identycznych scenariuszach, z opcją błędów grubych i analizą Dopplera.")

st.sidebar.header("Konfiguracja eksperymentu")

geom_mode = st.sidebar.selectbox("Geometria beaconów", ["Preset", "Ręcznie (tabela)"])

if geom_mode == "Preset":
    preset = st.sidebar.selectbox("Kształt", ["kwadrat", "pięciokąt", "trójkąt"])
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

with st.sidebar.expander("Trajektoria", expanded=True):
    traj_kind = st.selectbox("Typ", ["linia", "racetrack"], key="traj_kind")
    T = st.number_input("Czas symulacji T [s]", min_value=5.0, value=120.0, step=5.0, key="T")
    dt = st.number_input("Krok dt [s]", min_value=0.1, value=1.0, step=0.1, key="dt")
    speed = st.number_input("Prędkość [m/s]", min_value=0.0, value=1.5, step=0.1, key="speed")
    heading = st.number_input("Heading [deg] (kurs)", value=30.0, step=5.0, key="heading")
    start_x = st.number_input("Start x [m]", value=0.0, step=5.0, key="start_x")
    start_y = st.number_input("Start y [m]", value=0.0, step=5.0, key="start_y")
    obj_z = st.number_input("Głębokość obiektu z [m] (znana)", value=30.0, step=1.0, key="obj_z")

with st.sidebar.expander("Akustyka", expanded=False):
    c = st.number_input("Prędkość dźwięku c [m/s]", value=1500.0, step=10.0, key="c")

with st.sidebar.expander("Szumy pomiarów", expanded=True):
    sigma_tdoa = st.number_input("σ_tdoa (TDOA) [s]", value=1e-4, step=1e-5, format="%.6f", key="sigma_tdoa")
    sigma_vr = st.number_input("σ_vr (Doppler jako v_r) [m/s]", value=0.05, step=0.01, key="sigma_vr")

with st.sidebar.expander("Błędy grube (symulacja)", expanded=False):
    gross_enabled = st.checkbox("Włącz błędy grube", value=False, key="gross_enabled")
    p_gross = st.number_input("p_gross (prawdopodobieństwo)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="p_gross")
    gross_R_m = st.number_input("Skala błędu grubego TDOA [m]", min_value=0.0, value=10.0, step=1.0, key="gross_R_m")
    gross_vr_mps = st.number_input("Skala błędu grubego vr [m/s]", min_value=0.0, value=1.0, step=0.1, key="gross_vr_mps")

with st.sidebar.expander("EKF (strojenie)", expanded=False):
    q_acc = st.number_input("q_acc [m/s²]", value=0.05, step=0.01, key="q_acc")
    robust_k = st.number_input("robust_k [-]", value=3.0, step=0.5, key="robust_k")

with st.sidebar.expander("Monte-Carlo", expanded=False):
    do_mc = st.checkbox("Wykonaj Monte-Carlo", value=False, key="do_mc")
    M = st.number_input("Liczba prób M", min_value=5, value=100, step=10, key="M")
    seed0 = st.number_input("Seed startowy", min_value=1, value=1, step=1, key="seed0")

run_btn = st.sidebar.button("Uruchom", type="primary")

config = {
    "beacons": beacons.tolist(),
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
    "noise": {"sigma_tdoa": float(sigma_tdoa), "sigma_vr": float(sigma_vr)},
    "gross": {
        "enabled": bool(gross_enabled),
        "p_gross": float(p_gross),
        "gross_R_m": float(gross_R_m),
        "gross_vr_mps": float(gross_vr_mps),
    },
    "filter": {"q_acc": float(q_acc), "robust_k": float(robust_k)},
}

tabs = st.tabs([
    "Geometria i parametry",
    "VLS: TDOA",
    "VLS: TDOA + Doppler",
    "EKF: TDOA",
    "EKF: TDOA + Doppler",
    "Porównania",
    "Monte-Carlo",
    "Eksport",
])


if "out" not in st.session_state:
    st.session_state["out"] = None

if run_btn:
    st.session_state["out"] = run_single_experiment(config, seed=int(seed0))

out = st.session_state["out"]

def require_out():
    if out is None:
        st.warning("Kliknij **Uruchom** w panelu bocznym.")
        st.stop()


# ============================================================
# Tab: Geometry
# ============================================================

with tabs[0]:
    st.subheader("Geometria i parametry eksperymentu")

    col1, col2 = st.columns([1, 1])
    with col1:
        fig = _fig(6.8, 5.4)
        ax = fig.add_subplot(111)
        bxy = beacons[:, 0:2]
        ax.scatter(bxy[:, 0], bxy[:, 1], marker="^", s=90, label="Beacony")
        for i, (x, y) in enumerate(bxy):
            ax.text(x, y, f"B{i+1}", fontsize=9, ha="left", va="bottom")
        ax.set_title("Rzut XY – geometria beaconów")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        _grid(ax); _set_equal(ax)
        legend_outside(fig, ax, ncol=1, right=0.80)


        st.write("Tabela beaconów [m]")
        st.dataframe(pd.DataFrame(beacons, columns=["x", "y", "z"]), use_container_width=True)

    with col2:
        st.write("Tabela parametrów")
        st.dataframe(config_table(config), use_container_width=True)
        with st.expander("Konfiguracja JSON (reprodukowalność)"):
            st.code(json.dumps(config, indent=2), language="json")

    st.info("Wyniki pojawią się po kliknięciu **Uruchom**.")


# ============================================================
# Tab: VLS TDOA
# ============================================================

with tabs[1]:
    require_out()
    st.subheader("VLS: TDOA (bez Dopplera)")

    show_metrics(out["summary_vls_noD"])

    col1, col2 = st.columns(2)
    with col1:
        plot_xy(out["p_true"], out["p_vls_noD"], out["beacons"],
                "Trajektoria (XY) – VLS: TDOA", "Estymata VLS (TDOA)")
    with col2:
        plot_error(out["t"], out["e_vls_noD"],
                   "Błąd pozycji w czasie – VLS: TDOA",
                   "e(t) VLS(TDOA)")


# ============================================================
# Tab: VLS TDOA + Doppler
# ============================================================

with tabs[2]:
    require_out()
    st.subheader("VLS: TDOA + Doppler (per-epoka, LS ważony)")

    show_metrics(out["summary_vls_D"])

    show_vr = st.checkbox("Pokaż prędkości radialne v_r (true / pomiar / predykcja)", value=False, key="show_vr_vls")

    col1, col2 = st.columns(2)
    with col1:
        plot_xy(out["p_true"], out["p_vls_D"], out["beacons"],
                "Trajektoria (XY) – VLS: TDOA + Doppler", "Estymata VLS (TDOA + Doppler)")
    with col2:
        plot_error(out["t"], out["e_vls_D"],
                   "Błąd pozycji w czasie – VLS: TDOA + Doppler",
                   "e(t) VLS(TDOA+D)")

    if show_vr:
        st.write("Porównanie v_r dla pierwszych beaconów (czytelność ograniczona)")
        plot_vr_timeseries(out["t"], out["vr_true"], out["vr_hats"], out["vr_pred_vlsD"],
                           "v_r(t) – VLS: TDOA + Doppler")


# ============================================================
# Tab: EKF TDOA
# ============================================================

with tabs[3]:
    require_out()
    st.subheader("EKF: TDOA (bez Dopplera)")

    show_metrics(out["summary_ekf_noD"])

    col1, col2 = st.columns(2)
    with col1:
        plot_xy(out["p_true"], out["p_ekf_noD"], out["beacons"],
                "Trajektoria (XY) – EKF: TDOA", "Estymata EKF (TDOA)")
    with col2:
        plot_error(out["t"], out["e_ekf_noD"],
                   "Błąd pozycji w czasie – EKF: TDOA",
                   "e(t) EKF(TDOA)")


# ============================================================
# Tab: EKF TDOA + Doppler
# ============================================================

with tabs[4]:
    require_out()
    st.subheader("EKF: TDOA + Doppler (robust)")

    show_metrics(out["summary_ekf_D"])

    show_vr = st.checkbox("Pokaż prędkości radialne v_r (true / pomiar / predykcja)", value=False, key="show_vr_ekf")

    col1, col2 = st.columns(2)
    with col1:
        plot_xy(out["p_true"], out["p_ekf_D"], out["beacons"],
                "Trajektoria (XY) – EKF: TDOA + Doppler", "Estymata EKF (TDOA + Doppler)")
    with col2:
        plot_error(out["t"], out["e_ekf_D"],
                   "Błąd pozycji w czasie – EKF: TDOA + Doppler",
                   "e(t) EKF(TDOA+D)")

    if show_vr:
        st.write("Porównanie v_r dla pierwszych beaconów (czytelność ograniczona)")
        plot_vr_timeseries(out["t"], out["vr_true"], out["vr_hats"], out["vr_pred_ekfD"],
                           "v_r(t) – EKF: TDOA + Doppler")


# ============================================================
# Tab: Comparisons
# ============================================================

# ============================================================
# Tab: Comparisons
# ============================================================

with tabs[5]:
    require_out()
    st.subheader("Porównania metod")

    # tabela zbiorcza zostaje, bo to wygodne do pracy
    df_sum = pd.DataFrame([
        {"Metoda": "VLS: TDOA", **out["summary_vls_noD"]},
        {"Metoda": "VLS: TDOA + Doppler", **out["summary_vls_D"]},
        {"Metoda": "EKF: TDOA", **out["summary_ekf_noD"]},
        {"Metoda": "EKF: TDOA + Doppler", **out["summary_ekf_D"]},
    ])
    st.dataframe(df_sum, use_container_width=True)

    t = out["t"]

    # ------------------------------------------------------------
    # 1) VLS: bez vs z Dopplerem
    # ------------------------------------------------------------
    st.markdown("### 1) VLS: bez Dopplera vs z Dopplerem")

    col1, col2 = st.columns(2)
    with col1:
        fig = _fig(6.8, 5.0)
        ax = fig.add_subplot(111)
        ax.plot(t, out["e_vls_noD"], label="VLS: TDOA (bez Dopplera)", linewidth=2)
        ax.plot(t, out["e_vls_D"], label="VLS: TDOA + Doppler", linewidth=2)
        ax.set_title("Błąd pozycji e(t) – VLS")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("e(t) [m]")
        _grid(ax)
        ax.legend(loc="best")
        st.pyplot(fig, clear_figure=True)

    with col2:
        fig = _fig(6.8, 5.0)
        ax = fig.add_subplot(111)
        ax.hist(out["e_vls_noD"], bins=25, alpha=0.65, label="VLS: bez Dopplera")
        ax.hist(out["e_vls_D"], bins=25, alpha=0.65, label="VLS: z Dopplerem")
        ax.set_title("Histogram e – VLS")
        ax.set_xlabel("e [m]")
        ax.set_ylabel("liczność")
        _grid(ax)
        ax.legend(loc="best")
        st.pyplot(fig, clear_figure=True)

    # ------------------------------------------------------------
    # 2) EKF: bez vs z Dopplerem
    # ------------------------------------------------------------
    st.markdown("### 2) EKF: bez Dopplera vs z Dopplerem")

    col1, col2 = st.columns(2)
    with col1:
        fig = _fig(6.8, 5.0)
        ax = fig.add_subplot(111)
        ax.plot(t, out["e_ekf_noD"], label="EKF: TDOA (bez Dopplera)", linewidth=2)
        ax.plot(t, out["e_ekf_D"], label="EKF: TDOA + Doppler", linewidth=2)
        ax.set_title("Błąd pozycji e(t) – EKF")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("e(t) [m]")
        _grid(ax)
        ax.legend(loc="best")
        st.pyplot(fig, clear_figure=True)

    with col2:
        fig = _fig(6.8, 5.0)
        ax = fig.add_subplot(111)
        ax.hist(out["e_ekf_noD"], bins=25, alpha=0.65, label="EKF: bez Dopplera")
        ax.hist(out["e_ekf_D"], bins=25, alpha=0.65, label="EKF: z Dopplerem")
        ax.set_title("Histogram e – EKF")
        ax.set_xlabel("e [m]")
        ax.set_ylabel("liczność")
        _grid(ax)
        ax.legend(loc="best")
        st.pyplot(fig, clear_figure=True)

    # ------------------------------------------------------------
    # 3) VLS bez Dopplera vs EKF bez Dopplera
    # ------------------------------------------------------------
    st.markdown("### 3) VLS bez Dopplera vs EKF bez Dopplera")

    col1, col2 = st.columns(2)
    with col1:
        fig = _fig(6.8, 5.0)
        ax = fig.add_subplot(111)
        ax.plot(t, out["e_vls_noD"], label="VLS: TDOA (bez Dopplera)", linewidth=2)
        ax.plot(t, out["e_ekf_noD"], label="EKF: TDOA (bez Dopplera)", linewidth=2)
        ax.set_title("Błąd pozycji e(t) – porównanie bez Dopplera")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("e(t) [m]")
        _grid(ax)
        ax.legend(loc="best")
        st.pyplot(fig, clear_figure=True)

    with col2:
        fig = _fig(6.8, 5.0)
        ax = fig.add_subplot(111)
        ax.hist(out["e_vls_noD"], bins=25, alpha=0.65, label="VLS: bez Dopplera")
        ax.hist(out["e_ekf_noD"], bins=25, alpha=0.65, label="EKF: bez Dopplera")
        ax.set_title("Histogram e – porównanie bez Dopplera")
        ax.set_xlabel("e [m]")
        ax.set_ylabel("liczność")
        _grid(ax)
        ax.legend(loc="best")
        st.pyplot(fig, clear_figure=True)

    # ------------------------------------------------------------
    # 4) VLS z Dopplerem vs EKF z Dopplerem
    # ------------------------------------------------------------
    st.markdown("### 4) VLS z Dopplerem vs EKF z Dopplerem")

    col1, col2 = st.columns(2)
    with col1:
        fig = _fig(6.8, 5.0)
        ax = fig.add_subplot(111)
        ax.plot(t, out["e_vls_D"], label="VLS: TDOA + Doppler", linewidth=2)
        ax.plot(t, out["e_ekf_D"], label="EKF: TDOA + Doppler", linewidth=2)
        ax.set_title("Błąd pozycji e(t) – porównanie z Dopplerem")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("e(t) [m]")
        _grid(ax)
        ax.legend(loc="best")
        st.pyplot(fig, clear_figure=True)

    with col2:
        fig = _fig(6.8, 5.0)
        ax = fig.add_subplot(111)
        ax.hist(out["e_vls_D"], bins=25, alpha=0.65, label="VLS: z Dopplerem")
        ax.hist(out["e_ekf_D"], bins=25, alpha=0.65, label="EKF: z Dopplerem")
        ax.set_title("Histogram e – porównanie z Dopplerem")
        ax.set_xlabel("e [m]")
        ax.set_ylabel("liczność")
        _grid(ax)
        legend_outside(fig, ax, ncol=1, right=0.80)



# ============================================================
# Tab: Monte Carlo
# ============================================================

with tabs[6]:
    st.subheader("Monte-Carlo")
    st.caption("Wiele uruchomień z różnymi seedami, agregacja statystyk metryk.")
    if not do_mc:
        st.info("Zaznacz **Wykonaj Monte-Carlo** w panelu bocznym.")
    else:
        require_out()
        mc = run_monte_carlo(config, M=int(M), seed0=int(seed0))
        st.write("Wyniki per próba")
        st.dataframe(mc["runs"], use_container_width=True)
        st.write("Agregacja (mean/std/min/max)")
        st.dataframe(mc["agg"], use_container_width=True)


# ============================================================
# Tab: Export
# ============================================================

with tabs[7]:
    require_out()
    st.subheader("Eksport (reprodukowalność)")
    st.caption("Paczka ZIP: config + timeseries + summary.")

    df_sum = pd.DataFrame([
        {"Metoda": "VLS: TDOA", **out["summary_vls_noD"]},
        {"Metoda": "VLS: TDOA + Doppler", **out["summary_vls_D"]},
        {"Metoda": "EKF: TDOA", **out["summary_ekf_noD"]},
        {"Metoda": "EKF: TDOA + Doppler", **out["summary_ekf_D"]},
    ])

    df_ts = pd.DataFrame({
        "t": out["t"],

        "true_x": out["p_true"][:, 0],
        "true_y": out["p_true"][:, 1],
        "true_z": out["p_true"][:, 2],

        "vls0_x": out["p_vls_noD"][:, 0],
        "vls0_y": out["p_vls_noD"][:, 1],
        "vls0_z": out["p_vls_noD"][:, 2],

        "vlsD_x": out["p_vls_D"][:, 0],
        "vlsD_y": out["p_vls_D"][:, 1],
        "vlsD_z": out["p_vls_D"][:, 2],

        "ekf0_x": out["p_ekf_noD"][:, 0],
        "ekf0_y": out["p_ekf_noD"][:, 1],
        "ekf0_z": out["p_ekf_noD"][:, 2],

        "ekfD_x": out["p_ekf_D"][:, 0],
        "ekfD_y": out["p_ekf_D"][:, 1],
        "ekfD_z": out["p_ekf_D"][:, 2],

        "e_vls0": out["e_vls_noD"],
        "e_vlsD": out["e_vls_D"],
        "e_ekf0": out["e_ekf_noD"],
        "e_ekfD": out["e_ekf_D"],
    })

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.json", json.dumps(out["config"], indent=2))
        zf.writestr("summary.csv", df_sum.to_csv(index=False))
        zf.writestr("timeseries.csv", df_ts.to_csv(index=False))

    st.download_button(
        label="Pobierz paczkę ZIP (config + CSV)",
        data=mem.getvalue(),
        file_name="experiment_bundle.zip",
        mime="application/zip",
    )
