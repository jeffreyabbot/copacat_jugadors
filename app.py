# app.py
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import base64
from src.data_loader import CACHE_DIR, rebuild_cache
from src.utils import (
    compute_pbp_advanced_stats,
    get_assist_combos,
    get_defensive_actions,
    get_individual_four_factors,
    aggregate_lineup_data,
    get_5man_lineup_summary,
    is_player_in_lineup,
)

#python -m streamlit run app.py


st.set_page_config(
    page_title="CB Argentona 26-27",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)
# -------------------------------------------------------------
# CONTROL D'ACCÉS (LOGIN SCREEN)
# -------------------------------------------------------------
def check_password():
    """Returns True if the user is authenticated, otherwise stops execution."""
    if st.session_state.get("authenticated", False):
        return True

    st.title("🔒 Dashboard CB Argentona - Accés Staff")
    correct_password = st.secrets.get("auth", {}).get("password", None)

    if not correct_password:
        st.error("⚠️ La contrasenya no està configurada als Secrets de Streamlit.")
        st.stop()

    col_l1, col_l2, _ = st.columns([1.2, 1, 1])
    with col_l1:
        with st.form("login_form"):
            entered_password = st.text_input("Introdueix la contrasenya de l'Staff:", type="password")
            submit = st.form_submit_button("Entrar", use_container_width=True)
            if submit:
                if entered_password == correct_password:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("❌ Contrasenya incorrecta.")
    return False

if not check_password():
    st.stop()  # Hides all sidebar, data loading, and dashboards until authenticated
st.markdown(
    """
<style>
    /* 1. LIMITAR AMPLADA A TOTES LES VERSIONS DE STREAMLIT */
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    .block-container,
    .stMainBlockContainer,
    section[data-testid="stMain"] > div {
        max-width: 1300px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* 2. Targetes KPI de mida uniforme */
    .stMetric {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        min-height: 125px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem;
        font-weight: 700;
        color: #212529;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        font-weight: 600;
        color: #495057;
    }

    /* 3. Deltes: 🔴 VERMELL per a Positiu / Avantatge */
    div[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]) {
        color: #d00000 !important;
        background-color: rgba(208, 0, 0, 0.12) !important;
        border-radius: 6px;
        padding: 3px 8px;
        width: fit-content;
        font-weight: 600;
    }
    div[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]) svg {
        fill: #d00000 !important;
    }

    /* 4. Deltes: 🔵 BLAU per a Negatiu / Desavantatge */
    div[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) {
        color: #0077b6 !important;
        background-color: rgba(0, 119, 182, 0.12) !important;
        border-radius: 6px;
        padding: 3px 8px;
        width: fit-content;
        font-weight: 600;
    }
    div[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) svg {
        fill: #0077b6 !important;
    }
    /* Amplada intel·ligent del Sidebar: 360px en PC i adaptable en mòbil */
    [data-testid="stSidebar"] {
        min-width: min(360px, 85vw) !important;
    }

    /* Permetre que el text llarg del partit salti de línia còmodament */
    [data-testid="stSidebar"] div[data-baseweb="select"] div {
        white-space: normal !important;
        line-height: 1.3 !important;
    }
    /* Ajust adaptatiu perquè el text ON · OFF no es talli mai */
    div[data-testid="stMetricValue"] {
        font-size: clamp(1.2rem, 1.7vw, 1.55rem) !important;
        white-space: normal !important;
        line-height: 1.25 !important;
    }

    /* Pastilla neutra per a la quota de participació (sense fletxa enganyosa) */
    div[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Off"]) {
        color: #495057 !important;
        background-color: #e9ecef !important;
        border-radius: 6px;
        padding: 3px 8px;
        width: fit-content;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data
def get_cached_data():
    required_files = [
        CACHE_DIR / "games_index.parquet",
        CACHE_DIR / "all_boxscores.parquet",
        CACHE_DIR / "all_lineups.parquet",
        CACHE_DIR / "all_advanced.parquet",
        CACHE_DIR / "all_pbp.parquet",  # 5è fitxer
    ]
    if not all(f.exists() for f in required_files):
        return rebuild_cache()

    try:
        games = pd.read_parquet(CACHE_DIR / "games_index.parquet")
        box = pd.read_parquet(CACHE_DIR / "all_boxscores.parquet")
        lineups = pd.read_parquet(CACHE_DIR / "all_lineups.parquet")
        adv = pd.read_parquet(CACHE_DIR / "all_advanced.parquet")
        pbp = pd.read_parquet(CACHE_DIR / "all_pbp.parquet")  # Carregar pbp
        return games, box, lineups, adv, pbp  # Retornar els 5
    except Exception:
        return rebuild_cache()


# Desempaquetem exactament els 5 DataFrames:
games_df, box_df, lineups_df, adv_df, pbp_df = get_cached_data()
# Filtre de seguretat: assegurar que box_df només té jugadors individuals
if not box_df.empty and "Player" in box_df.columns:
    p_up = box_df["Player"].astype(str).str.upper().str.strip()
    box_df = box_df[
        ~p_up.isin(
            ["TOTALS", "TOTAL", "OPPONENT", "RIVAL", "TEAM TOTAL", "EQUIP"]
        )
    ].copy()

# --- BARRA LATERAL ---
# 1. Mostrar el logo si existeix a la carpeta assets o a l'arrel
logo_candidates = [
    Path("assets/logo.png"),
    Path("assets/logo.jpg"),
    Path("assets/logo.jpeg"),
    Path("logo.png"),
    Path("logo.jpg"),
]

for candidate in logo_candidates:
    if candidate.exists():
        with open(candidate, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        st.sidebar.markdown(
            f"""
            <div style="display: flex; justify-content: center; width: 100%; margin-bottom: 10px;">
                <img src="data:image/png;base64,{logo_b64}" style="width: 180px; border-radius: 14px;">
            </div>
            """,
            unsafe_allow_html=True,
        )
        break
st.sidebar.title("CB Argentona 2026-2027")

if st.sidebar.button("🔄 Sincronitzar Dades", use_container_width=True):
    st.cache_data.clear()
    rebuild_cache()
    st.sidebar.success("Dades actualitzades!")
    st.rerun()

if games_df.empty:
    st.warning("⚠️ No s'han trobat partits a `data/raw/`.")
    st.stop()

# Filtre de competició
all_comps = ["Totes les competicions"] + sorted(
    games_df["competition"].dropna().unique().tolist()
)

# Detecta automàticament 'Copa Catalunya' (o 'Copa_Catalunya') si existeix; si no, deixa el primer
default_idx = 0
for i, comp in enumerate(all_comps):
    c_clean = str(comp).lower().replace("_", " ")
    if "copa" in c_clean or "catalunya" in c_clean:
        default_idx = i
        break

selected_comp = st.sidebar.selectbox("Competició", all_comps, index=default_idx)

filtered_games = games_df.copy()
if selected_comp != "Totes les competicions":
    filtered_games = filtered_games[
        filtered_games["competition"] == selected_comp
    ]

view_mode = st.sidebar.radio(
    "Mode de visualització",
    ["Partit Individual", "Totals Acumulats"],
    horizontal=True,
)

# -------------------------------------------------------------
# FUNCIONS AUXILIARS D'ESTIL I OUTLIERS (AMB ALINEACIÓ A LA DRETA)
# -------------------------------------------------------------
STYLE_RED = "color: #d00000; font-weight: 700; text-align: right;"  # Bo / Destacat
STYLE_BLUE = "color: #0077b6; font-weight: 700; text-align: right;"  # Desfavorable
STYLE_NEUTRAL = "color: #212529; font-weight: 400; text-align: right;"
# Diccionari de traducció de mètriques avançades
METRIC_NAMES_CAT = {
    "Game Score": "Marcador del Partit",
    "1. Effective FG% (eFG%)": "1. % Tir de Camp Efectiu (eFG%)",
    "2. Turnover Rate (TO%)": "2. Ràtio de Pèrdues (TO%)",
    "3. Offensive Rebound% (OREB%)": "3. % Rebot Ofensiu (OREB%)",
    "4. Free Throw Rate (FT Rate)": "4. Freqüència de Tirs Lliures (FT Rate)",
    "Overall Points Per Shot (PPS)": "Punts per Tir Total (PPS)",
    "Paint Touch PPS": "PPS amb Toc a Pintura",
    "Non-Paint Touch PPS": "PPS sense Toc a Pintura",
    "Fastbreak PPS": "PPS en Transició (Contraatac)",
    "Half-Court PPS": "PPS en Atac Estàtic (5c5)",
    "Total Paint Touches": "Tocs a Pintura Totals",
    "Total Fastbreaks": "Transicions Totals",
    "2nd Chance Points": "Punts de 2a Oportunitat",
    "Defensive Kills (3 Stops)": "Kills Defensius (3 Aturades Seguides)",
    "Stocks (Steals + Blocks)": "Stocks (Recuperacions + Taps)",
    "Team Fouls": "Faltes d'Equip",
    "Points Off Live TO (PTS OFF TO)": "Punts de Pèrdua Viva (PTS OFF TO)",
    "Points Off Live TO": "Punts de Pèrdua Viva (PTS OFF TO)",
    "PTS OFF TO": "Punts de Pèrdua Viva (PTS OFF TO)",
    "Live Turnovers (LTO)": "Pèrdues Vives (LTO)",
    "Live Turnovers": "Pèrdues Vives (LTO)",
    "Dead Turnovers (DTO)": "Pèrdues Mortes (DTO)",
    "Dead Turnovers": "Pèrdues Mortes (DTO)",
}


# -------------------------------------------------------------
# FUNCIONS AUXILIARS D'ESTIL I OUTLIERS
# -------------------------------------------------------------
STYLE_RED = "color: #d00000; font-weight: 700;"  # Bo / Destacat
STYLE_BLUE = "color: #0077b6; font-weight: 700;"  # Desfavorable
STYLE_NEUTRAL = "color: #212529; font-weight: 400;"

def parse_min_to_float(min_val):
    """Converteix '24:30' o '00:11' a minuts decimals."""
    try:
        val_str = str(min_val).strip()
        if ":" in val_str:
            parts = val_str.split(":")
            return float(parts[0]) + float(parts[1]) / 60.0
        return float(val_str)
    except Exception:
        return 0.0
def parse_fraction(s):
    """Extreu encerts i intents d'un string tipus '2/3' o '2.0/3.0' de forma segura."""
    try:
        cleaned = str(s).replace("\u00a0", "").strip()
        parts = cleaned.split("/")
        if len(parts) == 2:
            return int(float(parts[0])), int(float(parts[1]))
        return 0, 0
    except Exception:
        return 0, 0

def render_boxscore(df, mode="Tradicional"):
    """Renderitza el Boxscore adaptant l'alçada automàticament per veure tots els jugadors sense scroll vertical."""
    calc_height = int((len(df) + 1) * 35.5) + 3
    
    # Configurem l'alineació a la dreta per a les columnes d'intents fraccionaris (A/I)
    col_cfg = {}
    for c in df.columns:
        if "(A/I)" in c or "2a Op" in c or "Transició" in c or "2a Oportunitat" in c:
            try:
                col_cfg[c] = st.column_config.Column(alignment="right")
            except Exception:
                pass

    st.dataframe(
        style_boxscore(df, mode),
        height=calc_height,
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg if col_cfg else None,
    )
    
STYLE_RED = "color: #d00000; font-weight: 700; text-align: right;"
STYLE_BLUE = "color: #0077b6; font-weight: 700; text-align: right;"
STYLE_NEUTRAL = "color: #212529; font-weight: 400; text-align: right;"


def style_boxscore(df_disp, mode="Tradicional"):
    def to_f(val, default=-999999.0):
        try:
            v = float(val)
            return v if not pd.isna(v) else default
        except Exception:
            return default

    def apply_row_styles(df):
        css_df = pd.DataFrame(
            "text-align: right;", index=df.index, columns=df.columns
        )

        for col_name in ["Jugador", "Passador", "Anotador", "Mètrica"]:
            if col_name in df.columns:
                css_df[col_name] = "text-align: left;"
        if "#" in df.columns:
            css_df["#"] = "text-align: center;"

        # Detectem si estem en acumulats (té PJ, o mode acumulat, o valors alts d'assistències totals)
        max_ast_val = to_f(df["AST"].max()) if "AST" in df.columns else 0.0
        is_accumulated = ("PJ" in df.columns) or ("acumulat" in str(mode).lower()) or (max_ast_val > 6.0)

        # Pre-càlcul d'outliers de plantilla per als totals acumulats (Opció A)
        col_outliers = {}
        if is_accumulated:
            for c in df.columns:
                num_s = pd.to_numeric(df[c], errors="coerce")
                if num_s.notna().sum() >= 3:
                    m = num_s.mean()
                    s = num_s.std()
                    if s > 0:
                        col_outliers[c] = (m, s)

        # Funcions per comprovar si un valor destaca positivament (vermell) o negativament (blau)
        def is_high_outlier(col_name, val, single_threshold):
            v = to_f(val)
            if is_accumulated and col_name in col_outliers:
                m, s = col_outliers[col_name]
                return v >= (m + 0.75 * s) and v > 0
            return v >= single_threshold

        def is_bad_outlier(col_name, val, single_threshold):
            v = to_f(val)
            if is_accumulated and col_name in col_outliers:
                m, s = col_outliers[col_name]
                return v >= (m + 0.75 * s) and v > 0
            return v >= single_threshold

        for i in df.index:
            row = df.loc[i]
            min_attempts = 6 if is_accumulated else 2

            # 1. Tradicional
            if mode == "Tradicional":
                if "%T2" in df.columns and "T2 (A/I)" in df.columns:
                    m, a = parse_fraction(row["T2 (A/I)"])
                    if a >= min_attempts:
                        pct = (m / a) * 100
                        if pct >= 70:
                            css_df.loc[i, "%T2"] = STYLE_RED
                        elif pct <= 25:
                            css_df.loc[i, "%T2"] = STYLE_BLUE

                if "%T3" in df.columns and "T3 (A/I)" in df.columns:
                    m, a = parse_fraction(row["T3 (A/I)"])
                    if a >= min_attempts:
                        pct = (m / a) * 100
                        if pct >= 50:
                            css_df.loc[i, "%T3"] = STYLE_RED
                        elif pct <= 20:
                            css_df.loc[i, "%T3"] = STYLE_BLUE

                if "%TL" in df.columns and "TL (A/I)" in df.columns:
                    m, a = parse_fraction(row["TL (A/I)"])
                    if a >= min_attempts:
                        pct = (m / a) * 100
                        if pct >= 85:
                            css_df.loc[i, "%TL"] = STYLE_RED
                        elif pct <= 40:
                            css_df.loc[i, "%TL"] = STYLE_BLUE

                if "REB_T" in df.columns and is_high_outlier("REB_T", row["REB_T"], 6):
                    css_df.loc[i, "REB_T"] = STYLE_RED
                if "AST" in df.columns and is_high_outlier("AST", row["AST"], 4):
                    css_df.loc[i, "AST"] = STYLE_RED
                if "REC" in df.columns and is_high_outlier("REC", row["REC"], 3):
                    css_df.loc[i, "REC"] = STYLE_RED
                if "PER" in df.columns and is_bad_outlier("PER", row["PER"], 3):
                    css_df.loc[i, "PER"] = STYLE_BLUE

            # 2. Zones de tir (Percentatges)
            elif mode == "Zones":
                for col in ["Aro (Rim)", "Pintura (Paint)", "Mitja Distància (MR)"]:
                    if col in df.columns:
                        m, a = parse_fraction(row[col])
                        if a >= min_attempts:
                            pct = (m / a) * 100
                            if pct >= 70:
                                css_df.loc[i, col] = STYLE_RED
                            elif pct == 0:
                                css_df.loc[i, col] = STYLE_BLUE

                for col in ["Triple Cantonada (C3)", "Triple Frontal (ATB3)"]:
                    if col in df.columns:
                        m, a = parse_fraction(row[col])
                        if a >= min_attempts:
                            pct = (m / a) * 100
                            if pct >= 50:
                                css_df.loc[i, col] = STYLE_RED
                            elif pct <= 15:
                                css_df.loc[i, col] = STYLE_BLUE

            # 3. 4 Factors Individuals (Percentatges i ràtios nacionals)
            elif mode == "4Factors":
                if "USG%" in df.columns and to_f(row["USG%"]) >= 28.0:
                    css_df.loc[i, "USG%"] = STYLE_RED
                if "1. eFG% (Tir)" in df.columns:
                    v = to_f(row["1. eFG% (Tir)"])
                    if v >= 60.0:
                        css_df.loc[i, "1. eFG% (Tir)"] = STYLE_RED
                    elif 0 < v <= 35.0:
                        css_df.loc[i, "1. eFG% (Tir)"] = STYLE_BLUE
                if "2. TOV% (Pèrdues)" in df.columns:
                    v = to_f(row["2. TOV% (Pèrdues)"])
                    if 0 <= v <= 8.0:
                        css_df.loc[i, "2. TOV% (Pèrdues)"] = STYLE_RED
                    elif v >= 25.0:
                        css_df.loc[i, "2. TOV% (Pèrdues)"] = STYLE_BLUE
                if "3. OREB% (Rebot)" in df.columns and to_f(row["3. OREB% (Rebot)"]) >= 12.0:
                    css_df.loc[i, "3. OREB% (Rebot)"] = STYLE_RED
                if "4. FT Rate (TL)" in df.columns and to_f(row["4. FT Rate (TL)"]) >= 0.25:
                    css_df.loc[i, "4. FT Rate (TL)"] = STYLE_RED

            # 4. Impacte i Context
            elif "context" in str(mode).lower() or "impacte" in str(mode).lower():
                if "Transició" in df.columns:
                    m, a = parse_fraction(row["Transició"])
                    if m >= min_attempts:
                        css_df.loc[i, "Transició"] = STYLE_RED
                    elif a >= min_attempts and m == 0:
                        css_df.loc[i, "Transició"] = STYLE_BLUE

                if "2a Oportunitat" in df.columns:
                    m, a = parse_fraction(row["2a Oportunitat"])
                    if m >= min_attempts:
                        css_df.loc[i, "2a Oportunitat"] = STYLE_RED
                    elif a >= min_attempts and m == 0:
                        css_df.loc[i, "2a Oportunitat"] = STYLE_BLUE

                for per_col in ["PER Totals", "Pèrdues"]:
                    if per_col in df.columns and is_bad_outlier(per_col, row[per_col], 3):
                        css_df.loc[i, per_col] = STYLE_BLUE

                if "Vives (LTO)" in df.columns and is_bad_outlier("Vives (LTO)", row["Vives (LTO)"], 2):
                    css_df.loc[i, "Vives (LTO)"] = STYLE_BLUE
                if "Faltes Com." in df.columns and is_bad_outlier("Faltes Com.", row["Faltes Com."], 4):
                    css_df.loc[i, "Faltes Com."] = STYLE_BLUE
                if "Faltes Reb." in df.columns and is_high_outlier("Faltes Reb.", row["Faltes Reb."], 4):
                    css_df.loc[i, "Faltes Reb."] = STYLE_RED

            # 5. Accions Defensives (Amb Outliers a Acumulats)
            elif mode == "Defensa":
                if "Defleccions" in df.columns and is_high_outlier("Defleccions", row["Defleccions"], 3):
                    css_df.loc[i, "Defleccions"] = STYLE_RED
                if "Robatoris (REC)" in df.columns and is_high_outlier("Robatoris (REC)", row["Robatoris (REC)"], 2):
                    css_df.loc[i, "Robatoris (REC)"] = STYLE_RED
                if "Taps Totals (TAP)" in df.columns and is_high_outlier("Taps Totals (TAP)", row["Taps Totals (TAP)"], 2):
                    css_df.loc[i, "Taps Totals (TAP)"] = STYLE_RED
                if "Impacte Defensiu (Total)" in df.columns and is_high_outlier("Impacte Defensiu (Total)", row["Impacte Defensiu (Total)"], 5):
                    css_df.loc[i, "Impacte Defensiu (Total)"] = STYLE_RED

            # 6. Assistències
            elif "assist" in str(mode).lower():
                for pts_col in ["PTS Generats", "Punts Generats"]:
                    if pts_col in df.columns and is_high_outlier(pts_col, row[pts_col], 8):
                        css_df.loc[i, pts_col] = STYLE_RED
                if "AST" in df.columns and is_high_outlier("AST", row["AST"], 4):
                    css_df.loc[i, "AST"] = STYLE_RED

            # 7. Rebot i Segona Oportunitat
            elif "rebot" in str(mode).lower():
                if "REB_T" in df.columns and is_high_outlier("REB_T", row["REB_T"], 6):
                    css_df.loc[i, "REB_T"] = STYLE_RED
                if "REB_O" in df.columns and is_high_outlier("REB_O", row["REB_O"], 3):
                    css_df.loc[i, "REB_O"] = STYLE_RED
                if "Pts 2a Op." in df.columns and is_high_outlier("Pts 2a Op.", row["Pts 2a Op."], 4):
                    css_df.loc[i, "Pts 2a Op."] = STYLE_RED
                if "Fallades 2a Op." in df.columns and is_bad_outlier("Fallades 2a Op.", row["Fallades 2a Op."], 2):
                    css_df.loc[i, "Fallades 2a Op."] = STYLE_BLUE
                if "% 2a Op." in df.columns:
                    v = to_f(row["% 2a Op."])
                    if v >= 65.0:
                        css_df.loc[i, "% 2a Op."] = STYLE_RED
                    elif 0 < v <= 30.0:
                        css_df.loc[i, "% 2a Op."] = STYLE_BLUE
                if "OREB%" in df.columns and to_f(row["OREB%"]) >= 12.0:
                    css_df.loc[i, "OREB%"] = STYLE_RED
                if "REB/Min" in df.columns:
                    v_rm = to_f(row["REB/Min"])
                    if v_rm >= 0.35:
                        css_df.loc[i, "REB/Min"] = STYLE_RED
                    elif 0 <= v_rm <= 0.12 and to_f(row.get("REB_T", 0)) <= 3:
                        css_df.loc[i, "REB/Min"] = STYLE_BLUE

        return css_df

    styler = df_disp.style.apply(apply_row_styles, axis=None)

    fmt_dict = {}
    for col in df_disp.columns:
        if col in ["%T2", "%T3", "%TL", "% Taps Recuperats", "% 2a Op."]:
            if df_disp[col].dtype in [float, np.float64, int, np.int64]:
                fmt_dict[col] = "{:.0f}%"
        elif (
            "%" in col
            or "eFG" in col
            or "TOV" in col
            or "OREB" in col
            or "USG" in col
        ):
            if df_disp[col].dtype in [float, np.float64, int, np.int64]:
                fmt_dict[col] = "{:.1f}%"
        elif col in ["FT Rate", "REB/Min"]:
            if df_disp[col].dtype in [float, np.float64]:
                fmt_dict[col] = "{:.2f}"
        elif col in ["Plays", "PPP", "Reb/P"]:
            if df_disp[col].dtype in [float, np.float64]:
                fmt_dict[col] = "{:.1f}"

    if fmt_dict:
        styler = styler.format(fmt_dict, na_rep="-")

    return styler
import unicodedata

def clean_txt(t):
    """Elimina accents, espais i passa a minúscules de forma segura."""
    return "".join(
        c for c in unicodedata.normalize("NFD", str(t))
        if unicodedata.category(c) != "Mn"
    ).lower().strip()

def render_on_off_dashboard(l_df, b_df, key_prefix="single"):
    """Renderitza el panell d'On/Off basat exclusivament en possessions reals."""
    if l_df.empty:
        st.info("No hi ha dades de quintets disponibles per aquesta selecció.")
        return

    # 1. Llista de jugadors
    clean_b = b_df[~b_df["Player"].astype(str).str.lower().isin(["totals", "total", "opponent", "team", "equip"])].copy()
    grouped_p = clean_b.groupby(["Player", "Name"]).size().reset_index()

    player_options = []
    for _, r in grouped_p.iterrows():
        p_num = str(r["Player"]).replace("#", "").strip()
        p_name = str(r["Name"]).strip()
        player_options.append((f"#{p_num} {p_name}", p_num, p_name))

    def sort_p(item):
        try:
            return int(item[1])
        except Exception:
            return 999
    player_options.sort(key=sort_p)

    if not player_options:
        st.warning("No s'han trobat jugadors per analitzar.")
        return

    col_sel, _ = st.columns([1.5, 1])
    with col_sel:
        selected_p_tuple = st.selectbox(
            "👤 Selecciona el Jugador per a l'anàlisi On / Off:",
            options=player_options,
            format_func=lambda x: x[0],
            key=f"{key_prefix}_player_select",
        )
    
    p_label, p_num, p_name = selected_p_tuple

    # 2. Filtrar quintets ON i OFF (Dades 100% de Lineups)
    is_on_mask = l_df.apply(lambda r: is_player_in_lineup(r, p_num, p_name), axis=1)
    df_on = l_df[is_on_mask].copy()
    df_off = l_df[~is_on_mask].copy()

    st_on = aggregate_lineup_data(df_on)
    st_off = aggregate_lineup_data(df_off)

    # 3. Càlcul de possessions i mètriques netes
    poss_on = st_on.get("poss", 0)
    poss_off = st_off.get("poss", 0)
    disp_on_poss = int(round(poss_on))
    disp_off_poss = int(round(poss_off))
    tot_poss = disp_on_poss + disp_off_poss

    pct_poss = (poss_on / tot_poss * 100.0) if tot_poss > 0 else 0.0

    net_on = st_on.get("net_rtg", 0)
    net_off = st_off.get("net_rtg", 0)
    oer_on = st_on.get("oer", 0)
    oer_off = st_off.get("oer", 0)
    der_on = st_on.get("der", 0)
    der_off = st_off.get("der", 0)
    pm_on = st_on.get("plus_minus", 0)
    pm_off = st_off.get("plus_minus", 0)

    # 4. TARGETES KPI (5 columnes completes: Volum, Net, OER, DER, +/-)
    k1, k2, k3, k4, k5 = st.columns(5)
    
    k1.metric(
        "🏀 Possessions Jugades",
        f"{disp_on_poss} ON · {disp_off_poss} OFF",
        delta=f"Participació: {pct_poss:.0f}% del total",
        delta_color="off",
    )
    k2.metric(
        "⚡ Net Rating (Pts/100)",
        f"{net_on:+.1f}",
        delta=f"{net_on - net_off:+.1f} vs quan seu",
    )
    k3.metric(
        "🎯 Eficiència Atac (OER)",
        f"{oer_on:.1f} pts",
        delta=f"{oer_on - oer_off:+.1f} pts/100 vs banqueta",
    )
    k4.metric(
        "🛡️ Eficiència Defensa (DER)",
        f"{der_on:.1f} pts",
        delta=f"{der_on - der_off:+.1f} pts/100 vs banqueta",
        delta_color="inverse",  # En defensa, encaixar menys punts és positiu
    )
    k5.metric(
        "⚖️ Marcador +/- Real",
        f"{int(pm_on):+d} pts",
        delta=f"{int(pm_on - pm_off):+d} pts marge net",
    )

    st.markdown("---")

    # 5. Taules d'Atac i Defensa (100% immunes a errors de minuts)
    st.subheader(f"📊 Impacte On vs Off de {p_label}")
    col_offense, col_defense = st.columns(2)

    with col_offense:
        st.markdown("#### 🔴 Atac d'Argentona")
        df_offense = pd.DataFrame([
            {"Mètrica": "1. Tir Efectiu (eFG%)", "Amb Jugador (ON)": f"{st_on.get('efg', 0):.1f}%", "Sense Jugador (OFF)": f"{st_off.get('efg', 0):.1f}%", "Diferencial": f"{st_on.get('efg',0)-st_off.get('efg',0):+.1f}%"},
            {"Mètrica": "2. % Tir de 2 (2P%)", "Amb Jugador (ON)": f"{st_on.get('pct_2p', 0):.1f}% ({int(st_on.get('2pm',0))}/{int(st_on.get('2pa',0))})", "Sense Jugador (OFF)": f"{st_off.get('pct_2p', 0):.1f}% ({int(st_off.get('2pm',0))}/{int(st_off.get('2pa',0))})", "Diferencial": f"{st_on.get('pct_2p',0)-st_off.get('pct_2p',0):+.1f}%"},
            {"Mètrica": "3. % Triples (3P%)", "Amb Jugador (ON)": f"{st_on.get('pct_3p', 0):.1f}% ({int(st_on.get('3pm',0))}/{int(st_on.get('3pa',0))})", "Sense Jugador (OFF)": f"{st_off.get('pct_3p', 0):.1f}% ({int(st_off.get('3pm',0))}/{int(st_off.get('3pa',0))})", "Diferencial": f"{st_on.get('pct_3p',0)-st_off.get('pct_3p',0):+.1f}%"},
            {"Mètrica": "4. Ràtio de Triples (3PAr)", "Amb Jugador (ON)": f"{st_on.get('3par', 0):.1f}% dels tirs", "Sense Jugador (OFF)": f"{st_off.get('3par', 0):.1f}% dels tirs", "Diferencial": f"{st_on.get('3par',0)-st_off.get('3par',0):+.1f}%"},
            {"Mètrica": "5. Eficiència Ofensiva (OER)", "Amb Jugador (ON)": f"{st_on.get('oer', 0):.1f} pts/100", "Sense Jugador (OFF)": f"{st_off.get('oer', 0):.1f} pts/100", "Diferencial": f"{st_on.get('oer',0)-st_off.get('oer',0):+.1f}"},
            {"Mètrica": "% Rebot Ofensiu (OREB%)", "Amb Jugador (ON)": f"{st_on.get('oreb_pct', 0):.1f}%", "Sense Jugador (OFF)": f"{st_off.get('oreb_pct', 0):.1f}%", "Diferencial": f"{st_on.get('oreb_pct',0)-st_off.get('oreb_pct',0):+.1f}%"},
            {"Mètrica": "Ràtio de Pèrdues (TOV%)", "Amb Jugador (ON)": f"{st_on.get('tov_pct', 0):.1f}%", "Sense Jugador (OFF)": f"{st_off.get('tov_pct', 0):.1f}%", "Diferencial": f"{st_on.get('tov_pct',0)-st_off.get('tov_pct',0):+.1f}%"},
            {"Mètrica": "Freqüència TL (FT Rate)", "Amb Jugador (ON)": f"{st_on.get('ft_rate', 0):.2f}", "Sense Jugador (OFF)": f"{st_off.get('ft_rate', 0):.2f}", "Diferencial": f"{st_on.get('ft_rate',0)-st_off.get('ft_rate',0):+.2f}"},
        ])
        st.dataframe(df_offense, use_container_width=True, hide_index=True)

    with col_defense:
        st.markdown("#### 🛡️ Defensa (Rivals)")
        df_defense = pd.DataFrame([
            {"Mètrica": "1. Tir Efectiu Rival (eFG%)", "Amb Jugador (ON)": f"{st_on.get('opp_efg', 0):.1f}%", "Sense Jugador (OFF)": f"{st_off.get('opp_efg', 0):.1f}%", "Diferencial": f"{st_on.get('opp_efg',0)-st_off.get('opp_efg',0):+.1f}%"},
            {"Mètrica": "2. % Tir de 2 Rival (2P%)", "Amb Jugador (ON)": f"{st_on.get('opp_pct_2p', 0):.1f}% ({int(st_on.get('opp_2pm',0))}/{int(st_on.get('opp_2pa',0))})", "Sense Jugador (OFF)": f"{st_off.get('opp_pct_2p', 0):.1f}% ({int(st_off.get('opp_2pm',0))}/{int(st_off.get('opp_2pa',0))})", "Diferencial": f"{st_on.get('opp_pct_2p',0)-st_off.get('opp_pct_2p',0):+.1f}%"},
            {"Mètrica": "3. % Triples Rival (3P%)", "Amb Jugador (ON)": f"{st_on.get('opp_pct_3p', 0):.1f}% ({int(st_on.get('opp_3pm',0))}/{int(st_on.get('opp_3pa',0))})", "Sense Jugador (OFF)": f"{st_off.get('opp_pct_3p', 0):.1f}% ({int(st_off.get('opp_3pm',0))}/{int(st_off.get('opp_3pa',0))})", "Diferencial": f"{st_on.get('opp_pct_3p',0)-st_off.get('opp_pct_3p',0):+.1f}%"},
            {"Mètrica": "4. Ràtio Triples Rival (3PAr)", "Amb Jugador (ON)": f"{st_on.get('opp_3par', 0):.1f}% dels tirs", "Sense Jugador (OFF)": f"{st_off.get('opp_3par', 0):.1f}% dels tirs", "Diferencial": f"{st_on.get('opp_3par',0)-st_off.get('opp_3par',0):+.1f}%"},
            {"Mètrica": "5. Eficiència Defensiva (DER)", "Amb Jugador (ON)": f"{st_on.get('der', 0):.1f} pts/100", "Sense Jugador (OFF)": f"{st_off.get('der', 0):.1f} pts/100", "Diferencial": f"{st_on.get('der',0)-st_off.get('der',0):+.1f}"},
            {"Mètrica": "% Rebot Ofensiu Rival (OREB%)", "Amb Jugador (ON)": f"{st_on.get('opp_oreb_pct', 0):.1f}%", "Sense Jugador (OFF)": f"{st_off.get('opp_oreb_pct', 0):.1f}%", "Diferencial": f"{st_on.get('opp_oreb_pct',0)-st_off.get('opp_oreb_pct',0):+.1f}%"},
            {"Mètrica": "Pèrdues Forçades Rival (TOV%)", "Amb Jugador (ON)": f"{st_on.get('opp_tov_pct', 0):.1f}%", "Sense Jugador (OFF)": f"{st_off.get('opp_tov_pct', 0):.1f}%", "Diferencial": f"{st_on.get('opp_tov_pct',0)-st_off.get('opp_tov_pct',0):+.1f}%"},
            {"Mètrica": "Freqüència TL Rival (FT Rate)", "Amb Jugador (ON)": f"{st_on.get('opp_ft_rate', 0):.2f}", "Sense Jugador (OFF)": f"{st_off.get('opp_ft_rate', 0):.2f}", "Diferencial": f"{st_on.get('opp_ft_rate',0)-st_off.get('opp_ft_rate',0):+.2f}"},
        ])
        st.dataframe(df_defense, use_container_width=True, hide_index=True)

    # 6. Taula de Quintets de 5 Jugadors
    st.markdown("---")
    st.subheader("👥 Quintets de 5 Jugadors")
    col_q1, col_q2 = st.columns([1, 2])
    with col_q1:
        filter_player_lineups = st.checkbox(
            f"Mostrar només quintets amb {p_label}",
            value=False,
            key=f"{key_prefix}_check_p_lineups"
        )
    with col_q2:
        min_poss_val = st.slider(
            "Possessions mínimes jugades pel quintet",
            min_value=0,
            max_value=100 if key_prefix == "season" else 30,
            value=5 if key_prefix == "season" else 1,
            step=1,
            key=f"{key_prefix}_slider_lineup_poss"
        )

    target_lineups = df_on if filter_player_lineups else l_df
    df_5man = get_5man_lineup_summary(target_lineups, b_df, min_minutes=0.0)

    # Filtrem la taula directament per possessions reals
    if not df_5man.empty and "POSS" in df_5man.columns:
        df_5man = df_5man[df_5man["POSS"] >= min_poss_val].reset_index(drop=True)
        # Eliminem la columna MIN de la taula si no és fiable
        if "MIN" in df_5man.columns:
            df_5man = df_5man.drop(columns=["MIN"])

    if not df_5man.empty:
        calc_h = int((len(df_5man) + 1) * 36) + 3
        st.dataframe(
            df_5man,
            height=calc_h,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No hi ha quintets que compleixin els filtres de possessions seleccionats.")
# =============================================================
# MODO 1: PARTIT INDIVIDUAL
# =============================================================
if view_mode == "Partit Individual":
    game_dict = dict(zip(filtered_games["game_id"], filtered_games["name"]))
    selected_game_id = st.sidebar.selectbox(
        "Seleccionar Partit",
        options=list(game_dict.keys()),
        format_func=lambda x: game_dict.get(x, x),
    )

    st.title(f"{game_dict.get(selected_game_id, 'Partit')}")

    # Barra superior de navegació persistent
    active_tab = st.segmented_control(
        "Navegació",
        options=["📈 Mètriques i 4 Factors", "📋 Box Score", "👥 Quintets i On/Off"],
        default="📈 Mètriques i 4 Factors",
        key="main_active_tab",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)


# -------------------------------------------------------------
# 1. PESTAÑA: MÈTRIQUES, 4 FACTORS I EFICIÈNCIA (PARTIT INDIVIDUAL)
# -------------------------------------------------------------

    if active_tab == "📈 Mètriques i 4 Factors":
        g_adv = adv_df[adv_df["game_id"] == str(selected_game_id)].copy()

        if not g_adv.empty:
            g_box_m = box_df[
                box_df["game_id"] == str(selected_game_id)
            ].drop_duplicates(subset=["Player"])
            g_line_m = lineups_df[
                lineups_df["game_id"] == str(selected_game_id)
            ].drop_duplicates(subset=["Lineup"])

            adv_dict_team = dict(zip(g_adv["Metric"], g_adv["Team_Value"]))
            adv_dict_opp = dict(zip(g_adv["Metric"], g_adv["Opp_Value"]))

            # 1. SUMATORIS REALS D'ARGENTONA
            s_team_rim_m = float(
                pd.to_numeric(g_box_m["RIM(M)"], errors="coerce").fillna(0).sum()
            )
            s_team_rim_a = float(
                pd.to_numeric(g_box_m["RIM(A)"], errors="coerce").fillna(0).sum()
            )
            s_team_paint_m = float(
                pd.to_numeric(g_box_m["PAINT(M)"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_team_paint_a = float(
                pd.to_numeric(g_box_m["PAINT(A)"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_team_mr_m = float(
                pd.to_numeric(g_box_m["MR(M)"], errors="coerce").fillna(0).sum()
            )
            s_team_mr_a = float(
                pd.to_numeric(g_box_m["MR(A)"], errors="coerce").fillna(0).sum()
            )
            s_team_c3_m = float(
                pd.to_numeric(g_box_m["C3(M)"], errors="coerce").fillna(0).sum()
            )
            s_team_c3_a = float(
                pd.to_numeric(g_box_m["C3(A)"], errors="coerce").fillna(0).sum()
            )
            s_team_atb3_m = float(
                pd.to_numeric(g_box_m["ATB3(M)"], errors="coerce").fillna(0).sum()
            )
            s_team_atb3_a = float(
                pd.to_numeric(g_box_m["ATB3(A)"], errors="coerce").fillna(0).sum()
            )

            s_team_2pm = s_team_rim_m + s_team_paint_m + s_team_mr_m
            s_team_2pa = s_team_rim_a + s_team_paint_a + s_team_mr_a
            s_team_3pm = s_team_c3_m + s_team_atb3_m
            s_team_3pa = s_team_c3_a + s_team_atb3_a
            s_team_fgm = s_team_2pm + s_team_3pm
            s_team_fga = s_team_2pa + s_team_3pa

            s_team_ftm = float(
                pd.to_numeric(g_box_m["FTM"], errors="coerce").fillna(0).sum()
            )
            s_team_fta = float(
                pd.to_numeric(g_box_m["FTA"], errors="coerce").fillna(0).sum()
            )
            s_team_oreb = float(
                pd.to_numeric(g_box_m["Off Reb"], errors="coerce").fillna(0).sum()
            )
            s_team_dreb = float(
                pd.to_numeric(g_box_m["Def Reb"], errors="coerce").fillna(0).sum()
            )
            s_team_tov = float(
                pd.to_numeric(g_box_m["Turnovers"], errors="coerce")
                .fillna(0)
                .sum()
            )

            s_team_paint_pts = (s_team_rim_m + s_team_paint_m) * 2.0
            s_team_paint_fga = s_team_rim_a + s_team_paint_a

            s_team_nopaint_pts = (
                s_team_mr_m * 2.0 + s_team_c3_m * 3.0 + s_team_atb3_m * 3.0
            )
            s_team_nopaint_fga = s_team_mr_a + s_team_c3_a + s_team_atb3_a
            s_team_fg_pts = s_team_paint_pts + s_team_nopaint_pts

            # Marcador oficial
            score_row = g_adv[
                g_adv["Metric"]
                .astype(str)
                .str.contains("Game Score|Score|Marcador", case=False, na=False)
            ]
            if not score_row.empty:
                s_team_pts = float(score_row["Team_Value"].values[0])
                s_opp_pts = float(score_row["Opp_Value"].values[0])
            else:
                s_team_pts = s_team_fg_pts + s_team_ftm
                s_opp_pts = float(
                    pd.to_numeric(g_line_m["PTS_Agn"], errors="coerce")
                    .fillna(0)
                    .sum()
                )

            # 2. SUMATORIS REALS DEL RIVAL
            s_opp_rim_m = float(
                pd.to_numeric(g_line_m["Rim_FGM_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_opp_rim_a = float(
                pd.to_numeric(g_line_m["Rim_FGA_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_opp_paint_m = float(
                pd.to_numeric(g_line_m["Paint_FGM_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_opp_paint_a = float(
                pd.to_numeric(g_line_m["Paint_FGA_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_opp_mr_m = float(
                pd.to_numeric(g_line_m["MR_FGM_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_opp_mr_a = float(
                pd.to_numeric(g_line_m["MR_FGA_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_opp_c3_m = float(
                pd.to_numeric(g_line_m["Cor3_FGM_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_opp_c3_a = float(
                pd.to_numeric(g_line_m["Cor3_FGA_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_opp_atb3_m = float(
                pd.to_numeric(g_line_m["ATB3_FGM_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_opp_atb3_a = float(
                pd.to_numeric(g_line_m["ATB3_FGA_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )

            s_opp_2pm = s_opp_rim_m + s_opp_paint_m + s_opp_mr_m
            s_opp_2pa = s_opp_rim_a + s_opp_paint_a + s_opp_mr_a
            s_opp_3pm = s_opp_c3_m + s_opp_atb3_m
            s_opp_3pa = s_opp_c3_a + s_opp_atb3_a
            s_opp_fgm = s_opp_2pm + s_opp_3pm
            s_opp_fga = s_opp_2pa + s_opp_3pa

            s_opp_ftm = float(
                pd.to_numeric(g_line_m["FTM_Agn"], errors="coerce").fillna(0).sum()
            )
            s_opp_fta = float(
                pd.to_numeric(g_line_m["FTA_Agn"], errors="coerce").fillna(0).sum()
            )
            s_opp_oreb = float(
                pd.to_numeric(g_line_m["OREB_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_opp_dreb = float(
                pd.to_numeric(g_line_m["DREB_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            s_opp_tov = float(
                pd.to_numeric(g_line_m["TOV_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )

            s_opp_paint_pts = (s_opp_rim_m + s_opp_paint_m) * 2.0
            s_opp_paint_fga = s_opp_rim_a + s_opp_paint_a
            s_opp_nopaint_pts = (
                s_opp_mr_m * 2.0 + s_opp_c3_m * 3.0 + s_opp_atb3_m * 3.0
            )
            s_opp_nopaint_fga = s_opp_mr_a + s_opp_c3_a + s_opp_atb3_a
            s_opp_fg_pts = s_opp_paint_pts + s_opp_nopaint_pts

            # 3. CÀLCULS MATEMÀTICS DELS 4 FACTORS (Dean Oliver)
            efg_team = adv_dict_team.get(
                "1. Effective FG% (eFG%)",
                ((s_team_fgm + 0.5 * s_team_3pm) / s_team_fga * 100)
                if s_team_fga > 0
                else 0.0,
            )
            efg_opp = adv_dict_opp.get(
                "1. Effective FG% (eFG%)",
                ((s_opp_fgm + 0.5 * s_opp_3pm) / s_opp_fga * 100)
                if s_opp_fga > 0
                else 0.0,
            )

            tov_team = adv_dict_team.get(
                "2. Turnover Rate (TO%)",
                (s_team_tov / (s_team_fga + 0.44 * s_team_fta + s_team_tov) * 100)
                if (s_team_fga + 0.44 * s_team_fta + s_team_tov) > 0
                else 0.0,
            )
            tov_opp = adv_dict_opp.get(
                "2. Turnover Rate (TO%)",
                (s_opp_tov / (s_opp_fga + 0.44 * s_opp_fta + s_opp_tov) * 100)
                if (s_opp_fga + 0.44 * s_opp_fta + s_opp_tov) > 0
                else 0.0,
            )

            oreb_team = adv_dict_team.get(
                "3. Offensive Rebound% (OREB%)",
                (s_team_oreb / (s_team_oreb + s_opp_dreb) * 100)
                if (s_team_oreb + s_opp_dreb) > 0
                else 0.0,
            )
            oreb_opp = adv_dict_opp.get(
                "3. Offensive Rebound% (OREB%)",
                (s_opp_oreb / (s_opp_oreb + s_team_dreb) * 100)
                if (s_opp_oreb + s_team_dreb) > 0
                else 0.0,
            )

            ft_rate_team = (
                (s_team_ftm / s_team_fga) if s_team_fga > 0 else 0.0
            )
            ft_rate_opp = (s_opp_ftm / s_opp_fga) if s_opp_fga > 0 else 0.0

            # 4. TARGETES KPI SUPERIORS (ELS 4 FACTORS)
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(
                "1. Tir Efectiu (eFG%)",
                f"{efg_team:.1f}%",
                delta=f"{efg_team - efg_opp:+.1f}% vs Rival",
            )
            kpi2.metric(
                "2. Control Pèrdues (TOV%)",
                f"{tov_team:.1f}%",
                delta=f"{tov_opp - tov_team:+.1f}% marge",
            )
            kpi3.metric(
                "3. Rebot Ofensiu (OREB%)",
                f"{oreb_team:.1f}%",
                delta=f"{oreb_team - oreb_opp:+.1f}% vs Rival",
            )
            kpi4.metric(
                "4. Freqüència TL (FT Rate)",
                f"{ft_rate_team:.2f}",
                delta=f"{ft_rate_team - ft_rate_opp:+.2f} vs Rival",
            )

            st.markdown("---")
            # 4.1 VOLUMS DE LLANÇAMENT DEL PARTIT
            st.subheader("📦 Volums de Llançament del Partit")
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                s_pct_2p = (s_team_2pm / s_team_2pa * 100) if s_team_2pa > 0 else 0.0
                s_pct_3p = (s_team_3pm / s_team_3pa * 100) if s_team_3pa > 0 else 0.0
                s_pct_ft = (s_team_ftm / s_team_fta * 100) if s_team_fta > 0 else 0.0
                s_pct_fg = (s_team_fgm / s_team_fga * 100) if s_team_fga > 0 else 0.0
                s_pct_3par = (s_team_3pa / s_team_fga * 100) if s_team_fga > 0 else 0.0

                df_vol_team_single = pd.DataFrame(
                    [
                        {
                            "Concepte": "Tirs de Camp Totals (TC)",
                            "Encertats / Intentats": f"{int(s_team_fgm)} / {int(s_team_fga)}",
                            "Percentatge": f"{s_pct_fg:.1f}%",
                            "Punts Aportats": f"{int(s_team_2pm*2 + s_team_3pm*3)} pts",
                        },
                        {
                            "Concepte": "Tirs de 2 Punts (T2)",
                            "Encertats / Intentats": f"{int(s_team_2pm)} / {int(s_team_2pa)}",
                            "Percentatge": f"{s_pct_2p:.1f}%",
                            "Punts Aportats": f"{int(s_team_2pm*2)} pts",
                        },
                        {
                            "Concepte": "Triples (T3)",
                            "Encertats / Intentats": f"{int(s_team_3pm)} / {int(s_team_3pa)}",
                            "Percentatge": f"{s_pct_3p:.1f}%",
                            "Punts Aportats": f"{int(s_team_3pm*3)} pts",
                        },
                        {
                            "Concepte": "Tirs Lliures (TL)",
                            "Encertats / Intentats": f"{int(s_team_ftm)} / {int(s_team_fta)}",
                            "Percentatge": f"{s_pct_ft:.1f}%",
                            "Punts Aportats": f"{int(s_team_ftm)} pts",
                        },
                        {
                            "Concepte": "Ràtio de Triples (3PAr)",
                            "Encertats / Intentats": f"{int(s_team_3pa)} / {int(s_team_fga)}",
                            "Percentatge": f"{s_pct_3par:.1f}%",
                            "Punts Aportats": f"{s_pct_3par:.1f}% dels TC",
                        },
                    ]
                )
                st.markdown("##### 🔴 Volums Argentona")
                st.dataframe(
                    df_vol_team_single, use_container_width=True, hide_index=True
                )

            with col_v2:
                s_opp_pct_2p = (s_opp_2pm / s_opp_2pa * 100) if s_opp_2pa > 0 else 0.0
                s_opp_pct_3p = (s_opp_3pm / s_opp_3pa * 100) if s_opp_3pa > 0 else 0.0
                s_opp_pct_ft = (s_opp_ftm / s_opp_fta * 100) if s_opp_fta > 0 else 0.0
                s_opp_pct_fg = (s_opp_fgm / s_opp_fga * 100) if s_opp_fga > 0 else 0.0
                s_opp_pct_3par = (s_opp_3pa / s_opp_fga * 100) if s_opp_fga > 0 else 0.0

                df_vol_opp_single = pd.DataFrame(
                    [
                        {
                            "Concepte": "Tirs de Camp Totals (TC)",
                            "Encertats / Intentats": f"{int(s_opp_fgm)} / {int(s_opp_fga)}",
                            "Percentatge": f"{s_opp_pct_fg:.1f}%",
                            "Punts Aportats": f"{int(s_opp_2pm*2 + s_opp_3pm*3)} pts",
                        },
                        {
                            "Concepte": "Tirs de 2 Punts (T2)",
                            "Encertats / Intentats": f"{int(s_opp_2pm)} / {int(s_opp_2pa)}",
                            "Percentatge": f"{s_opp_pct_2p:.1f}%",
                            "Punts Aportats": f"{int(s_opp_2pm*2)} pts",
                        },
                        {
                            "Concepte": "Triples (T3)",
                            "Encertats / Intentats": f"{int(s_opp_3pm)} / {int(s_opp_3pa)}",
                            "Percentatge": f"{s_opp_pct_3p:.1f}%",
                            "Punts Aportats": f"{int(s_opp_3pm*3)} pts",
                        },
                        {
                            "Concepte": "Tirs Lliures (TL)",
                            "Encertats / Intentats": f"{int(s_opp_ftm)} / {int(s_opp_fta)}",
                            "Percentatge": f"{s_opp_pct_ft:.1f}%",
                            "Punts Aportats": f"{int(s_opp_ftm)} pts",
                        },
                        {
                            "Concepte": "Ràtio de Triples (3PAr)",
                            "Encertats / Intentats": f"{int(s_opp_3pa)} / {int(s_opp_fga)}",
                            "Percentatge": f"{s_opp_pct_3par:.1f}%",
                            "Punts Aportats": f"{s_opp_pct_3par:.1f}% dels TC",
                        },
                    ]
                )
                st.markdown("##### 🛡️ Volums Rival")
                st.dataframe(
                    df_vol_opp_single, use_container_width=True, hide_index=True
                )

            st.markdown("---")
            # 5. TAULA UNIFICADA DELS 4 FACTORS
            win_efg = efg_team > efg_opp
            win_tov = tov_team < tov_opp
            win_oreb = oreb_team > oreb_opp
            win_ft = ft_rate_team > ft_rate_opp
            factors_guanyats = sum([win_efg, win_tov, win_oreb, win_ft])

            st.subheader(
                f"🎯 Els 4 Factors (Dean Oliver) — Balanç: {factors_guanyats}/4 Guanyats"
            )

            df_four_factors_unified = pd.DataFrame(
                [
                    {
                        "Factor": "1. Tir Efectiu (eFG%)",
                        "🔴 El nostre Atac": f"{efg_team:.1f}%",
                        "🛡️ La nostra Defensa (Rival)": f"{efg_opp:.1f}%",
                        "Diferencial Net": f"{efg_team - efg_opp:+.1f}%",
                        "Impacte": "🔴 Guanyat (Més encert efectiu)"
                        if win_efg
                        else "🔵 Perdut (Rival més efectiu)",
                    },
                    {
                        "Factor": "2. Control de Pèrdues (TOV%)",
                        "🔴 El nostre Atac": f"{tov_team:.1f}%",
                        "🛡️ La nostra Defensa (Rival)": f"{tov_opp:.1f}%",
                        "Diferencial Net": f"{tov_opp - tov_team:+.1f}% marge",
                        "Impacte": "🔴 Guanyat (Hem cuidat millor la pilota)"
                        if win_tov
                        else "🔵 Perdut (Hem perdut més possessions)",
                    },
                    {
                        "Factor": "3. Rebot Ofensiu (OREB%)",
                        "🔴 El nostre Atac": f"{oreb_team:.1f}%",
                        "🛡️ La nostra Defensa (Rival)": f"{oreb_opp:.1f}%",
                        "Diferencial Net": f"{oreb_team - oreb_opp:+.1f}%",
                        "Impacte": "🔴 Guanyat (Més segones opcions)"
                        if win_oreb
                        else "🔵 Perdut (Rival ha carregat millor el rebot)",
                    },
                    {
                        "Factor": "4. Freqüència TL (FT Rate)",
                        "🔴 El nostre Atac": f"{ft_rate_team:.2f}",
                        "🛡️ La nostra Defensa (Rival)": f"{ft_rate_opp:.2f}",
                        "Diferencial Net": f"{ft_rate_team - ft_rate_opp:+.2f}",
                        "Impacte": "🔴 Guanyat (Més punts sumats des del TL)"
                        if win_ft
                        else "🔵 Perdut (Rival ha anat més a la línia)",
                    },
                ]
            )

            st.dataframe(
                df_four_factors_unified,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")

            # 6. NOVES TARGETES: RITME, OER I DER
            # Possessions reals de partit amb protecció si falta el rival
            raw_t_poss = float(
                s_team_fga + 0.44 * s_team_fta - s_team_oreb + s_team_tov
            )
            raw_o_poss = float(
                s_opp_fga + 0.44 * s_opp_fta - s_opp_oreb + s_opp_tov
            )

            # Si tenim dades dels dos equips fem la mitjana; si el rival no té dades, usem les nostres
            if raw_t_poss >= 15.0 and raw_o_poss >= 15.0:
                match_poss = 0.5 * (raw_t_poss + raw_o_poss)
            elif raw_t_poss >= 15.0:
                match_poss = raw_t_poss
            elif raw_o_poss >= 15.0:
                match_poss = raw_o_poss
            else:
                match_poss = 80.0

            # Minuts reals de l'equip al partit
            match_team_min = g_box_m["MIN"].apply(parse_min_to_float).sum()

            # Si el partit és de 40 minuts regulars (sense pròrroga), el Ritme és exactament igual a les possessions
            if 190.0 <= match_team_min <= 210.0:
                match_pace = match_poss
            elif match_team_min > 210.0:
                match_pace = (match_poss / match_team_min) * 200.0
            else:
                match_pace = match_poss
            match_oer = (
                (s_team_pts / match_poss) * 100.0 if match_poss > 0 else 0.0
            )
            match_der = (
                (s_opp_pts / match_poss) * 100.0 if match_poss > 0 else 0.0
            )
            match_net = match_oer - match_der

            st.subheader("⚡ Eficiència Global de Possessió")
            c_pace, c_oer, c_der = st.columns(3)
            c_pace.metric(
                "⏱️ Ritme de Joc (Pace a 40 min)",
                f"{match_pace:.1f} poss",
                delta=f"{match_poss:.1f} possessions reals",
            )
            c_oer.metric(
                "🎯 OER (Eficiència Ofensiva)",
                f"{match_oer:.1f} pts/100",
                delta=f"{match_net:+.1f} Net Rating",
            )
            c_der.metric(
                "🛡️ DER (Eficiència Defensiva)",
                f"{match_der:.1f} pts/100",
                delta=f"{match_oer - match_der:+.1f} marge",
            )

            st.markdown("---")

            # 7. TAULA DE DESGLOSSAMENT D'EFICIÈNCIA (14 FILES FIXES I HOMOGÈNIES)
            st.subheader("📊 Desglossament d'Eficiència de Tir i Joc")
            col_table, _ = st.columns([3, 1])

            with col_table:
                # 1. Càlculs de PPS en joc
                pps_team_real = (
                    (s_team_fg_pts / s_team_fga) if s_team_fga > 0 else 0.0
                )
                pps_opp_real = (
                    (s_opp_fg_pts / s_opp_fga) if s_opp_fga > 0 else 0.0
                )

                paint_pps_t = (
                    (s_team_paint_pts / s_team_paint_fga)
                    if s_team_paint_fga > 0
                    else 0.0
                )
                paint_pps_o = (
                    (s_opp_paint_pts / s_opp_paint_fga)
                    if s_opp_paint_fga > 0
                    else 0.0
                )

                nopaint_pps_t = (
                    (s_team_nopaint_pts / s_team_nopaint_fga)
                    if s_team_nopaint_fga > 0
                    else 0.0
                )
                nopaint_pps_o = (
                    (s_opp_nopaint_pts / s_opp_nopaint_fga)
                    if s_opp_nopaint_fga > 0
                    else 0.0
                )

                # Funció auxiliar per cercar a adv_dict de forma flexible
                def get_m_val(adv_d, terms, default_val=0.0):
                    for k, v in adv_d.items():
                        k_low = k.lower()
                        if any(t.lower() in k_low for t in terms):
                            try:
                                return float(v)
                            except Exception:
                                pass
                    return default_val

                # 2. Extracció creuada de les mètriques complementàries
                fb_pps_t = get_m_val(
                    adv_dict_team, ["fastbreak pps", "transició pps"], 0.0
                )
                fb_pps_o = get_m_val(
                    adv_dict_opp, ["fastbreak pps", "transició pps"], 0.0
                )

                half_pps_t = get_m_val(
                    adv_dict_team, ["half-court pps", "estàtic", "estatic"], 0.0
                )
                half_pps_o = get_m_val(
                    adv_dict_opp, ["half-court pps", "estàtic", "estatic"], 0.0
                )

                paint_touches_t = get_m_val(
                    adv_dict_team,
                    ["total paint touches", "tocs a pintura"],
                    s_team_paint_fga,
                )
                paint_touches_o = get_m_val(
                    adv_dict_opp,
                    ["total paint touches", "tocs a pintura"],
                    s_opp_paint_fga,
                )

                pts_off_to_t = get_m_val(
                    adv_dict_team,
                    ["points off live to", "pts off to", "points off"],
                    0.0,
                )
                pts_off_to_o = get_m_val(
                    adv_dict_opp,
                    ["points off live to", "pts off to", "points off"],
                    0.0,
                )

                lto_t = get_m_val(
                    adv_dict_team,
                    ["live turnovers", "lto"],
                    float(
                        pd.to_numeric(g_box_m.get("LTO", 0), errors="coerce")
                        .fillna(0)
                        .sum()
                    ),
                )
                lto_o = get_m_val(
                    adv_dict_opp, ["live turnovers", "lto"], 0.0
                )

                dto_t = get_m_val(
                    adv_dict_team,
                    ["dead turnovers", "dto"],
                    float(
                        pd.to_numeric(g_box_m.get("DTO", 0), errors="coerce")
                        .fillna(0)
                        .sum()
                    ),
                )
                dto_o = get_m_val(
                    adv_dict_opp, ["dead turnovers", "dto"], 0.0
                )

                fb_tot_t = get_m_val(
                    adv_dict_team,
                    ["total fastbreaks", "transicions totals"],
                    float(
                        pd.to_numeric(g_box_m.get("FB(M)", 0), errors="coerce")
                        .fillna(0)
                        .sum()
                        + pd.to_numeric(
                            g_box_m.get("FB(Miss)", 0), errors="coerce"
                        )
                        .fillna(0)
                        .sum()
                    ),
                )
                fb_tot_o = get_m_val(
                    adv_dict_opp,
                    ["total fastbreaks", "transicions totals"],
                    0.0,
                )

                ch2_pts_t = get_m_val(
                    adv_dict_team,
                    ["2nd chance points", "2a oportunitat"],
                    float(
                        pd.to_numeric(
                            g_box_m.get("2ndCh(M)", 0), errors="coerce"
                        )
                        .fillna(0)
                        .sum()
                        * 2
                    ),
                )
                ch2_pts_o = get_m_val(
                    adv_dict_opp, ["2nd chance points", "2a oportunitat"], 0.0
                )

                kills_t = get_m_val(
                    adv_dict_team, ["defensive kills", "kills"], 0.0
                )
                kills_o = get_m_val(
                    adv_dict_opp, ["defensive kills", "kills"], 0.0
                )

                stocks_t = get_m_val(
                    adv_dict_team,
                    ["stocks", "steals + blocks"],
                    float(
                        pd.to_numeric(
                            g_box_m.get("Steals", 0), errors="coerce"
                        )
                        .fillna(0)
                        .sum()
                        + pd.to_numeric(
                            g_box_m.get("Blocks", 0), errors="coerce"
                        )
                        .fillna(0)
                        .sum()
                    ),
                )
                stocks_o = get_m_val(
                    adv_dict_opp, ["stocks", "steals + blocks"], 0.0
                )
                pbp_stats_m = compute_pbp_advanced_stats(pbp_df, selected_game_id)
                # 3. LES 14 FILES EXACTES I EN ORDRE FIX
                table_rows_single = [
                    (
                        "Marcador del Partit",
                        f"{int(s_team_pts)}",
                        f"{int(s_opp_pts)}",
                        s_team_pts,
                        s_opp_pts,
                        abs(s_team_pts - s_opp_pts) >= 2,
                        False,
                    ),
                    (
                        "Punts de Tir de Camp per Intent (PPS en joc)",
                        f"{pps_team_real:.2f}",
                        f"{pps_opp_real:.2f}",
                        pps_team_real,
                        pps_opp_real,
                        abs(pps_team_real - pps_opp_real) >= 0.15,
                        False,
                    ),
                    (
                        "PPS amb Toc a Pintura",
                        f"{paint_pps_t:.2f}",
                        f"{paint_pps_o:.2f}",
                        paint_pps_t,
                        paint_pps_o,
                        abs(paint_pps_t - paint_pps_o) >= 0.15,
                        False,
                    ),
                    (
                        "PPS sense Toc a Pintura",
                        f"{nopaint_pps_t:.2f}",
                        f"{nopaint_pps_o:.2f}",
                        nopaint_pps_t,
                        nopaint_pps_o,
                        abs(nopaint_pps_t - nopaint_pps_o) >= 0.15,
                        False,
                    ),
                    (
                        "PPS en Transició (Contraatac)",
                        f"{pbp_stats_m.get('fb_pps_t', 0.0):.2f}",
                        f"{pbp_stats_m.get('fb_pps_o', 0.0):.2f}",
                        pbp_stats_m.get("fb_pps_t", 0.0),
                        pbp_stats_m.get("fb_pps_o", 0.0),
                        abs(
                            pbp_stats_m.get("fb_pps_t", 0.0)
                            - pbp_stats_m.get("fb_pps_o", 0.0)
                        )
                        >= 0.15,
                        False,
                    ),
                    (
                        "PPS en Atac Estàtic (5c5)",
                        f"{pbp_stats_m.get('hc_pps_t', 0.0):.2f}",
                        f"{pbp_stats_m.get('hc_pps_o', 0.0):.2f}",
                        pbp_stats_m.get("hc_pps_t", 0.0),
                        pbp_stats_m.get("hc_pps_o", 0.0),
                        abs(
                            pbp_stats_m.get("hc_pps_t", 0.0)
                            - pbp_stats_m.get("hc_pps_o", 0.0)
                        )
                        >= 0.15,
                        False,
                    ),
                    (
                        "Punts de Pèrdua Viva (PTS OFF TO)",
                        f"{int(pbp_stats_m.get('pts_off_to_t', 0))}",
                        f"{int(pbp_stats_m.get('pts_off_to_o', 0))}",
                        pbp_stats_m.get("pts_off_to_t", 0.0),
                        pbp_stats_m.get("pts_off_to_o", 0.0),
                        abs(
                            pbp_stats_m.get("pts_off_to_t", 0.0)
                            - pbp_stats_m.get("pts_off_to_o", 0.0)
                        )
                        >= 3,
                        False,
                    ),
                    (
                        "Pèrdues Vives (LTO)",
                        f"{int(pbp_stats_m.get('lto_t', 0))}",
                        f"{int(pbp_stats_m.get('lto_o', 0))}",
                        pbp_stats_m.get("lto_t", 0.0),
                        pbp_stats_m.get("lto_o", 0.0),
                        abs(pbp_stats_m.get("lto_t", 0.0) - pbp_stats_m.get("lto_o", 0.0))
                        >= 2,
                        True,
                    ),
                    (
                        "Pèrdues Mortes (DTO)",
                        f"{int(pbp_stats_m.get('dto_t', 0))}",
                        f"{int(pbp_stats_m.get('dto_o', 0))}",
                        pbp_stats_m.get("dto_t", 0.0),
                        pbp_stats_m.get("dto_o", 0.0),
                        abs(pbp_stats_m.get("dto_t", 0.0) - pbp_stats_m.get("dto_o", 0.0))
                        >= 2,
                        True,
                    ),
                    (
                        "Transicions Totals",
                        f"{int(fb_tot_t)}",
                        f"{int(fb_tot_o)}",
                        fb_tot_t,
                        fb_tot_o,
                        abs(fb_tot_t - fb_tot_o) >= 3,
                        False,
                    ),
                    (
                        "Punts de 2a Oportunitat",
                        f"{int(ch2_pts_t)}",
                        f"{int(ch2_pts_o)}",
                        ch2_pts_t,
                        ch2_pts_o,
                        abs(ch2_pts_t - ch2_pts_o) >= 3,
                        False,
                    ),
                    (
                        "Kills Defensius (3 Aturades Seguidas)",
                        f"{int(kills_t)}",
                        f"{int(kills_o)}",
                        kills_t,
                        kills_o,
                        abs(kills_t - kills_o) >= 2,
                        False,
                    ),
                    (
                        "Stocks (Recuperacions + Taps)",
                        f"{int(stocks_t)}",
                        f"{int(stocks_o)}",
                        stocks_t,
                        stocks_o,
                        abs(stocks_t - stocks_o) >= 3,
                        False,
                    ),
                ]

                rows_s_disp = []
                styles_s_disp = []
                for (
                    m_nom,
                    tv_str,
                    ov_str,
                    tv_num,
                    ov_num,
                    is_outlier,
                    lower_is_better,
                ) in table_rows_single:
                    if not is_outlier or tv_num == ov_num:
                        uem_css, riv_css = STYLE_NEUTRAL, STYLE_NEUTRAL
                    elif lower_is_better:
                        if tv_num < ov_num:
                            uem_css, riv_css = STYLE_RED, STYLE_BLUE
                        else:
                            uem_css, riv_css = STYLE_BLUE, STYLE_RED
                    else:
                        if tv_num > ov_num:
                            uem_css, riv_css = STYLE_RED, STYLE_BLUE
                        else:
                            uem_css, riv_css = STYLE_BLUE, STYLE_RED

                    rows_s_disp.append(
                        {"Mètrica": m_nom, "Argentona": tv_str, "Rival": ov_str}
                    )
                    styles_s_disp.append((uem_css, riv_css))

                df_display_single = pd.DataFrame(rows_s_disp)

                def apply_text_styles_s(df):
                    css_df = pd.DataFrame("", index=df.index, columns=df.columns)
                    for i, (uem_css, riv_css) in enumerate(styles_s_disp):
                        css_df.loc[i, "Argentona"] = uem_css
                        css_df.loc[i, "Rival"] = riv_css
                    return css_df

                table_height = (len(df_display_single) + 1) * 36 + 3
                st.dataframe(
                    df_display_single.style.apply(
                        apply_text_styles_s, axis=None
                    ),
                    height=table_height,
                    use_container_width=True,
                    hide_index=True,
                )
    # -------------------------------------------------------------
    # 2. SECCIÓ BOX SCORE (PARTIT INDIVIDUAL)
    # -------------------------------------------------------------
    elif active_tab == "📋 Box Score":
        g_box = box_df[box_df["game_id"] == str(selected_game_id)].copy()

        if not g_box.empty:
            # 1. FILTREM LA FILA D'EQUIP PERQUÈ NO SURTI A CAP TAULA
            is_team_row = (
                g_box["Player"]
                .astype(str)
                .str.lower()
                .str.strip()
                .isin(
                    ["team", "totals", "total", "opponent", "rival", "equip"]
                )
                | g_box["Name"]
                .astype(str)
                .str.lower()
                .str.contains("team|total|opponent|equip|rival", na=False)
            )
            g_box = g_box[~is_team_row].copy()

            # 2. SELECTOR DE PASTILLES
            box_mode = st.segmented_control(
                "Tipus d'Estadística",
                options=[
                    "Tradicional",
                    "Tir per Zones",
                    "4 Factors Individuals",
                    "Accions Defensives",
                    "Impacte i Context",
                    "Xarxa d'Assistències",
                    "Rebot i 2a Oportunitat",
                ],
                default="Tradicional",
                key="box_mode_selector_single",
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if box_mode == "Tradicional":
                df_trad = pd.DataFrame()
                df_trad["#"] = g_box["Player"]
                df_trad["Jugador"] = g_box["Name"]
                df_trad["MIN"] = g_box["MIN"].astype(str).str.strip()
                df_trad["PTS"] = g_box["Points"].astype(int)

                t2m = pd.to_numeric(g_box["2PM"], errors="coerce").fillna(0)
                t2a = pd.to_numeric(g_box["2PA"], errors="coerce").fillna(0)
                df_trad["T2 (A/I)"] = (
                    t2m.astype(int).astype(str)
                    + "/"
                    + t2a.astype(int).astype(str)
                )
                df_trad["%T2"] = (
                    (t2m / t2a.replace(0, np.nan)) * 100.0
                ).fillna(0.0)

                t3m = pd.to_numeric(g_box["3PM"], errors="coerce").fillna(0)
                t3a = pd.to_numeric(g_box["3PA"], errors="coerce").fillna(0)
                df_trad["T3 (A/I)"] = (
                    t3m.astype(int).astype(str)
                    + "/"
                    + t3a.astype(int).astype(str)
                )
                df_trad["%T3"] = (
                    (t3m / t3a.replace(0, np.nan)) * 100.0
                ).fillna(0.0)

                ftm = pd.to_numeric(g_box["FTM"], errors="coerce").fillna(0)
                fta = pd.to_numeric(g_box["FTA"], errors="coerce").fillna(0)
                df_trad["TL (A/I)"] = (
                    ftm.astype(int).astype(str)
                    + "/"
                    + fta.astype(int).astype(str)
                )
                df_trad["%TL"] = (
                    (ftm / fta.replace(0, np.nan)) * 100.0
                ).fillna(0.0)

                df_trad["REB_O"] = (
                    pd.to_numeric(g_box["Off Reb"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                df_trad["REB_D"] = (
                    pd.to_numeric(g_box["Def Reb"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                df_trad["REB_T"] = df_trad["REB_O"] + df_trad["REB_D"]
                df_trad["AST"] = (
                    pd.to_numeric(g_box["Assists"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                df_trad["REC"] = (
                    pd.to_numeric(g_box["Steals"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                df_trad["TAP"] = (
                    pd.to_numeric(g_box["Blocks"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                df_trad["PER"] = (
                    pd.to_numeric(g_box["Turnovers"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                df_trad["FAL"] = (
                    pd.to_numeric(g_box["Fouls"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                df_trad["F_REC"] = (
                    pd.to_numeric(g_box["Fouls Drawn"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )

                df_trad["_dorsal_sort"] = (
                    df_trad["#"].astype(str).str.replace("#", "").str.strip()
                )
                df_trad["_dorsal_sort"] = pd.to_numeric(
                    df_trad["_dorsal_sort"], errors="coerce"
                ).fillna(999)
                df_sorted = (
                    df_trad.sort_values(by="_dorsal_sort", ascending=True)
                    .drop(columns=["_dorsal_sort"])
                    .reset_index(drop=True)
                )

                render_boxscore(df_sorted, "Tradicional")

            elif box_mode == "Tir per Zones":
                df_shots = pd.DataFrame()
                df_shots["#"] = g_box["Player"]
                df_shots["Jugador"] = g_box["Name"]
                df_shots["PTS"] = g_box["Points"].astype(int)
                df_shots["Aro (Rim)"] = (
                    g_box["RIM(M)"].astype(int).astype(str)
                    + "/"
                    + g_box["RIM(A)"].astype(int).astype(str)
                )
                df_shots["Pintura (Paint)"] = (
                    g_box["PAINT(M)"].astype(int).astype(str)
                    + "/"
                    + g_box["PAINT(A)"].astype(int).astype(str)
                )
                df_shots["Mitja Distància (MR)"] = (
                    g_box["MR(M)"].astype(int).astype(str)
                    + "/"
                    + g_box["MR(A)"].astype(int).astype(str)
                )
                df_shots["Triple Cantonada (C3)"] = (
                    g_box["C3(M)"].astype(int).astype(str)
                    + "/"
                    + g_box["C3(A)"].astype(int).astype(str)
                )
                df_shots["Triple Frontal (ATB3)"] = (
                    g_box["ATB3(M)"].astype(int).astype(str)
                    + "/"
                    + g_box["ATB3(A)"].astype(int).astype(str)
                )

                df_sorted = df_shots.sort_values(
                    by="PTS", ascending=False
                ).reset_index(drop=True)
                render_boxscore(df_sorted, "Zones")
            elif box_mode == "4 Factors Individuals":
                df_4f_ind = get_individual_four_factors(
                    box_df, lineups_df, pbp_df, selected_game_id
                )
                render_boxscore(df_4f_ind, "4Factors")
            elif (
                box_mode
                == "Accions Defensives"
            ):
                df_def = get_defensive_actions(g_box, pbp_df, selected_game_id)
                render_boxscore(df_def, "Defensa")

            elif box_mode == "Impacte i Context":
                df_ctx = pd.DataFrame()
                df_ctx["#"] = g_box["Player"]
                df_ctx["Jugador"] = g_box["Name"]
                df_ctx["MIN"] = g_box["MIN"].astype(str).str.strip()

                fb_m = (
                    pd.to_numeric(g_box["FB(M)"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                fb_tot = (
                    pd.to_numeric(
                        g_box["FB(M)"] + g_box["FB(Miss)"], errors="coerce"
                    )
                    .fillna(0)
                    .astype(int)
                )
                df_ctx["Transició"] = (
                    fb_m.astype(str) + "/" + fb_tot.astype(str)
                )

                ch2_m = (
                    pd.to_numeric(g_box["2ndCh(M)"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                ch2_tot = (
                    pd.to_numeric(
                        g_box["2ndCh(M)"] + g_box["2ndCh(Miss)"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype(int)
                )
                df_ctx["2a Oportunitat"] = (
                    ch2_m.astype(str) + "/" + ch2_tot.astype(str)
                )

                df_ctx["PER Totals"] = (
                    pd.to_numeric(g_box["Turnovers"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )

                if "LTO" in g_box.columns:
                    df_ctx["Vives (LTO)"] = (
                        pd.to_numeric(g_box["LTO"], errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )
                if "DTO" in g_box.columns:
                    df_ctx["Mortes (DTO)"] = (
                        pd.to_numeric(g_box["DTO"], errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )

                df_ctx["Faltes Com."] = (
                    pd.to_numeric(g_box["Fouls"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                df_ctx["Faltes Reb."] = (
                    pd.to_numeric(g_box["Fouls Drawn"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )

                # Ordenació per dorsal numèric
                df_ctx["_dorsal_sort"] = (
                    df_ctx["#"].astype(str).str.replace("#", "").str.strip()
                )
                df_ctx["_dorsal_sort"] = pd.to_numeric(
                    df_ctx["_dorsal_sort"], errors="coerce"
                ).fillna(999)

                df_sorted = (
                    df_ctx.sort_values(by="_dorsal_sort", ascending=True)
                    .drop(columns=["_dorsal_sort"])
                    .reset_index(drop=True)
                )

                render_boxscore(df_sorted, "Context")

            elif box_mode == "Xarxa d'Assistències":
                df_rank, df_creadors, _ = get_assist_combos(
                    pbp_df, box_df, selected_game_id
                )

                if not df_rank.empty:
                    tot_ast = int(df_creadors["Assistències_Totals"].sum())
                    tot_pts_gen = int(
                        df_creadors["Punts_Generats_Totals"].sum()
                    )
                    tot_t3 = int(df_creadors["T3_Generats"].sum())

                    m_a1, m_a2, m_a3 = st.columns(3)
                    m_a1.metric("Assistències Totals", f"{tot_ast}")
                    m_a2.metric(
                        "Punts Generats per Assistència",
                        f"{tot_pts_gen} pts",
                        delta=f"{(tot_pts_gen/tot_ast):.2f} pts/ast"
                        if tot_ast > 0
                        else None,
                    )
                    m_a3.metric(
                        "Triples Assistits",
                        f"{tot_t3}",
                        delta=f"{tot_t3*3} pts via T3",
                    )

                    st.markdown("---")

                    # 1. Taula de Creadors (CRIDANT A render_boxscore)
                    st.markdown(
                        "##### 🗲 Punts Totals Generats per Creador"
                    )
                    df_c_disp = df_creadors.rename(
                        columns={
                            "Assistències_Totals": "AST",
                            "Punts_Generats_Totals": "PTS Generats",
                            "T2_Generats": "T2 Assistits",
                            "T3_Generats": "T3 Assistits",
                        }
                    )
                    render_boxscore(df_c_disp, "Assistències")

                    st.markdown("<br>", unsafe_allow_html=True)

                    # 2. Taula de Duos (CRIDANT A render_boxscore)
                    st.markdown(
                        "##### 🤝 Connexions i Duos (Passador ➔ Anotador)"
                    )
                    df_r_disp = df_rank.rename(
                        columns={
                            "Assistències": "AST",
                            "Punts_Generats": "PTS Generats",
                            "T2": "T2 Assistits",
                            "T3": "T3 Assistits",
                        }
                    )
                    render_boxscore(df_r_disp, "Assistències")
                else:
                    st.info(
                        "No hi ha assistències registrades en aquest partit."
                    )
            elif box_mode == "Rebot i 2a Oportunitat":
                df_reb = pd.DataFrame()
                df_reb["#"] = g_box["Player"]
                df_reb["Jugador"] = g_box["Name"]

                # 1. Dades base de rebots
                reb_o = pd.to_numeric(g_box["Off Reb"], errors="coerce").fillna(0).astype(int)
                reb_d = pd.to_numeric(g_box["Def Reb"], errors="coerce").fillna(0).astype(int)
                reb_t = reb_o + reb_d
                df_reb["REB_O"] = reb_o
                df_reb["REB_D"] = reb_d
                df_reb["REB_T"] = reb_t

                # 2. Càlcul OREB% (Fórmula Dean Oliver / Imatge) i REB/Min
                p_mp = g_box["MIN"].apply(parse_min_to_float)
                t_mp = float(p_mp.sum())
                t_or = float(reb_o.sum())

                # Rebots defensius del rival extrets de lineups
                g_line = lineups_df[lineups_df["game_id"] == str(selected_game_id)]
                opp_dr = float(pd.to_numeric(g_line["DREB_Agn"], errors="coerce").fillna(0).sum()) if not g_line.empty else (t_or * 1.5)

                orb_denom = p_mp * (t_or + opp_dr)
                df_reb["OREB%"] = np.where(
                    (p_mp > 0) & (orb_denom > 0),
                    100.0 * (reb_o * (t_mp / 5.0)) / orb_denom,
                    0.0
                )
                df_reb["REB/Min"] = np.round(np.where(p_mp > 0, reb_t / p_mp, 0.0), 2)

                # 3. Segona Oportunitat
                ch2_m = pd.to_numeric(g_box.get("2ndCh(M)", 0), errors="coerce").fillna(0).astype(int)
                ch2_miss = pd.to_numeric(g_box.get("2ndCh(Miss)", 0), errors="coerce").fillna(0).astype(int)
                ch2_tot = ch2_m + ch2_miss

                df_reb["2a Op. (A/I)"] = [f"{m}/{tot}".rjust(10, '\u00A0') for m, tot in zip(ch2_m, ch2_tot)]
                df_reb["Pts 2a Op."] = ch2_m * 2
                df_reb["Fallades 2a Op."] = ch2_miss
                df_reb["% 2a Op."] = ((ch2_m / ch2_tot.replace(0, np.nan)) * 100.0).fillna(0.0)

                # Ordenem per rebots totals
                df_sorted = df_reb.sort_values(by=["REB_T", "REB_O"], ascending=False).reset_index(drop=True)
                render_boxscore(df_sorted, "Rebot")
        else:
            st.info("No hi ha dades de Boxscore disponibles.")

# -------------------------------------------------------------
    # 3. SECCIÓ QUINTETS I ON/OFF (PARTIT INDIVIDUAL)
    # -------------------------------------------------------------
    elif active_tab == "👥 Quintets i On/Off":
        g_line = lineups_df[lineups_df["game_id"].astype(str) == str(selected_game_id)].copy()
        g_box = box_df[box_df["game_id"].astype(str) == str(selected_game_id)].copy()
        render_on_off_dashboard(g_line, g_box, key_prefix="single")
        
# --- MODO 2: TOTALS ACUMULATS DE LA TEMPORADA ---
else:
    st.title(f"🏆 Totals Acumulats ({selected_comp})")

    valid_ids = filtered_games["game_id"].astype(str).tolist()
    num_partits = len(valid_ids)

    # Dades acumulades de tots els partits seleccionats
    comp_box = box_df[box_df["game_id"].astype(str).isin(valid_ids)].copy()
    comp_lineups = (
        lineups_df[lineups_df["game_id"].astype(str).isin(valid_ids)].copy()
    )
    comp_adv = adv_df[adv_df["game_id"].astype(str).isin(valid_ids)].copy()
    comp_pbp = (
        pbp_df[pbp_df["game_id"].astype(str).isin(valid_ids)].copy()
        if pbp_df is not None and not pbp_df.empty
        else pd.DataFrame()
    )

    active_tab_season = st.segmented_control(
        "Navegació Acumulats",
        options=["📈 Mètriques i 4 Factors", "📋 Box Score", "👥 Quintets i On/Off"],
        default="📈 Mètriques i 4 Factors",
        key="season_active_tab",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 1. PESTAÑA: MÈTRIQUES, 4 FACTORS I EFICIÈNCIA (ACUMULATS)
    # -------------------------------------------------------------
    if active_tab_season == "📈 Mètriques i 4 Factors":
        if not comp_lineups.empty or not comp_box.empty:

            def safe_col_sum(df, col):
                if col in df.columns:
                    return float(
                        pd.to_numeric(df[col], errors="coerce").fillna(0).sum()
                    )
                return 0.0

            # 1. SUMATORIS REALS D'ARGENTONA
            team_rim_m = int(safe_col_sum(comp_box, "RIM(M)"))
            team_rim_a = int(safe_col_sum(comp_box, "RIM(A)"))
            team_paint_m = int(safe_col_sum(comp_box, "PAINT(M)"))
            team_paint_a = int(safe_col_sum(comp_box, "PAINT(A)"))
            team_mr_m = int(safe_col_sum(comp_box, "MR(M)"))
            team_mr_a = int(safe_col_sum(comp_box, "MR(A)"))
            team_c3_m = int(safe_col_sum(comp_box, "C3(M)"))
            team_c3_a = int(safe_col_sum(comp_box, "C3(A)"))
            team_atb3_m = int(safe_col_sum(comp_box, "ATB3(M)"))
            team_atb3_a = int(safe_col_sum(comp_box, "ATB3(A)"))

            team_2pm = team_rim_m + team_paint_m + team_mr_m
            team_2pa = team_rim_a + team_paint_a + team_mr_a
            team_3pm = team_c3_m + team_atb3_m
            team_3pa = team_c3_a + team_atb3_a
            team_fgm = team_2pm + team_3pm
            team_fga = team_2pa + team_3pa

            team_ftm = int(safe_col_sum(comp_box, "FTM"))
            team_fta = int(safe_col_sum(comp_box, "FTA"))
            team_oreb = int(safe_col_sum(comp_box, "Off Reb"))
            team_dreb = int(safe_col_sum(comp_box, "Def Reb"))
            team_tov = int(safe_col_sum(comp_box, "Turnovers"))
            team_ast = int(safe_col_sum(comp_box, "Assists"))

            team_paint_pts = (team_rim_m + team_paint_m) * 2
            team_paint_fga = team_rim_a + team_paint_a
            team_nopaint_pts = team_mr_m * 2 + team_c3_m * 3 + team_atb3_m * 3
            team_nopaint_fga = team_mr_a + team_c3_a + team_atb3_a
            team_fg_pts = team_paint_pts + team_nopaint_pts
            team_pts = team_fg_pts + team_ftm

            # 2. SUMATORIS REALS DELS RIVALS
            opp_rim_m = int(safe_col_sum(comp_lineups, "Rim_FGM_Agn"))
            opp_rim_a = int(safe_col_sum(comp_lineups, "Rim_FGA_Agn"))
            opp_paint_m = int(safe_col_sum(comp_lineups, "Paint_FGM_Agn"))
            opp_paint_a = int(safe_col_sum(comp_lineups, "Paint_FGA_Agn"))
            opp_mr_m = int(safe_col_sum(comp_lineups, "MR_FGM_Agn"))
            opp_mr_a = int(safe_col_sum(comp_lineups, "MR_FGA_Agn"))
            opp_c3_m = int(safe_col_sum(comp_lineups, "Cor3_FGM_Agn"))
            opp_c3_a = int(safe_col_sum(comp_lineups, "Cor3_FGA_Agn"))
            opp_atb3_m = int(safe_col_sum(comp_lineups, "ATB3_FGM_Agn"))
            opp_atb3_a = int(safe_col_sum(comp_lineups, "ATB3_FGA_Agn"))

            opp_2pm = opp_rim_m + opp_paint_m + opp_mr_m
            opp_2pa = opp_rim_a + opp_paint_a + opp_mr_a
            opp_3pm = opp_c3_m + opp_atb3_m
            opp_3pa = opp_c3_a + opp_atb3_a
            opp_fgm = opp_2pm + opp_3pm
            opp_fga = opp_2pa + opp_3pa

            opp_ftm = int(safe_col_sum(comp_lineups, "FTM_Agn"))
            opp_fta = int(safe_col_sum(comp_lineups, "FTA_Agn"))
            opp_oreb = int(safe_col_sum(comp_lineups, "OREB_Agn"))
            opp_dreb = int(safe_col_sum(comp_lineups, "DREB_Agn"))
            opp_tov = int(safe_col_sum(comp_lineups, "TOV_Agn"))

            opp_paint_pts = (opp_rim_m + opp_paint_m) * 2
            opp_paint_fga = opp_rim_a + opp_paint_a
            opp_nopaint_pts = opp_mr_m * 2 + opp_c3_m * 3 + opp_atb3_m * 3
            opp_nopaint_fga = opp_mr_a + opp_c3_a + opp_atb3_a
            opp_fg_pts = opp_paint_pts + opp_nopaint_pts
            opp_pts = opp_fg_pts + opp_ftm

            # 3. 4 FACTORS PONDERATS
            efg_team = (
                ((team_fgm + 0.5 * team_3pm) / team_fga * 100)
                if team_fga > 0
                else 0.0
            )
            efg_opp = (
                ((opp_fgm + 0.5 * opp_3pm) / opp_fga * 100)
                if opp_fga > 0
                else 0.0
            )

            team_plays = float(team_fga + 0.44 * team_fta + team_tov)
            opp_plays = float(opp_fga + 0.44 * opp_fta + opp_tov)
            tov_team = (
                (team_tov / team_plays * 100) if team_plays > 0 else 0.0
            )
            tov_opp = (opp_tov / opp_plays * 100) if opp_plays > 0 else 0.0

            oreb_team = (
                (team_oreb / (team_oreb + opp_dreb) * 100)
                if (team_oreb + opp_dreb) > 0
                else 0.0
            )
            oreb_opp = (
                (opp_oreb / (opp_oreb + team_dreb) * 100)
                if (opp_oreb + team_dreb) > 0
                else 0.0
            )

            ft_rate_team = (team_ftm / team_fga) if team_fga > 0 else 0.0
            ft_rate_opp = (opp_ftm / opp_fga) if opp_fga > 0 else 0.0

            # 4. TARGETES KPI SUPERIORS (4 FACTORS)
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(
                "1. Tir Efectiu (eFG%)",
                f"{efg_team:.1f}%",
                delta=f"{efg_team - efg_opp:+.1f}% vs Rivals",
            )
            kpi2.metric(
                "2. Control Pèrdues (TOV%)",
                f"{tov_team:.1f}%",
                delta=f"{tov_opp - tov_team:+.1f}% marge",
            )
            kpi3.metric(
                "3. Rebot Ofensiu (OREB%)",
                f"{oreb_team:.1f}%",
                delta=f"{oreb_team - oreb_opp:+.1f}% vs Rivals",
            )
            kpi4.metric(
                "4. Freqüència TL (FT Rate)",
                f"{ft_rate_team:.2f}",
                delta=f"{ft_rate_team - ft_rate_opp:+.2f} vs Rivals",
            )

            st.markdown("---")

            # 5. VOLUMS DE LLANÇAMENT
            st.subheader("📦 Volums de Llançament de la Temporada")
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                pct_2p = (team_2pm / team_2pa * 100) if team_2pa > 0 else 0.0
                pct_3p = (team_3pm / team_3pa * 100) if team_3pa > 0 else 0.0
                pct_ft = (team_ftm / team_fta * 100) if team_fta > 0 else 0.0
                pct_fg = (team_fgm / team_fga * 100) if team_fga > 0 else 0.0
                pct_3par = (team_3pa / team_fga * 100) if team_fga > 0 else 0.0

                df_vol_team = pd.DataFrame(
                    [
                        {
                            "Concepte": "Tirs de Camp Totals (TC)",
                            "Encertats / Intentats": f"{team_fgm} / {team_fga}",
                            "Percentatge": f"{pct_fg:.1f}%",
                            "Mitjana / Partit": f"{team_fga/num_partits:.1f} tirs",
                        },
                        {
                            "Concepte": "Tirs de 2 Punts (T2)",
                            "Encertats / Intentats": f"{team_2pm} / {team_2pa}",
                            "Percentatge": f"{pct_2p:.1f}%",
                            "Mitjana / Partit": f"{team_2pa/num_partits:.1f} tirs",
                        },
                        {
                            "Concepte": "Triples (T3)",
                            "Encertats / Intentats": f"{team_3pm} / {team_3pa}",
                            "Percentatge": f"{pct_3p:.1f}%",
                            "Mitjana / Partit": f"{team_3pa/num_partits:.1f} tirs",
                        },
                        {
                            "Concepte": "Tirs Lliures (TL)",
                            "Encertats / Intentats": f"{team_ftm} / {team_fta}",
                            "Percentatge": f"{pct_ft:.1f}%",
                            "Mitjana / Partit": f"{team_fta/num_partits:.1f} TL",
                        },
                        {
                            "Concepte": "Ràtio de Triples (3PAr)",
                            "Encertats / Intentats": f"{team_3pa} / {team_fga}",
                            "Percentatge": f"{pct_3par:.1f}%",
                            "Mitjana / Partit": f"{pct_3par:.1f}% dels TC",
                        },
                    ]
                )
                st.markdown("##### 🔴 Volums Argentona")
                st.dataframe(
                    df_vol_team, use_container_width=True, hide_index=True
                )

            with col_v2:
                opp_pct_2p = (
                    (opp_2pm / opp_2pa * 100) if opp_2pa > 0 else 0.0
                )
                opp_pct_3p = (
                    (opp_3pm / opp_3pa * 100) if opp_3pa > 0 else 0.0
                )
                opp_pct_ft = (
                    (opp_ftm / opp_fta * 100) if opp_fta > 0 else 0.0
                )
                opp_pct_fg = (
                    (opp_fgm / opp_fga * 100) if opp_fga > 0 else 0.0
                )
                opp_pct_3par = (
                    (opp_3pa / opp_fga * 100) if opp_fga > 0 else 0.0
                )

                df_vol_opp = pd.DataFrame(
                    [
                        {
                            "Concepte": "Tirs de Camp Totals (TC)",
                            "Encertats / Intentats": f"{opp_fgm} / {opp_fga}",
                            "Percentatge": f"{opp_pct_fg:.1f}%",
                            "Mitjana / Partit": f"{opp_fga/num_partits:.1f} tirs",
                        },
                        {
                            "Concepte": "Tirs de 2 Punts (T2)",
                            "Encertats / Intentats": f"{opp_2pm} / {opp_2pa}",
                            "Percentatge": f"{opp_pct_2p:.1f}%",
                            "Mitjana / Partit": f"{opp_2pa/num_partits:.1f} tirs",
                        },
                        {
                            "Concepte": "Triples (T3)",
                            "Encertats / Intentats": f"{opp_3pm} / {opp_3pa}",
                            "Percentatge": f"{opp_pct_3p:.1f}%",
                            "Mitjana / Partit": f"{opp_3pa/num_partits:.1f} tirs",
                        },
                        {
                            "Concepte": "Tirs Lliures (TL)",
                            "Encertats / Intentats": f"{opp_ftm} / {opp_fta}",
                            "Percentatge": f"{opp_pct_ft:.1f}%",
                            "Mitjana / Partit": f"{opp_fta/num_partits:.1f} TL",
                        },
                        {
                            "Concepte": "Ràtio de Triples (3PAr)",
                            "Encertats / Intentats": f"{opp_3pa} / {opp_fga}",
                            "Percentatge": f"{opp_pct_3par:.1f}%",
                            "Mitjana / Partit": f"{opp_pct_3par:.1f}% dels TC",
                        },
                    ]
                )
                st.markdown("##### 🛡️ Volums Rivals")
                st.dataframe(
                    df_vol_opp, use_container_width=True, hide_index=True
                )

            st.markdown("---")

            # 6. TAULA ÚNICA DELS 4 FACTORS
            win_efg_s = efg_team > efg_opp
            win_tov_s = tov_team < tov_opp
            win_oreb_s = oreb_team > oreb_opp
            win_ft_s = ft_rate_team > ft_rate_opp
            factors_guanyats_s = sum(
                [win_efg_s, win_tov_s, win_oreb_s, win_ft_s]
            )

            st.subheader(
                f"🎯 Els 4 Factors Reals de la Temporada — Balanç: {factors_guanyats_s}/4 Guanyats"
            )

            df_ff_unified_season = pd.DataFrame(
                [
                    {
                        "Factor": "1. Tir Efectiu (eFG%)",
                        "🔴 Atac Argentona": f"{efg_team:.1f}%",
                        "🛡️ Defensa (Rivals)": f"{efg_opp:.1f}%",
                        "Diferencial Net": f"{efg_team - efg_opp:+.1f}%",
                        "Impacte": "🔴 Guanyat (Més efectivitat)"
                        if win_efg_s
                        else "🔵 Perdut",
                    },
                    {
                        "Factor": "2. Control de Pèrdues (TOV%)",
                        "🔴 Atac Argentona": f"{tov_team:.1f}%",
                        "🛡️ Defensa (Rivals)": f"{tov_opp:.1f}%",
                        "Diferencial Net": f"{tov_opp - tov_team:+.1f}% marge",
                        "Impacte": "🔴 Guanyat (Millor cura de pilota)"
                        if win_tov_s
                        else "🔵 Perdut",
                    },
                    {
                        "Factor": "3. Rebot Ofensiu (OREB%)",
                        "🔴 Atac Argentona": f"{oreb_team:.1f}%",
                        "🛡️ Defensa (Rivals)": f"{oreb_opp:.1f}%",
                        "Diferencial Net": f"{oreb_team - oreb_opp:+.1f}%",
                        "Impacte": "🔴 Guanyat (Més rebot ofensiu)"
                        if win_oreb_s
                        else "🔵 Perdut",
                    },
                    {
                        "Factor": "4. Freqüència TL (FT Rate)",
                        "🔴 Atac Argentona": f"{ft_rate_team:.2f}",
                        "🛡️ Defensa (Rivals)": f"{ft_rate_opp:.2f}",
                        "Diferencial Net": f"{ft_rate_team - ft_rate_opp:+.2f}",
                        "Impacte": "🔴 Guanyat (Més producció TL)"
                        if win_ft_s
                        else "🔵 Perdut",
                    },
                ]
            )

            st.dataframe(
                df_ff_unified_season, use_container_width=True, hide_index=True
            )

            st.markdown("---")

            # 7. TARGETES D'EFICIÈNCIA GLOBAL (CÀLCUL PARTIT A PARTIT GARANTIT)
            game_possessions_list = []
            team_pts_list = []
            opp_pts_list = []

            for gid in valid_ids:
                gb = comp_box[comp_box["game_id"].astype(str) == str(gid)]
                gl = comp_lineups[
                    comp_lineups["game_id"].astype(str) == str(gid)
                ]
                ga = comp_adv[comp_adv["game_id"].astype(str) == str(gid)]

                # Punts d'Argentona del partit
                g_tpts = float(
                    pd.to_numeric(gb["Points"], errors="coerce").fillna(0).sum()
                )
                if g_tpts == 0 and not ga.empty:
                    sc_r = ga[
                        ga["Metric"]
                        .astype(str)
                        .str.contains(
                            "Game Score|Score|Marcador",
                            case=False,
                            na=False,
                        )
                    ]
                    if not sc_r.empty:
                        g_tpts = float(sc_r["Team_Value"].values[0])

                # Punts del Rival del partit (agafem el marcador oficial de Advanced Metrics)
                g_opts = 0.0
                if not ga.empty:
                    sc_r = ga[
                        ga["Metric"]
                        .astype(str)
                        .str.contains(
                            "Game Score|Score|Marcador",
                            case=False,
                            na=False,
                        )
                    ]
                    if not sc_r.empty:
                        g_opts = float(sc_r["Opp_Value"].values[0])

                if g_opts == 0 and not gl.empty:
                    g_opts = float(
                        pd.to_numeric(gl["PTS_Agn"], errors="coerce")
                        .fillna(0)
                        .sum()
                    )

                team_pts_list.append(g_tpts)
                opp_pts_list.append(g_opts)

                # Possessions del partit
                g_tfga = (
                    safe_col_sum(gb, "2PA")
                    + safe_col_sum(gb, "3PA")
                    if not gb.empty
                    else 0.0
                )
                g_tfta = safe_col_sum(gb, "FTA") if not gb.empty else 0.0
                g_toreb = safe_col_sum(gb, "Off Reb") if not gb.empty else 0.0
                g_ttov = safe_col_sum(gb, "Turnovers") if not gb.empty else 0.0
                g_tposs = g_tfga + 0.44 * g_tfta - g_toreb + g_ttov

                g_ofga = (
                    safe_col_sum(gl, "Rim_FGA_Agn")
                    + safe_col_sum(gl, "Paint_FGA_Agn")
                    + safe_col_sum(gl, "MR_FGA_Agn")
                    + safe_col_sum(gl, "Cor3_FGA_Agn")
                    + safe_col_sum(gl, "ATB3_FGA_Agn")
                    if not gl.empty
                    else 0.0
                )
                g_ofta = safe_col_sum(gl, "FTA_Agn") if not gl.empty else 0.0
                g_ooreb = safe_col_sum(gl, "OREB_Agn") if not gl.empty else 0.0
                g_otov = safe_col_sum(gl, "TOV_Agn") if not gl.empty else 0.0
                g_oposs = g_ofga + 0.44 * g_ofta - g_ooreb + g_otov

                if g_tposs >= 15.0 and g_oposs >= 15.0:
                    g_match_poss = 0.5 * (g_tposs + g_oposs)
                elif g_tposs >= 15.0:
                    g_match_poss = g_tposs
                elif g_oposs >= 15.0:
                    g_match_poss = g_oposs
                else:
                    g_match_poss = 80.0

                game_possessions_list.append(g_match_poss)

            # Sumatoris totals reals
            team_pts = sum(team_pts_list)
            opp_pts = sum(opp_pts_list)
            norm_poss_tot = sum(game_possessions_list)
            norm_poss_game = (
                norm_poss_tot / num_partits if num_partits > 0 else 0.0
            )

            tot_team_min = comp_box["MIN"].apply(parse_min_to_float).sum()
            pace_season = (
                (norm_poss_tot / tot_team_min) * 200.0
                if tot_team_min >= 100.0 * num_partits
                else norm_poss_game
            )

            # OER i DER reals de la temporada
            season_oer = (
                (team_pts / norm_poss_tot) * 100.0 if norm_poss_tot > 0 else 0.0
            )
            season_der = (
                (opp_pts / norm_poss_tot) * 100.0 if norm_poss_tot > 0 else 0.0
            )
            season_net = season_oer - season_der
            ppg_team = team_pts / num_partits if num_partits > 0 else 0.0
            ppg_opp = opp_pts / num_partits if num_partits > 0 else 0.0

            st.subheader("⚡ Eficiència Global de la Temporada")
            c_pace_s, c_oer_s, c_der_s = st.columns(3)
            c_pace_s.metric(
                "⏱️ Ritme de Joc (Pace a 40 min)",
                f"{pace_season:.1f} poss",
                delta=f"{norm_poss_game:.1f} poss/partit",
            )
            c_oer_s.metric(
                "🎯 OER (Eficiència Ofensiva)",
                f"{season_oer:.1f} pts/100",
                delta=f"{season_net:+.1f} Net Rating",
            )
            c_der_s.metric(
                "🛡️ DER (Eficiència Defensiva)",
                f"{season_der:.1f} pts/100",
                delta=f"{season_oer - season_der:+.1f} marge",
            )

            st.markdown("---")

            # 8. TAULA DE DESGLOSSAMENT D'EFICIÈNCIA ACUMULADA (14 FILES FIXES)
            st.subheader("📊 Desglossament d'Eficiència de Tir i Joc")
            col_table, _ = st.columns([3, 1])

            with col_table:
                pps_team_real = (
                    (team_fg_pts / team_fga) if team_fga > 0 else 0.0
                )
                pps_opp_real = (opp_fg_pts / opp_fga) if opp_fga > 0 else 0.0

                paint_pps_t = (
                    (team_paint_pts / team_paint_fga)
                    if team_paint_fga > 0
                    else 0.0
                )
                paint_pps_o = (
                    (opp_paint_pts / opp_paint_fga) if opp_paint_fga > 0 else 0.0
                )

                nopaint_pps_t = (
                    (team_nopaint_pts / team_nopaint_fga)
                    if team_nopaint_fga > 0
                    else 0.0
                )
                nopaint_pps_o = (
                    (opp_nopaint_pts / opp_nopaint_fga)
                    if opp_nopaint_fga > 0
                    else 0.0
                )

                # Cercar sumatoris d'Advanced Metrics
                def get_adv_sum(metric_patterns):
                    t_val = 0.0
                    o_val = 0.0
                    for _, r in comp_adv.iterrows():
                        m_l = str(r["Metric"]).lower()
                        if any(p.lower() in m_l for p in metric_patterns):
                            try:
                                t_val += float(r["Team_Value"])
                                o_val += float(r["Opp_Value"])
                            except Exception:
                                pass
                    return t_val, o_val

                # Extreure valors acumulats
                fb_pps_t_sum, fb_pps_o_sum = get_adv_sum(
                    ["fastbreak pps", "transició pps"]
                )
                half_pps_t_sum, half_pps_o_sum = get_adv_sum(
                    ["half-court pps", "estàtic", "estatic"]
                )
                pt_t_sum, pt_o_sum = get_adv_sum(
                    ["total paint touches", "tocs a pintura"]
                )
                pto_t_sum, pto_o_sum = get_adv_sum(
                    ["points off live to", "pts off to", "points off"]
                )
                lto_t_sum, lto_o_sum = get_adv_sum(["live turnovers", "lto"])
                dto_t_sum, dto_o_sum = get_adv_sum(["dead turnovers", "dto"])
                fb_t_sum, fb_o_sum = get_adv_sum(
                    ["total fastbreaks", "transicions totals"]
                )
                ch2_t_sum, ch2_o_sum = get_adv_sum(
                    ["2nd chance points", "2a oportunitat"]
                )
                kills_t_sum, kills_o_sum = get_adv_sum(
                    ["defensive kills", "kills"]
                )
                stocks_t_sum, stocks_o_sum = get_adv_sum(
                    ["stocks", "steals + blocks"]
                )

                # Calcular mitjanes per partit
                pbp_stats_s = compute_pbp_advanced_stats(comp_pbp, game_id=None)
                n_p = float(num_partits) if num_partits > 0 else 1.0

                table_rows_season = [
                    (
                        "Marcador del Partit",
                        f"{ppg_team:.1f} ({int(team_pts)} pts)",
                        f"{ppg_opp:.1f} ({int(opp_pts)} pts)",
                        ppg_team,
                        ppg_opp,
                        abs(ppg_team - ppg_opp) >= 2.0,
                        False,
                    ),
                    (
                        "Punts de Tir de Camp per Intent (PPS en joc)",
                        f"{pps_team_real:.2f}",
                        f"{pps_opp_real:.2f}",
                        pps_team_real,
                        pps_opp_real,
                        abs(pps_team_real - pps_opp_real) >= 0.15,
                        False,
                    ),
                    (
                        "PPS amb Toc a Pintura",
                        f"{paint_pps_t:.2f}",
                        f"{paint_pps_o:.2f}",
                        paint_pps_t,
                        paint_pps_o,
                        abs(paint_pps_t - paint_pps_o) >= 0.15,
                        False,
                    ),
                    (
                        "PPS sense Toc a Pintura",
                        f"{nopaint_pps_t:.2f}",
                        f"{nopaint_pps_o:.2f}",
                        nopaint_pps_t,
                        nopaint_pps_o,
                        abs(nopaint_pps_t - nopaint_pps_o) >= 0.15,
                        False,
                    ),
                    (
                        "PPS en Transició (Contraatac)",
                        f"{pbp_stats_s.get('fb_pps_t', 0.0):.2f}",
                        f"{pbp_stats_s.get('fb_pps_o', 0.0):.2f}",
                        pbp_stats_s.get("fb_pps_t", 0.0),
                        pbp_stats_s.get("fb_pps_o", 0.0),
                        abs(
                            pbp_stats_s.get("fb_pps_t", 0.0)
                            - pbp_stats_s.get("fb_pps_o", 0.0)
                        )
                        >= 0.15,
                        False,
                    ),
                    (
                        "PPS en Atac Estàtic (5c5)",
                        f"{pbp_stats_s.get('hc_pps_t', 0.0):.2f}",
                        f"{pbp_stats_s.get('hc_pps_o', 0.0):.2f}",
                        pbp_stats_s.get("hc_pps_t", 0.0),
                        pbp_stats_s.get("hc_pps_o", 0.0),
                        abs(
                            pbp_stats_s.get("hc_pps_t", 0.0)
                            - pbp_stats_s.get("hc_pps_o", 0.0)
                        )
                        >= 0.15,
                        False,
                    ),
                    (
                        "Punts de Pèrdua Viva (PTS OFF TO)",
                        f"{(pbp_stats_s.get('pts_off_to_t', 0.0) / n_p):.1f} ({int(pbp_stats_s.get('pts_off_to_t', 0))})",
                        f"{(pbp_stats_s.get('pts_off_to_o', 0.0) / n_p):.1f} ({int(pbp_stats_s.get('pts_off_to_o', 0))})",
                        pbp_stats_s.get("pts_off_to_t", 0.0) / n_p,
                        pbp_stats_s.get("pts_off_to_o", 0.0) / n_p,
                        abs(
                            pbp_stats_s.get("pts_off_to_t", 0.0)
                            - pbp_stats_s.get("pts_off_to_o", 0.0)
                        )
                        / n_p
                        >= 2.0,
                        False,
                    ),
                    (
                        "Pèrdues Vives (LTO)",
                        f"{(pbp_stats_s.get('lto_t', 0.0) / n_p):.1f} ({int(pbp_stats_s.get('lto_t', 0))})",
                        f"{(pbp_stats_s.get('lto_o', 0.0) / n_p):.1f} ({int(pbp_stats_s.get('lto_o', 0))})",
                        pbp_stats_s.get("lto_t", 0.0) / n_p,
                        pbp_stats_s.get("lto_o", 0.0) / n_p,
                        abs(
                            pbp_stats_s.get("lto_t", 0.0) - pbp_stats_s.get("lto_o", 0.0)
                        )
                        / n_p
                        >= 1.5,
                        True,
                    ),
                    (
                        "Pèrdues Mortes (DTO)",
                        f"{(pbp_stats_s.get('dto_t', 0.0) / n_p):.1f} ({int(pbp_stats_s.get('dto_t', 0))})",
                        f"{(pbp_stats_s.get('dto_o', 0.0) / n_p):.1f} ({int(pbp_stats_s.get('dto_o', 0))})",
                        pbp_stats_s.get("dto_t", 0.0) / n_p,
                        pbp_stats_s.get("dto_o", 0.0) / n_p,
                        abs(
                            pbp_stats_s.get("dto_t", 0.0) - pbp_stats_s.get("dto_o", 0.0)
                        )
                        / n_p
                        >= 1.5,
                        True,
                    ),
                    (
                        "Transicions Totals",
                        f"{(fb_t_sum/n_p):.1f} ({int(fb_t_sum)})",
                        f"{(fb_o_sum/n_p):.1f} ({int(fb_o_sum)})",
                        fb_t_sum / n_p,
                        fb_o_sum / n_p,
                        abs(fb_t_sum - fb_o_sum) / n_p >= 2.0,
                        False,
                    ),
                    (
                        "Punts de 2a Oportunitat",
                        f"{(ch2_t_sum/n_p):.1f} ({int(ch2_t_sum)})",
                        f"{(ch2_o_sum/n_p):.1f} ({int(ch2_o_sum)})",
                        ch2_t_sum / n_p,
                        ch2_o_sum / n_p,
                        abs(ch2_t_sum - ch2_o_sum) / n_p >= 2.0,
                        False,
                    ),
                    (
                        "Kills Defensius (3 Aturades Seguidas)",
                        f"{(kills_t_sum/n_p):.1f} ({int(kills_t_sum)})",
                        f"{(kills_o_sum/n_p):.1f} ({int(kills_o_sum)})",
                        kills_t_sum / n_p,
                        kills_o_sum / n_p,
                        abs(kills_t_sum - kills_o_sum) / n_p >= 1.5,
                        False,
                    ),
                    (
                        "Stocks (Recuperacions + Taps)",
                        f"{(stocks_t_sum/n_p):.1f} ({int(stocks_t_sum)})",
                        f"{(stocks_o_sum/n_p):.1f} ({int(stocks_o_sum)})",
                        stocks_t_sum / n_p,
                        stocks_o_sum / n_p,
                        abs(stocks_t_sum - stocks_o_sum) / n_p >= 2.0,
                        False,
                    ),
                ]

                rows_s_disp = []
                styles_s_disp = []

                for (
                    m_nom,
                    tv_str,
                    ov_str,
                    tv_num,
                    ov_num,
                    is_outlier,
                    lower_is_better,
                ) in table_rows_season:
                    if not is_outlier or tv_num == ov_num:
                        uem_css, riv_css = STYLE_NEUTRAL, STYLE_NEUTRAL
                    elif lower_is_better:
                        if tv_num < ov_num:
                            uem_css, riv_css = STYLE_RED, STYLE_BLUE
                        else:
                            uem_css, riv_css = STYLE_BLUE, STYLE_RED
                    else:
                        if tv_num > ov_num:
                            uem_css, riv_css = STYLE_RED, STYLE_BLUE
                        else:
                            uem_css, riv_css = STYLE_BLUE, STYLE_RED

                    rows_s_disp.append(
                        {"Mètrica": m_nom, "Argentona": tv_str, "Rivals": ov_str}
                    )
                    styles_s_disp.append((uem_css, riv_css))

                df_display_season = pd.DataFrame(rows_s_disp)

                def apply_text_styles_season(df):
                    css_df = pd.DataFrame("", index=df.index, columns=df.columns)
                    for i, (uem_css, riv_css) in enumerate(styles_s_disp):
                        css_df.loc[i, "Argentona"] = uem_css
                        css_df.loc[i, "Rivals"] = riv_css
                    return css_df

                table_height = (len(df_display_season) + 1) * 36 + 3
                st.dataframe(
                    df_display_season.style.apply(
                        apply_text_styles_season, axis=None
                    ),
                    height=table_height,
                    use_container_width=True,
                    hide_index=True,
                )

    # -------------------------------------------------------------
    # 2. SECCIÓ BOX SCORE (TOTALS ACUMULATS)
    # -------------------------------------------------------------
    elif active_tab_season == "📋 Box Score":
        comp_box = box_df[box_df["game_id"].astype(str).isin(valid_ids)].copy()
        comp_lineups = (
            lineups_df[lineups_df["game_id"].astype(str).isin(valid_ids)].copy()
        )

        if not comp_box.empty and "Name" in comp_box.columns:
            # 1. Filtrem la fila d'equip en els acumulats
            is_team_row_s = (
                comp_box["Player"]
                .astype(str)
                .str.lower()
                .str.strip()
                .isin(
                    ["team", "totals", "total", "opponent", "rival", "equip"]
                )
                | comp_box["Name"]
                .astype(str)
                .str.lower()
                .str.contains("team|total|opponent|equip|rival", na=False)
            )
            comp_box = comp_box[~is_team_row_s].copy()

            # 2. Minuts totals per al slider
            comp_box["_min_num"] = comp_box["MIN"].apply(parse_min_to_float)
            player_tot_mins = comp_box.groupby(["Player", "Name"])[
                "_min_num"
            ].sum()
            max_m = (
                int(player_tot_mins.max()) if not player_tot_mins.empty else 40
            )

            # 3. Filtre de Minuts a sobre (alineat a l'esquerra)
            col_b_filt, col_sp = st.columns([1, 2])
            with col_b_filt:
                min_m_filt = st.slider(
                    "⏱️ Minuts mínims (Temporada)",
                    min_value=0,
                    max_value=max(max_m, 10),
                    value=0,
                    step=5,
                    key="season_box_min_slider",
                )

            # 4. Selector de pastilles (NOMÉS UNA SOLA VEGADA)
            box_mode_s = st.segmented_control(
                "Tipus d'Estadística Acumulada",
                options=[
                    "Tradicional",
                    "Tir per Zones",
                    "4 Factors Individuals",
                    "Accions Defensives",
                    "Impacte i Context",
                    "Xarxa d'Assistències",
                    "Rebot i 2a Oportunitat",
                ],
                default="Tradicional",
                key="box_mode_selector_season",
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # 5. Filtratge de jugadors segons el slider
            valid_players = player_tot_mins[
                player_tot_mins >= min_m_filt
            ].index.tolist()
            comp_box_f = comp_box[
                comp_box.set_index(["Player", "Name"]).index.isin(valid_players)
            ].copy()
            # Agrupació utilitzant el conjunt de dades filtrat
            p_agg = (
                comp_box_f.groupby(["Player", "Name"])
                .agg(
                    PJ=("game_id", "nunique"),
                    PTS=("Points", "sum"),
                    PPP=("Points", "mean"),
                    T2M=("2PM", "sum"),
                    T2A=("2PA", "sum"),
                    T3M=("3PM", "sum"),
                    T3A=("3PA", "sum"),
                    FTM=("FTM", "sum"),
                    FTA=("FTA", "sum"),
                    RIM_M=("RIM(M)", "sum")
                    if "RIM(M)" in comp_box.columns
                    else ("Points", "count"),
                    RIM_A=("RIM(A)", "sum")
                    if "RIM(A)" in comp_box.columns
                    else ("Points", "count"),
                    PAINT_M=("PAINT(M)", "sum")
                    if "PAINT(M)" in comp_box.columns
                    else ("Points", "count"),
                    PAINT_A=("PAINT(A)", "sum")
                    if "PAINT(A)" in comp_box.columns
                    else ("Points", "count"),
                    MR_M=("MR(M)", "sum")
                    if "MR(M)" in comp_box.columns
                    else ("Points", "count"),
                    MR_A=("MR(A)", "sum")
                    if "MR(A)" in comp_box.columns
                    else ("Points", "count"),
                    C3_M=("C3(M)", "sum")
                    if "C3(M)" in comp_box.columns
                    else ("Points", "count"),
                    C3_A=("C3(A)", "sum")
                    if "C3(A)" in comp_box.columns
                    else ("Points", "count"),
                    ATB3_M=("ATB3(M)", "sum")
                    if "ATB3(M)" in comp_box.columns
                    else ("Points", "count"),
                    ATB3_A=("ATB3(A)", "sum")
                    if "ATB3(A)" in comp_box.columns
                    else ("Points", "count"),
                    FBM=("FB(M)", "sum")
                    if "FB(M)" in comp_box.columns
                    else ("Points", "count"),
                    FBMiss=("FB(Miss)", "sum")
                    if "FB(Miss)" in comp_box.columns
                    else ("Points", "count"),
                    Ch2M=("2ndCh(M)", "sum")
                    if "2ndCh(M)" in comp_box.columns
                    else ("Points", "count"),
                    Ch2Miss=("2ndCh(Miss)", "sum")
                    if "2ndCh(Miss)" in comp_box.columns
                    else ("Points", "count"),
                    REB_O=("Off Reb", "sum"),
                    REB_D=("Def Reb", "sum"),
                    AST=("Assists", "sum"),
                    REC=("Steals", "sum"),
                    TAP=("Blocks", "sum"),
                    PER=("Turnovers", "sum"),
                    FAL=("Fouls", "sum"),
                    F_REC=("Fouls Drawn", "sum"),
                    STOCKS=("Stocks", "sum")
                    if "Stocks" in comp_box.columns
                    else ("Steals", "sum"),
                    DEFL=("Deflections", "sum")
                    if "Deflections" in comp_box.columns
                    else ("Points", "count"),
                )
                .reset_index()
            )

            if box_mode_s == "Tradicional":
                df_trad_s = pd.DataFrame()
                df_trad_s["#"] = p_agg["Player"]
                df_trad_s["Jugador"] = p_agg["Name"]
                df_trad_s["PJ"] = p_agg["PJ"].astype(int)
                df_trad_s["PTS"] = p_agg["PTS"].astype(int)

                # PPP com a número
                df_trad_s["PPP"] = (
                    p_agg["PTS"] / p_agg["PJ"].replace(0, np.nan)
                ).fillna(0.0)

                t2m = pd.to_numeric(p_agg["T2M"], errors="coerce").fillna(0)
                t2a = pd.to_numeric(p_agg["T2A"], errors="coerce").fillna(0)
                df_trad_s["T2 (A/I)"] = (
                    t2m.astype(int).astype(str)
                    + "/"
                    + t2a.astype(int).astype(str)
                )
                df_trad_s["%T2"] = (
                    (t2m / t2a.replace(0, np.nan)) * 100.0
                ).fillna(0.0)

                t3m = pd.to_numeric(p_agg["T3M"], errors="coerce").fillna(0)
                t3a = pd.to_numeric(p_agg["T3A"], errors="coerce").fillna(0)
                df_trad_s["T3 (A/I)"] = (
                    t3m.astype(int).astype(str)
                    + "/"
                    + t3a.astype(int).astype(str)
                )
                df_trad_s["%T3"] = (
                    (t3m / t3a.replace(0, np.nan)) * 100.0
                ).fillna(0.0)

                ftm = pd.to_numeric(p_agg["FTM"], errors="coerce").fillna(0)
                fta = pd.to_numeric(p_agg["FTA"], errors="coerce").fillna(0)
                df_trad_s["TL (A/I)"] = (
                    ftm.astype(int).astype(str)
                    + "/"
                    + fta.astype(int).astype(str)
                )
                df_trad_s["%TL"] = (
                    (ftm / fta.replace(0, np.nan)) * 100.0
                ).fillna(0.0)

                df_trad_s["REB_O"] = p_agg["REB_O"].astype(int)
                df_trad_s["REB_D"] = p_agg["REB_D"].astype(int)
                df_trad_s["REB_T"] = df_trad_s["REB_O"] + df_trad_s["REB_D"]
                df_trad_s["AST"] = p_agg["AST"].astype(int)
                df_trad_s["REC"] = p_agg["REC"].astype(int)
                df_trad_s["TAP"] = p_agg["TAP"].astype(int)
                df_trad_s["PER"] = p_agg["PER"].astype(int)
                df_trad_s["FAL"] = p_agg["FAL"].astype(int)
                df_trad_s["F_REC"] = p_agg["F_REC"].astype(int)

                df_trad_s["_dorsal_sort"] = (
                    df_trad_s["#"].astype(str).str.replace("#", "").str.strip()
                )
                df_trad_s["_dorsal_sort"] = pd.to_numeric(
                    df_trad_s["_dorsal_sort"], errors="coerce"
                ).fillna(999)
                df_sorted = (
                    df_trad_s.sort_values(by="_dorsal_sort", ascending=True)
                    .drop(columns=["_dorsal_sort"])
                    .reset_index(drop=True)
                )

                render_boxscore(df_sorted, "Tradicional")
                
            elif box_mode_s == "Tir per Zones":
                df_shots_s = pd.DataFrame()
                df_shots_s["#"] = p_agg["Player"]
                df_shots_s["Jugador"] = p_agg["Name"]
                df_shots_s["PJ"] = p_agg["PJ"].astype(int)
                df_shots_s["PTS"] = p_agg["PTS"].astype(int)

                # Neteja de fraccions (sense .0)
                df_shots_s["Aro (Rim)"] = (
                    p_agg["RIM_M"].astype(int).astype(str)
                    + "/"
                    + p_agg["RIM_A"].astype(int).astype(str)
                )
                df_shots_s["Pintura (Paint)"] = (
                    p_agg["PAINT_M"].astype(int).astype(str)
                    + "/"
                    + p_agg["PAINT_A"].astype(int).astype(str)
                )
                df_shots_s["Mitja Distància (MR)"] = (
                    p_agg["MR_M"].astype(int).astype(str)
                    + "/"
                    + p_agg["MR_A"].astype(int).astype(str)
                )
                df_shots_s["Triple Cantonada (C3)"] = (
                    p_agg["C3_M"].astype(int).astype(str)
                    + "/"
                    + p_agg["C3_A"].astype(int).astype(str)
                )
                df_shots_s["Triple Frontal (ATB3)"] = (
                    p_agg["ATB3_M"].astype(int).astype(str)
                    + "/"
                    + p_agg["ATB3_A"].astype(int).astype(str)
                )

                # Ordenar per dorsal numèric (#1, #7, #8, #9, #11...)
                df_shots_s["_dorsal_sort"] = (
                    df_shots_s["#"].astype(str).str.replace("#", "").str.strip()
                )
                df_shots_s["_dorsal_sort"] = pd.to_numeric(
                    df_shots_s["_dorsal_sort"], errors="coerce"
                ).fillna(999)

                df_sorted = (
                    df_shots_s.sort_values(by="_dorsal_sort", ascending=True)
                    .drop(columns=["_dorsal_sort"])
                    .reset_index(drop=True)
                )

                render_boxscore(df_sorted, "Zones")

            elif box_mode_s == "4 Factors Individuals":
                df_4f_s = get_individual_four_factors(
                    comp_box, comp_lineups, comp_pbp, game_id=None
                )
                render_boxscore(df_4f_s, "4Factors")

            elif box_mode_s == "Accions Defensives":
                df_def_s = get_defensive_actions(
                    comp_box, comp_pbp, game_id=None
                )
                render_boxscore(df_def_s, "Defensa")

            elif box_mode_s == "Impacte i Context":
                df_ctx_s = pd.DataFrame()
                df_ctx_s["#"] = p_agg["Player"]
                df_ctx_s["Jugador"] = p_agg["Name"]
                df_ctx_s["PJ"] = p_agg["PJ"].astype(int)

                # Neteja de fraccions a enters purs (ex: 1/5, 2/2)
                fbm_int = (
                    pd.to_numeric(p_agg["FBM"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                fbtot_int = (
                    pd.to_numeric(
                        p_agg["FBM"] + p_agg["FBMiss"], errors="coerce"
                    )
                    .fillna(0)
                    .astype(int)
                )
                df_ctx_s["Transició"] = (
                    fbm_int.astype(str) + "/" + fbtot_int.astype(str)
                )

                ch2m_int = (
                    pd.to_numeric(p_agg["Ch2M"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                ch2tot_int = (
                    pd.to_numeric(
                        p_agg["Ch2M"] + p_agg["Ch2Miss"], errors="coerce"
                    )
                    .fillna(0)
                    .astype(int)
                )
                df_ctx_s["2a Oportunitat"] = (
                    ch2m_int.astype(str) + "/" + ch2tot_int.astype(str)
                )

                df_ctx_s["PER Totals"] = p_agg["PER"].astype(int)

                if "LTO" in comp_box.columns:
                    lto_map = (
                        comp_box.groupby(["Player", "Name"])["LTO"]
                        .sum()
                        .to_dict()
                    )
                    df_ctx_s["Vives (LTO)"] = [
                        int(lto_map.get((r["Player"], r["Name"]), 0))
                        for _, r in p_agg.iterrows()
                    ]
                if "DTO" in comp_box.columns:
                    dto_map = (
                        comp_box.groupby(["Player", "Name"])["DTO"]
                        .sum()
                        .to_dict()
                    )
                    df_ctx_s["Mortes (DTO)"] = [
                        int(dto_map.get((r["Player"], r["Name"]), 0))
                        for _, r in p_agg.iterrows()
                    ]

                df_ctx_s["Faltes Com."] = p_agg["FAL"].astype(int)
                df_ctx_s["Faltes Reb."] = p_agg["F_REC"].astype(int)

                # Ordenar per dorsal numèric (#1, #7, #8...)
                df_ctx_s["_dorsal_sort"] = (
                    df_ctx_s["#"].astype(str).str.replace("#", "").str.strip()
                )
                df_ctx_s["_dorsal_sort"] = pd.to_numeric(
                    df_ctx_s["_dorsal_sort"], errors="coerce"
                ).fillna(999)

                df_sorted = (
                    df_ctx_s.sort_values(by="_dorsal_sort", ascending=True)
                    .drop(columns=["_dorsal_sort"])
                    .reset_index(drop=True)
                )

                render_boxscore(df_sorted, "Context")

            elif box_mode_s == "Xarxa d'Assistències":
                df_rank_s, df_creadors_s, _ = get_assist_combos(
                    comp_pbp, comp_box, game_id=None
                )

                if not df_rank_s.empty:
                    tot_ast_s = int(df_creadors_s["Assistències_Totals"].sum())
                    tot_pts_gen_s = int(
                        df_creadors_s["Punts_Generats_Totals"].sum()
                    )
                    tot_t3_s = int(df_creadors_s["T3_Generats"].sum())

                    m_a1, m_a2, m_a3 = st.columns(3)
                    m_a1.metric(
                        "Assistències Totals (Temporada)", f"{tot_ast_s}"
                    )
                    m_a2.metric(
                        "Punts Generats per Assistència",
                        f"{tot_pts_gen_s} pts",
                        delta=f"{(tot_pts_gen_s/tot_ast_s):.2f} pts/ast"
                        if tot_ast_s > 0
                        else None,
                    )
                    m_a3.metric(
                        "Triples Assistits (Temporada)",
                        f"{tot_t3_s}",
                        delta=f"{tot_t3_s*3} pts via T3",
                    )

                    st.markdown("---")

                    st.markdown(
                        "##### 👑 Punts Totals Generats per Creador (Temporada)"
                    )
                    df_c_disp_s = df_creadors_s.rename(
                        columns={
                            "Assistències_Totals": "AST",
                            "Punts_Generats_Totals": "PTS Generats",
                            "T2_Generats": "T2 Assistits",
                            "T3_Generats": "T3 Assistits",
                        }
                    )
                    render_boxscore(df_c_disp_s, "Assistències")

                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown(
                        "##### 🤝 Connexions i Duos (Passador ➔ Anotador -"
                        " Temporada)"
                    )
                    df_r_disp_s = df_rank_s.rename(
                        columns={
                            "Assistències": "AST",
                            "Punts_Generats": "PTS Generats",
                            "T2": "T2 Assistits",
                            "T3": "T3 Assistits",
                        }
                    )
                    render_boxscore(df_r_disp_s, "Assistències")
                else:
                    st.info(
                        "No hi ha assistències registrades en els partits"
                        " seleccionats."
                    )
            elif box_mode_s == "Rebot i 2a Oportunitat":
                df_reb_s = pd.DataFrame()
                df_reb_s["#"] = p_agg["Player"]
                df_reb_s["Jugador"] = p_agg["Name"]
                df_reb_s["PJ"] = p_agg["PJ"].astype(int)

                reb_o_s = p_agg["REB_O"].astype(int)
                reb_d_s = p_agg["REB_D"].astype(int)
                reb_t_s = reb_o_s + reb_d_s
                df_reb_s["REB_O"] = reb_o_s
                df_reb_s["REB_D"] = reb_d_s
                df_reb_s["REB_T"] = reb_t_s
                df_reb_s["Reb/P"] = (reb_t_s / p_agg["PJ"].replace(0, np.nan)).fillna(0.0)

                # Càlcul OREB% acumulat i REB/Min
                p_mins_map = comp_box.groupby(["Player", "Name"])["_min_num"].sum().to_dict()
                p_mp_s = p_agg.apply(lambda r: p_mins_map.get((r["Player"], r["Name"]), 0.0), axis=1)
                t_mp_s = float(comp_box["_min_num"].sum())
                t_or_s = float(comp_box["Off Reb"].sum())
                opp_dr_s = float(pd.to_numeric(comp_lineups["DREB_Agn"], errors="coerce").fillna(0).sum()) if not comp_lineups.empty else (t_or_s * 1.5)

                orb_denom_s = p_mp_s * (t_or_s + opp_dr_s)
                df_reb_s["OREB%"] = np.where(
                    (p_mp_s > 0) & (orb_denom_s > 0),
                    100.0 * (reb_o_s * (t_mp_s / 5.0)) / orb_denom_s,
                    0.0
                )
                df_reb_s["REB/Min"] = np.round(np.where(p_mp_s > 0, reb_t_s / p_mp_s, 0.0), 2)

                # Segona Oportunitat
                ch2_m_s = pd.to_numeric(p_agg.get("Ch2M", 0), errors="coerce").fillna(0).astype(int)
                ch2_miss_s = pd.to_numeric(p_agg.get("Ch2Miss", 0), errors="coerce").fillna(0).astype(int)
                ch2_tot_s = ch2_m_s + ch2_miss_s

                df_reb_s["2a Op. (A/I)"] = [f"{m}/{tot}".rjust(10, '\u00A0') for m, tot in zip(ch2_m_s, ch2_tot_s)]
                df_reb_s["Pts 2a Op."] = ch2_m_s * 2
                df_reb_s["Fallades 2a Op."] = ch2_miss_s
                df_reb_s["% 2a Op."] = ((ch2_m_s / ch2_tot_s.replace(0, np.nan)) * 100.0).fillna(0.0)

                df_sorted = df_reb_s.sort_values(by=["REB_T", "REB_O"], ascending=False).reset_index(drop=True)
                render_boxscore(df_sorted, "Rebot")
        else:
            st.info("No hi ha dades acumulades de Boxscore disponibles.")
    # -------------------------------------------------------------
    # 3. SECCIÓ QUINTETS I ON/OFF (TOTALS ACUMULATS)
    # -------------------------------------------------------------
    elif active_tab_season == "👥 Quintets i On/Off":
        render_on_off_dashboard(comp_lineups, comp_box, key_prefix="season")