# app.py
from pathlib import Path
import numpy as np
import pandas as pd
from src.data_loader import CACHE_DIR, rebuild_cache
import streamlit as st
import base64
from src.utils import get_assist_combos

#python -m streamlit run app.py


st.set_page_config(
    page_title="CB Argentona 26-27",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estils CSS: Targetes uniformes, accents vermell (bo) / blau (dolent)
st.markdown(
    """
<style>
    /* 1. LIMITAR AMPLADA A TOTES LES VERSIONS DE STREAMLIT */
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    .block-container,
    .stMainBlockContainer,
    section[data-testid="stMain"] > div {
        max-width: 1400px !important;
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
selected_comp = st.sidebar.selectbox("Competició", all_comps)

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
}


# -------------------------------------------------------------
# FUNCIONS AUXILIARS D'ESTIL I OUTLIERS
# -------------------------------------------------------------
STYLE_RED = "color: #d00000; font-weight: 700;"  # Bo / Destacat
STYLE_BLUE = "color: #0077b6; font-weight: 700;"  # Desfavorable
STYLE_NEUTRAL = "color: #212529; font-weight: 400;"


def parse_fraction(s):
    """Extreu encerts i intents d'un string tipus '2/3'."""
    try:
        parts = str(s).split("/")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def style_boxscore(df_disp, mode="Tradicional"):
    def apply_row_styles(df):
        css_df = pd.DataFrame("", index=df.index, columns=df.columns)
        for i in df.index:
            row = df.loc[i]

            # Punts (Outlier >= 12 pts)
            if "PTS" in df.columns:
                try:
                    if float(row["PTS"]) >= 12:
                        css_df.loc[i, "PTS"] = STYLE_RED
                except Exception:
                    pass

            # Tradicional
            if mode == "Tradicional":
                if "%T2" in df.columns and "T2 (A/I)" in df.columns:
                    m, a = parse_fraction(row["T2 (A/I)"])
                    if a >= 2:
                        pct = (m / a) * 100
                        if pct >= 70:
                            css_df.loc[i, "%T2"] = STYLE_RED
                        elif pct <= 25:
                            css_df.loc[i, "%T2"] = STYLE_BLUE

                if "%T3" in df.columns and "T3 (A/I)" in df.columns:
                    m, a = parse_fraction(row["T3 (A/I)"])
                    if a >= 2:
                        pct = (m / a) * 100
                        if pct >= 50:
                            css_df.loc[i, "%T3"] = STYLE_RED
                        elif pct <= 20:
                            css_df.loc[i, "%T3"] = STYLE_BLUE

                if "%TL" in df.columns and "TL (A/I)" in df.columns:
                    m, a = parse_fraction(row["TL (A/I)"])
                    if a >= 2:
                        pct = (m / a) * 100
                        if pct >= 85:
                            css_df.loc[i, "%TL"] = STYLE_RED
                        elif pct <= 40:
                            css_df.loc[i, "%TL"] = STYLE_BLUE

                if "REB_T" in df.columns:
                    try:
                        if float(row["REB_T"]) >= 6:
                            css_df.loc[i, "REB_T"] = STYLE_RED
                    except Exception:
                        pass

                if "AST" in df.columns:
                    try:
                        if float(row["AST"]) >= 4:
                            css_df.loc[i, "AST"] = STYLE_RED
                    except Exception:
                        pass

                if "REC" in df.columns:
                    try:
                        if float(row["REC"]) >= 3:
                            css_df.loc[i, "REC"] = STYLE_RED
                    except Exception:
                        pass

                if "PER" in df.columns:
                    try:
                        if float(row["PER"]) >= 3:
                            css_df.loc[i, "PER"] = STYLE_BLUE
                    except Exception:
                        pass

            # Zones de tir
            elif mode == "Zones":
                shot_cols = [
                    "Aro (Rim)",
                    "Pintura (Paint)",
                    "Mitja Distància (MR)",
                    "Triple Cantonada (C3)",
                    "Triple Frontal (ATB3)",
                ]
                for col in shot_cols:
                    if col in df.columns:
                        m, a = parse_fraction(row[col])
                        if a >= 2:
                            pct = (m / a) * 100
                            if pct >= 70:
                                css_df.loc[i, col] = STYLE_RED
                            elif pct == 0:
                                css_df.loc[i, col] = STYLE_BLUE

            # Context i Impacte
            elif mode == "Context":
                if "Stocks (Rec+Tap)" in df.columns:
                    try:
                        if float(row["Stocks (Rec+Tap)"]) >= 3:
                            css_df.loc[i, "Stocks (Rec+Tap)"] = STYLE_RED
                    except Exception:
                        pass
                if "Pèrdues" in df.columns:
                    try:
                        if float(row["Pèrdues"]) >= 3:
                            css_df.loc[i, "Pèrdues"] = STYLE_BLUE
                    except Exception:
                        pass

        return css_df

    return df_disp.style.apply(apply_row_styles, axis=None)


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
        options=["📈 Mètriques i 4 Factors", "📋 Box Score"],
        default="📈 Mètriques i 4 Factors",
        key="main_active_tab",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 1. PESTAÑA: MÈTRIQUES I 4 FACTORS
    # -------------------------------------------------------------
    if active_tab == "📈 Mètriques i 4 Factors":
        g_adv = adv_df[adv_df["game_id"] == str(selected_game_id)].copy()

        if not g_adv.empty:
            adv_dict_team = dict(zip(g_adv["Metric"], g_adv["Team_Value"]))
            adv_dict_opp = dict(zip(g_adv["Metric"], g_adv["Opp_Value"]))

            score_team = adv_dict_team.get("Game Score", 0)
            score_opp = adv_dict_opp.get("Game Score", 0)
            efg_team = adv_dict_team.get("1. Effective FG% (eFG%)", 0.0)
            efg_opp = adv_dict_opp.get("1. Effective FG% (eFG%)", 0.0)
            tov_team = adv_dict_team.get("2. Turnover Rate (TO%)", 0.0)
            tov_opp = adv_dict_opp.get("2. Turnover Rate (TO%)", 0.0)
            pps_team = adv_dict_team.get("Overall Points Per Shot (PPS)", 0.0)
            pps_opp = adv_dict_opp.get("Overall Points Per Shot (PPS)", 0.0)

            # Targetes KPI Superiors
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(
                "Marcador Final",
                f"{int(score_team)} - {int(score_opp)}",
                delta=f"{int(score_team - score_opp):+d} pts",
            )
            kpi2.metric(
                "eFG% Efectiu",
                f"{efg_team:.1f}%",
                delta=f"{efg_team - efg_opp:+.1f}% vs Rival",
            )
            kpi3.metric(
                "PPS (Punts/Tir)",
                f"{pps_team:.2f}",
                delta=f"{pps_team - pps_opp:+.2f} vs Rival",
            )
            kpi4.metric(
                "Control de Pèrdues (TOV%)",
                f"{tov_team:.1f}%",
                delta=f"{tov_opp - tov_team:+.1f}% marge",
            )

            st.markdown("---")

            # Càlcul del Free Throw Rate real d'Oliver (FTM / FGA) a partir de les dades reals
            # Càlcul del Free Throw Rate d'Oliver (FTM / FGA)
            g_box_match = box_df[box_df["game_id"] == str(selected_game_id)]
            g_line_match = lineups_df[
                lineups_df["game_id"] == str(selected_game_id)
            ]

            match_team_ftm = float(
                pd.to_numeric(g_box_match["FTM"], errors="coerce")
                .fillna(0)
                .sum()
            )
            match_team_fga = float(
                (
                    pd.to_numeric(g_box_match["2PA"], errors="coerce").fillna(0)
                    + pd.to_numeric(g_box_match["3PA"], errors="coerce").fillna(
                        0
                    )
                ).sum()
            )
            ft_rate_team = (
                (match_team_ftm / match_team_fga) if match_team_fga > 0 else 0.0
            )

            match_opp_ftm = float(
                pd.to_numeric(g_line_match["FTM_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            match_opp_fga = float(
                (
                    pd.to_numeric(
                        g_line_match["Rim_FGA_Agn"], errors="coerce"
                    ).fillna(0)
                    + pd.to_numeric(
                        g_line_match["Paint_FGA_Agn"], errors="coerce"
                    ).fillna(0)
                    + pd.to_numeric(
                        g_line_match["MR_FGA_Agn"], errors="coerce"
                    ).fillna(0)
                    + pd.to_numeric(
                        g_line_match["Cor3_FGA_Agn"], errors="coerce"
                    ).fillna(0)
                    + pd.to_numeric(
                        g_line_match["ATB3_FGA_Agn"], errors="coerce"
                    ).fillna(0)
                ).sum()
            )
            ft_rate_opp = (
                (match_opp_ftm / match_opp_fga) if match_opp_fga > 0 else 0.0
            )

            oreb_team = adv_dict_team.get("3. Offensive Rebound% (OREB%)", 0.0)
            oreb_opp = adv_dict_opp.get("3. Offensive Rebound% (OREB%)", 0.0)

            # Avaluació de qui ha guanyat cada factor
            win_efg = efg_team > efg_opp
            win_tov = (
                tov_team < tov_opp
            )  # Menys pèrdues és guanyar el factor
            win_oreb = oreb_team > oreb_opp
            win_ft = ft_rate_team > ft_rate_opp
            factors_guanyats = sum([win_efg, win_tov, win_oreb, win_ft])

            st.subheader(
                f"🎯 Els 4 Factors (Dean Oliver) — Balanç: {factors_guanyats}/4 Guanyats"
            )

            # TAULA ÚNICA UNIFICADA
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
                        "🔴 El nostre Atac": f"{ft_rate_team:.2f} ({int(match_team_ftm)} TL / {int(match_team_fga)} TC)",
                        "🛡️ La nostra Defensa (Rival)": f"{ft_rate_opp:.2f} ({int(match_opp_ftm)} TL / {int(match_opp_fga)} TC)",
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

            # Taula de desglossament amb colors a números atípics
            st.subheader("Desglossament d'Eficiència i Zones")
            col_table, _ = st.columns([3, 1])

            with col_table:
                rows = []
                styles = []

                for _, r in g_adv.iterrows():
                    m_raw = str(r["Metric"]).strip()
                    # Ometre faltes d'equip
                    if (
                        "team fouls" in m_raw.lower()
                        or "faltes" in m_raw.lower()
                    ):
                        continue
                    m = METRIC_NAMES_CAT.get(
                        m_raw, m_raw
                    )  # Traducció al català

                    tv = (
                        float(r["Team_Value"])
                        if pd.notna(r["Team_Value"])
                        else 0.0
                    )
                    ov = (
                        float(r["Opp_Value"])
                        if pd.notna(r["Opp_Value"])
                        else 0.0
                    )

                    if "%" in m_raw:
                        tv_str, ov_str = f"{tv:.1f}%", f"{ov:.1f}%"
                        is_outlier = abs(tv - ov) >= 4.0
                    elif "PPS" in m_raw or "Rate" in m_raw:
                        tv_str, ov_str = f"{tv:.2f}", f"{ov:.2f}"
                        is_outlier = abs(tv - ov) >= 0.20
                    else:
                        tv_str, ov_str = f"{int(tv)}", f"{int(ov)}"
                        is_outlier = abs(tv - ov) >= 2

                    m_lower = m_raw.lower()
                    lower_is_better = (
                        "turnover" in m_lower
                        or "fouls" in m_lower
                        or "pèrdua" in m_lower
                        or "pérdida" in m_lower
                    )

                    if not is_outlier or tv == ov:
                        uem_css, riv_css = STYLE_NEUTRAL, STYLE_NEUTRAL
                    elif lower_is_better:
                        if tv < ov:
                            uem_css, riv_css = STYLE_RED, STYLE_BLUE
                        else:
                            uem_css, riv_css = STYLE_BLUE, STYLE_RED
                    else:
                        if tv > ov:
                            uem_css, riv_css = STYLE_RED, STYLE_BLUE
                        else:
                            uem_css, riv_css = STYLE_BLUE, STYLE_RED

                    rows.append({"Mètrica": m, "UEM": tv_str, "Rival": ov_str})
                    styles.append((uem_css, riv_css))

                df_display = pd.DataFrame(rows)

                def apply_text_styles(df):
                    css_df = pd.DataFrame("", index=df.index, columns=df.columns)
                    for i, (uem_css, riv_css) in enumerate(styles):
                        css_df.loc[i, "UEM"] = uem_css
                        css_df.loc[i, "Rival"] = riv_css
                    return css_df

                styled_table = df_display.style.apply(
                    apply_text_styles, axis=None
                )

                # Alçada automàtica per mostrar totes les files sense scroll intern
                table_height = (len(df_display) + 1) * 36 + 3
                st.dataframe(
                    styled_table,
                    height=table_height,
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info(
                "No hi ha mètriques avançades disponibles per a aquest partit."
            )

    # -------------------------------------------------------------
    # 2. SECCIÓ BOX SCORE (PARTIT INDIVIDUAL)
    # -------------------------------------------------------------
    elif active_tab == "📋 Box Score":
        g_box = box_df[box_df["game_id"] == str(selected_game_id)].copy()

        if not g_box.empty:
            box_mode = st.segmented_control(
                "Tipus d'Estadística",
                options=[
                    "Tradicional (Estàndard)",
                    "Zones de Tir (Aro / Pintura / Mitja / 3P)",
                    "Context i Impacte (Transició / Segones Opcions / Faltes)",
                    "Xarxa de Passades (Assistències)",
                    "Complet (Totes les Columnes)",
                ],
                default="Tradicional (Estàndard)",
                key="box_mode_selector_single",
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if box_mode == "Tradicional (Estàndard)":
                df_trad = pd.DataFrame()
                df_trad["#"] = g_box["Player"]
                df_trad["Jugador"] = g_box["Name"]
                df_trad["MIN"] = g_box["MIN"].astype(str).str.strip()
                df_trad["PTS"] = g_box["Points"].astype(int)
                df_trad["T2 (A/I)"] = (
                    g_box["2PM"].astype(int).astype(str)
                    + "/"
                    + g_box["2PA"].astype(int).astype(str)
                )
                df_trad["%T2"] = g_box["2P%"].apply(lambda x: f"{x:.0f}%")
                df_trad["T3 (A/I)"] = (
                    g_box["3PM"].astype(int).astype(str)
                    + "/"
                    + g_box["3PA"].astype(int).astype(str)
                )
                df_trad["%T3"] = g_box["3P%"].apply(lambda x: f"{x:.0f}%")
                df_trad["TL (A/I)"] = (
                    g_box["FTM"].astype(int).astype(str)
                    + "/"
                    + g_box["FTA"].astype(int).astype(str)
                )
                df_trad["%TL"] = g_box["FT%"].apply(lambda x: f"{x:.0f}%")
                df_trad["REB_O"] = g_box["Off Reb"].astype(int)
                df_trad["REB_D"] = g_box["Def Reb"].astype(int)
                df_trad["REB_T"] = df_trad["REB_O"] + df_trad["REB_D"]
                df_trad["AST"] = g_box["Assists"].astype(int)
                df_trad["REC"] = g_box["Steals"].astype(int)
                df_trad["TAP"] = g_box["Blocks"].astype(int)
                df_trad["PER"] = g_box["Turnovers"].astype(int)
                df_trad["FAL"] = g_box["Fouls"].astype(int)
                df_trad["F_REC"] = g_box["Fouls Drawn"].astype(int)

                df_sorted = df_trad.sort_values(
                    by="PTS", ascending=False
                ).reset_index(drop=True)
                st.dataframe(
                    style_boxscore(df_sorted, "Tradicional"),
                    use_container_width=True,
                    hide_index=True,
                )

            elif box_mode == "Zones de Tir (Aro / Pintura / Mitja / 3P)":
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
                st.dataframe(
                    style_boxscore(df_sorted, "Zones"),
                    use_container_width=True,
                    hide_index=True,
                )

            elif (
                box_mode
                == "Context i Impacte (Transició / Segones Opcions / Faltes)"
            ):
                df_ctx = pd.DataFrame()
                df_ctx["#"] = g_box["Player"]
                df_ctx["Jugador"] = g_box["Name"]
                df_ctx["Transició"] = (
                    g_box["FB(M)"].astype(int).astype(str)
                    + "/"
                    + (g_box["FB(M)"] + g_box["FB(Miss)"]).astype(int).astype(str)
                )
                df_ctx["2a Oportunitat"] = (
                    g_box["2ndCh(M)"].astype(int).astype(str)
                    + "/"
                    + (g_box["2ndCh(M)"] + g_box["2ndCh(Miss)"])
                    .astype(int)
                    .astype(str)
                )
                df_ctx["Stocks (Rec+Tap)"] = g_box["Stocks"].astype(int)
                df_ctx["Defleccions"] = g_box["Deflections"].astype(int)
                df_ctx["Pèrdues"] = g_box["Turnovers"].astype(int)
                df_ctx["Faltes Com."] = g_box["Fouls"].astype(int)
                df_ctx["Faltes Reb."] = g_box["Fouls Drawn"].astype(int)

                df_sorted = df_ctx.sort_values(
                    by="Stocks (Rec+Tap)", ascending=False
                ).reset_index(drop=True)
                st.dataframe(
                    style_boxscore(df_sorted, "Context"),
                    use_container_width=True,
                    hide_index=True,
                )

            # 👇 TAULES D'ASSISTÈNCIES UNA SOTA L'ALTRA
            elif box_mode == "Xarxa de Passades (Assistències)":
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

                    # 1. Taula de Creadors (A dalt, amplada completa)
                    st.markdown(
                        "##### 👑 Punts Totals Generats per Creador"
                    )
                    st.dataframe(
                        df_creadors.rename(
                            columns={
                                "Assistències_Totals": "AST",
                                "Punts_Generats_Totals": "PTS Generats",
                                "T2_Generats": "T2 Assistits",
                                "T3_Generats": "T3 Assistits",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # 2. Taula de Duos (A sota, amplada completa)
                    st.markdown(
                        "##### 🤝 Connexions i Duos (Passador ➔ Anotador)"
                    )
                    st.dataframe(
                        df_rank.rename(
                            columns={
                                "Assistències": "AST",
                                "Punts_Generats": "PTS Generats",
                                "T2": "T2 Assistits",
                                "T3": "T3 Assistits",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info(
                        "No hi ha assistències registrades en aquest partit."
                    )

            else:
                st.dataframe(
                    g_box[[c for c in g_box.columns if c != "game_id"]],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No hi ha dades de Boxscore disponibles.")


# =============================================================
# MODO 2: TOTALS ACUMULATS (AMB LA MATEIXA ESTRUCTURA)
# =============================================================
else:
    st.title(f"Totals Acumulats ({selected_comp})")

    valid_ids = filtered_games["game_id"].astype(str).tolist()
    num_partits = len(valid_ids)

    # Barra superior de navegació als acumulats
    active_tab_season = st.segmented_control(
        "Navegació Acumulats",
        options=["📈 Mètriques i 4 Factors", "📋 Box Score"],
        default="📈 Mètriques i 4 Factors",
        key="season_active_tab",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 1. PESTAÑA: MÈTRIQUES, 4 FACTORS I VOLUMS ACUMULATS
    # -------------------------------------------------------------
    if active_tab_season == "📈 Mètriques i 4 Factors":
        comp_lineups = lineups_df[
            lineups_df["game_id"].astype(str).isin(valid_ids)
        ].copy()
        comp_box = box_df[box_df["game_id"].astype(str).isin(valid_ids)].copy()
        comp_adv = adv_df[adv_df["game_id"].astype(str).isin(valid_ids)].copy()

        if not comp_lineups.empty or not comp_box.empty:

            def safe_col_sum(df, col):
                if col in df.columns:
                    return float(
                        pd.to_numeric(df[col], errors="coerce").fillna(0).sum()
                    )
                return 0.0

            # 1. VOLUMS REALS ACUMULATS (UEM)
            team_pts = safe_col_sum(comp_box, "Points")
            team_2pm = int(safe_col_sum(comp_box, "2PM"))
            team_2pa = int(safe_col_sum(comp_box, "2PA"))
            team_3pm = int(safe_col_sum(comp_box, "3PM"))
            team_3pa = int(safe_col_sum(comp_box, "3PA"))
            team_fgm = team_2pm + team_3pm
            team_fga = team_2pa + team_3pa

            team_ftm = int(safe_col_sum(comp_box, "FTM"))
            team_fta = int(safe_col_sum(comp_box, "FTA"))
            team_oreb = int(safe_col_sum(comp_box, "Off Reb"))
            team_dreb = int(safe_col_sum(comp_box, "Def Reb"))
            team_tov = int(safe_col_sum(comp_box, "Turnovers"))
            team_ast = int(safe_col_sum(comp_box, "Assists"))
            team_rec = int(safe_col_sum(comp_box, "Steals"))
            team_tap = int(safe_col_sum(comp_box, "Blocks"))

            # 2. VOLUMS REALS ACUMULATS (RIVALS)
            opp_pts = safe_col_sum(comp_lineups, "PTS_Agn")
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

            opp_3pm = opp_c3_m + opp_atb3_m
            opp_3pa = opp_c3_a + opp_atb3_a
            opp_2pm = opp_rim_m + opp_paint_m + opp_mr_m
            opp_2pa = opp_rim_a + opp_paint_a + opp_mr_a
            opp_fgm = opp_2pm + opp_3pm
            opp_fga = opp_2pa + opp_3pa

            opp_ftm = int(safe_col_sum(comp_lineups, "FTM_Agn"))
            opp_fta = int(safe_col_sum(comp_lineups, "FTA_Agn"))
            opp_oreb = int(safe_col_sum(comp_lineups, "OREB_Agn"))
            opp_dreb = int(safe_col_sum(comp_lineups, "DREB_Agn"))
            opp_tov = int(safe_col_sum(comp_lineups, "TOV_Agn"))

            # 3. CÀLCUL DELS 4 FACTORS
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

            team_poss = float(team_fga + 0.44 * team_fta + team_tov)
            opp_poss = float(opp_fga + 0.44 * opp_fta + opp_tov)
            tov_team = (
                (team_tov / team_poss * 100) if team_poss > 0 else 0.0
            )
            tov_opp = (opp_tov / opp_poss * 100) if opp_poss > 0 else 0.0

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

            pps_team = (team_pts / team_fga) if team_fga > 0 else 0.0
            pps_opp = (opp_pts / opp_fga) if opp_fga > 0 else 0.0

            ppg_team = team_pts / num_partits if num_partits > 0 else 0.0
            ppg_opp = opp_pts / num_partits if num_partits > 0 else 0.0

            # Targetes KPI Superiors amb Volum
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(
                f"Mitjana Punts ({num_partits} Partits)",
                f"{ppg_team:.1f} - {ppg_opp:.1f}",
                delta=f"Total: {int(team_pts)}-{int(opp_pts)} pts",
            )
            kpi2.metric(
                "eFG% Efectiu Acumulat",
                f"{efg_team:.1f}%",
                delta=f"{team_fgm}/{team_fga} TC ({team_3pm} T3)",
            )
            kpi3.metric(
                "PPS Acumulat (Punts/Tir)",
                f"{pps_team:.2f}",
                delta=f"{int(team_pts)} pts / {team_fga} tirs",
            )
            kpi4.metric(
                "Control Pèrdues (TOV%)",
                f"{tov_team:.1f}%",
                delta=f"{team_tov} pèrdues en {team_poss:.0f} poss",
            )

            st.markdown("---")

            # -------------------------------------------------------------
            # QUADRE RESUM DE VOLUMS REALS DE LA TEMPORADA
            # -------------------------------------------------------------
            st.subheader("📦 Volums Totals de Llançament i Joc")
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                pct_2p = (team_2pm / team_2pa * 100) if team_2pa > 0 else 0.0
                pct_3p = (team_3pm / team_3pa * 100) if team_3pa > 0 else 0.0
                pct_ft = (team_ftm / team_fta * 100) if team_fta > 0 else 0.0
                pct_fg = (team_fgm / team_fga * 100) if team_fga > 0 else 0.0

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
                            "Concepte": "Rebots (Ofensius / Defensius)",
                            "Encertats / Intentats": f"{team_oreb} Reb.O / {team_dreb} Reb.D",
                            "Percentatge": f"{team_oreb+team_dreb} Totals",
                            "Mitjana / Partit": f"{(team_oreb+team_dreb)/num_partits:.1f} reb",
                        },
                        {
                            "Concepte": "Pèrdues / Assistències",
                            "Encertats / Intentats": f"{team_tov} PER / {team_ast} AST",
                            "Percentatge": f"Ràtio: {team_ast/team_tov:.2f}"
                            if team_tov > 0
                            else "-",
                            "Mitjana / Partit": f"{team_tov/num_partits:.1f} PER",
                        },
                    ]
                )
                st.markdown("##### 🔴 Volums UEM")
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
                            "Concepte": "Rebots (Ofensius / Defensius)",
                            "Encertats / Intentats": f"{opp_oreb} Reb.O / {opp_dreb} Reb.D",
                            "Percentatge": f"{opp_oreb+opp_dreb} Totals",
                            "Mitjana / Partit": f"{(opp_oreb+opp_dreb)/num_partits:.1f} reb",
                        },
                        {
                            "Concepte": "Pèrdues Totals Rivals",
                            "Encertats / Intentats": f"{opp_tov} PER",
                            "Percentatge": f"{opp_poss:.0f} Possessions",
                            "Mitjana / Partit": f"{opp_tov/num_partits:.1f} PER",
                        },
                    ]
                )
                st.markdown("##### 🛡️ Volums Rivals")
                st.dataframe(
                    df_vol_opp, use_container_width=True, hide_index=True
                )

            st.markdown("---")

            # -------------------------------------------------------------
            # 4 FACTORS AMB DESGLOSSAMENT DEL VOLUM REAL
            # -------------------------------------------------------------
            # Càlcul de quants factors s'han guanyat a la temporada
            win_efg_s = efg_team > efg_opp
            win_tov_s = (
                tov_team < tov_opp
            )  # Menys pèrdues és guanyar el factor
            win_oreb_s = oreb_team > oreb_opp
            win_ft_s = ft_rate_team > ft_rate_opp
            factors_guanyats_s = sum(
                [win_efg_s, win_tov_s, win_oreb_s, win_ft_s]
            )

            st.subheader(
                f"🎯 Els 4 Factors Reals de la Temporada — Balanç: {factors_guanyats_s}/4 Guanyats"
            )

            # TAULA ÚNICA UNIFICADA AMB VOLUMS
            df_ff_unified_season = pd.DataFrame(
                [
                    {
                        "Factor": "1. Tir Efectiu (eFG%)",
                        "🔴 Atac UEM": f"{efg_team:.1f}% ({team_fgm}/{team_fga} TC, {team_3pm} T3)",
                        "🛡️ Defensa (Rivals)": f"{efg_opp:.1f}% ({opp_fgm}/{opp_fga} TC, {opp_3pm} T3)",
                        "Diferencial Net": f"{efg_team - efg_opp:+.1f}%",
                        "Impacte": "🔴 Guanyat (Més efectivitat global)"
                        if win_efg_s
                        else "🔵 Perdut",
                    },
                    {
                        "Factor": "2. Control de Pèrdues (TOV%)",
                        "🔴 Atac UEM": f"{tov_team:.1f}% ({team_tov} PER / {team_poss:.1f} Poss)",
                        "🛡️ Defensa (Rivals)": f"{tov_opp:.1f}% ({opp_tov} PER / {opp_poss:.1f} Poss)",
                        "Diferencial Net": f"{tov_opp - tov_team:+.1f}% marge",
                        "Impacte": "🔴 Guanyat (Millor cura de pilota)"
                        if win_tov_s
                        else "🔵 Perdut",
                    },
                    {
                        "Factor": "3. Rebot Ofensiu (OREB%)",
                        "🔴 Atac UEM": f"{oreb_team:.1f}% ({team_oreb} Reb.O / {team_oreb + opp_dreb} Disp.)",
                        "🛡️ Defensa (Rivals)": f"{oreb_opp:.1f}% ({opp_oreb} Reb.O / {opp_oreb + team_dreb} Disp.)",
                        "Diferencial Net": f"{oreb_team - oreb_opp:+.1f}%",
                        "Impacte": "🔴 Guanyat (Més rebot ofensiu)"
                        if win_oreb_s
                        else "🔵 Perdut",
                    },
                    {
                        "Factor": "4. Freqüència TL (FT Rate)",
                        "🔴 Atac UEM": f"{ft_rate_team:.2f} ({team_ftm} TL / {team_fga} TC)",
                        "🛡️ Defensa (Rivals)": f"{ft_rate_opp:.2f} ({opp_ftm} TL / {opp_fga} TC)",
                        "Diferencial Net": f"{ft_rate_team - ft_rate_opp:+.2f}",
                        "Impacte": "🔴 Guanyat (Més producció en TL)"
                        if win_ft_s
                        else "🔵 Perdut",
                    },
                ]
            )

            st.dataframe(
                df_ff_unified_season, use_container_width=True, hide_index=True
            )

            st.markdown("---")

            # -------------------------------------------------------------
            # TAULA DE DESGLOSSAMENT D'EFICIÈNCIA
            # -------------------------------------------------------------
            st.subheader("📊 Desglossament d'Eficiència Ponderat")
            col_table, _ = st.columns([3, 1])

            with col_table:
                team_rim_a = int(safe_col_sum(comp_box, "RIM(A)"))
                team_rim_m = int(safe_col_sum(comp_box, "RIM(M)"))
                team_paint_a = int(safe_col_sum(comp_box, "PAINT(A)"))
                team_paint_m = int(safe_col_sum(comp_box, "PAINT(M)"))

                team_paint_fga = team_rim_a + team_paint_a
                team_paint_pts = (team_rim_m + team_paint_m) * 2
                paint_pps_team = (
                    (team_paint_pts / team_paint_fga)
                    if team_paint_fga > 0
                    else 0.0
                )

                opp_paint_fga = opp_rim_a + opp_paint_a
                opp_paint_pts = (opp_rim_m + opp_paint_m) * 2
                paint_pps_opp = (
                    (opp_paint_pts / opp_paint_fga) if opp_paint_fga > 0 else 0.0
                )

                team_nopaint_fga = team_fga - team_paint_fga
                team_nopaint_pts = team_pts - team_paint_pts - team_ftm
                nopaint_pps_team = (
                    (team_nopaint_pts / team_nopaint_fga)
                    if team_nopaint_fga > 0
                    else 0.0
                )

                opp_nopaint_fga = opp_fga - opp_paint_fga
                opp_nopaint_pts = opp_pts - opp_paint_pts - opp_ftm
                nopaint_pps_opp = (
                    (opp_nopaint_pts / opp_nopaint_fga)
                    if opp_nopaint_fga > 0
                    else 0.0
                )

                adv_sums = (
                    comp_adv.groupby("Metric")
                    .agg(
                        Team_Sum=("Team_Value", "sum"),
                        Opp_Sum=("Opp_Value", "sum"),
                    )
                    .reset_index()
                )
                adv_sums_dict_t = dict(
                    zip(adv_sums["Metric"], adv_sums["Team_Sum"])
                )
                adv_sums_dict_o = dict(
                    zip(adv_sums["Metric"], adv_sums["Opp_Sum"])
                )

                table_rows_data = [
                    (
                        "Game Score",
                        f"{ppg_team:.1f} ({int(team_pts)} pts)",
                        f"{ppg_opp:.1f} ({int(opp_pts)} pts)",
                        ppg_team,
                        ppg_opp,
                        abs(ppg_team - ppg_opp) >= 2.0,
                        False,
                    ),
                    (
                        "1. Effective FG% (eFG%)",
                        f"{efg_team:.1f}% ({team_fgm}/{team_fga} TC)",
                        f"{efg_opp:.1f}% ({opp_fgm}/{opp_fga} TC)",
                        efg_team,
                        efg_opp,
                        abs(efg_team - efg_opp) >= 3.0,
                        False,
                    ),
                    (
                        "2. Turnover Rate (TO%)",
                        f"{tov_team:.1f}% ({team_tov} PER)",
                        f"{tov_opp:.1f}% ({opp_tov} PER)",
                        tov_team,
                        tov_opp,
                        abs(tov_team - tov_opp) >= 3.0,
                        True,
                    ),
                    (
                        "3. Offensive Rebound% (OREB%)",
                        f"{oreb_team:.1f}% ({team_oreb} Reb.O)",
                        f"{oreb_opp:.1f}% ({opp_oreb} Reb.O)",
                        oreb_team,
                        oreb_opp,
                        abs(oreb_team - oreb_opp) >= 3.0,
                        False,
                    ),
                    (
                        "4. Free Throw Rate (FT Rate)",
                        f"{ft_rate_team:.2f} ({team_fta} TL)",
                        f"{ft_rate_opp:.2f} ({opp_fta} TL)",
                        ft_rate_team,
                        ft_rate_opp,
                        abs(ft_rate_team - ft_rate_opp) >= 0.10,
                        False,
                    ),
                    (
                        "Overall Points Per Shot (PPS)",
                        f"{pps_team:.2f}",
                        f"{pps_opp:.2f}",
                        pps_team,
                        pps_opp,
                        abs(pps_team - pps_opp) >= 0.15,
                        False,
                    ),
                    (
                        "Paint Touch PPS",
                        f"{paint_pps_team:.2f} ({team_paint_fga} tirs)",
                        f"{paint_pps_opp:.2f} ({opp_paint_fga} tirs)",
                        paint_pps_team,
                        paint_pps_opp,
                        abs(paint_pps_team - paint_pps_opp) >= 0.15,
                        False,
                    ),
                    (
                        "Non-Paint Touch PPS",
                        f"{nopaint_pps_team:.2f} ({team_nopaint_fga} tirs)",
                        f"{nopaint_pps_opp:.2f} ({opp_nopaint_fga} tirs)",
                        nopaint_pps_team,
                        nopaint_pps_opp,
                        abs(nopaint_pps_team - nopaint_pps_opp) >= 0.15,
                        False,
                    ),
                ]

                count_metrics = [
                    "Total Paint Touches",
                    "Total Fastbreaks",
                    "2nd Chance Points",
                    "Defensive Kills (3 Stops)",
                    "Stocks (Steals + Blocks)",
                ]
                for cm in count_metrics:
                    if cm in adv_sums_dict_t:
                        st_val = float(adv_sums_dict_t[cm] / num_partits)
                        so_val = float(adv_sums_dict_o[cm] / num_partits)
                        is_out = abs(st_val - so_val) >= 1.5
                        low_better = "fouls" in cm.lower()
                        table_rows_data.append(
                            (
                                cm,
                                f"{st_val:.1f} (Tot: {int(adv_sums_dict_t[cm])})",
                                f"{so_val:.1f} (Tot: {int(adv_sums_dict_o[cm])})",
                                st_val,
                                so_val,
                                is_out,
                                low_better,
                            )
                        )

                rows_season = []
                styles_season = []

                for (
                    m_raw,
                    tv_str,
                    ov_str,
                    tv_num,
                    ov_num,
                    is_outlier,
                    lower_is_better,
                ) in table_rows_data:
                    m = METRIC_NAMES_CAT.get(m_raw, m_raw)

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

                    rows_season.append(
                        {"Mètrica": m, "UEM": tv_str, "Rivals": ov_str}
                    )
                    styles_season.append((uem_css, riv_css))

                df_display_s = pd.DataFrame(rows_season)

                def apply_text_styles_season(df):
                    css_df = pd.DataFrame("", index=df.index, columns=df.columns)
                    for i, (uem_css, riv_css) in enumerate(styles_season):
                        css_df.loc[i, "UEM"] = uem_css
                        css_df.loc[i, "Rivals"] = riv_css
                    return css_df

                table_height = (len(df_display_s) + 1) * 36 + 3
                st.dataframe(
                    df_display_s.style.apply(
                        apply_text_styles_season, axis=None
                    ),
                    height=table_height,
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No hi ha dades acumulades disponibles.")

    # -------------------------------------------------------------
    # 2. SECCIÓ BOX SCORE (TOTALS ACUMULATS)
    # -------------------------------------------------------------
    elif active_tab_season == "📋 Box Score":
        comp_box = box_df[box_df["game_id"].astype(str).isin(valid_ids)].copy()

        if not comp_box.empty and "Name" in comp_box.columns:
            box_mode_s = st.segmented_control(
                "Tipus d'Estadística Acumulada",
                options=[
                    "Tradicional (Estàndard)",
                    "Zones de Tir (Aro / Pintura / Mitja / 3P)",
                    "Context i Impacte (Transició / Segones Opcions / Faltes)",
                ],
                default="Tradicional (Estàndard)",
                key="box_mode_selector_season",
            )

            st.markdown("<br>", unsafe_allow_html=True)

            p_agg = (
                comp_box.groupby(["Player", "Name"])
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

            if box_mode_s == "Tradicional (Estàndard)":
                df_trad_s = pd.DataFrame()
                df_trad_s["#"] = p_agg["Player"]
                df_trad_s["Jugador"] = p_agg["Name"]
                df_trad_s["PJ"] = p_agg["PJ"].astype(int)
                df_trad_s["PTS"] = p_agg["PTS"].astype(int)
                df_trad_s["PPP"] = p_agg["PPP"].round(1)
                df_trad_s["T2 (A/I)"] = (
                    p_agg["T2M"].astype(str) + "/" + p_agg["T2A"].astype(str)
                )
                df_trad_s["%T2"] = (
                    (p_agg["T2M"] / p_agg["T2A"].replace(0, np.nan)) * 100
                ).fillna(0).apply(lambda x: f"{x:.0f}%")
                df_trad_s["T3 (A/I)"] = (
                    p_agg["T3M"].astype(str) + "/" + p_agg["T3A"].astype(str)
                )
                df_trad_s["%T3"] = (
                    (p_agg["T3M"] / p_agg["T3A"].replace(0, np.nan)) * 100
                ).fillna(0).apply(lambda x: f"{x:.0f}%")
                df_trad_s["TL (A/I)"] = (
                    p_agg["FTM"].astype(str) + "/" + p_agg["FTA"].astype(str)
                )
                df_trad_s["%TL"] = (
                    (p_agg["FTM"] / p_agg["FTA"].replace(0, np.nan)) * 100
                ).fillna(0).apply(lambda x: f"{x:.0f}%")
                df_trad_s["REB_O"] = p_agg["REB_O"].astype(int)
                df_trad_s["REB_D"] = p_agg["REB_D"].astype(int)
                df_trad_s["REB_T"] = df_trad_s["REB_O"] + df_trad_s["REB_D"]
                df_trad_s["AST"] = p_agg["AST"].astype(int)
                df_trad_s["REC"] = p_agg["REC"].astype(int)
                df_trad_s["TAP"] = p_agg["TAP"].astype(int)
                df_trad_s["PER"] = p_agg["PER"].astype(int)
                df_trad_s["FAL"] = p_agg["FAL"].astype(int)
                df_trad_s["F_REC"] = p_agg["F_REC"].astype(int)

                df_sorted = df_trad_s.sort_values(
                    by="PTS", ascending=False
                ).reset_index(drop=True)
                st.dataframe(
                    style_boxscore(df_sorted, "Tradicional"),
                    use_container_width=True,
                    hide_index=True,
                )

            elif box_mode_s == "Zones de Tir (Aro / Pintura / Mitja / 3P)":
                df_shots_s = pd.DataFrame()
                df_shots_s["#"] = p_agg["Player"]
                df_shots_s["Jugador"] = p_agg["Name"]
                df_shots_s["PJ"] = p_agg["PJ"].astype(int)
                df_shots_s["PTS"] = p_agg["PTS"].astype(int)
                df_shots_s["Aro (Rim)"] = (
                    p_agg["RIM_M"].astype(str) + "/" + p_agg["RIM_A"].astype(str)
                )
                df_shots_s["Pintura (Paint)"] = (
                    p_agg["PAINT_M"].astype(str)
                    + "/"
                    + p_agg["PAINT_A"].astype(str)
                )
                df_shots_s["Mitja Distància (MR)"] = (
                    p_agg["MR_M"].astype(str) + "/" + p_agg["MR_A"].astype(str)
                )
                df_shots_s["Triple Cantonada (C3)"] = (
                    p_agg["C3_M"].astype(str) + "/" + p_agg["C3_A"].astype(str)
                )
                df_shots_s["Triple Frontal (ATB3)"] = (
                    p_agg["ATB3_M"].astype(str)
                    + "/"
                    + p_agg["ATB3_A"].astype(str)
                )

                df_sorted = df_shots_s.sort_values(
                    by="PTS", ascending=False
                ).reset_index(drop=True)
                st.dataframe(
                    style_boxscore(df_sorted, "Zones"),
                    use_container_width=True,
                    hide_index=True,
                )

            elif (
                box_mode_s
                == "Context i Impacte (Transició / Segones Opcions / Faltes)"
            ):
                df_ctx_s = pd.DataFrame()
                df_ctx_s["#"] = p_agg["Player"]
                df_ctx_s["Jugador"] = p_agg["Name"]
                df_ctx_s["PJ"] = p_agg["PJ"].astype(int)
                df_ctx_s["Transició"] = (
                    p_agg["FBM"].astype(str)
                    + "/"
                    + (p_agg["FBM"] + p_agg["FBMiss"]).astype(str)
                )
                df_ctx_s["2a Oportunitat"] = (
                    p_agg["Ch2M"].astype(str)
                    + "/"
                    + (p_agg["Ch2M"] + p_agg["Ch2Miss"]).astype(str)
                )
                df_ctx_s["Stocks (Rec+Tap)"] = p_agg["STOCKS"].astype(int)
                df_ctx_s["Defleccions"] = p_agg["DEFL"].astype(int)
                df_ctx_s["Pèrdues"] = p_agg["PER"].astype(int)
                df_ctx_s["Faltes Com."] = p_agg["FAL"].astype(int)
                df_ctx_s["Faltes Reb."] = p_agg["F_REC"].astype(int)

                df_sorted = df_ctx_s.sort_values(
                    by="Stocks (Rec+Tap)", ascending=False
                ).reset_index(drop=True)
                st.dataframe(
                    style_boxscore(df_sorted, "Context"),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No hi ha dades acumulades de Boxscore disponibles.")