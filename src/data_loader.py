# src/data_loader.py
import json
import os
from pathlib import Path
import re
import csv
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CACHE_DIR = BASE_DIR / "data" / "cache"

MESOS_CAT = {
    "01": "Gener", "02": "Febrer", "03": "Març", "04": "Abril",
    "05": "Maig", "06": "Juny", "07": "Juliol", "08": "Agost",
    "09": "Setembre", "10": "Octubre", "11": "Novembre", "12": "Desembre"
}


def clean_num_str(series: pd.Series) -> pd.Series:
    """Neteja decimals europeus ('1,00'), percentatges i espais."""
    return (
        series.astype(str)
        .str.strip()
        .str.rstrip("%")
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def safe_read_csv(file_path: Path, keyword: str) -> pd.DataFrame:
    """Lector antibales per NBN23: corregeix accents i línies amb comes de més."""
    # 1. Buscar la línia de capçalera i el separador
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return pd.DataFrame()
        
    sep = ","
    header_idx = -1
    for i, line in enumerate(lines):
        cleaned = line.strip().strip('"').lower()
        if keyword in cleaned:
            header_idx = i
            if "\t" in line: sep = "\t"
            elif ";" in line: sep = ";"
            break
            
    if header_idx == -1:
        return pd.DataFrame()

    # 2. Llegir dades respectant accents (utf-8-sig) o latin-1 si falla
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=sep)
            data = list(reader)
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            reader = csv.reader(f, delimiter=sep)
            data = list(reader)
            
    if not data or header_idx >= len(data):
        return pd.DataFrame()
        
    data = data[header_idx:]
    header = [str(x).strip() for x in data[0]]
    # Prevenir columnes sense nom
    header = [h if h else f"Unnamed_{i}" for i, h in enumerate(header)]
    
    # 3. Retallar comes sobrants de cada línia en lloc de descartar-la
    clean_rows = []
    for r in data[1:]:
        if not any(r): continue  # Omet files 100% buides
        if len(r) > len(header):
            clean_rows.append(r[:len(header)])  # Retalla les comes de més
        else:
            clean_rows.append(r + [''] * (len(header) - len(r)))  # Omple si falten
            
    return pd.DataFrame(clean_rows, columns=header)


def parse_metadata(folder_path: Path, score_team=None, score_opp=None) -> dict:
    """Genera les metadades incloent data, mes i pista per als filtres."""
    folder_name = folder_path.name

    if folder_path.parent.resolve() != RAW_DATA_DIR.resolve():
        comp_str = folder_path.parent.name.replace("_", " ")
    else:
        comp_str = "Copa Catalunya"

    date_str = ""
    our_team = "CB Argentona"
    opponent = ""
    round_str = ""

    # 1. Llegir metadata.json si existeix
    meta_json_file = folder_path / "metadata.json"
    if meta_json_file.exists():
        try:
            with open(meta_json_file, "r", encoding="utf-8") as f:
                j_data = json.load(f)
                comp_str = j_data.get("competition", comp_str)
                date_str = j_data.get("date", date_str)
                our_team = j_data.get("our_team", our_team)
                opponent = j_data.get("opponent", opponent)
                round_str = j_data.get("round", round_str)
        except Exception as e:
            print(f"Avís llegint metadata.json a {folder_name}: {e}")

    # 2. Llegir capçalera de fitxers CSV si falten dades
    if not date_str or not opponent:
        csv_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".csv"]
        for csv_file in csv_files:
            try:
                with open(csv_file, "r", encoding="utf-8-sig", errors="ignore") as f:
                    first_line = f.readline().strip()
                if "game date:" in first_line.lower():
                    delim = ";" if ";" in first_line else ","
                    for item in first_line.split(delim):
                        il = item.lower()
                        if "game date:" in il and not date_str:
                            date_str = item.split(":", 1)[1].strip()
                        elif "opponent:" in il and not opponent:
                            val = item.split(":", 1)[1].strip()
                            if val.lower() not in ["opponent", "rival"]:
                                opponent = val
            except Exception:
                continue

    # 3. Fallback intel·ligent del nom de carpeta
    fn_clean = folder_name.replace("-", "_")
    parts = fn_clean.split("_")

    if not date_str:
        if len(parts) >= 3 and len(parts[0]) == 4 and parts[0].isdigit():
            date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"
        else:
            date_str = parts[0]

    # Determinació de Pista (Casa / Fora)
    pista = "Casa"
    fn_low = folder_name.lower()
    if "fora" in fn_low or "away" in fn_low:
        pista = "Fora"

    # Determinació del Mes
    month_str = "Altres"
    if date_str:
        m_match = re.search(r"[-_/](\d{2})[-_/]", date_str) or re.search(r"^\d{4}[-_/](\d{2})", date_str)
        if m_match and m_match.group(1) in MESOS_CAT:
            month_str = MESOS_CAT[m_match.group(1)]

    if not opponent or opponent.lower() in ["opponent", "rival"]:
        parts_clean = [p for p in parts if p.lower() not in ["cb", "argentona", "j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9", "j10"]]
        opponent = " ".join(parts_clean[3:]) if len(parts_clean) > 3 else "Rival"

    if not round_str:
        match_j = re.search(r"[_\b]J(\d+)[_\b]", folder_name.upper()) or re.search(r"\bJ(\d+)\b", folder_name.upper())
        if match_j:
            round_str = f"Jornada {int(match_j.group(1))}"

    score_txt = f"{int(score_team)}-{int(score_opp)}" if (score_team is not None and score_opp is not None) else "vs"

    if round_str:
        readable_name = f"{date_str} - {our_team} {score_txt} {opponent} ({round_str})"
    else:
        readable_name = f"{date_str} - {our_team} {score_txt} {opponent}"

    return {
        "game_id": folder_name,
        "date": date_str,
        "month": month_str,
        "pista": pista,
        "competition": comp_str,
        "our_team": our_team,
        "opponent": opponent,
        "score_team": score_team,
        "score_opp": score_opp,
        "round": round_str,
        "name": readable_name,
    }


def parse_advanced_metrics(file_path: Path, game_id: str) -> pd.DataFrame:
    df = safe_read_csv(file_path, "metric")
    if df.empty or len(df.columns) < 3:
        return pd.DataFrame()

    df = df.iloc[:, :3]
    df.columns = ["Metric", "Team_Value", "Opp_Value"]
    df = df[df["Metric"].astype(str).str.lower() != "metric"].copy()

    df["Metric"] = df["Metric"].astype(str).str.strip()
    df["Team_Value"] = clean_num_str(df["Team_Value"]).fillna(0.0)
    df["Opp_Value"] = clean_num_str(df["Opp_Value"]).fillna(0.0)
    df["game_id"] = str(game_id)
    return df


def parse_boxscore(file_path: Path, game_id: str) -> pd.DataFrame:
    df = safe_read_csv(file_path, "player")
    if df.empty:
        return pd.DataFrame()

    if "Name" in df.columns:
        n_upper = df["Name"].astype(str).str.upper().str.strip()
        is_team_name = n_upper.isin([
            "TOTALS", "TOTAL", "OPPONENT", "RIVAL", "CB ARGENTONA", "EQUIP", "TEAM"
        ])
        df = df[~is_team_name].copy()

    text_cols = ["Player", "Name", "MIN"]
    for col in df.columns:
        if col in text_cols:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = clean_num_str(df[col]).fillna(0.0)

    df["game_id"] = str(game_id)
    return df


def parse_lineups(file_path: Path, game_id: str) -> pd.DataFrame:
    df = safe_read_csv(file_path, "competition")
    if df.empty:
        df = safe_read_csv(file_path, "lineup")
    
    if df.empty:
        return pd.DataFrame()

    text_cols = [
        "Competition", "Week", "Team", "Win", "P1", "P2", "P3", "P4", "P5",
        "Lineup", "Rival", "game_id", "MIN", "Min", "min", "Duration",
        "Duration(s)", "Time", "Seconds",
    ]
    for col in df.columns:
        if col in text_cols:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = clean_num_str(df[col]).fillna(0.0)

    df["game_id"] = str(game_id)
    return df


def parse_pbp(file_path: Path, game_id: str) -> pd.DataFrame:
    df = safe_read_csv(file_path, "period")
    if df.empty:
        return pd.DataFrame()

    for col in df.columns:
        if col in ["Duration(s)"]:
            df[col] = clean_num_str(df[col]).fillna(0.0)
        else:
            df[col] = df[col].astype(str).str.strip()

    df["game_id"] = str(game_id)
    return df


def rebuild_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    games_meta = []
    adv_list = []
    box_list = []
    lineup_list = []
    pbp_list = []

    all_csvs = [
        f for f in RAW_DATA_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() == ".csv"
    ]
    game_folders = sorted(list({csv_file.parent.resolve() for csv_file in all_csvs}))

    print(f"🔍 S'han trobat {len(game_folders)} carpetes de partits a data/raw/")

    for folder in game_folders:
        folder = Path(folder)
        game_id = folder.name

        adv_done = False
        box_done = False
        line_done = False
        pbp_done = False
        found = False

        score_t = None
        score_o = None

        files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".csv"])

        # 1. Primer pas: cercar marcadors a Advanced Metrics
        for f in files:
            name = f.name.lower()
            if ("advanced" in name or "metric" in name) and not adv_done and "box" not in name:
                df = parse_advanced_metrics(f, game_id)
                if not df.empty:
                    adv_list.append(df)
                    adv_done = True
                    found = True
                    sc_row = df[df["Metric"].astype(str).str.contains("Game Score|Score|Marcador", case=False, na=False)]
                    if not sc_row.empty:
                        score_t = float(sc_row["Team_Value"].values[0])
                        score_o = float(sc_row["Opp_Value"].values[0])

        # 2. Metadades
        meta = parse_metadata(folder, score_t, score_o)

        # 3. Altres fitxers
        for f in files:
            name = f.name.lower()
            if ("boxscore" in name or "box" in name or "player" in name) and not box_done and "line" not in name:
                df = parse_boxscore(f, game_id)
                if not df.empty:
                    box_list.append(df)
                    box_done = True
                    found = True
            elif ("lineup" in name or "5man" in name or "line" in name) and not line_done and "box" not in name:
                df = parse_lineups(f, game_id)
                if not df.empty:
                    lineup_list.append(df)
                    line_done = True
                    found = True
            elif ("playbyplay" in name or "pbp" in name or "jugada" in name) and not pbp_done and "box" not in name:
                df = parse_pbp(f, game_id)
                if not df.empty:
                    pbp_list.append(df)
                    pbp_done = True
                    found = True

        if found:
            games_meta.append(meta)
            print(f"  -> Partit processat: {meta['name']}")

    df_games = pd.DataFrame(games_meta).drop_duplicates(subset=["game_id"]) if games_meta else pd.DataFrame()
    df_box = pd.concat(box_list, ignore_index=True).drop_duplicates(subset=["game_id", "Player"]) if box_list else pd.DataFrame()
    df_lineup = pd.concat(lineup_list, ignore_index=True).drop_duplicates(subset=["game_id", "Lineup"]) if lineup_list else pd.DataFrame()
    df_adv = pd.concat(adv_list, ignore_index=True).drop_duplicates(subset=["game_id", "Metric"]) if adv_list else pd.DataFrame()
    df_pbp = pd.concat(pbp_list, ignore_index=True).drop_duplicates() if pbp_list else pd.DataFrame()

    df_games.to_parquet(CACHE_DIR / "games_index.parquet", index=False)
    df_box.to_parquet(CACHE_DIR / "all_boxscores.parquet", index=False)
    df_lineup.to_parquet(CACHE_DIR / "all_lineups.parquet", index=False)
    df_adv.to_parquet(CACHE_DIR / "all_advanced.parquet", index=False)
    df_pbp.to_parquet(CACHE_DIR / "all_pbp.parquet", index=False)

    print(f"✅ Memòria cau neta generada correctament: {len(games_meta)} partits.")
    return df_games, df_box, df_lineup, df_adv, df_pbp