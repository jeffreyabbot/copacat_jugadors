# src/data_loader.py
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CACHE_DIR = BASE_DIR / "data" / "cache"


def clean_num_str(series: pd.Series) -> pd.Series:
    """Neteja decimals europeus ('1,00'), percentatges i espais."""
    return (
        series.astype(str)
        .str.strip()
        .str.rstrip("%")
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def detect_file_sep_and_header(file_path: Path, keyword: str):
    """Detecta automàticament el separador (, ; o tab) i la línia de capçalera."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return 0, ","

    header_idx = 0
    for idx, line in enumerate(lines):
        cleaned = line.strip().strip('"').lower()
        if keyword in cleaned:
            header_idx = idx
            break

    target_line = lines[header_idx] if header_idx < len(lines) else ""
    if "\t" in target_line:
        sep = "\t"
    elif ";" in target_line:
        sep = ";"
    else:
        sep = ","

    return header_idx, sep


def parse_metadata(folder_path: Path, score_team=None, score_opp=None) -> dict:
    """Genera les metadades prioritzant metadata.json, la carpeta pare i els noms dels equips."""
    folder_name = folder_path.name

    # 1. Competició des de la carpeta pare
    if folder_path.parent.resolve() != RAW_DATA_DIR.resolve():
        comp_str = folder_path.parent.name.replace("_", " ")
    else:
        comp_str = "Competicio"

    date_str = ""
    our_team = ""
    opponent = ""
    round_str = ""

    # 2. Si existeix metadata.json a la carpeta, LLEGIR-LO PRIMER
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
            print(f"Avis llegint metadata.json a {folder_name}: {e}")

    # 3. Si encara no tenim data ni equips, llegir de la primera línia dels CSVs
    if not date_str or not our_team or not opponent:
        csv_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".csv"]
        for csv_file in csv_files:
            try:
                with open(csv_file, "r", encoding="latin-1", errors="ignore") as f:
                    first_line = f.readline().strip()
                if "game date:" in first_line.lower():
                    delim = ";" if ";" in first_line else ","
                    for item in first_line.split(delim):
                        il = item.lower()
                        if "game date:" in il and not date_str:
                            date_str = item.split(":", 1)[1].strip()
                        elif "our team:" in il and not our_team:
                            val = item.split(":", 1)[1].strip()
                            if val.lower() not in ["our team", "team"]:
                                our_team = val
                        elif "opponent:" in il and not opponent:
                            val = item.split(":", 1)[1].strip()
                            if val.lower() not in ["opponent", "rival"]:
                                opponent = val
            except Exception:
                continue

    # 4. Fallback intel·ligent a partir del nom de la carpeta (ex: 2026_09_26_CB_J1_Argentona_FC_Martinenc)
    fn_clean = folder_name.replace("-", "_")
    parts = fn_clean.split("_")
    
    if not date_str:
        if len(parts) >= 3 and len(parts[0]) == 4 and parts[0].isdigit():
            date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"
        else:
            date_str = parts[0]

    # Detectar equips pel nom de la carpeta si venien com a "Our Team"
    if not our_team or our_team.lower() in ["our team", "team"]:
        if "argentona" in folder_name.lower():
            our_team = "CB Argentona"
        else:
            our_team = "CB Argentona"

    if not opponent or opponent.lower() in ["opponent", "rival"]:
        fn_l = folder_name.lower()
        if "martinenc" in fn_l:
            opponent = "FC Martinenc"
        else:
            opponent = "Rival"

    # 5. Detectar Jornada (ex: J1, J01)
    if not round_str:
        match_j = re.search(r"[_\b]J(\d+)[_\b]", folder_name.upper()) or re.search(r"\bJ(\d+)\b", folder_name.upper())
        if match_j:
            round_str = f"Jornada {int(match_j.group(1))}"

    # 6. Títol net final
    score_txt = f"{int(score_team)}-{int(score_opp)}" if score_team and score_opp else "vs"
    
    if round_str:
        readable_name = f"{date_str} - {our_team} {score_txt} {opponent} ({round_str})"
    else:
        readable_name = f"{date_str} - {our_team} {score_txt} {opponent}"

    return {
        "game_id": folder_name,
        "date": date_str,
        "competition": comp_str,
        "our_team": our_team,
        "opponent": opponent,
        "score_team": score_team,
        "score_opp": score_opp,
        "round": round_str,
        "name": readable_name,
    }

def parse_advanced_metrics(file_path: Path, game_id: str) -> pd.DataFrame:
    try:
        header_idx, sep = detect_file_sep_and_header(file_path, "metric")
        df = pd.read_csv(file_path, skiprows=header_idx, sep=sep)
        df = df.dropna(how="all").copy()

        if len(df.columns) < 3:
            return pd.DataFrame()

        df = df.iloc[:, :3]
        df.columns = ["Metric", "Team_Value", "Opp_Value"]
        df = df[df["Metric"].astype(str).str.lower() != "metric"].copy()

        df["Metric"] = df["Metric"].astype(str).str.strip()
        df["Team_Value"] = clean_num_str(df["Team_Value"]).fillna(0.0)
        df["Opp_Value"] = clean_num_str(df["Opp_Value"]).fillna(0.0)
        df["game_id"] = str(game_id)
        return df
    except Exception as e:
        print(f"Avis Advanced a {file_path.name}: {e}")
        return pd.DataFrame()


def parse_boxscore(file_path: Path, game_id: str) -> pd.DataFrame:
    try:
        header_idx, sep = detect_file_sep_and_header(file_path, "player")
        df = pd.read_csv(file_path, skiprows=header_idx, sep=sep)
        df = df.dropna(how="all").copy()

        # FILTRAR AUTOMÀTICAMENT FILES DE TOTALS I RIVALS DEL BOXSCORE
        if "Player" in df.columns:
            p_upper = df["Player"].astype(str).str.upper().str.strip()
            # Ignorem files que siguin TOTALS, TOTAL, OPPONENT, RIVAL o buides
            is_summary = p_upper.isin(
                ["TOTALS", "TOTAL", "OPPONENT", "RIVAL", "TEAM TOTAL", "EQUIP"]
            )
            df = df[~is_summary].copy()

        if "Name" in df.columns:
            n_upper = df["Name"].astype(str).str.upper().str.strip()
            is_team_name = n_upper.isin(
                [
                    "TOTALS",
                    "TOTAL",
                    "OPPONENT",
                    "RIVAL",
                    "CB ARGENTONA",
                    "MARISTES ADEMAR",
                ]
            )
            df = df[~is_team_name].copy()

        text_cols = ["Player", "Name", "MIN"]
        for col in df.columns:
            if col in text_cols:
                df[col] = df[col].astype(str).str.strip()
            else:
                df[col] = clean_num_str(df[col]).fillna(0.0)

        df["game_id"] = str(game_id)
        return df
    except Exception as e:
        print(f"Avis Boxscore a {file_path.name}: {e}")
        return pd.DataFrame()

def parse_lineups(file_path: Path, game_id: str) -> pd.DataFrame:
    try:
        header_idx, sep = detect_file_sep_and_header(file_path, "competition")
        if header_idx == 0:
            h_alt, s_alt = detect_file_sep_and_header(file_path, "lineup")
            if h_alt > 0:
                header_idx, sep = h_alt, s_alt

        df = pd.read_csv(file_path, skiprows=header_idx, sep=sep)
        df = df.dropna(how="all").copy()

        text_cols = [
            "Competition",
            "Week",
            "Team",
            "Win",
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "Lineup",
            "Rival",
            "game_id",
            "MIN",
            "Min",
            "min",
            "Duration",
            "Duration(s)",
            "Time",
            "Seconds",
        ]
        for col in df.columns:
            if col in text_cols:
                df[col] = df[col].astype(str).str.strip()
            else:
                df[col] = clean_num_str(df[col]).fillna(0.0)

        df["game_id"] = str(game_id)
        return df
    except Exception as e:
        print(f"Avis Lineups a {file_path.name}: {e}")
        return pd.DataFrame()


def parse_pbp(file_path: Path, game_id: str) -> pd.DataFrame:
    try:
        header_idx, sep = detect_file_sep_and_header(file_path, "period")
        df = pd.read_csv(file_path, skiprows=header_idx, sep=sep)
        df = df.dropna(how="all").copy()

        for col in df.columns:
            if col in ["Duration(s)"]:
                df[col] = clean_num_str(df[col]).fillna(0.0)
            else:
                df[col] = df[col].astype(str).str.strip()

        df["game_id"] = str(game_id)
        return df
    except Exception as e:
        print(f"Avis PBP a {file_path.name}: {e}")
        return pd.DataFrame()


def rebuild_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    games_meta = []
    adv_list = []
    box_list = []
    lineup_list = []
    pbp_list = []

    all_csvs = [
        f
        for f in RAW_DATA_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() == ".csv"
    ]
    game_folders = sorted(list({csv_file.parent.resolve() for csv_file in all_csvs}))

    print(
        f"🔍 S'han trobat {len(game_folders)} carpetes de partits a data/raw/"
    )

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

        files = sorted(
            [
                f
                for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() == ".csv"
            ]
        )

        # 1. Primer llegim Advanced Metrics per extreure el marcador oficial
        for f in files:
            name = f.name.lower()
            if (
                ("advanced" in name or "metric" in name)
                and not adv_done
                and "box" not in name
            ):
                df = parse_advanced_metrics(f, game_id)
                if not df.empty:
                    adv_list.append(df)
                    adv_done = True
                    found = True
                    # Extreure marcador
                    sc_row = df[
                        df["Metric"]
                        .astype(str)
                        .str.contains(
                            "Game Score|Score|Marcador", case=False, na=False
                        )
                    ]
                    if not sc_row.empty:
                        score_t = float(sc_row["Team_Value"].values[0])
                        score_o = float(sc_row["Opp_Value"].values[0])

        # 2. Generar metadades amb el marcador real
        meta = parse_metadata(folder, score_t, score_o)

        # 3. Llegir la resta de fitxers (Boxscore, Lineups, PBP)
        for f in files:
            name = f.name.lower()
            if (
                ("boxscore" in name or "box" in name or "player" in name)
                and not box_done
                and "line" not in name
            ):
                df = parse_boxscore(f, game_id)
                if not df.empty:
                    box_list.append(df)
                    box_done = True
                    found = True
            elif (
                ("lineup" in name or "5man" in name or "line" in name)
                and not line_done
                and "box" not in name
            ):
                df = parse_lineups(f, game_id)
                if not df.empty:
                    lineup_list.append(df)
                    line_done = True
                    found = True
            elif (
                ("playbyplay" in name or "pbp" in name or "jugada" in name)
                and not pbp_done
                and "box" not in name
            ):
                df = parse_pbp(f, game_id)
                if not df.empty:
                    pbp_list.append(df)
                    pbp_done = True
                    found = True

        if found:
            games_meta.append(meta)
            print(f"  -> Partit registrat: {meta['name']}")

    # 4. Generar DataFrames nets
    df_games = (
        pd.DataFrame(games_meta).drop_duplicates(subset=["game_id"])
        if games_meta
        else pd.DataFrame()
    )
    df_box = (
        pd.concat(box_list, ignore_index=True).drop_duplicates(
            subset=["game_id", "Player"]
        )
        if box_list
        else pd.DataFrame()
    )
    df_lineup = (
        pd.concat(lineup_list, ignore_index=True).drop_duplicates(
            subset=["game_id", "Lineup"]
        )
        if lineup_list
        else pd.DataFrame()
    )
    df_adv = (
        pd.concat(adv_list, ignore_index=True).drop_duplicates(
            subset=["game_id", "Metric"]
        )
        if adv_list
        else pd.DataFrame()
    )
    df_pbp = (
        pd.concat(pbp_list, ignore_index=True).drop_duplicates()
        if pbp_list
        else pd.DataFrame()
    )

    df_games.to_parquet(CACHE_DIR / "games_index.parquet", index=False)
    df_box.to_parquet(CACHE_DIR / "all_boxscores.parquet", index=False)
    df_lineup.to_parquet(CACHE_DIR / "all_lineups.parquet", index=False)
    df_adv.to_parquet(CACHE_DIR / "all_advanced.parquet", index=False)
    df_pbp.to_parquet(CACHE_DIR / "all_pbp.parquet", index=False)

    print(
        f"✅ Memòria cau neta generada correctament: {len(games_meta)} partits."
    )
    return df_games, df_box, df_lineup, df_adv, df_pbp

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    games_meta = []
    adv_list = []
    box_list = []
    lineup_list = []
    pbp_list = []

    # 1. Cerca única de carpetes
    all_csvs = [
        f
        for f in RAW_DATA_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() == ".csv"
    ]
    game_folders = sorted(list({csv_file.parent.resolve() for csv_file in all_csvs}))

    print(
        f"🔍 S'han trobat {len(game_folders)} carpetes de partits a data/raw/"
    )

    for folder in game_folders:
        folder = Path(folder)
        meta = parse_metadata(folder)
        game_id = meta["game_id"]

        adv_done = False
        box_done = False
        line_done = False
        pbp_done = False
        found = False

        files = sorted(
            [
                f
                for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() == ".csv"
            ]
        )

        for f in files:
            name = f.name.lower()
            if (
                ("advanced" in name or "metric" in name)
                and not adv_done
                and "box" not in name
            ):
                df = parse_advanced_metrics(f, game_id)
                if not df.empty:
                    adv_list.append(df)
                    adv_done = True
                    found = True
            elif (
                ("boxscore" in name or "box" in name or "player" in name)
                and not box_done
                and "line" not in name
            ):
                df = parse_boxscore(f, game_id)
                if not df.empty:
                    box_list.append(df)
                    box_done = True
                    found = True
            elif (
                ("lineup" in name or "5man" in name or "line" in name)
                and not line_done
                and "box" not in name
            ):
                df = parse_lineups(f, game_id)
                if not df.empty:
                    lineup_list.append(df)
                    line_done = True
                    found = True
            elif (
                ("playbyplay" in name or "pbp" in name or "jugada" in name)
                and not pbp_done
                and "box" not in name
            ):
                df = parse_pbp(f, game_id)
                if not df.empty:
                    pbp_list.append(df)
                    pbp_done = True
                    found = True

        if found:
            games_meta.append(meta)
            print(f"  -> Partit processat: {meta['name']}")

    # 2. Generar DataFrames únics nets
    df_games = (
        pd.DataFrame(games_meta).drop_duplicates(subset=["game_id"])
        if games_meta
        else pd.DataFrame()
    )
    df_box = (
        pd.concat(box_list, ignore_index=True).drop_duplicates(
            subset=["game_id", "Player"]
        )
        if box_list
        else pd.DataFrame()
    )
    df_lineup = (
        pd.concat(lineup_list, ignore_index=True).drop_duplicates(
            subset=["game_id", "Lineup"]
        )
        if lineup_list
        else pd.DataFrame()
    )
    df_adv = (
        pd.concat(adv_list, ignore_index=True).drop_duplicates(
            subset=["game_id", "Metric"]
        )
        if adv_list
        else pd.DataFrame()
    )
    df_pbp = (
        pd.concat(pbp_list, ignore_index=True).drop_duplicates()
        if pbp_list
        else pd.DataFrame()
    )

    # 3. Guardar en Parquet
    df_games.to_parquet(CACHE_DIR / "games_index.parquet", index=False)
    df_box.to_parquet(CACHE_DIR / "all_boxscores.parquet", index=False)
    df_lineup.to_parquet(CACHE_DIR / "all_lineups.parquet", index=False)
    df_adv.to_parquet(CACHE_DIR / "all_advanced.parquet", index=False)
    df_pbp.to_parquet(CACHE_DIR / "all_pbp.parquet", index=False)

    print(
        f"✅ Memòria cau neta generada correctament: {len(games_meta)} partits."
    )
    return df_games, df_box, df_lineup, df_adv, df_pbp