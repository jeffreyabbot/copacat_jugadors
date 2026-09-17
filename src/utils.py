# src/utils.py
import re
import pandas as pd
import numpy as np



def get_defensive_actions(box_df, pbp_df=None, game_id=None):
    """Calcula les accions defensives i detecta quants taps han generat canvi de possessió."""
    if box_df is None or box_df.empty:
        return pd.DataFrame()

    if game_id:
        b_df = box_df[box_df["game_id"].astype(str) == str(game_id)].copy()
        p_df = (
            pbp_df[pbp_df["game_id"].astype(str) == str(game_id)].copy()
            if pbp_df is not None and not pbp_df.empty
            else pd.DataFrame()
        )
    else:
        b_df = box_df.copy()
        p_df = (
            pbp_df.copy()
            if pbp_df is not None and not pbp_df.empty
            else pd.DataFrame()
        )

    # 1. Analitzar al Play-by-Play quins taps acaben en canvi de possessió
    blocks_with_change = {}
    if not p_df.empty:
        p_df = p_df.reset_index(drop=True)
        n_rows = len(p_df)
        for i in range(n_rows):
            row = p_df.iloc[i]
            act = str(row.get("Action", "")).lower()
            if "block" in act or "tap" in act:
                player_raw = str(row.get("Player", "")).replace("#", "").strip()
                # Comprovem les següents 2 jugades per veure si recuperem la pilota
                recovered = False
                for offset in [1, 2]:
                    if i + offset < n_rows:
                        next_row = p_df.iloc[i + offset]
                        next_act = str(next_row.get("Action", "")).lower()
                        next_state = str(
                            next_row.get("Possession State", "")
                        ).lower()

                        if (
                            "def rebound" in next_act
                            or "steal" in next_act
                            or "offense" in next_state
                        ):
                            recovered = True
                            break
                if recovered:
                    blocks_with_change[player_raw] = (
                        blocks_with_change.get(player_raw, 0) + 1
                    )

    # 2. Generar taula individual o acumulada
    if game_id:
        df_def = pd.DataFrame()
        df_def["#"] = b_df["Player"]
        df_def["Jugador"] = b_df["Name"]
        df_def["MIN"] = b_df["MIN"].astype(str).str.strip()
        df_def["Defleccions"] = (
            pd.to_numeric(b_df["Deflections"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        df_def["Robatoris (REC)"] = (
            pd.to_numeric(b_df["Steals"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        df_def["Taps Totals (TAP)"] = (
            pd.to_numeric(b_df["Blocks"], errors="coerce").fillna(0).astype(int)
        )

        def get_rec(num_str):
            p = str(num_str).replace("#", "").strip()
            return blocks_with_change.get(p, 0)

        df_def["Taps amb Canvi Poss."] = df_def["#"].apply(get_rec)
        df_def["% Taps Recuperats"] = (
            (
                df_def["Taps amb Canvi Poss."]
                / df_def["Taps Totals (TAP)"].replace(0, np.nan)
            )
            * 100
        ).fillna(0).apply(lambda x: f"{x:.0f}%" if x > 0 else "0%")
        df_def["Stocks (REC+TAP)"] = (
            df_def["Robatoris (REC)"] + df_def["Taps Totals (TAP)"]
        )
        df_def["Impacte Defensiu (Total)"] = (
            df_def["Defleccions"] + df_def["Stocks (REC+TAP)"]
        )

        return df_def.sort_values(
            by="Impacte Defensiu (Total)", ascending=False
        ).reset_index(drop=True)
    else:
        agg_def = (
            b_df.groupby(["Player", "Name"])
            .agg(
                PJ=("game_id", "nunique"),
                Defleccions=("Deflections", "sum"),
                Robatoris=("Steals", "sum"),
                Taps=("Blocks", "sum"),
            )
            .reset_index()
        )

        df_def_s = pd.DataFrame()
        df_def_s["#"] = agg_def["Player"]
        df_def_s["Jugador"] = agg_def["Name"]
        df_def_s["PJ"] = agg_def["PJ"].astype(int)
        df_def_s["Defleccions"] = agg_def["Defleccions"].astype(int)
        df_def_s["Robatoris (REC)"] = agg_def["Robatoris"].astype(int)
        df_def_s["Taps Totals (TAP)"] = agg_def["Taps"].astype(int)

        def get_rec_s(num_str):
            p = str(num_str).replace("#", "").strip()
            return blocks_with_change.get(p, 0)

        df_def_s["Taps amb Canvi Poss."] = df_def_s["#"].apply(get_rec_s)
        df_def_s["% Taps Recuperats"] = (
            (
                df_def_s["Taps amb Canvi Poss."]
                / df_def_s["Taps Totals (TAP)"].replace(0, np.nan)
            )
            * 100
        ).fillna(0).apply(lambda x: f"{x:.0f}%" if x > 0 else "0%")
        df_def_s["Stocks (REC+TAP)"] = (
            df_def_s["Robatoris (REC)"] + df_def_s["Taps Totals (TAP)"]
        )
        df_def_s["Impacte Defensiu (Total)"] = (
            df_def_s["Defleccions"] + df_def_s["Stocks (REC+TAP)"]
        )

        return df_def_s.sort_values(
            by="Impacte Defensiu (Total)", ascending=False
        ).reset_index(drop=True)
# src/utils.py


def get_individual_four_factors(
    box_df, lineups_df=None, pbp_df=None, game_id=None
):
    """Calcula els 4 Factors Individuals de Dean Oliver + Usage Rate (USG%) + Plays."""
    if box_df is None or box_df.empty:
        return pd.DataFrame()

    def parse_m(v):
        try:
            s = str(v).strip()
            if ":" in s:
                parts = s.split(":")
                return float(parts[0]) + float(parts[1]) / 60.0
            return float(s)
        except Exception:
            return 0.0

    if game_id:
        # 1. PARTIT INDIVIDUAL
        b_df = box_df[box_df["game_id"].astype(str) == str(game_id)].copy()
        l_df = (
            lineups_df[lineups_df["game_id"].astype(str) == str(game_id)].copy()
            if lineups_df is not None and not lineups_df.empty
            else pd.DataFrame()
        )

        team_oreb_tot = float(
            pd.to_numeric(b_df["Off Reb"], errors="coerce").fillna(0).sum()
        )
        opp_dreb_tot = (
            float(
                pd.to_numeric(l_df["DREB_Agn"], errors="coerce").fillna(0).sum()
            )
            if not l_df.empty
            else 20.0
        )
        total_reb_opps = team_oreb_tot + opp_dreb_tot

        team_fga_tot = float(
            (
                pd.to_numeric(b_df["2PA"], errors="coerce").fillna(0)
                + pd.to_numeric(b_df["3PA"], errors="coerce").fillna(0)
            ).sum()
        )
        team_fta_tot = float(
            pd.to_numeric(b_df["FTA"], errors="coerce").fillna(0).sum()
        )
        team_tov_tot = float(
            pd.to_numeric(b_df["Turnovers"], errors="coerce").fillna(0).sum()
        )
        team_plays_tot = team_fga_tot + 0.44 * team_fta_tot + team_tov_tot
        game_min = 40.0

        rows = []
        for _, r in b_df.iterrows():
            p_num = str(r.get("Player", "")).strip()
            p_name = str(r.get("Name", "")).strip()

            # Ometre files d'equip o totals
            if (
                "team" in p_num.lower()
                or "team" in p_name.lower()
                or "total" in p_num.lower()
                or "total" in p_name.lower()
            ):
                continue

            min_str = str(r.get("MIN", "00:00")).strip()
            min_f = parse_m(min_str)

            t2m = int(pd.to_numeric(r.get("2PM", 0), errors="coerce") or 0)
            t2a = int(pd.to_numeric(r.get("2PA", 0), errors="coerce") or 0)
            t3m = int(pd.to_numeric(r.get("3PM", 0), errors="coerce") or 0)
            t3a = int(pd.to_numeric(r.get("3PA", 0), errors="coerce") or 0)
            ftm = int(pd.to_numeric(r.get("FTM", 0), errors="coerce") or 0)
            fta = int(pd.to_numeric(r.get("FTA", 0), errors="coerce") or 0)
            tov = int(
                pd.to_numeric(r.get("Turnovers", 0), errors="coerce") or 0
            )
            oreb = int(
                pd.to_numeric(r.get("Off Reb", 0), errors="coerce") or 0
            )

            fgm = t2m + t3m
            fga = t2a + t3a
            plays = fga + 0.44 * fta + tov

            efg = ((fgm + 0.5 * t3m) / fga * 100) if fga > 0 else 0.0
            tov_pct = (tov / plays * 100) if plays > 0 else 0.0

            if min_f >= 1.0 and total_reb_opps > 0:
                oreb_pct = (oreb * game_min) / (min_f * total_reb_opps) * 100.0
            else:
                oreb_pct = 0.0

            ft_rate = (ftm / fga) if fga > 0 else 0.0

            if min_f >= 1.0 and team_plays_tot > 0:
                usg = (plays * game_min) / (min_f * team_plays_tot) * 100.0
            else:
                usg = 0.0

            # Guardem com a NÚMEROS (floats) per permetre l'ordenació de Streamlit
            rows.append(
                {
                    "#": p_num,
                    "Jugador": p_name,
                    "MIN": min_str,
                    "USG%": round(usg, 1),
                    "1. eFG% (Tir)": round(efg, 1),
                    "2. TOV% (Pèrdues)": round(tov_pct, 1),
                    "3. OREB% (Rebot)": round(oreb_pct, 1),
                    "4. FT Rate (TL)": round(ft_rate, 2),
                    "Plays": round(plays, 1),
                }
            )

        df_res = pd.DataFrame(rows)
        return df_res.sort_values(by="Plays", ascending=False).reset_index(
            drop=True
        )

    else:
        # 2. TOTALS ACUMULATS
        # Filtrar files d'equip abans d'agrupar
        b_clean = box_df[
            ~box_df["Player"]
            .astype(str)
            .str.lower()
            .isin(["team", "totals", "total", "opponent"])
        ].copy()

        p_agg = (
            b_clean.groupby(["Player", "Name"])
            .agg(
                PJ=("game_id", "nunique"),
                T2M=("2PM", "sum"),
                T2A=("2PA", "sum"),
                T3M=("3PM", "sum"),
                T3A=("3PA", "sum"),
                FTM=("FTM", "sum"),
                FTA=("FTA", "sum"),
                OREB=("Off Reb", "sum"),
                TOV=("Turnovers", "sum"),
            )
            .reset_index()
        )

        num_partits = box_df["game_id"].nunique()
        tot_game_min = 40.0 * num_partits

        team_oreb_s = float(
            pd.to_numeric(b_clean["Off Reb"], errors="coerce").fillna(0).sum()
        )
        opp_dreb_s = (
            float(
                pd.to_numeric(lineups_df["DREB_Agn"], errors="coerce")
                .fillna(0)
                .sum()
            )
            if lineups_df is not None and not lineups_df.empty
            else (team_oreb_s * 1.2)
        )
        tot_reb_opps_s = team_oreb_s + opp_dreb_s

        team_fga_s = float(
            (
                pd.to_numeric(b_clean["2PA"], errors="coerce").fillna(0)
                + pd.to_numeric(b_clean["3PA"], errors="coerce").fillna(0)
            ).sum()
        )
        team_fta_s = float(
            pd.to_numeric(b_clean["FTA"], errors="coerce").fillna(0).sum()
        )
        team_tov_s = float(
            pd.to_numeric(b_clean["Turnovers"], errors="coerce").fillna(0).sum()
        )
        team_plays_s = team_fga_s + 0.44 * team_fta_s + team_tov_s

        p_min_map = {}
        for (p, n), g in b_clean.groupby(["Player", "Name"]):
            p_min_map[(p, n)] = g["MIN"].apply(parse_m).sum()

        rows_s = []
        for _, r in p_agg.iterrows():
            p_num = str(r["Player"]).strip()
            p_name = str(r["Name"]).strip()

            if (
                "team" in p_num.lower()
                or "team" in p_name.lower()
                or "total" in p_num.lower()
                or "total" in p_name.lower()
            ):
                continue

            pj = int(r["PJ"])
            t2m = int(r["T2M"])
            t2a = int(r["T2A"])
            t3m = int(r["T3M"])
            t3a = int(r["T3A"])
            ftm = int(r["FTM"])
            fta = int(r["FTA"])
            oreb = int(r["OREB"])
            tov = int(r["TOV"])

            min_tot = p_min_map.get((r["Player"], r["Name"]), 0.0)
            fgm = t2m + t3m
            fga = t2a + t3a
            plays = fga + 0.44 * fta + tov

            efg = ((fgm + 0.5 * t3m) / fga * 100) if fga > 0 else 0.0
            tov_pct = (tov / plays * 100) if plays > 0 else 0.0

            if min_tot >= 2.0 and tot_reb_opps_s > 0:
                oreb_pct = (
                    (oreb * tot_game_min) / (min_tot * tot_reb_opps_s) * 100.0
                )
            else:
                oreb_pct = 0.0

            ft_rate = (ftm / fga) if fga > 0 else 0.0

            if min_tot >= 2.0 and team_plays_s > 0:
                usg = (plays * tot_game_min) / (min_tot * team_plays_s) * 100.0
            else:
                usg = 0.0

            # Guardem com a NÚMEROS purs
            rows_s.append(
                {
                    "#": p_num,
                    "Jugador": p_name,
                    "PJ": pj,
                    "USG%": round(usg, 1),
                    "1. eFG% (Tir)": round(efg, 1),
                    "2. TOV% (Pèrdues)": round(tov_pct, 1),
                    "3. OREB% (Rebot)": round(oreb_pct, 1),
                    "4. FT Rate (TL)": round(ft_rate, 2),
                    "Plays": round(plays, 1),
                }
            )

        df_res_s = pd.DataFrame(rows_s)
        return df_res_s.sort_values(by="Plays", ascending=False).reset_index(
            drop=True
        )


# src/utils.py


def compute_pbp_advanced_stats(pbp_df, game_id=None):
    """Calcula des del Play-by-Play vinculant assistències i tirs de contraatac."""
    if pbp_df is None or pbp_df.empty:
        return {}

    if game_id:
        pbp = (
            pbp_df[pbp_df["game_id"].astype(str) == str(game_id)]
            .copy()
            .reset_index(drop=True)
        )
    else:
        pbp = pbp_df.copy().reset_index(drop=True)

    if pbp.empty:
        return {}

    t_fb_pts, t_fb_fga = 0, 0
    t_hc_pts, t_hc_fga = 0, 0
    o_fb_pts, o_fb_fga = 0, 0
    o_hc_pts, o_hc_fga = 0, 0

    t_lto, t_dto = 0, 0
    o_lto, o_dto = 0, 0
    t_pts_off_to, o_pts_off_to = 0, 0

    n = len(pbp)
    for i in range(n):
        row = pbp.iloc[i]
        act_low = str(row.get("Action", "")).strip().lower()
        res = str(row.get("Result", "")).strip().lower()
        zone = str(row.get("Zone", "")).strip().lower()
        fb = str(row.get("Fastbreak", "")).strip().lower()
        state = str(row.get("Possession State", "")).strip().lower()

        is_3p = "3" in zone or "3" in act_low or "c3" in zone or "atb3" in zone
        pts_made = (3 if is_3p else 2) if "make" in res else 0

        # DETECCIÓ INTEL·LIGENT DE CONTRAATAC (Tir o Assistència prèvia)
        is_fb_play = fb == "yes"
        if not is_fb_play and i > 0:
            prev_r = pbp.iloc[i - 1]
            prev_act = str(prev_r.get("Action", "")).lower()
            prev_fb = str(prev_r.get("Fastbreak", "")).lower()
            if prev_fb == "yes" and (
                "assist" in prev_act or "steal" in prev_act
            ):
                is_fb_play = True

        # 1. Tirs d'Argentona (Transició vs 5c5)
        if state == "offense" and (
            "shot" in act_low or zone in ["rim", "paint", "mr", "c3", "atb3"]
        ):
            if is_fb_play:
                t_fb_fga += 1
                t_fb_pts += pts_made
            else:
                t_hc_fga += 1
                t_hc_pts += pts_made

        # 2. Tirs del Rival (Transició vs 5c5)
        elif state == "defense" and "opponent" in act_low and "shot" in act_low:
            if is_fb_play:
                o_fb_fga += 1
                o_fb_pts += pts_made
            else:
                o_hc_fga += 1
                o_hc_pts += pts_made

        # 3. Pèrdues d'Argentona i Punts de pèrdua del rival
        if state == "offense" and "turnover" in act_low:
            if "live" in act_low:
                t_lto += 1
                for off in [1, 2, 3]:
                    if i + off < n:
                        nr = pbp.iloc[i + off]
                        if (
                            "opponent" in str(nr.get("Action", "")).lower()
                            and "make" in str(nr.get("Result", "")).lower()
                        ):
                            nz = str(nr.get("Zone", "")).lower()
                            o_pts_off_to += (
                                3
                                if ("3" in nz or "c3" in nz or "atb3" in nz)
                                else 2
                            )
                            break
                        if (
                            str(nr.get("Possession State", "")).lower()
                            == "offense"
                        ):
                            break
            else:
                t_dto += 1

        # 4. Pèrdues del Rival i Punts de pèrdua d'Argentona
        if (
            state == "defense"
            and "steal" in act_low
            or ("opponent turnover" in act_low and "steal" in res)
        ):
            o_lto += 1
            for off in [1, 2, 3]:
                if i + off < n:
                    nr = pbp.iloc[i + off]
                    if (
                        str(nr.get("Possession State", "")).lower() == "offense"
                        and "make" in str(nr.get("Result", "")).lower()
                    ):
                        nz = str(nr.get("Zone", "")).lower()
                        t_pts_off_to += (
                            3
                            if ("3" in nz or "c3" in nz or "atb3" in nz)
                            else 2
                        )
                        break
                    if (
                        str(nr.get("Possession State", "")).lower() == "defense"
                    ):
                        break
        elif state == "defense" and "opponent turnover" in act_low:
            o_dto += 1

    return {
        "fb_pps_t": (t_fb_pts / t_fb_fga) if t_fb_fga > 0 else 0.0,
        "fb_pps_o": (o_fb_pts / o_fb_fga) if o_fb_fga > 0 else 0.0,
        "hc_pps_t": (t_hc_pts / t_hc_fga) if t_hc_fga > 0 else 0.0,
        "hc_pps_o": (o_hc_pts / o_hc_fga) if o_hc_fga > 0 else 0.0,
        "pts_off_to_t": t_pts_off_to,
        "pts_off_to_o": o_pts_off_to,
        "lto_t": t_lto,
        "lto_o": o_lto,
        "dto_t": t_dto,
        "dto_o": o_dto,
    }
       
def get_assist_combos(pbp_df, box_df=None, game_id=None):
    """Extreu el rànquing de duos, punts generats i matriu a partir del Play-by-Play."""
    if pbp_df is None or pbp_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if game_id:
        pbp = (
            pbp_df[pbp_df["game_id"].astype(str) == str(game_id)]
            .copy()
            .reset_index(drop=True)
        )
    else:
        pbp = pbp_df.copy().reset_index(drop=True)

    assists_idx = pbp[
        pbp["Action"].astype(str).str.lower().str.contains("assist", na=False)
    ].index

    if len(assists_idx) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    name_map = {}
    if box_df is not None and not box_df.empty:
        box_sub = (
            box_df[box_df["game_id"].astype(str) == str(game_id)]
            if game_id
            else box_df
        )
        for _, r in box_sub.iterrows():
            if "Player" in r and "Name" in r:
                num = str(r["Player"]).replace("#", "").strip()
                nom = str(r["Name"]).strip()
                if num and nom:
                    name_map[num] = f"{nom} (#{num})"

    pairs = []
    n_rows = len(pbp)

    for i in assists_idx:
        row = pbp.iloc[i]

        # 1. Passador
        passador_num = str(row.get("Player", "")).replace("#", "").strip()
        passador_nom = name_map.get(
            passador_num, f"Jugador #{passador_num}"
        )

        # 2. Anotador
        res_txt = str(row.get("Result", ""))
        match = re.search(r"#+(\d+)", res_txt)
        if match:
            anotador_num = match.group(1).strip()
            anotador_nom = name_map.get(
                anotador_num, f"Jugador #{anotador_num}"
            )
        else:
            anotador_nom = res_txt.replace("To", "").strip()
            if not anotador_nom or anotador_nom.lower() == "none":
                anotador_nom = "Desconegut"

        # 3. Determinar si la cistella assistida és de 2 o 3 punts
        pts = 2
        is_3p = False

        for offset in [1, -1, 2, -2]:
            idx = i + offset
            if 0 <= idx < n_rows:
                cand_row = pbp.iloc[idx]
                cand_act = str(cand_row.get("Action", "")).lower()
                cand_zone = str(cand_row.get("Zone", "")).lower()
                cand_res = str(cand_row.get("Result", "")).lower()

                if "make" in cand_res or "shot" in cand_act:
                    if (
                        "3" in cand_zone
                        or "3" in cand_act
                        or "c3" in cand_zone
                        or "atb3" in cand_zone
                    ):
                        pts = 3
                        is_3p = True
                    else:
                        pts = 2
                        is_3p = False
                    break

        pairs.append(
            {
                "Passador": passador_nom,
                "Anotador": anotador_nom,
                "Punts": pts,
                "T3": 1 if is_3p else 0,
                "T2": 0 if is_3p else 1,
            }
        )

    if not pairs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_pairs = pd.DataFrame(pairs)

    # 1. Rànquing per parelles (Duos)
    df_ranking = (
        df_pairs.groupby(["Passador", "Anotador"])
        .agg(
            Assistències=("Punts", "count"),
            Punts_Generats=("Punts", "sum"),
            T2=("T2", "sum"),
            T3=("T3", "sum"),
        )
        .reset_index()
        .sort_values(by=["Punts_Generats", "Assistències"], ascending=False)
        .reset_index(drop=True)
    )

    # 2. Resum total per Passador (Generador de joc)
    df_passador_totals = (
        df_pairs.groupby("Passador")
        .agg(
            Assistències_Totals=("Punts", "count"),
            Punts_Generats_Totals=("Punts", "sum"),
            T2_Generats=("T2", "sum"),
            T3_Generats=("T3", "sum"),
        )
        .reset_index()
        .sort_values(by="Punts_Generats_Totals", ascending=False)
        .reset_index(drop=True)
    )
    df_passador_totals["PTS / AST"] = (
    df_passador_totals["Punts_Generats_Totals"]
    / df_passador_totals["Assistències_Totals"]
).apply(lambda x: f"{x:.2f}")

    # 3. Matriu de Punts Generats (Passador ➔ Anotador)
    df_matrix = df_pairs.pivot_table(
        index="Passador",
        columns="Anotador",
        values="Punts",
        aggfunc="sum",
        fill_value=0,
    ).astype(int)

    return df_ranking, df_passador_totals, df_matrix
# -------------------------------------------------------------
# FUNCIONS D'ANÀLISI DE QUINTETS I ON / OFF (DEFINITIU)
# -------------------------------------------------------------

def parse_duration_to_min(val):
    """Converteix '04:30', segons purs o minuts decimals a minuts float."""
    try:
        s = str(val).strip()
        if ":" in s:
            parts = s.split(":")
            return float(parts[0]) + float(parts[1]) / 60.0
        f = float(s)
        if f > 60.0:
            return f / 60.0
        return f
    except Exception:
        return 0.0


import unicodedata
import re

def clean_txt(t):
    """Elimina accents, caràcters especials i passa a minúscules."""
    return "".join(
        c for c in unicodedata.normalize("NFD", str(t))
        if unicodedata.category(c) != "Mn"
    ).lower().strip()


def is_player_in_lineup(row, p_num, p_name):
    """Detecta de forma unívoca si el jugador és al quintet evitant confusions amb posicions 1-5."""
    clean_p_name = clean_txt(p_name)
    clean_p_num = re.sub(r"[^\d]", "", str(p_num)).lstrip("0")
    
    # Obtenim els tokens del nom (ex: "victor", "domingo")
    tokens = [t for t in clean_p_name.split() if len(t) >= 3]
    last_name = tokens[-1] if tokens else ""

    # Cel·les P1..P5 i el text complet del quintet
    p_cells = [str(row.get(f"P{i}", "")).strip() for i in range(1, 6)]
    lineup_text = str(row.get("Lineup", "")).strip()
    
    # Text combinat normalitzat (sense accents)
    combined_raw = " | ".join(p_cells + [lineup_text])
    combined_clean = clean_txt(combined_raw)

    # 1. MÈTODE PRINCIPAL: COINCIDÈNCIA EXACTA DE COGNOM
    # (El cognom "domingo", "blanco", "martori", "basterra", "tena" mai es confon amb una posició)
    if last_name and len(last_name) >= 3:
        if re.search(rf"\b{re.escape(last_name)}\b", combined_clean):
            return True

    # 2. NOM COMPLET
    if clean_p_name and clean_p_name in combined_clean:
        return True

    # 3. DORSAL AMB COIXINET '#': obligatori per als dorsals de l'1 al 5 (#1, #2, #3, #4, #5)
    # per evitar confondre'ls amb la posició de Base (1), Escolta (2), etc.
    if clean_p_num:
        if re.search(rf"#\s*0*{clean_p_num}\b", combined_raw):
            return True

    # 4. Si i només si el dorsal és > 5 (ex: #8, #9, #15, #32), pot emparellar com a número pur
    if clean_p_num and int(clean_p_num) > 5:
        for cell in p_cells:
            c_strip = cell.strip()
            if c_strip.isdigit() and int(c_strip) == int(clean_p_num):
                return True

    return False

def aggregate_lineup_data(df_slice, forced_min=None, total_game_min=None, total_game_poss=None):
    """Calcula mètriques d'atac i defensa dels quintets amb ritme equilibrat."""
    if df_slice.empty:
        return {}

    # 1. Càlcul de minuts
    tot_min = 0.0
    for min_col in ["MIN", "Min", "min", "Duration", "Duration(s)", "Time", "Seconds"]:
        if min_col in df_slice.columns:
            m_s = df_slice[min_col].apply(parse_duration_to_min).sum()
            if m_s > 0:
                tot_min = float(m_s)
                break

    # 2. ATAC D'ARGENTONA
    t_rim_m = find_lineup_col_val(df_slice, ["Rim_FGM_For", "Rim_FGM", "RIM(M)", "Rim(M)", "Rim_FGM_Tm"])
    t_rim_a = find_lineup_col_val(df_slice, ["Rim_FGA_For", "Rim_FGA", "RIM(A)", "Rim(A)", "Rim_FGA_Tm"])
    t_paint_m = find_lineup_col_val(df_slice, ["Paint_FGM_For", "Paint_FGM", "PAINT(M)", "Paint(M)", "Paint_FGM_Tm"])
    t_paint_a = find_lineup_col_val(df_slice, ["Paint_FGA_For", "Paint_FGA", "PAINT(A)", "Paint(A)", "Paint_FGA_Tm"])
    t_mr_m = find_lineup_col_val(df_slice, ["MR_FGM_For", "MR_FGM", "MR(M)", "MR(M)", "MR_FGM_Tm"])
    t_mr_a = find_lineup_col_val(df_slice, ["MR_FGA_For", "MR_FGA", "MR(A)", "MR(A)", "MR_FGA_Tm"])

    t_2pm = find_lineup_col_val(df_slice, ["2PM_For", "2PM", "2PM_Tm", "2FGM_For", "2FGM"])
    t_2pa = find_lineup_col_val(df_slice, ["2PA_For", "2PA", "2PA_Tm", "2FGA_For", "2FGA"])
    if t_2pa == 0:
        t_2pm = t_rim_m + t_paint_m + t_mr_m
        t_2pa = t_rim_a + t_paint_a + t_mr_a

    t_c3_m = find_lineup_col_val(df_slice, ["Cor3_FGM_For", "Cor3_FGM", "C3(M)", "C3_M", "C3_FGM_For", "C3_FGM"])
    t_c3_a = find_lineup_col_val(df_slice, ["Cor3_FGA_For", "Cor3_FGA", "C3(A)", "C3_A", "C3_FGA_For", "C3_FGA"])
    t_atb3_m = find_lineup_col_val(df_slice, ["ATB3_FGM_For", "ATB3_FGM", "ATB3(M)", "ATB3_M", "ATB3_FGM_Tm"])
    t_atb3_a = find_lineup_col_val(df_slice, ["ATB3_FGA_For", "ATB3_FGA", "ATB3(A)", "ATB3_A", "ATB3_FGA_Tm"])

    t_3pm = find_lineup_col_val(df_slice, ["3PM_For", "3PM", "3PM_Tm", "3FGM_For", "3FGM"])
    t_3pa = find_lineup_col_val(df_slice, ["3PA_For", "3PA", "3PA_Tm", "3FGA_For", "3FGA"])
    if t_3pa == 0:
        t_3pm = t_c3_m + t_atb3_m
        t_3pa = t_c3_a + t_atb3_a

    t_fgm = t_2pm + t_3pm
    t_fga = t_2pa + t_3pa

    t_ftm = find_lineup_col_val(df_slice, ["FTM_For", "FTM", "FTM_Tm", "FT_M"])
    t_fta = find_lineup_col_val(df_slice, ["FTA_For", "FTA", "FTA_Tm", "FT_A"])
    t_oreb = find_lineup_col_val(df_slice, ["OREB_For", "OREB", "Off Reb", "Off_Reb", "OffReb", "OREB_Tm"])
    t_dreb = find_lineup_col_val(df_slice, ["DREB_For", "DREB", "Def Reb", "Def_Reb", "DefReb", "DREB_Tm"])
    t_tov = find_lineup_col_val(df_slice, ["TOV_For", "TOV", "Turnovers", "TOV_Tm", "TO_For"])

    t_pts = find_lineup_col_val(df_slice, ["PTS_For", "PTS", "Points", "Points_For", "PTS_Tm"])
    if t_pts == 0:
        t_pts = t_2pm * 2.0 + t_3pm * 3.0 + t_ftm

    # 3. DEFENSA (RIVAL)
    o_rim_m = find_lineup_col_val(df_slice, ["Rim_FGM_Agn"])
    o_rim_a = find_lineup_col_val(df_slice, ["Rim_FGA_Agn"])
    o_paint_m = find_lineup_col_val(df_slice, ["Paint_FGM_Agn"])
    o_paint_a = find_lineup_col_val(df_slice, ["Paint_FGA_Agn"])
    o_mr_m = find_lineup_col_val(df_slice, ["MR_FGM_Agn"])
    o_mr_a = find_lineup_col_val(df_slice, ["MR_FGA_Agn"])

    o_2pm = find_lineup_col_val(df_slice, ["2PM_Agn"])
    o_2pa = find_lineup_col_val(df_slice, ["2PA_Agn"])
    if o_2pa == 0:
        o_2pm = o_rim_m + o_paint_m + o_mr_m
        o_2pa = o_rim_a + o_paint_a + o_mr_a

    o_c3_m = find_lineup_col_val(df_slice, ["Cor3_FGM_Agn"])
    o_c3_a = find_lineup_col_val(df_slice, ["Cor3_FGA_Agn"])
    o_atb3_m = find_lineup_col_val(df_slice, ["ATB3_FGM_Agn"])
    o_atb3_a = find_lineup_col_val(df_slice, ["ATB3_FGA_Agn"])

    o_3pm = find_lineup_col_val(df_slice, ["3PM_Agn"])
    o_3pa = find_lineup_col_val(df_slice, ["3PA_Agn"])
    if o_3pa == 0:
        o_3pm = o_c3_m + o_atb3_m
        o_3pa = o_c3_a + o_atb3_a

    o_fgm = o_2pm + o_3pm
    o_fga = o_2pa + o_3pa

    o_ftm = find_lineup_col_val(df_slice, ["FTM_Agn"])
    o_fta = find_lineup_col_val(df_slice, ["FTA_Agn"])
    o_oreb = find_lineup_col_val(df_slice, ["OREB_Agn"])
    o_dreb = find_lineup_col_val(df_slice, ["DREB_Agn"])
    o_tov = find_lineup_col_val(df_slice, ["TOV_Agn"])

    o_pts = find_lineup_col_val(df_slice, ["PTS_Agn"])
    if o_pts == 0:
        o_pts = o_2pm * 2.0 + o_3pm * 3.0 + o_ftm

    # 4. Possessions
    t_poss_raw = t_fga + 0.44 * t_fta - t_oreb + t_tov
    o_poss_raw = o_fga + 0.44 * o_fta - o_oreb + o_tov
    if t_poss_raw > 0 and o_poss_raw > 0:
        poss = 0.5 * (t_poss_raw + o_poss_raw)
    else:
        poss = max(t_poss_raw, o_poss_raw)

    # 5. Integració de minuts
    if tot_min == 0.0 and forced_min is not None and forced_min > 0:
        tot_min = float(forced_min)
    elif tot_min == 0.0 and total_game_min is not None and total_game_poss is not None and total_game_poss > 0:
        tot_min = poss * (total_game_min / total_game_poss)

    p40 = (40.0 / tot_min) if tot_min > 0 else 0.0
    pace = (poss * p40) if tot_min > 0 else poss

    return {
        "minutes": tot_min,
        "poss": poss,
        "pace": pace,
        "plus_minus": t_pts - o_pts,
        "plus_minus_40": (t_pts - o_pts) * p40,
        "oer": (t_pts / poss * 100.0) if poss > 0 else 0.0,
        "der": (o_pts / poss * 100.0) if poss > 0 else 0.0,
        "net_rtg": ((t_pts - o_pts) / poss * 100.0) if poss > 0 else 0.0,
        # Volums Atac per 40 min
        "fga_40": t_fga * p40,
        "2pa_40": t_2pa * p40,
        "3pa_40": t_3pa * p40,
        "fta_40": t_fta * p40,
        "oreb_40": t_oreb * p40,
        "dreb_40": t_dreb * p40,
        "tov_40": t_tov * p40,
        "3par": (t_3pa / t_fga * 100.0) if t_fga > 0 else 0.0,
        # Eficiència Atac
        "2pm": t_2pm, "2pa": t_2pa,
        "pct_2p": (t_2pm / t_2pa * 100.0) if t_2pa > 0 else 0.0,
        "3pm": t_3pm, "3pa": t_3pa,
        "pct_3p": (t_3pm / t_3pa * 100.0) if t_3pa > 0 else 0.0,
        "efg": ((t_fgm + 0.5 * t_3pm) / t_fga * 100.0) if t_fga > 0 else 0.0,
        "ftm": t_ftm, "fta": t_fta,
        "pct_ft": (t_ftm / t_fta * 100.0) if t_fta > 0 else 0.0,
        "ft_rate": (t_ftm / t_fga) if t_fga > 0 else 0.0,
        "oreb_pct": (t_oreb / (t_oreb + o_dreb) * 100.0) if (t_oreb + o_dreb) > 0 else 0.0,
        "tov_pct": (t_tov / (t_fga + 0.44 * t_fta + t_tov) * 100.0) if (t_fga + 0.44 * t_fta + t_tov) > 0 else 0.0,
        # Volums Defensa Rival per 40 min
        "opp_fga_40": o_fga * p40,
        "opp_2pa_40": o_2pa * p40,
        "opp_3pa_40": o_3pa * p40,
        "opp_fta_40": o_fta * p40,
        "opp_oreb_40": o_oreb * p40,
        "opp_tov_40": o_tov * p40,
        "opp_3par": (o_3pa / o_fga * 100.0) if o_fga > 0 else 0.0,
        # Eficiència Defensa Rival
        "opp_2pm": o_2pm, "opp_2pa": o_2pa,
        "opp_pct_2p": (o_2pm / o_2pa * 100.0) if o_2pa > 0 else 0.0,
        "opp_3pm": o_3pm, "opp_3pa": o_3pa,
        "opp_pct_3p": (o_3pm / o_3pa * 100.0) if o_3pa > 0 else 0.0,
        "opp_efg": ((o_fgm + 0.5 * o_3pm) / o_fga * 100.0) if o_fga > 0 else 0.0,
        "opp_ftm": o_ftm, "opp_fta": o_fta,
        "opp_pct_ft": (o_ftm / o_fta * 100.0) if o_fta > 0 else 0.0,
        "opp_ft_rate": (o_ftm / o_fga) if o_fga > 0 else 0.0,
        "opp_oreb_pct": (o_oreb / (o_oreb + t_dreb) * 100.0) if (o_oreb + t_dreb) > 0 else 0.0,
        "opp_tov_pct": (o_tov / (o_fga + 0.44 * o_fta + o_tov) * 100.0) if (o_fga + 0.44 * o_fta + o_tov) > 0 else 0.0,
    }

def find_lineup_col_val(df_slice, candidates):
    """Cerca tolerant entre columnes _For, directes o de boxscore."""
    if df_slice.empty:
        return 0.0

    existing_cols = {str(c).strip(): c for c in df_slice.columns}
    existing_lower = {str(c).strip().lower(): c for c in df_slice.columns}
    existing_stripped = {
        re.sub(r"[_\s\-\(\)]", "", str(c).lower()): c for c in df_slice.columns
    }

    for cand in candidates:
        if cand in existing_cols:
            return float(pd.to_numeric(df_slice[existing_cols[cand]], errors="coerce").fillna(0).sum())
        cand_low = cand.strip().lower()
        if cand_low in existing_lower:
            return float(pd.to_numeric(df_slice[existing_lower[cand_low]], errors="coerce").fillna(0).sum())
        cand_strip = re.sub(r"[_\s\-\(\)]", "", cand_low)
        if cand_strip in existing_stripped:
            return float(pd.to_numeric(df_slice[existing_stripped[cand_strip]], errors="coerce").fillna(0).sum())

    return 0.0


def aggregate_lineup_data(df_slice, forced_min=None, total_game_min=None, total_game_poss=None):
    """Calcula mètriques d'atac i defensa utilitzant minuts reals o proporcionals."""
    if df_slice.empty:
        return {}

    # 1. Càlcul de minuts
    tot_min = 0.0
    for min_col in ["MIN", "Min", "min", "Duration", "Duration(s)", "Time", "Seconds"]:
        if min_col in df_slice.columns:
            m_s = df_slice[min_col].apply(parse_duration_to_min).sum()
            if m_s > 0:
                tot_min = float(m_s)
                break

    # 2. ATAC D'ARGENTONA
    t_rim_m = find_lineup_col_val(df_slice, ["Rim_FGM_For", "Rim_FGM", "RIM(M)", "Rim(M)", "Rim_FGM_Tm"])
    t_rim_a = find_lineup_col_val(df_slice, ["Rim_FGA_For", "Rim_FGA", "RIM(A)", "Rim(A)", "Rim_FGA_Tm"])
    t_paint_m = find_lineup_col_val(df_slice, ["Paint_FGM_For", "Paint_FGM", "PAINT(M)", "Paint(M)", "Paint_FGM_Tm"])
    t_paint_a = find_lineup_col_val(df_slice, ["Paint_FGA_For", "Paint_FGA", "PAINT(A)", "Paint(A)", "Paint_FGA_Tm"])
    t_mr_m = find_lineup_col_val(df_slice, ["MR_FGM_For", "MR_FGM", "MR(M)", "MR(M)", "MR_FGM_Tm"])
    t_mr_a = find_lineup_col_val(df_slice, ["MR_FGA_For", "MR_FGA", "MR(A)", "MR(A)", "MR_FGA_Tm"])

    t_2pm = find_lineup_col_val(df_slice, ["2PM_For", "2PM", "2PM_Tm", "2FGM_For", "2FGM"])
    t_2pa = find_lineup_col_val(df_slice, ["2PA_For", "2PA", "2PA_Tm", "2FGA_For", "2FGA"])
    if t_2pa == 0:
        t_2pm = t_rim_m + t_paint_m + t_mr_m
        t_2pa = t_rim_a + t_paint_a + t_mr_a

    t_c3_m = find_lineup_col_val(df_slice, ["Cor3_FGM_For", "Cor3_FGM", "C3(M)", "C3_M", "C3_FGM_For", "C3_FGM"])
    t_c3_a = find_lineup_col_val(df_slice, ["Cor3_FGA_For", "Cor3_FGA", "C3(A)", "C3_A", "C3_FGA_For", "C3_FGA"])
    t_atb3_m = find_lineup_col_val(df_slice, ["ATB3_FGM_For", "ATB3_FGM", "ATB3(M)", "ATB3_M", "ATB3_FGM_Tm"])
    t_atb3_a = find_lineup_col_val(df_slice, ["ATB3_FGA_For", "ATB3_FGA", "ATB3(A)", "ATB3_A", "ATB3_FGA_Tm"])

    t_3pm = find_lineup_col_val(df_slice, ["3PM_For", "3PM", "3PM_Tm", "3FGM_For", "3FGM"])
    t_3pa = find_lineup_col_val(df_slice, ["3PA_For", "3PA", "3PA_Tm", "3FGA_For", "3FGA"])
    if t_3pa == 0:
        t_3pm = t_c3_m + t_atb3_m
        t_3pa = t_c3_a + t_atb3_a

    t_fgm = t_2pm + t_3pm
    t_fga = t_2pa + t_3pa

    t_ftm = find_lineup_col_val(df_slice, ["FTM_For", "FTM", "FTM_Tm", "FT_M"])
    t_fta = find_lineup_col_val(df_slice, ["FTA_For", "FTA", "FTA_Tm", "FT_A"])
    t_oreb = find_lineup_col_val(df_slice, ["OREB_For", "OREB", "Off Reb", "Off_Reb", "OffReb", "OREB_Tm"])
    t_dreb = find_lineup_col_val(df_slice, ["DREB_For", "DREB", "Def Reb", "Def_Reb", "DefReb", "DREB_Tm"])
    t_tov = find_lineup_col_val(df_slice, ["TOV_For", "TOV", "Turnovers", "TOV_Tm", "TO_For"])

    t_pts = find_lineup_col_val(df_slice, ["PTS_For", "PTS", "Points", "Points_For", "PTS_Tm"])
    if t_pts == 0:
        t_pts = t_2pm * 2.0 + t_3pm * 3.0 + t_ftm

    # 3. DEFENSA (RIVAL)
    o_rim_m = find_lineup_col_val(df_slice, ["Rim_FGM_Agn"])
    o_rim_a = find_lineup_col_val(df_slice, ["Rim_FGA_Agn"])
    o_paint_m = find_lineup_col_val(df_slice, ["Paint_FGM_Agn"])
    o_paint_a = find_lineup_col_val(df_slice, ["Paint_FGA_Agn"])
    o_mr_m = find_lineup_col_val(df_slice, ["MR_FGM_Agn"])
    o_mr_a = find_lineup_col_val(df_slice, ["MR_FGA_Agn"])

    o_2pm = find_lineup_col_val(df_slice, ["2PM_Agn"])
    o_2pa = find_lineup_col_val(df_slice, ["2PA_Agn"])
    if o_2pa == 0:
        o_2pm = o_rim_m + o_paint_m + o_mr_m
        o_2pa = o_rim_a + o_paint_a + o_mr_a

    o_c3_m = find_lineup_col_val(df_slice, ["Cor3_FGM_Agn"])
    o_c3_a = find_lineup_col_val(df_slice, ["Cor3_FGA_Agn"])
    o_atb3_m = find_lineup_col_val(df_slice, ["ATB3_FGM_Agn"])
    o_atb3_a = find_lineup_col_val(df_slice, ["ATB3_FGA_Agn"])

    o_3pm = find_lineup_col_val(df_slice, ["3PM_Agn"])
    o_3pa = find_lineup_col_val(df_slice, ["3PA_Agn"])
    if o_3pa == 0:
        o_3pm = o_c3_m + o_atb3_m
        o_3pa = o_c3_a + o_atb3_a

    o_fgm = o_2pm + o_3pm
    o_fga = o_2pa + o_3pa

    o_ftm = find_lineup_col_val(df_slice, ["FTM_Agn"])
    o_fta = find_lineup_col_val(df_slice, ["FTA_Agn"])
    o_oreb = find_lineup_col_val(df_slice, ["OREB_Agn"])
    o_dreb = find_lineup_col_val(df_slice, ["DREB_Agn"])
    o_tov = find_lineup_col_val(df_slice, ["TOV_Agn"])

    o_pts = find_lineup_col_val(df_slice, ["PTS_Agn"])
    if o_pts == 0:
        o_pts = o_2pm * 2.0 + o_3pm * 3.0 + o_ftm

    # 4. Possessions
    t_poss_raw = t_fga + 0.44 * t_fta - t_oreb + t_tov
    o_poss_raw = o_fga + 0.44 * o_fta - o_oreb + o_tov
    if t_poss_raw > 0 and o_poss_raw > 0:
        poss = 0.5 * (t_poss_raw + o_poss_raw)
    else:
        poss = max(t_poss_raw, o_poss_raw)

    # 5. Integració de minuts oficials de Boxscore o proporcionals
    if tot_min == 0.0 and forced_min is not None and forced_min > 0:
        tot_min = float(forced_min)
    elif tot_min == 0.0 and total_game_min is not None and total_game_poss is not None and total_game_poss > 0:
        tot_min = poss * (total_game_min / total_game_poss)

    p40 = (40.0 / tot_min) if tot_min > 0 else 0.0
    pace = (poss * p40) if tot_min > 0 else poss

    return {
        "minutes": tot_min,
        "poss": poss,
        "pace": pace,
        "plus_minus": t_pts - o_pts,
        "plus_minus_40": (t_pts - o_pts) * p40,
        "oer": (t_pts / poss * 100.0) if poss > 0 else 0.0,
        "der": (o_pts / poss * 100.0) if poss > 0 else 0.0,
        "net_rtg": ((t_pts - o_pts) / poss * 100.0) if poss > 0 else 0.0,
        # Volums Atac per 40 min
        "fga_40": t_fga * p40,
        "2pa_40": t_2pa * p40,
        "3pa_40": t_3pa * p40,
        "fta_40": t_fta * p40,
        "oreb_40": t_oreb * p40,
        "dreb_40": t_dreb * p40,
        "tov_40": t_tov * p40,
        "3par": (t_3pa / t_fga * 100.0) if t_fga > 0 else 0.0,
        # Eficiència Atac
        "2pm": t_2pm, "2pa": t_2pa,
        "pct_2p": (t_2pm / t_2pa * 100.0) if t_2pa > 0 else 0.0,
        "3pm": t_3pm, "3pa": t_3pa,
        "pct_3p": (t_3pm / t_3pa * 100.0) if t_3pa > 0 else 0.0,
        "efg": ((t_fgm + 0.5 * t_3pm) / t_fga * 100.0) if t_fga > 0 else 0.0,
        "ftm": t_ftm, "fta": t_fta,
        "pct_ft": (t_ftm / t_fta * 100.0) if t_fta > 0 else 0.0,
        "ft_rate": (t_ftm / t_fga) if t_fga > 0 else 0.0,
        "oreb_pct": (t_oreb / (t_oreb + o_dreb) * 100.0) if (t_oreb + o_dreb) > 0 else 0.0,
        "tov_pct": (t_tov / (t_fga + 0.44 * t_fta + t_tov) * 100.0) if (t_fga + 0.44 * t_fta + t_tov) > 0 else 0.0,
        # Volums Defensa Rival per 40 min
        "opp_fga_40": o_fga * p40,
        "opp_2pa_40": o_2pa * p40,
        "opp_3pa_40": o_3pa * p40,
        "opp_fta_40": o_fta * p40,
        "opp_oreb_40": o_oreb * p40,
        "opp_tov_40": o_tov * p40,
        "opp_3par": (o_3pa / o_fga * 100.0) if o_fga > 0 else 0.0,
        # Eficiència Defensa Rival
        "opp_2pm": o_2pm, "opp_2pa": o_2pa,
        "opp_pct_2p": (o_2pm / o_2pa * 100.0) if o_2pa > 0 else 0.0,
        "opp_3pm": o_3pm, "opp_3pa": o_3pa,
        "opp_pct_3p": (o_3pm / o_3pa * 100.0) if o_3pa > 0 else 0.0,
        "opp_efg": ((o_fgm + 0.5 * o_3pm) / o_fga * 100.0) if o_fga > 0 else 0.0,
        "opp_ftm": o_ftm, "opp_fta": o_fta,
        "opp_pct_ft": (o_ftm / o_fta * 100.0) if o_fta > 0 else 0.0,
        "opp_ft_rate": (o_ftm / o_fga) if o_fga > 0 else 0.0,
        "opp_oreb_pct": (o_oreb / (o_oreb + t_dreb) * 100.0) if (o_oreb + t_dreb) > 0 else 0.0,
        "opp_tov_pct": (o_tov / (o_fga + 0.44 * o_fta + o_tov) * 100.0) if (o_fga + 0.44 * o_fta + o_tov) > 0 else 0.0,
    }


def get_5man_lineup_summary(lineups_df, box_df=None, min_minutes=0.0):
    """Genera la taula de tots els quintets de 5 jugadors amb volums i eficiències."""
    if lineups_df is None or lineups_df.empty:
        return pd.DataFrame()

    tot_game_min = 40.0
    if box_df is not None and not box_df.empty and "MIN" in box_df.columns:
        m_s = box_df["MIN"].apply(parse_duration_to_min).sum()
        if m_s > 0:
            tot_game_min = m_s / 5.0

    total_l_data = aggregate_lineup_data(lineups_df)
    tot_game_poss = total_l_data.get("poss", 75.0)

    lineups_list = []
    grouped = lineups_df.groupby("Lineup", as_index=False)
    for l_name, group in grouped:
        st_data = aggregate_lineup_data(
            group,
            total_game_min=tot_game_min,
            total_game_poss=tot_game_poss
        )
        if st_data.get("minutes", 0) < min_minutes:
            continue

        clean_name = str(l_name).replace('"', '').strip()
        lineups_list.append({
            "Quintet": clean_name,
            "MIN": f"{int(st_data['minutes'])}:{int((st_data['minutes'] % 1)*60):02d}",
            "_min_num": st_data["minutes"],
            "POSS": round(st_data["poss"], 1),
            "+/-": int(st_data["plus_minus"]),
            "+/- /40m": round(st_data["plus_minus_40"], 1),
            "Net Rtg": round(st_data["net_rtg"], 1),
            "OER": round(st_data["oer"], 1),
            "DER": round(st_data["der"], 1),
            "eFG%": round(st_data["efg"], 1),
            "eFG% Riv": round(st_data["opp_efg"], 1),
            "T2 (A/I)": f"{int(st_data['2pm'])}/{int(st_data['2pa'])}",
            "%T2": round(st_data["pct_2p"], 1),
            "T3 (A/I)": f"{int(st_data['3pm'])}/{int(st_data['3pa'])}",
            "%T3": round(st_data["pct_3p"], 1),
            "3PAr": round(st_data["3par"], 1),
            "REB_O": int(st_data.get("oreb_40", 0) * (st_data["minutes"] / 40.0)),
            "PER": int(st_data.get("tov_40", 0) * (st_data["minutes"] / 40.0)),
        })

    df_res = pd.DataFrame(lineups_list)
    if not df_res.empty:
        df_res = df_res.sort_values(by="_min_num", ascending=False).drop(columns=["_min_num"]).reset_index(drop=True)
    return df_res