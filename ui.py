import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Match Predictor",
    page_icon="⚽",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #060e1e; }
[data-testid="stHeader"]           { background: transparent; }
div[data-testid="metric-container"] {
    background: #0d1f3c;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 10px;
}
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white; border: none; border-radius: 12px;
    font-size: 1.2rem; font-weight: bold; padding: 14px 30px;
    transition: all 0.2s ease;
}
.stButton > button:hover { background: linear-gradient(135deg, #1976d2, #1565c0); }
h1, h2, h3 { color: #4fc3f7 !important; }
p, label, div { color: #cfd8dc; }
hr { border-color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────
@st.cache_data
def load_clubs():
    return pd.read_csv("ClubsId.csv")

@st.cache_data
def load_players():
    return pd.read_csv("Players.csv")

@st.cache_data
def load_games():
    df = pd.read_csv("games_df.csv")
    df = df.sort_values(['season', 'month']).reset_index(drop=True)
    df['is_draw'] = (df['result_num'] == 0).astype(int)
    df['home_draw_rate_5'] = (
        df.groupby('home_rank_id')['is_draw']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df['away_draw_rate_5'] = (
        df.groupby('away_rank_id')['is_draw']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df['rank_diff_abs']   = (df['home_rank_id'] - df['away_rank_id']).abs()
    df['rank_closeness']  = 1 / (1 + df['rank_diff_abs'])
    df['home_win_rate_5'] = (
        df.groupby('home_rank_id')['result_num']
        .transform(lambda x: (x.shift(1) == 1).rolling(5, min_periods=1).mean())
    )
    return df.drop(columns=['is_draw'])

def get_historical_stats(home_rank_id, away_rank_id, gdf):
    hg = gdf[gdf['home_rank_id'] == home_rank_id]
    ag = gdf[gdf['away_rank_id'] == away_rank_id]
    home_draw = float(hg['home_draw_rate_5'].iloc[-1]) if len(hg) > 0 else 0.20
    away_draw = float(ag['away_draw_rate_5'].iloc[-1]) if len(ag) > 0 else 0.20
    home_win  = float(hg['home_win_rate_5'].iloc[-1])  if len(hg) > 0 else 0.40
    r_diff    = abs(home_rank_id - away_rank_id)
    return {
        'home_draw_rate_5': round(home_draw, 4),
        'away_draw_rate_5': round(away_draw, 4),
        'rank_diff_abs':    r_diff,
        'rank_closeness':   round(1 / (1 + r_diff), 4),
        'home_win_rate_5':  round(home_win, 4),
    }

def compute_stats(names, pdf):
    sel = pdf[pdf['name'].isin(names)]
    if len(sel) == 0:
        return {k: 0.0 for k in ['goals','assists','yellow_cards','red_card','age','market_value']}
    return {
        'goals':        sel['goals'].mean(),
        'assists':      sel['assists'].mean(),
        'yellow_cards': sel['yellow_cards'].mean(),
        'red_card':     sel['red_cards'].mean(),
        'age':          sel['age'].mean(),
        'market_value': sel['market_value'].mean(),
    }

# ─────────────────────────────────────────────
# Formations
# ─────────────────────────────────────────────
FORMATIONS = {
    '4-3-3': (4, 3, 3),
    '4-4-2': (4, 4, 2),
    '4-5-1': (4, 5, 1),
    '3-5-2': (3, 5, 2),
    '3-4-3': (3, 4, 3),
    '3-3-4': (3, 3, 4),
    '5-4-1': (5, 4, 1),
    '5-1-4': (5, 1, 4),
    '4-2-4': (4, 2, 4),
}

# ─────────────────────────────────────────────
# Pitch Drawing
# ─────────────────────────────────────────────
def player_positions(d, m, a, side):
    if side == 'home':
        xs = {'GK': 0.05, 'DEF': 0.21, 'MID': 0.40, 'ATT': 0.61}
    else:
        xs = {'GK': 0.95, 'DEF': 0.79, 'MID': 0.60, 'ATT': 0.39}
    pos = [(xs['GK'], 0.5, 'GK')]
    for i in range(d): pos.append((xs['DEF'], (i + 1) / (d + 1), 'DEF'))
    for i in range(m): pos.append((xs['MID'], (i + 1) / (m + 1), 'MID'))
    for i in range(a): pos.append((xs['ATT'], (i + 1) / (a + 1), 'ATT'))
    return pos

def draw_pitch(hf, af, hp, ap, hn, an):
    d_h, m_h, a_h = FORMATIONS[hf]
    d_a, m_a, a_a = FORMATIONS[af]
    hpos = player_positions(d_h, m_h, a_h, 'home')
    apos = player_positions(d_a, m_a, a_a, 'away')

    fig, ax = plt.subplots(figsize=(20, 11))
    fig.patch.set_facecolor('#060e1e')
    ax.set_facecolor('#060e1e')

    # Grass stripes
    for i in range(12):
        c = '#1a4028' if i % 2 == 0 else '#1f4d30'
        ax.add_patch(patches.Rectangle((i / 12, 0.03), 1 / 12, 0.94, color=c, zorder=0))

    # Pitch outline
    ax.add_patch(patches.Rectangle((0.02, 0.03), 0.96, 0.94,
                                    fill=False, edgecolor='white', lw=2.5, zorder=3))
    # Halfway line
    ax.plot([0.5, 0.5], [0.03, 0.97], color='white', lw=2, zorder=3)

    # Centre circle
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.5 + 0.09 * np.cos(t), 0.5 + 0.09 * np.sin(t), 'white', lw=2, zorder=3)
    ax.plot(0.5, 0.5, 'wo', ms=5, zorder=4)

    # Penalty boxes
    for lx, sign in [(0.02, 1), (0.98, -1)]:
        bw = 0.14 * sign
        ax.add_patch(patches.Rectangle((lx, 0.28), bw, 0.44,
                                        fill=False, edgecolor='white', lw=2, zorder=3))
        sw = 0.06 * sign
        ax.add_patch(patches.Rectangle((lx, 0.38), sw, 0.24,
                                        fill=False, edgecolor='white', lw=2, zorder=3))
        ax.plot(lx + 0.10 * sign, 0.5, 'wo', ms=4, zorder=4)

    # Corner arcs
    for cx, cy in [(0.02, 0.03), (0.02, 0.97), (0.98, 0.03), (0.98, 0.97)]:
        dx = 1 if cx < 0.5 else -1
        dy = 1 if cy < 0.5 else -1
        t2 = np.linspace(0, np.pi / 2, 50)
        r  = 0.025
        base_angle = np.arctan2(dy, dx) - np.pi / 4
        ax.plot(cx + r * np.cos(t2 + base_angle), cy + r * np.sin(t2 + base_angle),
                'white', lw=1.5, zorder=3)

    # Team headers
    ax.text(0.25, 0.993, hn, ha='center', va='top',
            fontsize=13, fontweight='bold', color='#64b5f6', zorder=5)
    ax.text(0.75, 0.993, an, ha='center', va='top',
            fontsize=13, fontweight='bold', color='#ef9a9a', zorder=5)
    ax.text(0.25, 0.963, f'[{hf}]', ha='center', va='top',
            fontsize=10, color='#90caf9', zorder=5)
    ax.text(0.75, 0.963, f'[{af}]', ha='center', va='top',
            fontsize=10, color='#ffcdd2', zorder=5)

    # Home players (blue)
    for idx, (x, y, _) in enumerate(hpos):
        name  = hp[idx] if idx < len(hp) and hp[idx] else '?'
        short = name.split()[-1][:10] if name != '?' else '?'
        ax.add_patch(plt.Circle((x, y), 0.033, color='#0d47a1', zorder=5))
        ax.add_patch(plt.Circle((x, y), 0.033, fill=False, color='#64b5f6', lw=1.5, zorder=6))
        ax.text(x, y, str(idx + 1), ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=7)
        ax.text(x, y - 0.068, short, ha='center', va='top', fontsize=6.5, color='white', zorder=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='#0a2e6e', alpha=0.85, ec='none'))

    # Away players (red)
    for idx, (x, y, _) in enumerate(apos):
        name  = ap[idx] if idx < len(ap) and ap[idx] else '?'
        short = name.split()[-1][:10] if name != '?' else '?'
        ax.add_patch(plt.Circle((x, y), 0.033, color='#7f0000', zorder=5))
        ax.add_patch(plt.Circle((x, y), 0.033, fill=False, color='#ef9a9a', lw=1.5, zorder=6))
        ax.text(x, y, str(idx + 1), ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=7)
        ax.text(x, y - 0.068, short, ha='center', va='top', fontsize=6.5, color='white', zorder=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='#5c0000', alpha=0.85, ec='none'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

# ─────────────────────────────────────────────
# Load CSVs
# ─────────────────────────────────────────────
clubs_df   = load_clubs()
players_df = load_players()
games_df   = load_games()

club_opts   = ['— Select Club —']   + sorted(clubs_df['club_name'].tolist())
player_opts = ['— Select Player —'] + sorted(players_df['name'].tolist())

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:30px 0 15px 0;'>
    <h1 style='font-size:3rem; color:#4fc3f7; margin:0;'>⚽ Match Predictor</h1>
    <p style='color:#90caf9; font-size:1.1rem; margin:10px 0 0 0;'>
        Pre-match AI prediction — set the lineup, get your forecast
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 1 — Match Details
# ─────────────────────────────────────────────
st.markdown("### 📋 Match Details")
c1, c2, c3, c4 = st.columns(4)

with c1:
    season    = st.number_input("Season", 2013, 2030, 2026, step=1)
    month     = st.number_input("Month", 1, 12, 5, step=1)
    dayofweek = st.number_input("Day of Week  (1=Mon, 7=Sun)", 1, 7, 1, step=1)

with c2:
    cup_choice  = st.radio("Cup Game?", ["No ❌", "Yes ✅"], horizontal=True)
    is_cup_game = 1 if "Yes" in cup_choice else 0
    att_raw     = st.number_input("Attendance", 0, 120_000, 35_000, step=500)
    attendance  = float(np.log1p(att_raw))
    st.caption(f"Log-scaled value sent to model: **{attendance:.4f}**")

with c3:
    country_display = st.selectbox("Country / Competition",
        ['France', 'Germany', 'Italy', 'Netherlands',
         'Portugal', 'Spain', 'Türkiye', 'Europa'])
    match_type = st.radio("Match Type",
        ['domestic_league', 'international_cup', 'other'])

with c4:
    round_type = st.radio("Round", ['league', 'knockout', 'rounds'])

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 2 — Teams & Positions
# ─────────────────────────────────────────────
st.markdown("### 🏟️ Team Selection")
tc1, tc2 = st.columns(2)

with tc1:
    home_club = st.selectbox("🔵 Home Team", club_opts)
    home_pos  = st.number_input(
        "Home League Position (0 = cup / unknown)",
        0, 30, 0, step=1,
        disabled=(is_cup_game == 1)
    )

with tc2:
    away_club = st.selectbox("🔴 Away Team", club_opts)
    away_pos  = st.number_input(
        "Away League Position (0 = cup / unknown)",
        0, 30, 0, step=1,
        disabled=(is_cup_game == 1)
    )

# Resolve rank IDs
home_rank_id = int(
    clubs_df.loc[clubs_df['club_name'] == home_club, 'club_rank_id'].values[0]
) if home_club != '— Select Club —' else 0

away_rank_id = int(
    clubs_df.loc[clubs_df['club_name'] == away_club, 'club_rank_id'].values[0]
) if away_club != '— Select Club —' else 0

pos_diff = int(home_pos) - int(away_pos)

if home_club != '— Select Club —' and away_club != '— Select Club —':
    st.caption(f"Home rank ID: **{home_rank_id}** | Away rank ID: **{away_rank_id}** | "
               f"Position diff (home − away): **{pos_diff}**")

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 3 — Tactics
# ─────────────────────────────────────────────
st.markdown("### ⚙️ Tactics")
tf1, tf2 = st.columns(2)

with tf1:
    home_form = st.selectbox("🔵 Home Formation", list(FORMATIONS.keys()))
    hd, hm, ha = FORMATIONS[home_form]
    st.caption(f"GK · DEF×{hd} · MID×{hm} · ATT×{ha} = **11 players**")

with tf2:
    away_form = st.selectbox("🔴 Away Formation", list(FORMATIONS.keys()))
    ad, am, aa = FORMATIONS[away_form]
    st.caption(f"GK · DEF×{ad} · MID×{am} · ATT×{aa} = **11 players**")

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 4 — Lineup Selection
# ─────────────────────────────────────────────
hn_label = home_club if home_club != '— Select Club —' else 'Home Team'
an_label = away_club if away_club != '— Select Club —' else 'Away Team'

st.markdown("### 👥 Lineup Selection")
lc1, lc2 = st.columns(2)

home_selected = []
away_selected = []

# ── Home Lineup ──
with lc1:
    st.markdown(f"**🔵 {hn_label}**")

    st.markdown("🧤 *Goalkeeper*")
    p = st.selectbox("GK", player_opts, key="h_gk", label_visibility="collapsed")
    home_selected.append(p if p != '— Select Player —' else None)

    st.markdown("🛡️ *Defenders*")
    cols = st.columns(hd)
    for i in range(hd):
        with cols[i]:
            p = st.selectbox(f"DEF {i+1}", player_opts, key=f"h_d_{i}")
            home_selected.append(p if p != '— Select Player —' else None)

    st.markdown("⚙️ *Midfielders*")
    cols = st.columns(hm)
    for i in range(hm):
        with cols[i]:
            p = st.selectbox(f"MID {i+1}", player_opts, key=f"h_m_{i}")
            home_selected.append(p if p != '— Select Player —' else None)

    st.markdown("⚡ *Attackers*")
    cols = st.columns(ha)
    for i in range(ha):
        with cols[i]:
            p = st.selectbox(f"ATT {i+1}", player_opts, key=f"h_a_{i}")
            home_selected.append(p if p != '— Select Player —' else None)

# ── Away Lineup ──
with lc2:
    st.markdown(f"**🔴 {an_label}**")

    st.markdown("🧤 *Goalkeeper*")
    p = st.selectbox("GK", player_opts, key="a_gk", label_visibility="collapsed")
    away_selected.append(p if p != '— Select Player —' else None)

    st.markdown("🛡️ *Defenders*")
    cols = st.columns(ad)
    for i in range(ad):
        with cols[i]:
            p = st.selectbox(f"DEF {i+1}", player_opts, key=f"a_d_{i}")
            away_selected.append(p if p != '— Select Player —' else None)

    st.markdown("⚙️ *Midfielders*")
    cols = st.columns(am)
    for i in range(am):
        with cols[i]:
            p = st.selectbox(f"MID {i+1}", player_opts, key=f"a_m_{i}")
            away_selected.append(p if p != '— Select Player —' else None)

    st.markdown("⚡ *Attackers*")
    cols = st.columns(aa)
    for i in range(aa):
        with cols[i]:
            p = st.selectbox(f"ATT {i+1}", player_opts, key=f"a_a_{i}")
            away_selected.append(p if p != '— Select Player —' else None)

# ─────────────────────────────────────────────
# SECTION 5 — Live Pitch
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🟩 Live Pitch View")
st.caption("Updates automatically as you change formations and player selections above.")

pitch_fig = draw_pitch(
    home_form, away_form,
    home_selected, away_selected,
    hn_label, an_label
)
st.pyplot(pitch_fig, use_container_width=True)
plt.close(pitch_fig)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Compute All Features
# ─────────────────────────────────────────────
h_names = [p for p in home_selected if p]
a_names = [p for p in away_selected if p]

hs  = compute_stats(h_names, players_df)
as_ = compute_stats(a_names, players_df)

goals_diff   = hs['goals']        - as_['goals']
assists_diff = hs['assists']      - as_['assists']
yellow_diff  = hs['yellow_cards'] - as_['yellow_cards']
red_diff     = hs['red_card']     - as_['red_card']
age_diff     = hs['age']          - as_['age']
mval_diff    = hs['market_value'] - as_['market_value']

hist = get_historical_stats(home_rank_id, away_rank_id, games_df)

# Country / type / round one-hot
COUNTRY_COLS = {
    'France':      'country_France',
    'Germany':     'country_Germany',
    'Italy':       'country_Italy',
    'Netherlands': 'country_Netherlands',
    'Portugal':    'country_Portugal',
    'Spain':       'country_Spain',
    'Türkiye':     'country_Türkiye',
    'Europa':      'country_europa',
}
country_enc = {v: (1 if country_display == k else 0) for k, v in COUNTRY_COLS.items()}
type_enc = {
    'type_domestic_league':   1 if match_type == 'domestic_league'   else 0,
    'type_international_cup': 1 if match_type == 'international_cup' else 0,
    'type_other':             1 if match_type == 'other'             else 0,
}
round_enc = {
    'round_knockout': 1 if round_type == 'knockout' else 0,
    'round_league':   1 if round_type == 'league'   else 0,
    'round_rounds':   1 if round_type == 'rounds'   else 0,
}

# Expandable feature inspector
with st.expander("📊 Inspect All Computed Features"):
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("Goals Diff",        f"{goals_diff:.2f}")
        st.metric("Assists Diff",      f"{assists_diff:.2f}")
        st.metric("Yellow Cards Diff", f"{yellow_diff:.2f}")
    with e2:
        st.metric("Red Cards Diff",    f"{red_diff:.2f}")
        st.metric("Age Diff",          f"{age_diff:.2f}")
        st.metric("Market Value Diff", f"{mval_diff:,.0f}")
    with e3:
        st.metric("Position Diff",     f"{pos_diff}")
        st.metric("Home Draw Rate 5",  f"{hist['home_draw_rate_5']:.3f}")
        st.metric("Away Draw Rate 5",  f"{hist['away_draw_rate_5']:.3f}")
    with e4:
        st.metric("Home Win Rate 5",   f"{hist['home_win_rate_5']:.3f}")
        st.metric("Rank Diff (abs)",   f"{hist['rank_diff_abs']}")
        st.metric("Rank Closeness",    f"{hist['rank_closeness']:.3f}")

# ─────────────────────────────────────────────
# Build Payload — exact column order
# ─────────────────────────────────────────────
payload = {
    'home_rank_id':             home_rank_id,
    'away_rank_id':             away_rank_id,
    'season':                   int(season),
    'month':                    int(month),
    'dayofweek':                int(dayofweek),
    'home_defense':             hd,
    'home_midfield':            hm,
    'home_attack':              ha,
    'away_defense':             ad,
    'away_midfield':            am,
    'away_attack':              aa,
    'is_cup_game':              is_cup_game,
    'attendance':               attendance,
    'lineup_goals_diff':        goals_diff,
    'lineup_assists_diff':      assists_diff,
    'lineup_yellow_diff':       yellow_diff,
    'lineup_red_diff':          red_diff,
    'pos_diff':                 pos_diff,
    'lineup_age_diff':          age_diff,
    'lineup_market_value_diff': mval_diff,
    **hist,
    **country_enc,
    **type_enc,
    **round_enc,
}

# ─────────────────────────────────────────────
# SECTION 6 — Predict
# ─────────────────────────────────────────────
st.markdown("### 🤖 Prediction")

if st.button("🔮  Predict Match Result", use_container_width=True, type="primary"):

    if home_club == '— Select Club —' or away_club == '— Select Club —':
        st.error("⚠️  Please select both Home and Away teams first.")
    elif home_club == away_club:
        st.error("⚠️  Home and Away teams must be different!")
    else:
        with st.spinner("Running CatBoost model..."):
            try:
                res = requests.post(
                    "https://football-predictor-2bqr.onrender.com/predict",
                    json=payload,
                    timeout=10
                )

                if res.status_code == 200:
                    data  = res.json()
                    label = data['result_label']
                    probs = data['probabilities']

                    color_map = {
                        "Home Win": "#1565c0",
                        "Away Win": "#b71c1c",
                        "Draw":     "#e65100"
                    }
                    emoji_map = {
                        "Home Win": "🔵",
                        "Away Win": "🔴",
                        "Draw":     "🟡"
                    }

                    # Big result banner
                    st.markdown(f"""
                    <div style='
                        text-align:center;
                        background: linear-gradient(135deg,
                            {color_map[label]}cc, {color_map[label]}44);
                        border: 2px solid {color_map[label]};
                        padding: 35px 20px;
                        border-radius: 20px;
                        margin: 20px 0;
                    '>
                        <div style='font-size:4rem;'>{emoji_map[label]}</div>
                        <h1 style='color:white; font-size:2.8rem;
                                   margin:10px 0 5px 0; letter-spacing:3px;'>
                            {label.upper()}
                        </h1>
                        <p style='color:rgba(255,255,255,0.75); font-size:1.2rem; margin:0;'>
                            {hn_label}  vs  {an_label}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability metrics
                    pc1, pc2, pc3 = st.columns(3)
                    pc1.metric("🔵 Home Win", f"{probs['home_win']*100:.1f}%")
                    pc2.metric("🟡 Draw",     f"{probs['draw']*100:.1f}%")
                    pc3.metric("🔴 Away Win", f"{probs['away_win']*100:.1f}%")

                    # Stacked bar
                    fig2, ax2 = plt.subplots(figsize=(12, 1.4))
                    fig2.patch.set_facecolor('#060e1e')
                    ax2.set_facecolor('#060e1e')
                    vals_  = [probs['home_win'], probs['draw'], probs['away_win']]
                    cols_  = ['#1565c0', '#e65100', '#b71c1c']
                    left_  = 0
                    for v, c, lbl in zip(vals_, cols_,
                                          ['Home Win', 'Draw', 'Away Win']):
                        ax2.barh(0, v, left=left_, color=c, height=0.55,
                                 edgecolor='#060e1e', linewidth=2)
                        if v > 0.06:
                            ax2.text(left_ + v / 2, 0,
                                     f'{lbl}\n{v*100:.1f}%',
                                     ha='center', va='center',
                                     color='white', fontsize=10,
                                     fontweight='bold')
                        left_ += v
                    ax2.set_xlim(0, 1)
                    ax2.axis('off')
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

                else:
                    st.error(f"API Error {res.status_code}: {res.json()}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot reach the prediction API.  "
                    "Make sure `python app.py` is running in a separate terminal."
                )
            except Exception as e:
                st.error(f"Unexpected error: {e}")
