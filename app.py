import streamlit as st
import pandas as pd
import joblib
import textwrap

def render_html(html_str):
    cleaned = "\n".join(line.strip() for line in html_str.split("\n"))
    st.markdown(cleaned, unsafe_allow_html=True)

artifacts = joblib.load("fifa_model.joblib")
model = artifacts["model"]
le_team = artifacts["le_team"]
le_result = artifacts["le_result"]
team_stats = artifacts.get("team_stats", {})

_encoded_teams = set(le_team.classes_)

_name_map = {
    "South Korea":   "Korea Republic",
    "North Korea":   "Korea DPR",
    "Iran":          "IR Iran",
    "Turkey":        "T\uFFFDrkiye",
    "Ivory Coast":   "C\uFFFDte d'Ivoire",
    "Ireland":       "Republic of Ireland",
}
_reverse_map = {v: k for k, v in _name_map.items()}


teams = sorted(
    _reverse_map.get(t, t)
    for t in _encoded_teams
    if t not in {"Cuba", "Angola", "Togo", "Haiti", "Kuwait",
                 "Korea DPR", "El Salvador", "Trinidad and Tobago",
                 "United Arab Emirates", "Israel", "China PR"}
)


_flag_map = {
    "Afghanistan": "af", "Albania": "al", "Algeria": "dz", "Angola": "ao",
    "Argentina": "ar", "Armenia": "am", "Australia": "au", "Austria": "at",
    "Azerbaijan": "az", "Bahrain": "bh", "Bangladesh": "bd", "Belarus": "by",
    "Belgium": "be", "Benin": "bj", "Bolivia": "bo",
    "Bosnia and Herzegovina": "ba", "Botswana": "bw", "Brazil": "br",
    "Bulgaria": "bg", "Burkina Faso": "bf", "Cameroon": "cm", "Canada": "ca",
    "Chile": "cl", "Colombia": "co", "Costa Rica": "cr", "Croatia": "hr",
    "Czech Republic": "cz", "Denmark": "dk", "DR Congo": "cd", "Ecuador": "ec",
    "Egypt": "eg", "England": "gb-eng", "Ethiopia": "et", "Finland": "fi",
    "France": "fr", "Georgia": "ge", "Germany": "de", "Ghana": "gh",
    "Greece": "gr", "Guatemala": "gt", "Guinea": "gn", "Honduras": "hn",
    "Hungary": "hu", "India": "in", "Indonesia": "id", "Iran": "ir",
    "Iraq": "iq", "Ireland": "ie", "Italy": "it", "Ivory Coast": "ci",
    "Jamaica": "jm", "Japan": "jp", "Jordan": "jo", "Kenya": "ke",
    "Kuwait": "kw", "Libya": "ly", "Mali": "ml", "Mexico": "mx",
    "Morocco": "ma", "Mozambique": "mz", "Netherlands": "nl",
    "New Zealand": "nz", "Nigeria": "ng", "North Korea": "kp",
    "Norway": "no", "Oman": "om", "Panama": "pa", "Paraguay": "py",
    "Peru": "pe", "Poland": "pl", "Portugal": "pt", "Qatar": "qa",
    "Romania": "ro", "Russia": "ru", "Saudi Arabia": "sa", "Scotland": "gb-sct",
    "Senegal": "sn", "Serbia": "rs", "Slovakia": "sk", "Slovenia": "si",
    "South Africa": "za", "South Korea": "kr", "Spain": "es", "Sudan": "sd",
    "Sweden": "se", "Switzerland": "ch", "Syria": "sy", "Tanzania": "tz",
    "Tunisia": "tn", "Turkey": "tr", "Uganda": "ug", "Ukraine": "ua",
    "Uruguay": "uy", "USA": "us", "United States": "us", "Venezuela": "ve",
    "Vietnam": "vn", "Wales": "gb-wls", "Zambia": "zm", "Zimbabwe": "zw",
    "Korea Republic": "kr", "IR Iran": "ir", "Republic of Ireland": "ie",
}


def get_flag_url(team):
    code = _flag_map.get(team, "").lower()
    if code:
        return f"https://flagcdn.com/w160/{code}.png"
    return ""

st.set_page_config(page_title="FIFA Match Predictor", page_icon="⚽", layout="centered")


render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050a14 !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #0d1f3c 0%, #050a14 50%),
                radial-gradient(ellipse at 80% 80%, #1a0a2e 0%, transparent 60%) !important;
    background-blend-mode: screen !important;
    min-height: 100vh;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stVerticalBlock"] { position: relative; z-index: 1; }
[data-testid="stHeader"] { background: transparent !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: linear-gradient(#00d4ff, #7b2fff); border-radius: 3px; }

.hero-header {
    text-align: center;
    padding: 3rem 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 50px;
    padding: 6px 18px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #00d4ff;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}
.hero-badge::before { content: "●"; font-size: 0.5rem; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #00d4ff 40%, #7b2fff 80%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.75rem;
}
.hero-subtitle {
    font-size: 1rem;
    font-weight: 400;
    color: rgba(255,255,255,0.45);
    letter-spacing: 0.01em;
}

.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(20px);
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(0,212,255,0.2);
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,212,255,0.08), inset 0 1px 0 rgba(255,255,255,0.1);
}
.card-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(0,212,255,0.7);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,212,255,0.2), transparent);
}

.vs-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin: 1rem 0;
    color: rgba(255,255,255,0.15);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.2em;
}
.vs-divider::before, .vs-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}
.vs-badge {
    background: linear-gradient(135deg, #ff6b35, #ff3366);
    color: white;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    box-shadow: 0 0 20px rgba(255,107,53,0.4);
}

.stat-row {
    display: flex;
    gap: 12px;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.55);
    flex: 1;
    min-width: 100px;
    text-align: center;
}
.stat-pill strong {
    display: block;
    color: #00d4ff;
    font-size: 1.1rem;
    font-weight: 700;
}

.prob-section { margin-top: 0.5rem; }
.prob-row { margin-bottom: 1.4rem; }
.prob-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.prob-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: rgba(255,255,255,0.85);
    display: flex;
    align-items: center;
    gap: 8px;
}
.prob-value {
    font-size: 1.1rem;
    font-weight: 800;
    color: white;
}
.prob-bar-track {
    height: 10px;
    background: rgba(255,255,255,0.06);
    border-radius: 50px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}
.prob-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    position: relative;
}
.prob-bar-fill::after {
    content: "";
    position: absolute;
    top: 0; right: 0;
    width: 20px; height: 100%;
    background: rgba(255,255,255,0.3);
    border-radius: 50px;
    filter: blur(4px);
}
.bar-team1 { background: linear-gradient(90deg, #00d4ff, #0099ff); box-shadow: 0 0 12px rgba(0,212,255,0.5); }
.bar-draw  { background: linear-gradient(90deg, #f59e0b, #fbbf24); box-shadow: 0 0 12px rgba(245,158,11,0.5); }
.bar-team2 { background: linear-gradient(90deg, #7b2fff, #a855f7); box-shadow: 0 0 12px rgba(123,47,255,0.5); }

.winner-banner {
    background: linear-gradient(135deg, rgba(0,212,255,0.12) 0%, rgba(123,47,255,0.12) 100%);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-top: 1.5rem;
}
.winner-banner::before {
    content: "";
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(0,212,255,0.05) 0%, transparent 60%);
    animation: rotate 8s linear infinite;
}
@keyframes rotate { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }

.winner-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(0,212,255,0.6);
    margin-bottom: 0.5rem;
}
.winner-name {
    font-size: 2.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ffffff, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    z-index: 1;
}
.winner-trophy {
    font-size: 3rem;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
    filter: drop-shadow(0 0 20px rgba(255,215,0,0.6));
}
.draw-name {
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    z-index: 1;
}

.custom-warning {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: rgba(245,158,11,0.9);
    font-size: 0.9rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 10px;
}

[data-testid="stSelectbox"] label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.4) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    color: white !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(0,212,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.08) !important;
}

[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00d4ff 0%, #0099ff 50%, #7b2fff 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    transition: opacity 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.3), 0 0 0 1px rgba(255,255,255,0.08) !important;
    margin-top: 0.5rem !important;
    text-transform: none !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,212,255,0.45) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stSpinner"] { color: #00d4ff !important; }


@media (max-width: 640px) {
    .hero-header {
        padding: 2rem 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .hero-title-row {
        flex-direction: column !important;
        gap: 0.5rem !important;
    }
    .hero-ball {
        font-size: 2.2rem !important;
    }
    .hero-title {
        font-size: clamp(1.7rem, 8vw, 2.5rem);
        text-align: center;
    }
    .hero-subtitle {
        font-size: 0.88rem;
    }
    .glass-card {
        padding: 1.25rem 1rem;
        border-radius: 16px;
    }
    .flag-matchup {
        flex-direction: column !important;
        gap: 1rem !important;
    }
    .flag-matchup .vs-col {
        padding: 0 !important;
    }
    .flag-img {
        width: 72px !important;
    }
    .prob-label { font-size: 0.82rem; }
    .prob-value { font-size: 0.95rem; }
    .winner-name { font-size: 1.8rem; }
    .draw-name   { font-size: 1.5rem; }
    .winner-trophy { font-size: 2.2rem; }
    [data-testid="stColumns"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > * {
        min-width: 100% !important;
        width: 100% !important;
    }
}

</style>
""")


render_html("""
<div class="hero-header">
    <div class="hero-badge">⚽ AI-Powered Predictor</div>
    <div class="hero-title-row" style="display:flex; align-items:center; justify-content:center; gap:1rem; flex-wrap:wrap; margin-bottom:0.75rem;">
        <span class="hero-ball" style="font-size:3rem; line-height:1; filter:drop-shadow(0 0 18px rgba(0,212,255,0.5));">⚽</span>
        <div class="hero-title" style="margin-bottom:0;">FIFA Match Predictor</div>
    </div>
    <div class="hero-subtitle">Powered by Machine Learning &amp; Historical World Cup Data</div>
</div>
""")


def _resolve(name):
    return _name_map.get(name, name)



def get_team_stats(team):
    internal = _resolve(team)
    stats = team_stats.get(internal, {"win_rate": 0, "avg_goals_scored": 0, "total_matches": 0})
    wr = f"{stats.get('win_rate', 0)*100:.1f}%"
    goals = f"{stats.get('avg_goals_scored', 0):.2f}"
    return (wr, goals)



col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", teams, key="team1")
with col2:
    team2 = st.selectbox("Team 2", teams, index=1, key="team2")


flag1 = get_flag_url(team1)
flag2 = get_flag_url(team2)
flag1_img = f'<img class="flag-img" src="{flag1}" alt="{team1}" style="width:96px; height:auto; border-radius:8px; box-shadow:0 4px 24px rgba(0,212,255,0.25); margin-bottom:0.65rem;"/>' if flag1 else "\U0001f3f3\ufe0f"
flag2_img = f'<img class="flag-img" src="{flag2}" alt="{team2}" style="width:96px; height:auto; border-radius:8px; box-shadow:0 4px 24px rgba(123,47,255,0.25); margin-bottom:0.65rem;"/>' if flag2 else "\U0001f3f3\ufe0f"

render_html(f"""
<div class="glass-card" style="padding:1.75rem 1.5rem; margin-top:1rem;">
  <div class="flag-matchup" style="display:flex; gap:1rem; align-items:center;">
    <div style="flex:1; text-align:center; display:flex; flex-direction:column; align-items:center; gap:0.5rem;">
      {flag1_img}
      <div style="font-size:1.05rem; font-weight:700; color:white; letter-spacing:-0.01em;">{team1}</div>
    </div>
    <div class="vs-col" style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:0 0.25rem;">
      <div class="vs-badge">VS</div>
    </div>
    <div style="flex:1; text-align:center; display:flex; flex-direction:column; align-items:center; gap:0.5rem;">
      {flag2_img}
      <div style="font-size:1.05rem; font-weight:700; color:white; letter-spacing:-0.01em;">{team2}</div>
    </div>
  </div>
</div>
""")



predict_clicked = st.button("⚡ Predict Match Outcome")


def predict_match(t1, t2):
    i1, i2 = _resolve(t1), _resolve(t2)

    enc1 = le_team.transform([i1])[0]
    enc2 = le_team.transform([i2])[0]

    s1 = team_stats.get(i1, {"win_rate": 0.33, "avg_goals_scored": 1.0})
    s2 = team_stats.get(i2, {"win_rate": 0.33, "avg_goals_scored": 1.0})

    win_rate_diff = s1["win_rate"] - s2["win_rate"]
    avg_goals_diff = s1["avg_goals_scored"] - s2["avg_goals_scored"]

    features = pd.DataFrame([{
        "home_encoded": enc1,
        "away_encoded": enc2,
        "win_rate_diff": win_rate_diff,
        "avg_goals_diff": avg_goals_diff,
        "is_home_host": 0,
        "is_away_host": 0
    }])

    probabilities = model.predict_proba(features)[0]
    prob_dict = dict(zip(le_result.classes_, probabilities))

    t1_prob   = prob_dict.get("HomeWin", 0) * 100
    draw_prob = prob_dict.get("Draw",    0) * 100
    t2_prob   = prob_dict.get("AwayWin", 0) * 100

    return t1_prob, draw_prob, t2_prob



if predict_clicked:
    if team1 == team2:
        render_html("""
        <div class="custom-warning">
            ⚠️ &nbsp;Please select two <strong>different</strong> teams to predict a match outcome.
        </div>
        """)
    else:
        t1_prob, draw_prob, t2_prob = predict_match(team1, team2)

        if t1_prob > t2_prob and t1_prob > draw_prob:
            winner = team1
        elif t2_prob > t1_prob and t2_prob > draw_prob:
            winner = team2
        else:
            winner = "Draw"

        render_html(f"""
        <div class="glass-card">
          <div class="card-label">📊 Match Probability Breakdown</div>
          <div class="prob-section">

            <div class="prob-row">
              <div class="prob-header">
                <span class="prob-label">⚽ {team1}</span>
                <span class="prob-value">{t1_prob:.1f}%</span>
              </div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill bar-team1" style="width:{t1_prob}%"></div>
              </div>
            </div>

            <div class="prob-row">
              <div class="prob-header">
                <span class="prob-label">🤝 Draw</span>
                <span class="prob-value">{draw_prob:.1f}%</span>
              </div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill bar-draw" style="width:{draw_prob}%"></div>
              </div>
            </div>

            <div class="prob-row" style="margin-bottom:0">
              <div class="prob-header">
                <span class="prob-label">⚽ {team2}</span>
                <span class="prob-value">{t2_prob:.1f}%</span>
              </div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill bar-team2" style="width:{t2_prob}%"></div>
              </div>
            </div>

          </div>
        </div>
        """)

        if winner == "Draw":
            render_html(f"""
            <div class="winner-banner">
              <div class="winner-trophy">🤝</div>
              <div class="winner-label">Predicted Outcome</div>
              <div class="draw-name">It's a Draw!</div>
              <div style="color:rgba(255,255,255,0.4); font-size:0.85rem; margin-top:0.5rem; position:relative; z-index:1;">
                Both teams are evenly matched
              </div>
            </div>
            """)
        else:
            render_html(f"""
            <div class="winner-banner">
              <div class="winner-trophy">🏆</div>
              <div class="winner-label">Predicted Winner</div>
              <div class="winner-name">{winner}</div>
              <div style="color:rgba(255,255,255,0.4); font-size:0.85rem; margin-top:0.5rem; position:relative; z-index:1;">
                Based on historical World Cup match data
              </div>
            </div>
            """)


render_html("""
<div style="text-align:center; padding: 2.5rem 0 1rem; color: rgba(255,255,255,0.15); font-size:0.78rem; letter-spacing:0.05em;">
    FIFA WORLD CUP · MATCH PREDICTOR · BUILT WITH MACHINE LEARNING
</div>
""")
