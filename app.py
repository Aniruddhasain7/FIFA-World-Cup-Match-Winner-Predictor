import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fifa_match_model.pkl")
le_team = joblib.load("team_encoder.pkl")
le_result = joblib.load("result_encoder.pkl")

ranking = pd.read_csv("fifa_ranking_2022-10-06.csv")
ranking = ranking[["team", "rank", "points"]]

qualified_teams = [
    "Argentina","Brazil","France","Germany","Spain",
    "England","Portugal","Netherlands","Belgium",
    "Croatia","USA","Mexico","Canada",
    "Japan","South Korea","Morocco"
]

teams = sorted(qualified_teams)

st.set_page_config(page_title="FIFA Match Winner Predictor", layout="centered")

st.title("⚽ FIFA Match Winner Predictor")

team1 = st.selectbox("⚽ Select Team 1", teams)
team2 = st.selectbox("⚽ Select Team 2", teams)

def predict_match(team1, team2):

    rank1 = ranking.loc[ranking["team"] == team1, "rank"].values[0]
    rank2 = ranking.loc[ranking["team"] == team2, "rank"].values[0]

    points1 = ranking.loc[ranking["team"] == team1, "points"].values[0]
    points2 = ranking.loc[ranking["team"] == team2, "points"].values[0]

    if rank1 <= rank2:
        stronger = team1
        weaker = team2
        strong_rank = rank1
        weak_rank = rank2
        strong_points = points1
        weak_points = points2
    else:
        stronger = team2
        weaker = team1
        strong_rank = rank2
        weak_rank = rank1
        strong_points = points2
        weak_points = points1

    rank_diff = abs(strong_rank - weak_rank)
    points_diff = abs(strong_points - weak_points)

    strong_enc = le_team.transform([stronger])[0]
    weak_enc = le_team.transform([weaker])[0]

    features = pd.DataFrame([{
        "home_encoded": strong_enc,
        "away_encoded": weak_enc,
        "rank_diff": rank_diff,
        "points_diff": points_diff
    }])

    probabilities = model.predict_proba(features)[0]

    prob_dict = dict(zip(le_result.classes_, probabilities))

    strong_prob = prob_dict.get("HomeWin", 0) * 100
    draw_prob = prob_dict.get("Draw", 0) * 100
    weak_prob = prob_dict.get("AwayWin", 0) * 100

    if stronger == team1:
        team1_prob = strong_prob
        team2_prob = weak_prob
    else:
        team1_prob = weak_prob
        team2_prob = strong_prob

    return team1_prob, draw_prob, team2_prob


if st.button("🔮 Predict Match Result"):

    if team1 == team2:
        st.warning("Please select two different teams.")
    else:
        team1_prob, draw_prob, team2_prob = predict_match(team1, team2)

        st.subheader("📊 Match Probability")

        st.write(f"⚽ {team1}: {team1_prob:.2f}%")
        st.progress(team1_prob / 100)

        st.write(f"🤝 Draw: {draw_prob:.2f}%")
        st.progress(draw_prob / 100)

        st.write(f"⚽ {team2}: {team2_prob:.2f}%")
        st.progress(team2_prob / 100)

        if team1_prob > team2_prob and team1_prob > draw_prob:
            winner = team1
        elif team2_prob > team1_prob and team2_prob > draw_prob:
            winner = team2
        else:
            winner = "Draw"

        st.success(f"🏆 Most Likely Winner: {winner}")
