import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import numpy as np

# -----------------------------
# Load datasets
# -----------------------------
@st.cache_data
def load_data():
    deliveries = pd.read_csv("data/deliveries.csv")
    matches = pd.read_csv("data/matches.csv")
    return deliveries, matches

deliveries, matches = load_data()

st.title("🏏 IPL Analytics Suite")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")
season = st.sidebar.selectbox("Select Season", sorted(matches["season"].unique()))
team = st.sidebar.selectbox("Select Team", sorted(set(matches["team1"].unique()) | set(matches["team2"].unique())))

# -----------------------------
# Tabs for features
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predict Player Runs",
    "🔮 Match Outcome Predictor",
    "⚔️ Head-to-Head Analysis",
    "📊 Player Performance"
])

# -----------------------------
# Tab 1: Predict Player Runs
# -----------------------------
with tab1:
    st.subheader("🎯 Predict Player Runs")

    batsman = st.selectbox("Select Player", deliveries["batsman"].unique())
    balls_faced = st.slider("Expected Balls Faced", 1, 60, 20)
    strike_rate = st.slider("Expected Strike Rate", 50, 200, 120)

    if st.button("Predict Runs"):
        # Simple regression-style formula (replace with trained model if desired)
        predicted_runs = int((balls_faced * strike_rate) / 100)
        st.success(f"Predicted Runs for {batsman}: {predicted_runs}")

# -----------------------------
# Tab 2: Match Outcome Predictor
# -----------------------------
with tab2:
    st.subheader("🔮 Match Outcome Predictor")

    team1 = st.selectbox("Select Team 1", matches["team1"].unique())
    team2 = st.selectbox("Select Team 2", matches["team2"].unique())
    venue = st.selectbox("Select Venue", matches["venue"].unique())
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
    toss_decision = st.radio("Toss Decision", ["bat", "field"])

    if st.button("Predict Winner"):
        # Dummy logic (replace with trained classifier if you want)
        predicted_winner = team1 if toss_winner == team1 else team2
        st.success(f"Predicted Winner: {predicted_winner}")

# -----------------------------
# Tab 3: Head-to-Head Analysis
# -----------------------------
with tab3:
    st.subheader("⚔️ Head-to-Head Analysis")

    opponent = st.selectbox("Select Opponent", sorted(set(matches["team1"].unique()) | set(matches["team2"].unique())))
    h2h = matches[((matches["team1"] == team) & (matches["team2"] == opponent)) |
                  ((matches["team1"] == opponent) & (matches["team2"] == team)) &
                  (matches["season"] == season)]

    st.write(f"Total Matches Played: {h2h.shape[0]}")
    if not h2h.empty:
        st.bar_chart(h2h["winner"].value_counts())
    else:
        st.info("No matches found for this matchup in the selected season.")

# -----------------------------
# Tab 4: Player Performance
# -----------------------------
with tab4:
    st.subheader("📊 Player Performance Analysis")

    player = st.selectbox("Select Player", deliveries["batsman"].unique(), key="perf_player")
    player_data = deliveries[deliveries["batsman"] == player]

    # Player summary stats
    total_runs = player_data["batsman_runs"].sum()
    balls_faced = player_data.shape[0]
    strike_rate = round((total_runs / balls_faced) * 100, 2) if balls_faced > 0 else 0
    matches_played = player_data["match_id"].nunique()
    avg_runs = round(total_runs / matches_played, 2) if matches_played > 0 else 0

    st.markdown(f"""
    **Total Runs:** {total_runs}  
    **Balls Faced:** {balls_faced}  
    **Strike Rate:** {strike_rate}  
    **Matches Played:** {matches_played}  
    **Average Runs per Match:** {avg_runs}  
    """)

    # Match-wise performance trend
    runs_per_match = player_data.groupby("match_id")["batsman_runs"].sum().reset_index()
    if not runs_per_match.empty:
        fig1 = px.line(runs_per_match, x="match_id", y="batsman_runs",
                       title=f"Runs per Match - {player}", markers=True)
        st.plotly_chart(fig1)

    # Season-wise performance trend
    merged = deliveries.merge(matches[["id", "season"]], left_on="match_id", right_on="id")
    season_runs = merged[merged["batsman"] == player].groupby("season")["batsman_runs"].sum().reset_index()
    if not season_runs.empty:
        fig2 = px.bar(season_runs, x="season", y="batsman_runs",
                      title=f"Season-wise Runs - {player}")
        st.plotly_chart(fig2)
