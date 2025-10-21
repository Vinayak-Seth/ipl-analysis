import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Load datasets
# -----------------------------
@st.cache_data
def load_data():
    deliveries = pd.read_csv("data/deliveries.csv")
    matches = pd.read_csv("data/matches.csv")

    # Handle missing 'season' column (derive from 'date' if necessary)
    if "Season" not in matches.columns:
        if "date" in matches.columns:
            matches["Season"] = pd.to_datetime(matches["date"], errors="coerce").dt.year
        else:
            matches["Season"] = 0  # default if date not available

    return deliveries, matches

deliveries, matches = load_data()

st.title("🏏 IPL Analytics Suite")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")
season = st.sidebar.selectbox("Select Season", sorted(matches["Season"].dropna().unique()))
team = st.sidebar.selectbox(
    "Select Team", sorted(set(matches["team1"].dropna().unique()) | set(matches["team2"].dropna().unique()))
)

# Filter matches for selected season and team
filtered_matches = matches[
    (matches["Season"] == season) &
    ((matches["team1"] == team) | (matches["team2"] == team))
]

match_ids = filtered_matches["id"].unique()
filtered_deliveries = deliveries[deliveries["match_id"].isin(match_ids)]

# Available batsmen and bowlers for selected team+season
available_batsmen = sorted(filtered_deliveries[filtered_deliveries["batting_team"] == team]["batsman"].unique())
available_bowlers = sorted(filtered_deliveries[filtered_deliveries["bowling_team"] == team]["bowler"].unique())

# -----------------------------
# Tabs for features
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Predict Player Runs",
    "🔮 Match Outcome Predictor",
    "⚔️ Head-to-Head Analysis",
    "📊 Player Performance",
    "🎯 Predict Player Wickets",
    "📊 Bowler Performance"
])

# -----------------------------
# Tab 1: Predict Player Runs
# -----------------------------
with tab1:
    st.subheader("🎯 Predict Player Runs")

    if available_batsmen:
        batsman = st.selectbox("Select Player", available_batsmen, key="batsman_tab1")
        balls_faced = st.slider("Expected Balls Faced", 1, 60, 20)
        strike_rate = st.slider("Expected Strike Rate", 50, 200, 120)

        if st.button("Predict Runs", key="btn_predict_runs"):
            predicted_runs = int((balls_faced * strike_rate) / 100)
            st.success(f"Predicted Runs for {batsman}: {predicted_runs}")
    else:
        st.info("No batsmen available for the selected team and season.")

# -----------------------------
# Tab 2: Match Outcome Predictor
# -----------------------------
with tab2:
    st.subheader("🔮 Match Outcome Predictor")

    team1 = st.selectbox("Select Team 1", matches["team1"].unique(), key="team1_tab2")
    team2 = st.selectbox("Select Team 2", matches["team2"].unique(), key="team2_tab2")
    venue = st.selectbox("Select Venue", matches["venue"].unique(), key="venue_tab2")
    toss_winner = st.selectbox("Toss Winner", [team1, team2], key="toss_tab2")
    toss_decision = st.radio("Toss Decision", ["bat", "field"], key="toss_decision_tab2")

    if st.button("Predict Winner", key="btn_predict_winner"):
        # Dummy logic (replace with ML model if desired)
        predicted_winner = team1 if toss_winner == team1 else team2
        st.success(f"Predicted Winner: {predicted_winner}")

# -----------------------------
# Tab 3: Head-to-Head Analysis
# -----------------------------
with tab3:
    st.subheader("⚔️ Head-to-Head Analysis")

    opponent = st.selectbox(
        "Select Opponent",
        sorted(set(matches["team1"].unique()) | set(matches["team2"].unique())),
        key="opponent_tab3"
    )

    h2h = matches[
        (
            ((matches["team1"] == team) & (matches["team2"] == opponent)) |
            ((matches["team1"] == opponent) & (matches["team2"] == team))
        )
        & (matches["Season"] == season)
    ]

    st.write(f"Total Matches Played: {h2h.shape[0]}")
    if not h2h.empty:
        st.bar_chart(h2h["winner"].value_counts())
    else:
        st.info("No matches found for this matchup in the selected season.")

# -----------------------------
# Tab 4: Player Performance (Batsman)
# -----------------------------
with tab4:
    st.subheader("📊 Player Performance Analysis")

    if available_batsmen:
        player = st.selectbox("Select Player", available_batsmen, key="perf_player")
        player_data = filtered_deliveries[filtered_deliveries["batsman"] == player]

        # Summary stats
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

        # Runs per match
        runs_per_match = player_data.groupby("match_id")["batsman_runs"].sum().reset_index()
        if not runs_per_match.empty:
            fig1 = px.line(runs_per_match, x="match_id", y="batsman_runs",
                           title=f"Runs per Match - {player}", markers=True)
            st.plotly_chart(fig1)

        # Season-wise runs
        merged = deliveries.merge(matches[['id','Season']], left_on='match_id', right_on='id')
        season_runs = merged[merged['batsman']==player].groupby('Season')['batsman_runs'].sum().reset_index()
        if not season_runs.empty:
            fig2 = px.bar(season_runs, x='Season', y='batsman_runs', title=f"Season-wise Runs - {player}")
            st.plotly_chart(fig2)
    else:
        st.info("No batsmen available for the selected team and season.")

# -----------------------------
# Tab 5: Predict Player Wickets
# -----------------------------
with tab5:
    st.subheader("🎯 Predict Player Wickets")

    if available_bowlers:
        bowler = st.selectbox("Select Bowler", available_bowlers, key="bowler_tab5")
        balls_bowled = st.slider("Expected Balls Bowled", 1, 120, 24)

        if st.button("Predict Wickets", key="btn_predict_wickets"):
            predicted_wickets = round(balls_bowled / 6 * 0.3, 1)
            st.success(f"Predicted Wickets for {bowler}: {predicted_wickets}")
    else:
        st.info("No bowlers available for the selected team and season.")

# -----------------------------
# Tab 6: Bowler Performance
# -----------------------------
with tab6:
    st.subheader("📊 Bowler Performance Analysis")

    if available_bowlers:
        bowler = st.selectbox("Select Bowler", available_bowlers, key="bowler_tab6")
        bowler_data = filtered_deliveries[filtered_deliveries["bowler"] == bowler]

        wicket_kinds = ['bowled','caught','lbw','stumped','caught and bowled','hit wicket']
        total_wickets = bowler_data['dismissal_kind'].isin(wicket_kinds).sum()
        balls_bowled = bowler_data.shape[0]
        matches_played = bowler_data['match_id'].nunique()
        avg_wickets = round(total_wickets / matches_played, 2) if matches_played > 0 else 0

        st.markdown(f"""
        **Total Wickets:** {total_wickets}  
        **Balls Bowled:** {balls_bowled}  
        **Matches Played:** {matches_played}  
        **Average Wickets per Match:** {avg_wickets}  
        """)

        # Wickets per match
        wickets_per_match = bowler_data.groupby('match_id')['dismissal_kind'].apply(lambda x: x.isin(wicket_kinds).sum()).reset_index()
        if not wickets_per_match.empty:
            fig1 = px.line(wickets_per_match, x='match_id', y='dismissal_kind',
                           title=f"Wickets per Match - {bowler}", markers=True)
            st.plotly_chart(fig1)

        # Season-wise wickets
        merged = bowler_data.merge(matches[['id','Season']], left_on='match_id', right_on='
