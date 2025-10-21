# app.py
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
    
    # Handle missing 'season' column
    if 'Season' not in matches.columns:
        matches['Season'] = pd.to_datetime(matches['date']).dt.year
    return deliveries, matches

deliveries, matches = load_data()

st.title("🏏 IPL Analytics Suite")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")
season = st.sidebar.selectbox("Select Season", sorted(matches["Season"].unique()))
team = st.sidebar.selectbox(
    "Select Team", sorted(set(matches["team1"].unique()) | set(matches["team2"].unique()))
)

# Filter matches for selected season + team
filtered_matches = matches[
    (matches["Season"] == season) &
    ((matches["team1"] == team) | (matches["team2"] == team))
]
match_ids = filtered_matches["id"].unique()
filtered_deliveries = deliveries[deliveries["match_id"].isin(match_ids)]

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Predict Player Runs",
    "🔮 Match Outcome Predictor",
    "⚔️ Head-to-Head Analysis",
    "📊 Player Performance (Batsman)",
    "🎯 Predict Player Wickets",
    "📊 Player Performance (Bowler)"
])

# -----------------------------
# Tab 1: Predict Player Runs
# -----------------------------
with tab1:
    st.subheader("🎯 Predict Player Runs")

    # Opponent selection for this tab
    opponent_list = sorted(set(matches["team1"].unique()) | set(matches["team2"].unique()) - {team})
    selected_opponent = st.selectbox("Select Opponent",opponent_list,key="opponent_runs")  

    # Filter deliveries for team + season + opponent
    filtered_matches_opponent = matches[
        (matches["Season"] == season) &
        (((matches["team1"] == team) & (matches["team2"] == selected_opponent)) |
         ((matches["team1"] == selected_opponent) & (matches["team2"] == team)))
    ]
    match_ids_opponent = filtered_matches_opponent["id"].unique()
    filtered_deliveries_opponent = deliveries[deliveries["match_id"].isin(match_ids_opponent)]
    available_players = sorted(filtered_deliveries_opponent["batsman"].unique())

    if available_players:
        batsman = st.selectbox("Select Player", available_players)
        balls_faced = st.slider("Expected Balls Faced", 1, 120, 20)
        strike_rate = st.slider("Expected Strike Rate", 50, 200, 120)

        if st.button("Predict Runs", key="btn_predict_runs"):
            predicted_runs = int((balls_faced * strike_rate) / 100)
            st.success(f"Predicted Runs for {batsman} vs {selected_opponent} in {season}: {predicted_runs}")
    else:
        st.info("No players available for the selected team, season, and opponent.")

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

    if st.button("Predict Winner", key="btn_predict_match"):
        predicted_winner = team1 if toss_winner == team1 else team2
        st.success(f"Predicted Winner: {predicted_winner}")

# -----------------------------
# Tab 3: Head-to-Head Analysis
# -----------------------------
with tab3:
    st.subheader("⚔️ Head-to-Head Analysis")

    opponent_h2h = st.selectbox(
        "Select Opponent",
        sorted(set(matches["team1"].unique()) | set(matches["team2"].unique())),
        key="h2h_opponent"
    )

    h2h = matches[
        (((matches["team1"] == team) & (matches["team2"] == opponent_h2h)) |
         ((matches["team1"] == opponent_h2h) & (matches["team2"] == team))) &
        (matches["Season"] == season)
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
    st.subheader("📊 Player Performance Analysis (Batsman)")
    available_players = sorted(filtered_deliveries["batsman"].unique())
    if available_players:
        player = st.selectbox("Select Player", available_players, key="perf_player")
        player_data = filtered_deliveries[filtered_deliveries["batsman"] == player]

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

        runs_per_match = player_data.groupby("match_id")["batsman_runs"].sum().reset_index()
        if not runs_per_match.empty:
            fig1 = px.line(runs_per_match, x="match_id", y="batsman_runs",
                           title=f"Runs per Match - {player}", markers=True)
            st.plotly_chart(fig1)

        merged = deliveries.merge(matches[['id', 'Season']], left_on='match_id', right_on='id')
        season_runs = merged[merged["batsman"] == player].groupby("Season")["batsman_runs"].sum().reset_index()
        if not season_runs.empty:
            fig2 = px.bar(season_runs, x="Season", y="batsman_runs",
                          title=f"Season-wise Runs - {player}")
            st.plotly_chart(fig2)
    else:
        st.info("No players available for the selected team and season.")

# -----------------------------
# Tab: Predict Player Wickets
# -----------------------------
with tab_bowler_predict:
    st.subheader("🎯 Predict Player Wickets")

    if available_bowlers:
        bowler = st.selectbox("Select Bowler", available_bowlers)
        opponent_list = sorted(set(matches["team1"].unique()) | set(matches["team2"].unique()))
        selected_opponent_wk = st.selectbox(
            "Select Opponent",
            opponent_list,
            key="opponent_wickets"
        )

        balls_bowled = st.slider("Expected Balls Bowled", 1, 120, 24)
        # Compute predicted wickets
        # (for example, use average wickets per match of this bowler in selected season)
        season_data = filtered_deliveries[
            (filtered_deliveries["bowler"] == bowler)
            & (filtered_deliveries["season"] == season)
        ]
        avg_wickets = season_data["player_dismissed"].notna().sum() / season_data["match_id"].nunique()
        predicted_wickets = balls_bowled / 6 * avg_wickets
        predicted_wickets = int(predicted_wickets + 0.5)  # round to nearest whole number

        if st.button("Predict Wickets"):
            st.success(f"Predicted Wickets for {bowler} vs {selected_opponent_wk}: {predicted_wickets}")
    else:
        st.info("No bowlers available for the selected team and season.")

# -----------------------------
# Tab 6: Player Performance (Bowler)
# -----------------------------
with tab6:
    st.subheader("📊 Player Performance Analysis (Bowler)")
    available_bowlers = sorted(filtered_deliveries["bowler"].unique())
    if available_bowlers:
        bowler = st.selectbox("Select Bowler", available_bowlers, key="perf_bowler")
        bowler_data = filtered_deliveries[filtered_deliveries["bowler"] == bowler]

        wicket_kinds = ['bowled','caught','lbw','stumped','caught and bowled','hit wicket']
        total_wickets = bowler_data['dismissal_kind'].isin(wicket_kinds).sum()
        balls_bowled = bowler_data.shape[0]
        matches_played = bowler_data['match_id'].nunique()
        wickets_per_match = round(total_wickets / matches_played, 2) if matches_played > 0 else 0

        st.markdown(f"""
        **Total Wickets:** {total_wickets}  
        **Balls Bowled:** {balls_bowled}  
        **Matches Played:** {matches_played}  
        **Wickets per Match:** {wickets_per_match}  
        """)

        wickets_per_match_graph = bowler_data.groupby('match_id')['dismissal_kind'].apply(
            lambda x: x.isin(wicket_kinds).sum()
        ).reset_index(name='wickets')
        if not wickets_per_match_graph.empty:
            fig1 = px.line(wickets_per_match_graph, x='match_id', y='wickets',
                           title=f"Wickets per Match - {bowler}", markers=True)
            st.plotly_chart(fig1)

        season_wickets = bowler_data.groupby('match_id').first().merge(matches[['id','Season']],
                                                                     left_on='match_id', right_on='id')
        season_wickets = season_wickets.groupby('Season')['dismissal_kind'].apply(
            lambda x: x.isin(wicket_kinds).sum()
        ).reset_index(name='wickets')
        if not season_wickets.empty:
            fig2 = px.bar(season_wickets, x='Season', y='wickets', title=f"Season-wise Wickets - {bowler}")
            st.plotly_chart(fig2)
    else:
        st.info("No bowlers available for the selected team and season.")
