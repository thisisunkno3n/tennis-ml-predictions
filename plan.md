Overall goal: finding out the probability of a player's odd to win the match against the opponent

This can be based on:
- Age difference (older oppponent, worse stamina)

- Tournament level: higher prestige tournaments (Grand Slams, Masters) attract better players and have higher stakes, affecting match dynamics and player motivation
    - Correlated with best of sets and tournament round

- Players's height: there's strong correlation that taller players tend to win more

- Ranking & ranking points: player's current ATP rank and ranking points (absolute values and differences) - lower rank/higher points indicate better skill and recent form

- Surface type: different court surfaces (Hard, Grass, Clay) favor different playing styles - some players perform significantly better on specific surfaces

- Tournament round: later rounds (QF, SF, F) have stronger competition and higher pressure, affecting match outcomes

- Best of sets: 3-set vs 5-set matches affect stamina requirements and match length, with 5-set matches (typically Grand Slams) favoring players with better endurance
    - correlated with age difference & ranking/ranking points

- Hand dominance: left-handed vs right-handed matchups create different tactical challenges - left-handed players are often more comfortable against right-handed opponents

- Month/Seasonality: time of year can affect player form, surface conditions, and tournament scheduling patterns



Feature engineering:
- remove Davis Cup as it's a team competition & not good indicator of ATP match outcomes

- Set each match into 2 rows for player's view
    - 1 row for winner = 1
    - 1 row for loser = 1
    this ensures that the model sees balanced win/loss examples from both player perspectives, creating a perfect 50/50 label distribution by design (not random). This allows the model to learn from both perspectives of each match and prevents bias toward always predicting from the "winner's" perspective.

- Remove player_ioc & opponent_ioc as it can cause bias:
    - if only 1 top player is from SUI, then the model may learn "SUI = wins"
    - cofoundng with other factors where if Spain has strong players, it would already be acounted for in rank/rank_points
    - if SUI (switzerland) dominates in the train like Federer, then we want to ensure this doesn't get carried over when validating & testing

- Remove (player_id & opponent_id) and (player_name & opponent_name):
    - direct identity features that allow the model to memorize specific players rather than learn generalizable patterns
    - if a player dominates in training data, the model may overfit to that player's ID rather than their actual skill attributes (rank, age, etc.)
    - prevents generalization to new players or players not seen in training

- Remove tourney_id & tourney_name:
    - allows model to memorize tournament-specific patterns rather than learn general tournament level effects
    - if a specific tournament (e.g., "Wimbledon") has certain characteristics, those should be captured by tourney_level and surface, not the tournament name
    - prevents overfitting to specific tournaments that may not exist in validation/test sets

- One-hot encode the following:
    - Surface: encoded as surface_Grass, surface_Hard (Clay is the base/dropped category)
    - Tournament level: encoded as tourney_level_F, tourney_level_G, tourney_level_M, tourney_level_O (one category dropped to avoid multicollinearity)
    - Note: round and player_hand/opponent_hand are currently NOT encoded (still categorical strings) and need encoding before model training

- Time-based train/val/test split: split by year (2013-2019 train, 2020-2021 val, 2022-2024 test) to ensure temporal generalization. Year is used for splitting but not as a model feature.

Model outcome: percentage in the form of a decimal, where 1.0 means the player will win 100%
