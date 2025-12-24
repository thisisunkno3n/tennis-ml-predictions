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

- Turn Players hands into binarys (similar to one-hot encode but create more columns): this ensures Left vs Right favourable is also visible

- One-hot encode the following:
    - Surface: encoded as surface_Grass, surface_Hard (Clay is the base/dropped category)
    - Tournament level: encoded as tourney_level_F, tourney_level_G, tourney_level_M, tourney_level_O (one category dropped to avoid multicollinearity)
    - Round: turn into round_to_remaining. There's no need to normalize based on tournament level because tree-based models will usually learn interactions automatically

- Time-based train/val/test split: split by year (2013-2019 train, 2020-2021 val, 2022-2024 test) to ensure temporal generalization. Year is used for splitting but not as a model feature.

Exploratory Data Analysis:
- load & basic info
- target variable analysis: should be 50/50 split
- missing values check
- feature distributions (key features only)
- verify train/val/test split validation are all 50/50 across splits
- confirm time-based split (train years < val years < test years)
- data leakeage check (verify identity columns removed)

Model Training
- Use a boosting model
- LightGBM > XGBoost bc it's faster (lower accurarcy though). Will use XGBoost if I use GPU

Validating Model:
First 10 binary predictions: [0 1 1 0 1 0 0 1 1 0]
First 10 probabilities: [0.43619152 0.53319805 0.60352292 0.36149893 0.63269947 0.33141845
 0.43724568 0.58860748 0.80917888 0.13962837]

Prediction #	Binary	Probability	Meaning
1	0	0.436	The "player" in row 1 is predicted to lose (43.6% win chance)
2	1	0.533	The "player" in row 2 is predicted to win (53.3% win chance)
-> these should add up to ~100% win chance
-> symmetric relationships

3	1	0.604	The "player" in row 3 is predicted to win (60.4% win chance)
4	0	0.362	The "player" in row 4 is predicted to lose (36.2% win chance)
5	1	0.633	The "player" in row 5 is predicted to win (63.3% win chance)

GOSS (gradient-based one-side sampling) is already calculated

- initial accurary on validation dataset: 63.89% 
- 64.11% by changing to learning rate = 0.05 and n_estimator = 200
- slightly overfits
- changed baseline court to Hard instead of Clay as there's more data and clay & grass will adjust based on it
    - accuray became 64.15%
- feature importance: difference features dominator the top 6 for importance. 

- AUC-ROC: "Area Under the ROC Curve"
    - how well the model can separate between wins and losses
- AUC-ROC vs Accuracy:
    - Accuracy goes off from certain threshold (i.e. 0.5): "what % did we get right"
    - AUC-ROC uses all possible thresholds: how well can we rank predictions
    - better for binary classficiation

- next time try to have complete baseline model and modify hyperparameters on a different cell




Desired model outcome: 
- percentage in the form of a decimal, where 1.0 means the player will win 100%
- accurary rate in %
 