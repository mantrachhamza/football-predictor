from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model & scaler once at startup
model  = joblib.load("catboost_model.pkl")
scaler = joblib.load("scaler.pkl")

DRAW_BOOST  = 1.5
DRAW_IDX    = 1
reverse_map = {0: -1, 1: 0, 2: 1}
result_map  = {-1: "Away Win", 0: "Draw", 1: "Home Win"}

# These must match EXACTLY the columns you trained on
FEATURE_COLS = [
    'home_rank_id', 'away_rank_id', 'season', 'month', 'dayofweek',
    'home_defense', 'home_midfield', 'home_attack',
    'away_defense', 'away_midfield', 'away_attack',
    'is_cup_game', 'attendance',
    'lineup_goals_diff', 'lineup_assists_diff',
    'lineup_yellow_diff', 'lineup_red_diff',
    'pos_diff', 'lineup_age_diff', 'lineup_market_value_diff',
    'home_draw_rate_5', 'away_draw_rate_5', 'rank_diff_abs',
    'rank_closeness', 'home_win_rate_5',
    'country_France', 'country_Germany', 'country_Italy',
    'country_Netherlands', 'country_Portugal', 'country_Spain',
    'country_Trkiye', 'country_europa',
    'type_domestic_league', 'type_international_cup', 'type_other',
    'round_knockout', 'round_league', 'round_rounds'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Build input row from JSON
        row = pd.DataFrame([{col: data.get(col, 0) for col in FEATURE_COLS}])

        # Scale → predict with draw boost
        row_scaled = scaler.transform(row)
        proba      = model.predict_proba(row_scaled).copy()
        proba[:, DRAW_IDX] *= DRAW_BOOST
        pred_mapped = int(np.argmax(proba, axis=1)[0])
        prediction  = reverse_map[pred_mapped]

        return jsonify({
            "prediction":   prediction,
            "result_label": result_map[prediction],
            "probabilities": {
                "away_win": round(float(proba[0][0]), 3),
                "draw":     round(float(proba[0][1]), 3),
                "home_win": round(float(proba[0][2]), 3),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True)