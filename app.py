from flask import Flask, request, jsonify
import json
from model4 import TravelRecommender

app = Flask(__name__)

with open("expanded_data.json") as f:
    data = json.load(f)

recommender = TravelRecommender(data)

@app.route("/recommend", methods=["GET"])
def get_recommendations():
    user_name = request.args.get("user_name")
    top_n = request.args.get("top_n", default=5, type=int)
    
    if not user_name:
        return jsonify({"error": "Missing user_name parameter"}), 400
    
    recommendations = recommender.recommend(user_name, top_n)
    
    if isinstance(recommendations, str):
        return jsonify({"error": recommendations}), 404
    
    return jsonify({"user": user_name, "recommendations": recommendations})

@app.route("/evaluate", methods=["GET"])
def evaluate_system():
    n_splits = request.args.get("n_splits", default=2, type=int)
    k = request.args.get("k", default=5, type=int)
    
    results = recommender.evaluate(n_splits, k)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)