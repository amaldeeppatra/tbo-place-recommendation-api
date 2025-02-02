from flask import Flask, request, jsonify
import json
from model5 import TravelRecommender  # Make sure this file contains your updated TravelRecommender

app = Flask(__name__)

# Load data from JSON file and initialize the recommender
with open("expanded_data.json") as f:
    data = json.load(f)

recommender = TravelRecommender(data)

@app.route("/recommend", methods=["GET"])
def get_recommendations():
    """
    GET /recommend supports two modes:
    1. If 'user_name' is provided in the query parameters, generate recommendations
       for an existing user.
    2. If 'placesVisited' is provided (as a comma-separated list), generate recommendations
       based on the provided places.
    """
    top_n = request.args.get("top_n", default=5, type=int)
    
    # Mode 1: Existing user recommendation using user_name
    user_name = request.args.get("user_name")
    if user_name:
        recommendations = recommender.recommend(user_name, top_n)
        if isinstance(recommendations, str):
            # This means an error occurred (for example, user not found)
            return jsonify({"error": recommendations}), 404
        return jsonify({"user": user_name, "recommendations": recommendations})
    
    # Mode 2: Virtual user recommendation using placesVisited query parameter
    places_str = request.args.get("placesVisited")
    if places_str:
        # Expecting a comma separated list of place names
        places_visited = [place.strip() for place in places_str.split(",")]
        recommendations = recommender.recommend_from_places(places_visited, top_n)
        return jsonify({
            "placesVisited": places_visited,
            "recommendations": recommendations
        })
    
    # If neither parameter is provided, return an error
    return jsonify({"error": "Missing required parameter: either 'user_name' or 'placesVisited'"}), 400

@app.route("/recommend_from_places", methods=["POST"])
def recommend_from_places():
    """
    POST /recommend_from_places expects a JSON payload with the following structure:
    {
        "placesVisited": ["Place_A", "Place_B", "Place_C"],
        "top_n": 5  // Optional
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request payload must be in JSON format"}), 400

    payload = request.get_json()
    places_visited = payload.get("placesVisited")
    top_n = payload.get("top_n", 5)
    
    if not places_visited or not isinstance(places_visited, list):
        return jsonify({"error": "Missing or invalid 'placesVisited' field in JSON payload"}), 400
    
    recommendations = recommender.recommend_from_places(places_visited, top_n)
    return jsonify({
        "placesVisited": places_visited,
        "recommendations": recommendations
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
