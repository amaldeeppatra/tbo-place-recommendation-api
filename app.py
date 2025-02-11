from flask import Flask, request, jsonify
from model9 import EnhancedTravelRecommender
import numpy as np
from functools import wraps
import json
from typing import Dict, List, Any

app = Flask(__name__)

with open('new_dataset.json', 'r') as f:
    training_data = json.load(f)

recommender = EnhancedTravelRecommender(training_data)
recommender.train_model()

def validate_coordinates(coords: List[float]) -> bool:
    """Validate if coordinates are in correct format and range"""
    if not isinstance(coords, list) or len(coords) != 2:
        return False
    lat, lon = coords
    return -90 <= lat <= 90 and -180 <= lon <= 180

def validate_request(f):
    """Decorator to validate request body"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Required fields
        required_fields = ["age", "origin", "travel_style", "placesVisited"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
            
        # Age validation
        if not isinstance(data["age"], (int, float)) or data["age"] < 0 or data["age"] > 120:
            return jsonify({"error": "Invalid age"}), 400
            
        # Origin validation
        if not isinstance(data["origin"], dict) or \
           not all(k in data["origin"] for k in ["country", "region", "coordinates"]):
            return jsonify({"error": "Invalid origin format"}), 400
            
        if not validate_coordinates(data["origin"]["coordinates"]):
            return jsonify({"error": "Invalid coordinates in origin"}), 400
            
        # Travel style validation
        if not isinstance(data["travel_style"], list) or not data["travel_style"]:
            return jsonify({"error": "Invalid travel_style format"}), 400
            
        # Places visited validation
        if not isinstance(data["placesVisited"], list) or not data["placesVisited"]:
            return jsonify({"error": "Invalid placesVisited format"}), 400
            
        for place in data["placesVisited"]:
            if not isinstance(place, dict):
                return jsonify({"error": "Invalid place format"}), 400
                
            required_place_fields = [
                "place", "tags", "duration_days", 
                "budget_category", "coordinates", "region"
            ]
            
            missing_place_fields = [
                field for field in required_place_fields 
                if field not in place
            ]
            
            if missing_place_fields:
                return jsonify({
                    "error": f"Missing required fields in place: {', '.join(missing_place_fields)}"
                }), 400
                
            if not validate_coordinates(place["coordinates"]):
                return jsonify({
                    "error": f"Invalid coordinates for place: {place['place']}"
                }), 400
                
            if not isinstance(place["duration_days"], (int, float)) or place["duration_days"] <= 0:
                return jsonify({
                    "error": f"Invalid duration_days for place: {place['place']}"
                }), 400

        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_status": "loaded",
        "vocab_size": len(recommender.word2vec_model.wv.key_to_index)
    })

@app.route('/api/recommend', methods=['POST'])
@validate_request
def get_recommendations():
    """Get travel recommendations based on user profile and history"""
    try:
        user_input = request.get_json()
        recommendations = recommender.recommend_places(user_input)
        
        return jsonify({
            "status": "success",
            "recommendations": recommendations,
            "metadata": {
                "user_age_group": recommender._get_age_group(user_input["age"]),
                "num_recommendations": len(recommendations),
                "considered_places": len(recommender.all_places)
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/places', methods=['GET'])
def get_available_places():
    """Get all available places and their metadata"""
    return jsonify({
        "places": list(recommender.all_places),
        "total_places": len(recommender.all_places),
        "available_regions": list(recommender.regions),
        "available_tags": list(recommender.all_tags),
        "budget_categories": list(recommender.budget_categories)
    })

@app.route('/api/similarity', methods=['GET'])
def get_place_similarity():
    """Get similarity between two places"""
    place1 = request.args.get('place1')
    place2 = request.args.get('place2')
    
    if not place1 or not place2:
        return jsonify({
            "error": "Both place1 and place2 parameters are required"
        }), 400
        
    if place1 not in recommender.all_places or place2 not in recommender.all_places:
        return jsonify({
            "error": "One or both places not found in database"
        }), 404
        
    similarity = recommender.get_place_similarity(place1, place2)
    
    return jsonify({
        "place1": place1,
        "place2": place2,
        "similarity_score": similarity,
        "place1_metadata": recommender.place_profiles[place1],
        "place2_metadata": recommender.place_profiles[place2]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)