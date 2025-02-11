from flask import Flask, request, jsonify
from model7 import TravelRecommender
import json
from functools import wraps
import numpy as np

app = Flask(__name__)

# Global variable to store our model
recommender = None

def initialize_model():
    """Initialize the recommender model with data"""
    global recommender
    try:
        with open("even_larger_dataset.json", "r") as file:
            data = json.load(file)
        recommender = TravelRecommender(data)
        recommender.train_model(vector_size=100, window=5, min_count=1)
        return True
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return False

def require_model(f):
    """Decorator to ensure model is initialized before handling requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if recommender is None:
            return jsonify({
                "error": "Model not initialized",
                "message": "Please wait for the model to initialize"
            }), 503
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint to check API health and model status"""
    return jsonify({
        "status": "healthy",
        "model_loaded": recommender is not None
    })

@app.route('/api/places', methods=['GET'])
@require_model
def get_available_places():
    """Get list of all available places in the model"""
    return jsonify({
        "places": list(recommender.all_places)
    })

@app.route('/api/recommend', methods=['POST'])
@require_model
def get_recommendations():
    """Get travel recommendations based on input places"""
    try:
        data = request.get_json()
        
        if not data or 'places' not in data:
            return jsonify({
                "error": "Invalid request",
                "message": "Please provide a list of places in the request body"
            }), 400
            
        input_places = data['places']
        top_n = data.get('top_n', 5)  # Default to 5 recommendations
        
        # Validate input places
        invalid_places = [place for place in input_places if place not in recommender.all_places]
        if invalid_places:
            return jsonify({
                "error": "Invalid places",
                "message": f"The following places were not found in our database: {', '.join(invalid_places)}",
                "valid_places": list(recommender.all_places)
            }), 400
            
        # Get recommendations
        recommendations = recommender.recommend_from_places(input_places, top_n=top_n)
        
        # Format recommendations without tags
        formatted_recommendations = [
            {
                'place': rec['place'],
                'similarity': rec['similarity']
            }
            for rec in recommendations
        ]
        
        return jsonify({
            "input_places": input_places,
            "recommendations": formatted_recommendations
        })
        
    except Exception as e:
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500

@app.route('/api/place_info/<place>', methods=['GET'])
@require_model
def get_place_info(place):
    """Get information about a specific place"""
    if place not in recommender.all_places:
        return jsonify({
            "error": "Place not found",
            "message": f"'{place}' was not found in our database"
        }), 404
        
    return jsonify({
        "place": place,
        "tags": recommender.place_profiles[place]
    })

if __name__ == '__main__':
    # Initialize the model before starting the server
    if initialize_model():
        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to initialize model. Exiting...")