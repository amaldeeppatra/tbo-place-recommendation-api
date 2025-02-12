from flask import Flask, request, jsonify
import json
from model10 import DeepTravelRecommender

app = Flask(__name__)

# Load dataset and train model once at startup
with open('expanded_travel_dataset_v2.json', 'r') as f:
    travel_data = json.load(f)

recommender = DeepTravelRecommender(travel_data)
recommender.train_model()

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Endpoint to get travel recommendations"""
    try:
        user_data = request.get_json()
        if not user_data:
            return jsonify({"error": "Invalid input, please provide user travel data"}), 400
        
        recommendations = recommender.get_recommendations(user_data)

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)