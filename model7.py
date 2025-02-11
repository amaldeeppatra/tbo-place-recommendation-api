import numpy as np
from gensim.models import Word2Vec
import json
from collections import defaultdict

class TravelRecommender:
    def __init__(self, data):
        self.data = data
        self.user_profiles = {}
        self.place_profiles = defaultdict(list)
        self.word2vec_model = None
        self.all_places = set()
        self.all_tags = set()
        
    def preprocess_data(self):
        """Process raw data to create training sequences"""
        sequences = []
        
        # Create sequences for each user's travel history
        for user in self.data:
            user_seq = []
            for visit in user['placesVisited']:
                self.all_places.add(visit['place'])
                user_seq.append(visit['place'])
                # Add tags to sequence
                for tag in visit['tags']:
                    self.all_tags.add(tag)
                    user_seq.append(tag)
            sequences.append(user_seq)
            
            # Store user profile
            self.user_profiles[user['name']] = {
                'gender': user['gender'],
                'age': user['age'],
                'places': [v['place'] for v in user['placesVisited']],
                'tags': [tag for v in user['placesVisited'] for tag in v['tags']]
            }
            
            # Store place profiles
            for visit in user['placesVisited']:
                self.place_profiles[visit['place']].extend(visit['tags'])
        
        return sequences
    
    def train_model(self, vector_size=100, window=5, min_count=1):
        """Train Word2Vec model on travel sequences"""
        sequences = self.preprocess_data()
        self.word2vec_model = Word2Vec(sentences=sequences,
                                     vector_size=vector_size,
                                     window=window,
                                     min_count=min_count,
                                     workers=4)
        
    def get_place_embedding(self, place):
        """Get combined embedding for a place and its tags"""
        if place not in self.place_profiles:
            return None
            
        vectors = [self.word2vec_model.wv[place]]
        for tag in self.place_profiles[place]:
            if tag in self.word2vec_model.wv:
                vectors.append(self.word2vec_model.wv[tag])
                
        return np.mean(vectors, axis=0)
    
    def get_places_embedding(self, places):
        """Get combined embedding for multiple places"""
        vectors = []
        
        for place in places:
            if place not in self.all_places:
                print(f"Warning: {place} not found in training data")
                continue
                
            place_embedding = self.get_place_embedding(place)
            if place_embedding is not None:
                vectors.append(place_embedding)
        
        return np.mean(vectors, axis=0) if vectors else None
    
    def recommend_from_places(self, input_places, top_n=5):
        """Recommend places based on a list of input places"""
        combined_embedding = self.get_places_embedding(input_places)
        if combined_embedding is None:
            return []
            
        # Calculate similarity with all places
        place_scores = []
        input_places_set = set(input_places)
        
        for place in self.all_places:
            if place in input_places_set:
                continue
                
            place_embedding = self.get_place_embedding(place)
            if place_embedding is not None:
                similarity = np.dot(combined_embedding, place_embedding) / (
                    np.linalg.norm(combined_embedding) * np.linalg.norm(place_embedding)
                )
                place_scores.append((place, similarity))
        
        # Sort by similarity and return top N recommendations
        recommendations = sorted(place_scores, key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {
                'place': place,
                'similarity': float(score),
                'tags': self.place_profiles[place]
            }
            for place, score in recommendations
        ]

# Example usage
def main():
    with open("even_larger_dataset.json", "r") as file:
        data = json.load(file)
    
    # Initialize and train the recommender
    recommender = TravelRecommender(data)
    recommender.train_model(vector_size=100, window=5, min_count=1)
    
    # Get recommendations based on input places
    input_places = ["Ajmer", "Pondicherry", "Dandeli"]
    recommendations = recommender.recommend_from_places(input_places, top_n=3)
    
    print(f"\nRecommendations based on places: {', '.join(input_places)}")
    for rec in recommendations:
        print(f"Place: {rec['place']}")
        print(f"Similarity Score: {rec['similarity']:.3f}")
        # print(f"Tags: {', '.join(rec['tags'])}")
        print()

if __name__ == "__main__":
    main()