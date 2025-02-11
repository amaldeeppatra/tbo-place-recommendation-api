import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
import json
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic

class EnhancedTravelRecommender:
    def __init__(self, data):
        self.data = data
        self.user_profiles = {}
        self.place_profiles = defaultdict(dict)
        self.word2vec_model = None
        self.all_places = set()
        self.all_tags = set()
        self.all_travel_styles = set()
        self.budget_categories = set()
        self.regions = set()
        self.scaler = MinMaxScaler()
        
    def preprocess_data(self):
        """Process data to create enhanced training sequences and profiles"""
        sequences = []
        locations = []
        durations = []
        
        for user in self.data:
            # Collect travel styles
            self.all_travel_styles.update(user['travel_style'])
            
            user_seq = []
            for visit in user['placesVisited']:
                # Collect basic info
                self.all_places.add(visit['place'])
                self.all_tags.update(visit['tags'])
                self.budget_categories.add(visit['budget_category'])
                self.regions.add(visit['region'])
                
                # Create sequence with place and features
                user_seq.extend([
                    visit['place'],
                    visit['budget_category'],
                    visit['region'],
                    *visit['tags']
                ])
                
                # Store location data for scaling
                locations.append(visit['coordinates'])
                durations.append(visit['duration_days'])
                
                # Build comprehensive place profile
                self.place_profiles[visit['place']] = {
                    'tags': visit['tags'],
                    'budget_category': visit['budget_category'],
                    'coordinates': visit['coordinates'],
                    'region': visit['region'],
                    'duration_days': visit['duration_days']
                }
            
            sequences.append(user_seq)
            
            # Build comprehensive user profile
            self.user_profiles[user['name']] = {
                'gender': user['gender'],
                'age': user['age'],
                'origin': user['origin'],
                'travel_style': user['travel_style'],
                'visited_places': [v['place'] for v in user['placesVisited']],
                'preferred_regions': set(v['region'] for v in user['placesVisited']),
                'preferred_budget': self._get_preferred_budget(user['placesVisited']),
                'avg_trip_duration': np.mean([v['duration_days'] for v in user['placesVisited']])
            }
        
        # Scale location coordinates
        self.locations_scaler = MinMaxScaler()
        self.locations_scaler.fit(locations)
        
        # Scale durations
        self.duration_scaler = MinMaxScaler()
        self.duration_scaler.fit(np.array(durations).reshape(-1, 1))
        
        return sequences
    
    def _get_preferred_budget(self, visits):
        """Determine user's preferred budget category"""
        budget_counts = defaultdict(int)
        for visit in visits:
            budget_counts[visit['budget_category']] += 1
        return max(budget_counts.items(), key=lambda x: x[1])[0]
    
    def train_model(self, vector_size=100, window=5, min_count=1):
        """Train Word2Vec model on enhanced sequences"""
        sequences = self.preprocess_data()
        self.word2vec_model = Word2Vec(
            sentences=sequences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
    
    def get_place_similarity(self, place1, place2):
        """Calculate comprehensive similarity between two places"""
        if place1 not in self.place_profiles or place2 not in self.place_profiles:
            return 0.0
            
        p1 = self.place_profiles[place1]
        p2 = self.place_profiles[place2]
        
        # Calculate various similarity components
        tag_sim = len(set(p1['tags']) & set(p2['tags'])) / len(set(p1['tags']) | set(p2['tags']))
        budget_sim = 1.0 if p1['budget_category'] == p2['budget_category'] else 0.0
        region_sim = 1.0 if p1['region'] == p2['region'] else 0.0
        
        # Geographic distance similarity (inverse of normalized distance)
        geo_dist = geodesic(p1['coordinates'], p2['coordinates']).kilometers
        max_dist = 3000  # approximate max distance in India
        geo_sim = 1 - min(geo_dist / max_dist, 1)
        
        # Duration similarity
        dur_diff = abs(p1['duration_days'] - p2['duration_days'])
        dur_sim = 1 - min(dur_diff / 7, 1)  # normalize by a week
        
        # Combine similarities with weights
        weights = {
            'tag': 0.3,
            'budget': 0.2,
            'region': 0.2,
            'geo': 0.2,
            'duration': 0.1
        }
        
        total_sim = (
            weights['tag'] * tag_sim +
            weights['budget'] * budget_sim +
            weights['region'] * region_sim +
            weights['geo'] * geo_sim +
            weights['duration'] * dur_sim
        )
        
        return total_sim
    
    def recommend_places(self, input_places, user_name=None, top_n=5):
        """Generate recommendations based on input places and optionally user profile"""
        if not input_places:
            return []
        
        # Calculate base similarities
        place_scores = defaultdict(float)
        input_places_set = set(input_places)
        
        for candidate_place in self.all_places:
            if candidate_place in input_places_set:
                continue
                
            # Calculate average similarity to input places
            similarities = [
                self.get_place_similarity(input_place, candidate_place)
                for input_place in input_places
            ]
            base_score = np.mean(similarities)
            
            # Apply user preferences if user_name is provided
            if user_name and user_name in self.user_profiles:
                user_profile = self.user_profiles[user_name]
                candidate_profile = self.place_profiles[candidate_place]
                
                # Calculate preference boosts
                region_boost = 1.2 if candidate_profile['region'] in user_profile['preferred_regions'] else 1.0
                budget_boost = 1.2 if candidate_profile['budget_category'] == user_profile['preferred_budget'] else 1.0
                
                # Calculate distance from user's origin
                origin_dist = geodesic(
                    user_profile['origin']['coordinates'],
                    candidate_profile['coordinates']
                ).kilometers
                distance_penalty = 1.0 - (min(origin_dist, 3000) / 3000) * 0.2
                
                # Apply all modifiers
                base_score *= region_boost * budget_boost * distance_penalty
            
            place_scores[candidate_place] = base_score
        
        # Sort and return top recommendations
        recommendations = sorted(
            place_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return [
            {
                'place': place,
                'similarity_score': score,
                'tags': self.place_profiles[place]['tags'],
                'budget_category': self.place_profiles[place]['budget_category'],
                'region': self.place_profiles[place]['region'],
                'duration_days': self.place_profiles[place]['duration_days']
            }
            for place, score in recommendations
        ]

def main():
    # Load your data
    with open("new_dataset.json", "r") as file:
        data = json.load(file)
    
    # Initialize and train recommender
    recommender = EnhancedTravelRecommender(data)
    recommender.train_model()
    
    # Example recommendation
    input_places = ["Sundarbans", "Dandeli"]
    user_name = "Neha_776"  # Optional: provide user name for personalized recommendations
    
    recommendations = recommender.recommend_places(
        input_places=input_places,
        user_name=user_name,
        top_n=3
    )
    
    print(f"\nRecommendations based on {', '.join(input_places)}")
    print(f"Personalized for user: {user_name if user_name else 'No user'}")
    print("\nRecommended Places:")
    for rec in recommendations:
        print(f"\nPlace: {rec['place']}")
        print(f"Similarity Score: {rec['similarity_score']:.3f}")
        print(f"Tags: {', '.join(rec['tags'])}")
        print(f"Budget Category: {rec['budget_category']}")
        print(f"Region: {rec['region']}")
        print(f"Recommended Duration: {rec['duration_days']} days")

if __name__ == "__main__":
    main()