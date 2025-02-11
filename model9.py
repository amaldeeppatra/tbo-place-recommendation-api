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
        self.age_scaler = MinMaxScaler()
        
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
        print("Model training completed!")
        print(f"Vocabulary size: {len(self.word2vec_model.wv.key_to_index)}")
        
    def preprocess_data(self):
        """Process data to create enhanced training sequences and profiles"""
        sequences = []
        locations = []
        durations = []
        ages = []
        
        for user in self.data:
            self.all_travel_styles.update(user['travel_style'])
            ages.append(user['age'])
            
            user_seq = []
            for visit in user['placesVisited']:
                self.all_places.add(visit['place'])
                self.all_tags.update(visit['tags'])
                self.budget_categories.add(visit['budget_category'])
                self.regions.add(visit['region'])
                
                user_seq.extend([
                    visit['place'],
                    visit['budget_category'],
                    visit['region'],
                    *visit['tags'],
                    f"age_{self._get_age_group(user['age'])}",
                    *[f"style_{style}" for style in user['travel_style']]
                ])
                
                locations.append(visit['coordinates'])
                durations.append(visit['duration_days'])
                
                self.place_profiles[visit['place']] = {
                    'tags': visit['tags'],
                    'budget_category': visit['budget_category'],
                    'coordinates': visit['coordinates'],
                    'region': visit['region'],
                    'duration_days': visit['duration_days'],
                    'popular_age_groups': set(),
                    'travel_styles': set()
                }
            
            sequences.append(user_seq)
            
            self.user_profiles[user['name']] = {
                'gender': user['gender'],
                'age': user['age'],
                'age_group': self._get_age_group(user['age']),
                'origin': user['origin'],
                'travel_style': user['travel_style'],
                'visited_places': [v['place'] for v in user['placesVisited']],
                'preferred_regions': set(v['region'] for v in user['placesVisited']),
                'preferred_budget': self._get_preferred_budget(user['placesVisited']),
                'avg_trip_duration': np.mean([v['duration_days'] for v in user['placesVisited']])
            }
            
            age_group = self._get_age_group(user['age'])
            for visit in user['placesVisited']:
                self.place_profiles[visit['place']]['popular_age_groups'].add(age_group)
                self.place_profiles[visit['place']]['travel_styles'].update(user['travel_style'])
        
        self.locations_scaler = MinMaxScaler()
        self.locations_scaler.fit(locations)
        
        self.duration_scaler = MinMaxScaler()
        self.duration_scaler.fit(np.array(durations).reshape(-1, 1))
        
        self.age_scaler = MinMaxScaler()
        self.age_scaler.fit(np.array(ages).reshape(-1, 1))
        
        return sequences
    
    def _get_age_group(self, age):
        if age < 25:
            return 'young'
        elif age < 40:
            return 'adult'
        elif age < 60:
            return 'middle_aged'
        else:
            return 'senior'
    
    def _get_preferred_budget(self, visits):
        budget_counts = defaultdict(int)
        for visit in visits:
            budget_counts[visit['budget_category']] += 1
        return max(budget_counts.items(), key=lambda x: x[1])[0]
    
    def get_place_similarity(self, place1, place2, user_profile=None):
        if place1 not in self.place_profiles or place2 not in self.place_profiles:
            return 0.0
            
        p1 = self.place_profiles[place1]
        p2 = self.place_profiles[place2]
        
        tag_sim = len(set(p1['tags']) & set(p2['tags'])) / len(set(p1['tags']) | set(p2['tags']))
        budget_sim = 1.0 if p1['budget_category'] == p2['budget_category'] else 0.0
        region_sim = 1.0 if p1['region'] == p2['region'] else 0.0
        
        geo_dist = geodesic(p1['coordinates'], p2['coordinates']).kilometers
        max_dist = 3000
        geo_sim = 1 - min(geo_dist / max_dist, 1)
        
        dur_diff = abs(p1['duration_days'] - p2['duration_days'])
        dur_sim = 1 - min(dur_diff / 7, 1)
        
        age_sim = 0.0
        if user_profile:
            age_group = user_profile['age_group']
            if age_group in p2['popular_age_groups']:
                age_sim = 1.0
            
        weights = {
            'tag': 0.25,
            'budget': 0.15,
            'region': 0.15,
            'geo': 0.15,
            'duration': 0.1,
            'age': 0.2
        }
        
        total_sim = (
            weights['tag'] * tag_sim +
            weights['budget'] * budget_sim +
            weights['region'] * region_sim +
            weights['geo'] * geo_sim +
            weights['duration'] * dur_sim +
            weights['age'] * age_sim
        )
        
        return total_sim
    
    def recommend_places(self, user_input):
        if not user_input.get('placesVisited'):
            return []
        
        temp_user_profile = {
            'age': user_input['age'],
            'age_group': self._get_age_group(user_input['age']),
            'origin': user_input['origin'],
            'travel_style': user_input['travel_style'],
            'visited_places': [v['place'] for v in user_input['placesVisited']],
            'preferred_regions': set(v['region'] for v in user_input['placesVisited']),
            'preferred_budget': self._get_preferred_budget(user_input['placesVisited'])
        }
        
        place_scores = defaultdict(float)
        input_places_set = set(temp_user_profile['visited_places'])
        
        for candidate_place in self.all_places:
            if candidate_place in input_places_set:
                continue
            
            similarities = [
                self.get_place_similarity(
                    input_place, 
                    candidate_place, 
                    temp_user_profile
                )
                for input_place in temp_user_profile['visited_places']
            ]
            base_score = np.mean(similarities)
            
            candidate_profile = self.place_profiles[candidate_place]
            
            style_overlap = len(set(temp_user_profile['travel_style']) & 
                              set(candidate_profile.get('travel_styles', set())))
            style_boost = 1 + (style_overlap * 0.1)
            
            origin_dist = geodesic(
                temp_user_profile['origin']['coordinates'],
                candidate_profile['coordinates']
            ).kilometers
            distance_penalty = 1.0 - (min(origin_dist, 3000) / 3000) * 0.2
            
            region_boost = 1.2 if candidate_profile['region'] in temp_user_profile['preferred_regions'] else 1.0
            
            final_score = base_score * style_boost * distance_penalty * region_boost
            place_scores[candidate_place] = final_score
        
        recommendations = sorted(
            place_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
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

# Sample training data
training_data = [
    {
        "name": "User1",
        "gender": "F",
        "age": 28,
        "origin": {
            "country": "India",
            "region": "South",
            "coordinates": [12.9716, 77.5946]
        },
        "travel_style": ["adventure", "culture"],
        "placesVisited": [
            {
                "place": "Sundarbans",
                "tags": ["nature", "wildlife"],
                "duration_days": 4,
                "budget_category": "medium",
                "coordinates": [21.9497, 89.1833],
                "region": "East"
            },
            {
                "place": "Goa",
                "tags": ["beach", "party"],
                "duration_days": 5,
                "budget_category": "high",
                "coordinates": [15.2993, 74.1240],
                "region": "West"
            }
        ]
    },
    # Add more training data here...
]

# Create and train the recommender
recommender = EnhancedTravelRecommender(training_data)
recommender.train_model()

# Test user input
user_input = {
    "age": 20,
    "origin": {
        "country": "India",
        "region": "East",
        "coordinates": [20.2960, 85.8246]
    },
    "travel_style": ["nature", "cultural"],
    "placesVisited": [
        {
                "place": "Puri",
                "tags": [
                    "Beach",
                    "Spiritual"
                ],
                "duration_days": 3,
                "budget_category": "budget",
                "coordinates": [
                    19.8135,
                    85.8315
                ],
                "region": "East"
            }
    ]
}

# Get and print recommendations
recommendations = recommender.recommend_places(user_input)
print("\nTravel Recommendations:")
print("----------------------")
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec['place']}")
    print(f"   Similarity Score: {rec['similarity_score']:.2f}")
    print(f"   Tags: {', '.join(rec['tags'])}")
    print(f"   Budget Category: {rec['budget_category']}")
    print(f"   Region: {rec['region']}")
    print(f"   Recommended Duration: {rec['duration_days']} days")