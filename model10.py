import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
import statistics
import json

class DeepTravelRecommender:
    def __init__(self, data):
        self.data = data
        self.user_profiles = {}
        self.place_profiles = defaultdict(dict)
        self.word2vec_model = None
        self.median_travel_radius = None
        self.place_embeddings = {}
        
    def calculate_median_radius(self, user):
        """Calculate median travel radius from user's origin"""
        distances = []
        origin_coords = user['origin']['coordinates']
        
        for visit in user['placesVisited']:
            dest_coords = visit['coordinates']
            distance = geodesic(origin_coords, dest_coords).kilometers
            distances.append(distance)
            
        return statistics.median(distances) if distances else 0
    
    def create_travel_sequence(self, user):
        """Create enhanced sequence for Word2Vec training with stronger emphasis on place characteristics"""
        sequence = []
        
        # Add user demographic info with less weight
        sequence.extend([
            f"age_{self._get_age_group(user['age'])}",
            f"region_{user['origin']['region']}"
        ])
        
        # Add travel style with higher weight by repeating
        for style in user['travel_style']:
            sequence.extend([f"style_{style}"] * 3)
        
        # Add visited places info with enhanced weighting for tags
        for visit in user['placesVisited']:
            # Repeat place and tags to increase their importance
            sequence.extend([visit['place']] * 2)
            for tag in visit['tags']:
                sequence.extend([f"tag_{tag.lower()}"] * 3)
            
            sequence.extend([
                f"budget_{visit['budget_category']}",
                f"region_{visit['region']}",
                f"duration_{self._get_duration_category(visit['duration_days'])}"
            ])
            
        return sequence
    
    def train_model(self, vector_size=100, window=5, min_count=1, epochs=200):
        """Train Word2Vec model with enhanced parameters"""
        # Calculate median travel radius for all users
        user_radii = [self.calculate_median_radius(user) for user in self.data]
        self.median_travel_radius = statistics.median(user_radii)
        
        # Create training sequences
        sequences = []
        for user in self.data:
            sequences.append(self.create_travel_sequence(user))
            self._update_profiles(user)
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=sequences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=epochs,
            sg=1  # Skip-gram model
        )
        
        # Create place embeddings dictionary
        self._create_place_embeddings()
        
        print(f"Model trained successfully!")
        print(f"Median travel radius: {self.median_travel_radius:.2f} km")
        print(f"Vocabulary size: {len(self.word2vec_model.wv.key_to_index)}")
    
    def _create_place_embeddings(self):
        """Create enhanced place embeddings"""
        for place in self.place_profiles:
            if place in self.word2vec_model.wv:
                place_vector = self.word2vec_model.wv[place]
                
                # Enhance embedding with place profile information
                profile = self.place_profiles[place]
                tag_vectors = [self.word2vec_model.wv[f"tag_{tag.lower()}"]
                             for tag in profile['tags']
                             if f"tag_{tag.lower()}" in self.word2vec_model.wv]
                
                if tag_vectors:
                    tag_vector = np.mean(tag_vectors, axis=0)
                    enhanced_vector = np.mean([place_vector, tag_vector], axis=0)
                    self.place_embeddings[place] = enhanced_vector
                else:
                    self.place_embeddings[place] = place_vector
    
    def _update_profiles(self, user):
        """Update user and place profiles"""
        for visit in user['placesVisited']:
            place_data = {
                'tags': visit['tags'],
                'budget_category': visit['budget_category'],
                'coordinates': visit['coordinates'],
                'region': visit['region'],
                'duration_days': visit['duration_days']
            }
            # Update existing profile or create new one
            if visit['place'] in self.place_profiles:
                self.place_profiles[visit['place']].update(place_data)
            else:
                self.place_profiles[visit['place']] = place_data
    
    def _get_age_group(self, age):
        """Categorize user age into groups"""
        if age < 25:
            return 'young'
        elif age < 35:
            return 'adult'
        elif age < 50:
            return 'middle'
        else:
            return 'senior'
    
    def _get_duration_category(self, days):
        """Categorize trip duration"""
        if days <= 2:
            return 'short'
        elif days <= 5:
            return 'medium'
        else:
            return 'long'
    
    def _calculate_tag_similarity(self, user_tags, place_tags):
        """Calculate similarity between user's preferred tags and place tags"""
        user_tags = set(tag.lower() for tag in user_tags)
        place_tags = set(tag.lower() for tag in place_tags)
        
        # Create tag categories with expanded nature-related tags
        nature_tags = {'nature', 'hill station', 'mountain', 'wildlife', 'trek', 'hiking', 
                      'valleys', 'scenic', 'landscape', 'waterfall'}
        cultural_tags = {'cultural', 'spiritual', 'historical', 'heritage', 'temple', 'museum'}
        adventure_tags = {'adventure', 'trekking', 'hiking', 'camping', 'rafting', 'climbing'}
        
        # Calculate category matches
        user_nature = bool(user_tags & nature_tags)
        user_cultural = bool(user_tags & cultural_tags)
        user_adventure = bool(user_tags & adventure_tags)
        
        place_nature = bool(place_tags & nature_tags)
        place_cultural = bool(place_tags & cultural_tags)
        place_adventure = bool(place_tags & adventure_tags)
        
        # Calculate weighted similarity with increased nature weight
        similarity = 0
        if user_nature and place_nature:
            similarity += 0.6  # Increased weight for nature matches
        if user_cultural and place_cultural:
            similarity += 0.2  # Reduced weight for cultural matches
        if user_adventure and place_adventure:
            similarity += 0.2  # Reduced weight for adventure matches
            
        return similarity
    
    def _analyze_user_preferences(self, user):
        """Analyze user's preferences based on visited places"""
        tag_counts = defaultdict(int)
        region_counts = defaultdict(int)
        budget_counts = defaultdict(int)
        
        for visit in user['placesVisited']:
            for tag in visit['tags']:
                tag_counts[tag.lower()] += 1
            region_counts[visit['region']] += 1
            budget_counts[visit['budget_category']] += 1
            
        return {
            'preferred_tags': sorted(tag_counts.items(), key=lambda x: x[1], reverse=True),
            'preferred_regions': sorted(region_counts.items(), key=lambda x: x[1], reverse=True),
            'preferred_budget': sorted(budget_counts.items(), key=lambda x: x[1], reverse=True)
        }
    
    def get_recommendations(self, user_input, num_recommendations=5):
        """Get personalized travel recommendations with enhanced preference matching."""
        if not user_input.get('placesVisited'):
            return []
        
        # Analyze user preferences (tags, regions, etc.)
        preferences = self._analyze_user_preferences(user_input)
        preferred_tags = [tag for tag, _ in preferences['preferred_tags']]
        preferred_regions = [region for region, _ in preferences['preferred_regions']]
        
        # ----- Compute a Travel Style Vector for the User -----
        # Build tokens for the user's travel style (these were also used in training)
        travel_style_tokens = [f"style_{style.lower()}" for style in user_input.get('travel_style', [])]
        travel_style_vectors = [
            self.word2vec_model.wv[token] for token in travel_style_tokens
            if token in self.word2vec_model.wv
        ]
        if travel_style_vectors:
            user_style_vector = np.mean(travel_style_vectors, axis=0)
        else:
            user_style_vector = None
        
        # ----- Compute the User Embedding -----
        # Create the travel sequence (which already includes style tokens, demographic info, etc.)
        user_sequence = self.create_travel_sequence(user_input)
        sequence_vectors = [
            self.word2vec_model.wv[token] for token in user_sequence
            if token in self.word2vec_model.wv
        ]
        if not sequence_vectors:
            return []
        user_vector = np.mean(sequence_vectors, axis=0)
        
        # Incorporate visited places into the user vector to better capture travel history
        visited_vectors = []
        for visit in user_input['placesVisited']:
            if visit['place'] in self.word2vec_model.wv:
                visited_vectors.append(self.word2vec_model.wv[visit['place']])
        if visited_vectors:
            visited_places_vector = np.mean(visited_vectors, axis=0)
            # Blend the general sequence vector with the visited places vector
            user_vector = (user_vector + visited_places_vector) / 2
        
        # ----- Calculate a Typical Travel Distance -----
        # Use the average distance from the origin to each visited place
        distances = [
            geodesic(user_input['origin']['coordinates'], visit['coordinates']).kilometers 
            for visit in user_input['placesVisited']
        ]
        typical_distance = np.mean(distances) if distances else 500  # default value if no history

        # ----- Score Each Candidate Place -----
        place_scores = []
        visited_places_set = {visit['place'] for visit in user_input['placesVisited']}
        
        for place, embedding in self.place_embeddings.items():
            if place in visited_places_set:
                continue  # Skip already visited places
            
            # (a) Base Cosine Similarity between user vector and place embedding
            cosine_similarity = np.dot(user_vector, embedding) / (
                np.linalg.norm(user_vector) * np.linalg.norm(embedding)
            )
            
            # (b) Distance Penalty using a logistic function
            distance = geodesic(
                user_input['origin']['coordinates'],
                self.place_profiles[place]['coordinates']
            ).kilometers
            # sigma controls the steepness of the penalty; adjust as needed
            sigma = typical_distance / 2 if typical_distance > 0 else 100  
            distance_penalty = 1 / (1 + np.exp((distance - typical_distance) / sigma))
            
            # (c) Tag Similarity (as before, with your custom weighting)
            tag_similarity = self._calculate_tag_similarity(preferred_tags, self.place_profiles[place]['tags'])
            
            # (d) Region and Budget Preferences
            region_factor = 1.2 if self.place_profiles[place]['region'] in preferred_regions else 1.0
            budget_factor = 1.1 if self.place_profiles[place]['budget_category'] == user_input['placesVisited'][-1]['budget_category'] else 1.0
            
            # (e) Travel Style Similarity: Compare the user's style vector to any travel-style tokens
            # that might be present in the place profile's tags (if available).
            if user_style_vector is not None:
                style_tokens = [
                    f"style_{token.lower()}" for token in self.place_profiles[place]['tags']
                    if f"style_{token.lower()}" in self.word2vec_model.wv
                ]
                if style_tokens:
                    style_vectors = [self.word2vec_model.wv[token] for token in style_tokens]
                    place_style_vector = np.mean(style_vectors, axis=0)
                    style_similarity = np.dot(user_style_vector, place_style_vector) / (
                        np.linalg.norm(user_style_vector) * np.linalg.norm(place_style_vector)
                    )
                else:
                    style_similarity = 0.5  # A neutral value if no style info is found
            else:
                style_similarity = 1.0
            
            # ----- Combine Factors into a Final Score -----
            # Here we weight four components: cosine similarity, tag similarity, distance, and style similarity.
            final_score = (
                (cosine_similarity * 0.3) +
                (tag_similarity * 0.3) +
                (distance_penalty * 0.2) +
                (style_similarity * 0.2)
            ) * region_factor * budget_factor
            
            place_scores.append((place, final_score))
        
        # Sort by score and return the top recommendations
        recommendations = sorted(place_scores, key=lambda x: x[1], reverse=True)[:num_recommendations]
        
        return [
            {
                'place': place,
                'score': score,
                'details': self.place_profiles[place]
            }
            for place, score in recommendations
        ]


def test_recommender(data):
    """Test the recommender with sample user data"""
    # Initialize and train the recommender
    recommender = DeepTravelRecommender(data)
    recommender.train_model()
    
    # Test with a sample user
    test_user = {
        "age": 28,
        "origin": {
            "country": "India",
            "region": "Central",
            "coordinates": [23.725033240776586, 77.23898337236996]
        },
        "travel_style": ["Hill Station"],
        "placesVisited": [
    #        {
    #     "place": "Khajuraho",
    #     "tags": [
    #       "Historical",
    #       "Architecture",
    #       "Cultural"
    #     ],
    #     "duration_days": 8,
    #     "budget_category": "midrange",
    #     "coordinates": [
    #       24.8318,
    #       79.9199
    #     ],
    #     "region": "Central"
    #   },
    #         {
    #     "place": "Bodh Gaya",
    #     "tags": [
    #       "Spiritual",
    #       "Heritage",
    #       "Historical"
    #     ],
    #     "duration_days": 3,
    #     "budget_category": "budget",
    #     "coordinates": [
    #       24.6952,
    #       84.9914
    #     ],
    #     "region": "East"
    #   }
    {
        "place": "Mahabaleshwar",
        "tags": [
          "Hill Station",
          "Nature",
          "Trekking"
        ],
        "duration_days": 5,
        "budget_category": "luxury",
        "coordinates": [
          17.9237,
          73.6586
        ],
        "region": "West"
      },
      {
        "place": "Coorg",
        "tags": [
          "Nature",
          "Hill Station",
          "Wildlife"
        ],
        "duration_days": 5,
        "budget_category": "budget",
        "coordinates": [
          12.3375,
          75.8069
        ],
        "region": "South"
      }
        ]
    }
    
    recommendations = recommender.get_recommendations(test_user)
    
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['place']}")
        print(f"   Score: {rec['score']:.3f}")
        print(f"   Tags: {', '.join(rec['details']['tags'])}")
        print(f"   Budget: {rec['details']['budget_category']}")
        print(f"   Region: {rec['details']['region']}")
        
    return recommender



# Example usage
if __name__ == "__main__":
    # Load your data
    with open('expanded_travel_dataset_v2.json', 'r') as f:
        travel_data = json.load(f)

    # Create and train the recommender
    recommender = test_recommender(travel_data)