import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json
import time

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

class TravelRecommenderEvaluator:
    def __init__(self, data, test_size=0.2, random_state=42):
        self.full_data = data
        self.test_size = test_size
        self.random_state = random_state
        self.train_data = []
        self.test_data = []
        self.recommender = None
        
    def split_user_data(self):
        """Split each user's visited places into train and test sets"""
        train_data = []
        test_data = []
        
        # Process all users at once instead of one by one
        for user in self.full_data:
            places = user['placesVisited']
            if len(places) < 2:
                train_data.append(user)
                continue
                
            # Split places
            train_places, test_places = train_test_split(
                places, 
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            if train_places and test_places:
                train_user = dict(user)
                test_user = dict(user)
                train_user['placesVisited'] = train_places
                test_user['placesVisited'] = test_places
                train_data.append(train_user)
                test_data.append(test_user)
        
        return train_data, test_data
    
    def calculate_metrics(self, k_values=[5, 10]):  # Reduced k values
        """Calculate metrics with optimization"""
        print("Splitting data...")
        start_time = time.time()
        self.train_data, self.test_data = self.split_user_data()
        
        print("Training model...")
        self.recommender = TravelRecommender(self.train_data)
        self.recommender.train_model()
        
        print("Calculating metrics...")
        metrics = {
            'precision': defaultdict(list),
            'recall': defaultdict(list),
            'hit_rate': defaultdict(list)
        }
        
        # Pre-calculate place embeddings
        place_embeddings = {}
        for place in self.recommender.all_places:
            place_embeddings[place] = self.recommender.get_place_embedding(place)
        
        total_users = len(self.test_data)
        for idx, test_user in enumerate(self.test_data, 1):
            if idx % 10 == 0:  # Progress indicator
                print(f"Processing user {idx}/{total_users}")
                
            actual_places = set(visit['place'] for visit in test_user['placesVisited'])
            
            # Get training places
            train_user = next(u for u in self.train_data if u['name'] == test_user['name'])
            input_places = [visit['place'] for visit in train_user['placesVisited']]
            
            # Get recommendations using pre-calculated embeddings
            max_k = max(k_values)
            recs = self._get_recommendations_optimized(
                input_places, 
                place_embeddings, 
                max_k
            )
            
            # Calculate metrics for each k
            for k in k_values:
                top_k_recs = recs[:k]
                relevant_and_recommended = len(set(top_k_recs) & actual_places)
                
                metrics['precision'][k].append(
                    relevant_and_recommended / k if k > 0 else 0
                )
                metrics['recall'][k].append(
                    relevant_and_recommended / len(actual_places) if actual_places else 0
                )
                metrics['hit_rate'][k].append(
                    1 if relevant_and_recommended > 0 else 0
                )
        
        # Calculate average metrics
        results = {}
        for metric_name, k_values_dict in metrics.items():
            results[metric_name] = {
                k: np.mean(values) for k, values in k_values_dict.items()
            }
        
        print(f"\nTotal evaluation time: {time.time() - start_time:.2f} seconds")
        return results
    
    def _get_recommendations_optimized(self, input_places, place_embeddings, top_n):
        """Optimized recommendation calculation using pre-computed embeddings"""
        vectors = []
        for place in input_places:
            if place in place_embeddings:
                vectors.append(place_embeddings[place])
        
        if not vectors:
            return []
            
        combined_embedding = np.mean(vectors, axis=0)
        
        # Calculate similarities
        place_scores = []
        input_places_set = set(input_places)
        
        for place, embedding in place_embeddings.items():
            if place not in input_places_set:
                similarity = np.dot(combined_embedding, embedding) / (
                    np.linalg.norm(combined_embedding) * np.linalg.norm(embedding)
                )
                place_scores.append((place, similarity))
        
        # Sort and return top N places
        return [place for place, _ in sorted(
            place_scores, 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]]

def main():
    print("Loading data...")
    start_time = time.time()
    
    with open("even_larger_dataset.json", "r") as file:
        data = json.load(file)
    
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    # Initialize evaluator with smaller test size
    evaluator = TravelRecommenderEvaluator(data, test_size=0.1)  # Reduced test size
    
    # Calculate and print metrics
    print("\nStarting evaluation...")
    metrics = evaluator.calculate_metrics()
    
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric_name, k_values in metrics.items():
        print(f"\n{metric_name.upper()}:")
        for k, value in k_values.items():
            print(f"{metric_name}@{k}: {value:.3f}")
    
    print(f"\nDataset Statistics:")
    print(f"Total users: {len(data)}")
    print(f"Training users: {len(evaluator.train_data)}")
    print(f"Test users: {len(evaluator.test_data)}")

if __name__ == "__main__":
    main()