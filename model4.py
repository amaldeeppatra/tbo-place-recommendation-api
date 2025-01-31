import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, ndcg_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import json
from typing import List, Tuple, Dict, Union, Set
from dataclasses import dataclass
from collections import Counter

@dataclass
class Place:
    name: str
    tags: List[str]
    popularity: float = 0.0
    
@dataclass
class User:
    name: str
    places_visited: List[Place]
    
class TravelRecommender:
    def __init__(self, data: List[dict], alpha: float = 0.3):
        """
        Initialize the recommender system with user data and hyperparameters.
        
        Args:
            data: List of user data dictionaries
            alpha: Weight for popularity in hybrid scoring (0-1)
        """
        self.alpha = alpha
        self.users: List[User] = []
        self.places: Dict[str, Place] = {}
        self.all_tags: Set[str] = set()
        self.vectorizer = TfidfVectorizer()
        
        self._process_data(data)
        self._compute_similarities()
        
    def _process_data(self, data: List[dict]) -> None:
        """Process raw data into structured format and compute place popularity."""
        # Process places and compute popularity
        visit_counts = Counter()
        for user_data in data:
            for place_data in user_data["placesVisited"]:
                visit_counts[place_data["place"]] += 1
                self.all_tags.update(place_data["tags"])
        
        # Create normalized popularity scores
        max_visits = max(visit_counts.values())
        for place_name, count in visit_counts.items():
            self.places[place_name] = Place(
                name=place_name,
                tags=[],  # Temporary, will be updated
                popularity=count / max_visits
            )
        
        # Process users and update place tags
        for user_data in data:
            places_visited = []
            for place_data in user_data["placesVisited"]:
                place_name = place_data["place"]
                self.places[place_name].tags = place_data["tags"]
                places_visited.append(self.places[place_name])
            
            self.users.append(User(
                name=user_data["name"],
                places_visited=places_visited
            ))
    
    def _compute_similarities(self) -> None:
        """Compute TF-IDF vectors and similarity matrices."""
        # Create place tag documents
        place_docs = [" ".join(place.tags) for place in self.places.values()]
        self.place_vectors = self.vectorizer.fit_transform(place_docs)
        
        # Create user preference vectors (based on visited places)
        user_docs = []
        for user in self.users:
            user_tags = []
            for place in user.places_visited:
                user_tags.extend(place.tags)
            user_docs.append(" ".join(set(user_tags)))
        
        self.user_vectors = self.vectorizer.transform(user_docs)
        self.similarity_matrix = cosine_similarity(self.user_vectors, self.place_vectors)
    
    def recommend(self, user_name: str, top_n: int = 5) -> Union[str, List[Tuple[str, float]]]:
        """
        Generate personalized recommendations for a user.
        
        Args:
            user_name: Name of the user
            top_n: Number of recommendations to generate
            
        Returns:
            List of (place_name, score) tuples or error message
        """
        try:
            # Find user index
            user_idx = next((i for i, u in enumerate(self.users) if u.name == user_name), None)
            if user_idx is None:
                return f"User '{user_name}' not found."
            
            user = self.users[user_idx]
            visited_places = {p.name for p in user.places_visited}
            
            # Calculate hybrid scores
            recommendations = []
            similarities = self.similarity_matrix[user_idx]
            
            for place_name, place in self.places.items():
                if place_name not in visited_places:
                    idx = list(self.places.keys()).index(place_name)
                    # Hybrid scoring combining similarity and popularity
                    similarity_score = similarities[idx]
                    hybrid_score = (1 - self.alpha) * similarity_score + self.alpha * place.popularity
                    recommendations.append((place_name, float(hybrid_score)))
            
            # Sort and return top recommendations
            return sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return f"Error: {str(e)}"
    
    def evaluate(self, n_splits: int = 2, k: int = 5) -> Dict[str, float]:
        """
        Evaluate recommender performance using k-fold cross-validation.
        
        Args:
            n_splits: Number of cross-validation splits
            k: Number of recommendations to generate
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'ndcg': [],
            'hit_rate': [],
            'precision': [],
            'recall': [],
            'coverage': set()
        }
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for user in self.users:
            places_visited = [p.name for p in user.places_visited]
            if len(places_visited) < 2:
                continue
                
            for train_idx, test_idx in kf.split(places_visited):
                test_places = {places_visited[i] for i in test_idx}
                
                # Generate recommendations
                recommendations = self.recommend(user.name, top_n=k)
                if isinstance(recommendations, str):
                    continue
                
                recommended_places = {place for place, _ in recommendations}
                
                # Calculate metrics
                hits = len(test_places & recommended_places)
                metrics['hit_rate'].append(hits / len(test_places))
                metrics['precision'].append(hits / k)
                metrics['recall'].append(hits / len(test_places))
                metrics['coverage'].update(recommended_places)
                
                # Calculate NDCG
                relevance = []
                for place, _ in recommendations:
                    relevance.append(1.0 if place in test_places else 0.0)
                metrics['ndcg'].append(ndcg_score([relevance], [[1.0] * len(relevance)]))
        
        if not metrics['hit_rate']:
            return {metric: 0.0 for metric in metrics.keys()}
        
        return {
            'ndcg': np.mean(metrics['ndcg']),
            'hit_rate': np.mean(metrics['hit_rate']),
            'precision': np.mean(metrics['precision']),
            'recall': np.mean(metrics['recall']),
            'coverage': len(metrics['coverage']) / len(self.places),
            'n_evaluations': len(metrics['hit_rate'])
        }

# Example usage
if __name__ == "__main__":
    # Load data
    with open("expanded_data.json") as f:
        data = json.load(f)
    
    # Initialize and test recommender
    recommender = TravelRecommender(data)
    
    # Generate recommendations for a user
    user_name = "Ramesh_970"
    print(f"\nGenerating recommendations for {user_name}...")
    recommendations = recommender.recommend(user_name, top_n=5)
    
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print(f"\nRecommended places for {user_name}:")
        for place, score in recommendations:
            print(f"- {place} (Score: {score:.3f})")
    
    # Evaluate system
    print("\nEvaluating recommendation system...")
    evaluation_results = recommender.evaluate()
    
    print("\nRecommendation System Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.3f}")