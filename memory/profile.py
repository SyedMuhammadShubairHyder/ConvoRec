from typing import Dict, List, Optional
from collections import Counter
import numpy as np

class UserProfile:
    """
    Aggregates user preferences from interaction history.
    
    Purpose:
    - Build comprehensive user profile
    - Track genre preferences
    - Identify patterns in user behavior
    """
    
    def __init__(self, user_id: int, memory_system):
        """
        Initialize profile for a user.
        
        Args:
            user_id: User identifier
            memory_system: UserMemory instance
        """
        self.user_id = user_id
        self.memory = memory_system
        self.profile_data = None
        self._build_profile()
    
    def _build_profile(self):
        """Build profile from user's interaction history."""
        history = self.memory.get_user_history(self.user_id, limit=100)
        
        if not history:
            self.profile_data = {
                'user_id': self.user_id,
                'total_ratings': 0,
                'avg_rating': 0.0,
                'favorite_genres': [],
                'disliked_genres': [],
                'preference_strength': 'unknown',
                'interaction_count': 0,
                'rating_std': 0.0,
                'all_genres': {},
                'most_common_genres': []
            }
            return
        
        # Extract data
        ratings = [h['rating'] for h in history if 'rating' in h]
        genres_list = [h['genres'].split('|') for h in history if 'genres' in h]
        
        # Flatten genres
        all_genres = [genre for sublist in genres_list for genre in sublist if genre]
        
        # Count genre occurrences
        genre_counter = Counter(all_genres)
        
        # Calculate genre preferences (weighted by rating)
        genre_ratings = {}
        for interaction in history:
            if 'genres' in interaction and 'rating' in interaction:
                for genre in interaction['genres'].split('|'):
                    if genre:
                        if genre not in genre_ratings:
                            genre_ratings[genre] = []
                        genre_ratings[genre].append(interaction['rating'])
        
        # Calculate average rating per genre
        genre_avg_ratings = {
            genre: np.mean(ratings_list) 
            for genre, ratings_list in genre_ratings.items()
        }
        
        # Sort genres by average rating
        sorted_genres = sorted(
            genre_avg_ratings.items(), 
            key=lambda x: (x[1], genre_counter[x[0]]),  # Sort by rating, then frequency
            reverse=True
        )
        
        # Identify favorites (avg rating >= 4.0)
        favorite_genres = [
            genre for genre, avg_rating in sorted_genres 
            if avg_rating >= 4.0
        ]
        
        # Identify dislikes (avg rating < 3.0)
        disliked_genres = [
            genre for genre, avg_rating in sorted_genres 
            if avg_rating < 3.0
        ]
        
        # Determine preference strength
        rating_std = np.std(ratings) if len(ratings) > 1 else 0
        if rating_std < 0.5:
            preference_strength = 'strong'  # Consistent ratings
        elif rating_std < 1.0:
            preference_strength = 'moderate'
        else:
            preference_strength = 'varied'  # Diverse tastes
        
        # Build profile
        self.profile_data = {
            'user_id': self.user_id,
            'total_ratings': len(ratings),
            'avg_rating': np.mean(ratings) if ratings else 0.0,
            'rating_std': rating_std,
            'favorite_genres': favorite_genres[:5],  # Top 5
            'disliked_genres': disliked_genres[:3],  # Top 3 dislikes
            'all_genres': dict(genre_avg_ratings),
            'preference_strength': preference_strength,
            'interaction_count': len(history),
            'most_common_genres': [g for g, _ in genre_counter.most_common(5)]
        }
    
    def get_profile(self) -> Dict:
        """Return complete profile."""
        if not self.profile_data:
            self._build_profile()
        return self.profile_data
    
    def get_favorite_genres(self) -> List[str]:
        """Get user's favorite genres."""
        return self.profile_data.get('favorite_genres', [])
    
    def get_disliked_genres(self) -> List[str]:
        """Get genres user dislikes."""
        return self.profile_data.get('disliked_genres', [])
    
    def is_cold_start(self) -> bool:
        """Check if user is in cold-start (few interactions)."""
        return self.profile_data['interaction_count'] < 5
    
    def get_confidence(self) -> str:
        """
        Get confidence level in profile accuracy.
        
        Returns:
            'high', 'medium', or 'low'
        """
        count = self.profile_data['interaction_count']
        
        if count >= 20:
            return 'high'
        elif count >= 10:
            return 'medium'
        else:
            return 'low'
    
    def to_text(self) -> str:
        """
        Convert profile to natural language text.
        
        Purpose: Feed to LLM for conversational AI
        """
        profile = self.profile_data
        
        if profile['interaction_count'] == 0:
            return f"User {self.user_id} is new with no interaction history."
        
        text = f"""
User Profile Summary:
- User ID: {self.user_id}
- Total Ratings: {profile['total_ratings']}
- Average Rating: {profile['avg_rating']:.2f} stars
- Preference Pattern: {profile['preference_strength']}

Favorite Genres: {', '.join(profile['favorite_genres']) if profile['favorite_genres'] else 'None yet'}
Disliked Genres: {', '.join(profile['disliked_genres']) if profile['disliked_genres'] else 'None'}

Profile Confidence: {self.get_confidence()}
Cold Start Status: {'Yes' if self.is_cold_start() else 'No'}
        """
        
        return text.strip()
    
    def refresh(self):
        """Rebuild profile from latest data."""
        self._build_profile()


# Example usage
if __name__ == "__main__":
    from user_memory import UserMemory
    
    # Initialize
    memory = UserMemory()
    
    # Create profile
    profile = UserProfile(user_id=1, memory_system=memory)
    
    # Display profile
    print(profile.to_text())
    
    print("\nFavorite Genres:", profile.get_favorite_genres())
    print("Is Cold Start?", profile.is_cold_start())
