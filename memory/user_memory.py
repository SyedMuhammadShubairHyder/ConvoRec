import chromadb
from chromadb.config import Settings
import json
import time
from typing import List, Dict, Optional
import numpy as np
import os

class UserMemory:
    """
    Stores and retrieves user interaction history using ChromaDB vector database.
    
    Purpose:
    - Store every user-movie interaction
    - Retrieve similar past interactions
    - Enable semantic search over user history
    """
    
    def __init__(self, persist_directory="data/memory"):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Where to store the database
        """
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Create persistent ChromaDB client
        try:
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False,
                is_persistent=True
            ))
        except TypeError:
            # Fallback for older/newer versions of chromadb if Settings differ
            self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection("user_interactions")
            print("Loaded existing memory collection")
        except:
            self.collection = self.client.create_collection(
                name="user_interactions",
                metadata={"description": "User-movie interactions with context"}
            )
            print("Created new memory collection")
    
    def store_interaction(
        self, 
        user_id: int, 
        movie_id: int, 
        rating: float, 
        context: str = "",
        movie_title: str = "",
        genres: str = ""
    ):
        """
        Store a user-movie interaction in memory.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
            rating: User's rating (0-5)
            context: Additional context (e.g., "User said they love action")
            movie_title: Title of the movie
            genres: Movie genres
            
        Example:
            memory.store_interaction(
                user_id=123,
                movie_id=456,
                rating=4.5,
                context="User loved the intense action scenes",
                movie_title="John Wick",
                genres="Action|Thriller"
            )
        """
        # Create unique ID
        interaction_id = f"u{user_id}_m{movie_id}_{int(time.time())}"
        
        # Create document text for semantic search
        document = f"""
        User {user_id} rated {movie_title} ({genres}) with {rating} stars.
        Context: {context}
        """
        
        # Create embedding (simple average for now, could use sentence transformers)
        # For now, ChromaDB will use default embedding function
        
        # Store in database
        self.collection.add(
            documents=[document],
            metadatas=[{
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": rating,
                "timestamp": time.time(),
                "movie_title": movie_title,
                "genres": genres,
                "context": context
            }],
            ids=[interaction_id]
        )
        
        print(f"✅ Stored interaction: User {user_id} → {movie_title} ({rating}★)")
    
    def get_user_history(self, user_id: int, limit: int = 50) -> List[Dict]:
        """
        Get all interactions for a specific user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of interactions to return
            
        Returns:
            List of interaction dictionaries
            
        Example:
            history = memory.get_user_history(user_id=123, limit=10)
            # [{'movie_title': 'John Wick', 'rating': 4.5, ...}, ...]
        """
        results = self.collection.get(
            where={"user_id": user_id},
            limit=limit
        )
        
        interaction_list = []
        if results['metadatas']:
             # Combine metadatas with documents
            for i in range(len(results['ids'])):
                meta = results['metadatas'][i]
                doc = results['documents'][i] if results['documents'] else ""
                interaction_list.append({
                    **meta,
                    'document': doc
                })
        
        # Sort by timestamp (most recent first)
        interaction_list.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        return interaction_list
    
    def retrieve_similar_interactions(
        self, 
        query: str, 
        user_id: Optional[int] = None, 
        k: int = 5
    ) -> List[Dict]:
        """
        Find similar past interactions using semantic search.
        
        Args:
            query: Text query (e.g., "action movies with high ratings")
            user_id: Optional - filter to specific user
            k: Number of results to return
            
        Returns:
            List of similar interactions
            
        Example:
            similar = memory.retrieve_similar_interactions(
                query="intense action movies",
                user_id=123,
                k=3
            )
        """
        # Build where filter
        where_filter = {"user_id": user_id} if user_id else None
        
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_filter
        )
        
        similar_interactions = []
        if results['metadatas'] and results['metadatas'][0]:
            # Format results
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                doc = results['documents'][0][i]
                distance = results['distances'][0][i]
                
                similar_interactions.append({
                    **meta,
                    'document': doc,
                    'similarity': 1 - distance  # Convert distance to similarity
                })
        
        return similar_interactions
    
    def get_top_rated_movies(self, user_id: int, min_rating: float = 4.0) -> List[Dict]:
        """
        Get user's favorite movies (highly rated).
        
        Args:
            user_id: User identifier
            min_rating: Minimum rating threshold
            
        Returns:
            List of top-rated movies
        """
        all_interactions = self.get_user_history(user_id, limit=100)
        
        # Filter by rating
        top_rated = [
            interaction for interaction in all_interactions 
            if interaction.get('rating', 0) >= min_rating
        ]
        
        # Sort by rating
        top_rated.sort(key=lambda x: x.get('rating', 0), reverse=True)
        
        return top_rated
    
    def clear_user_history(self, user_id: int):
        """Delete all interactions for a user."""
        # Get all interaction IDs for this user
        results = self.collection.get(where={"user_id": user_id})
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"🗑️ Deleted {len(results['ids'])} interactions for user {user_id}")
        else:
            print(f"No interactions found for user {user_id}")


# Example usage
if __name__ == "__main__":
    # Initialize memory
    memory = UserMemory()
    
    # Store some test interactions
    memory.store_interaction(
        user_id=1,
        movie_id=100,
        rating=4.5,
        context="User mentioned loving intense action sequences",
        movie_title="John Wick",
        genres="Action|Thriller"
    )
    
    memory.store_interaction(
        user_id=1,
        movie_id=101,
        rating=3.0,
        context="User found it too slow",
        movie_title="The Godfather",
        genres="Crime|Drama"
    )
    
    # Retrieve user history
    print("\n--- User History ---")
    history = memory.get_user_history(user_id=1)
    for interaction in history:
        print(f"{interaction['movie_title']}: {interaction['rating']}★")
    
    # Find similar interactions
    print("\n--- Similar to 'action movies' ---")
    similar = memory.retrieve_similar_interactions(
        query="action movies with high intensity",
        user_id=1,
        k=2
    )
    for interaction in similar:
        print(f"{interaction['movie_title']}: {interaction['rating']}★ (similarity: {interaction['similarity']:.2f})")
