import os
from typing import List, Dict, Any
import openai
from openai import OpenAI
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts."""
        pass

    # def get_embedding(self, text: str) -> List[float]:
    #     embeddings = self.get_embeddings([text])
    #     return embeddings[0] if embeddings else []

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI models."""
    def __init__(self, api_key: str, model: str, dimension: int):
        print(f"Initializing OpenAIEmbeddingProvider with model: {model}, dimension: {dimension}")
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAIEmbeddingProvider.")
        try:
            self.client = OpenAI(api_key=api_key)
            self.model = model
            self.dimension = dimension
            # TODO : Verify model compatibility with dimension early? for final godfin reproducability
            # extra bs  -- might require an API call or maintaining a known list.
            print(f"OpenAIEmbeddingProvider initialized for model {self.model}.")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts using the configured OpenAI model."""
        if not texts:
            print("Warning: get_embeddings called with empty list.")
            return []
        processed_texts = [text.replace("\n", " ").strip() for text in texts]
        # Filter out empty strings after cleaning, as OpenAI API might error on them
        non_empty_texts = [text for text in processed_texts if text]
        if not non_empty_texts:
            print("Warning: All texts were empty after cleaning; returning no embeddings.")
            return []
        if len(non_empty_texts) < len(texts):
            print(f"Warning: Filtered out {len(texts) - len(non_empty_texts)} empty texts before sending to OpenAI.")

        try:
            print(f"Requesting OpenAI embeddings for {len(non_empty_texts)} texts using model {self.model}...")
            response = self.client.embeddings.create(
                input=non_empty_texts,
                model=self.model
                # TODO : Potentially specify dimensions if using models that support it, e.g., text-embedding-3-large
                # dimensions=self.dimension # Uncomment if model supports/requires it, can depend here 
            )

            embeddings = [item.embedding for item in response.data]
            print(f"Received {len(embeddings)} embeddings from OpenAI.")
            if embeddings:
                actual_dimension = len(embeddings[0])
                if actual_dimension != self.dimension:
                    print(f"CRITICAL WARNING: OpenAI returned dimension {actual_dimension}, but expected {self.dimension} for model {self.model}. Check model name and configuration. Returning potentially incompatible embeddings.")
                # else:
                #     print(f"Embedding dimension verified: {actual_dimension}")
            if len(embeddings) != len(non_empty_texts):
                 print(f"Warning: OpenAI returned {len(embeddings)} embeddings, but {len(non_empty_texts)} non-empty texts were sent. Result length mismatch.")
            # TODO : Consumer/ Userbse of goofind  must be aware of potential length difference if filtering happened.
            return embeddings

        except openai.APIConnectionError as e:
            print(f"OpenAI API Connection Error: {e}")
            raise
        except openai.RateLimitError as e:
            print(f"OpenAI API Rate Limit Error: {e}")
            raise
        except openai.APIStatusError as e:
            print(f"OpenAI API Status Error: {e.status_code} - {e.response}")
            raise
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during OpenAI embedding generation: {e}")
            import traceback
            traceback.print_exc()
            raise

# TODO : Support for other Embeddings can be added as a feuture work ( lets keep it simple for now)
# class VoyageAIEmbeddingProvider(EmbeddingProvider): ...
# class HuggingFaceEmbeddingProvider(EmbeddingProvider): ...