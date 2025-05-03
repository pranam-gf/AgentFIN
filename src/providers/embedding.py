import os
from typing import List, Dict, Any, Optional, Union, Tuple
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

class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Hugging Face models."""
    
    def __init__(self, model_name: str, device: str = "cpu", normalize_embeddings: bool = True):
        """
        Initialize the Hugging Face embedding provider.
        
        Args:
            model_name: Name of the model from Hugging Face Hub
            device: Device to run the model on ('cpu' or 'cuda')
            normalize_embeddings: Whether to normalize the embeddings
        """
        print(f"Initializing HuggingFaceEmbeddingProvider with model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self.normalize_embeddings = normalize_embeddings
            self.model_name = model_name
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"HuggingFaceEmbeddingProvider initialized for model {model_name} with dimension {self.dimension}.")
        except ImportError:
            print("Error: sentence-transformers package is required for HuggingFaceEmbeddingProvider.")
            raise
        except Exception as e:
            print(f"Error initializing Hugging Face model: {e}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts using the configured Hugging Face model."""
        if not texts:
            print("Warning: get_embeddings called with empty list.")
            return []
        
        processed_texts = [text.replace("\n", " ").strip() for text in texts]
        non_empty_texts = [text for text in processed_texts if text]
        
        if not non_empty_texts:
            print("Warning: All texts were empty after cleaning; returning no embeddings.")
            return []
        
        if len(non_empty_texts) < len(texts):
            print(f"Warning: Filtered out {len(texts) - len(non_empty_texts)} empty texts.")
        
        try:
            print(f"Generating embeddings for {len(non_empty_texts)} texts using model {self.model_name}...")
            embeddings = self.model.encode(
                non_empty_texts, 
                normalize_embeddings=self.normalize_embeddings,
                convert_to_tensor=False, 
                show_progress_bar=True if len(non_empty_texts) > 10 else False
            )
            
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            print(f"Generated {len(embeddings_list)} embeddings with dimension {len(embeddings_list[0]) if embeddings_list else 0}.")
            
            return embeddings_list
            
        except Exception as e:
            print(f"An unexpected error occurred during Hugging Face embedding generation: {e}")
            import traceback
            traceback.print_exc()
            raise

# Model catalog organized by categories from MTEB leaderboard
MODEL_CATALOG = {
    "english": {
        "E5-large-v2": {
            "model_id": "intfloat/e5-large-v2",
            "dimension": 1024,
            "provider": "huggingface",
            "description": "High-performance English embedding model"
        },
        "bge-large-en-v1.5": {
            "model_id": "BAAI/bge-large-en-v1.5",
            "dimension": 1024,
            "provider": "huggingface",
            "description": "BGE Large English model v1.5"
        },
        "gte-large": {
            "model_id": "thenlper/gte-large",
            "dimension": 1024,
            "provider": "huggingface",
            "description": "General Text Embeddings Large model"
        }
    },
    "multilingual": {
        "E5-large-v2-multilingual": {
            "model_id": "intfloat/multilingual-e5-large",
            "dimension": 1024,
            "provider": "huggingface",
            "description": "Multilingual version of E5-large"
        },
        "bge-large-en-v1.5-multilingual": {
            "model_id": "BAAI/bge-m3",
            "dimension": 1024,
            "provider": "huggingface",
            "description": "Multilingual version of BGE Large"
        }
    },
    "code": {
        "CodeBERT": {
            "model_id": "microsoft/codebert-base",
            "dimension": 768,
            "provider": "huggingface",
            "description": "Specialized for code embeddings"
        },
        "StarCoder": {
            "model_id": "bigcode/starcoder",
            "dimension": 4096,
            "provider": "huggingface",
            "description": "Large language model for code"
        }
    },
    "legal": {
        "Legal-BERT": {
            "model_id": "nlpaueb/legal-bert-base-uncased",
            "dimension": 768,
            "provider": "huggingface",
            "description": "BERT model fine-tuned on legal text"
        }
    },
    "medical": {
        "BioBERT": {
            "model_id": "dmis-lab/biobert-v1.1",
            "dimension": 768,
            "provider": "huggingface",
            "description": "BERT model fine-tuned on biomedical text"
        }
    },
    "openai": {
        "text-embedding-3-small": {
            "model_id": "text-embedding-3-small",
            "dimension": 1536,
            "provider": "openai",
            "description": "OpenAI's small embedding model"
        },
        "text-embedding-3-large": {
            "model_id": "text-embedding-3-large",
            "dimension": 3072,
            "provider": "openai",
            "description": "OpenAI's large embedding model"
        },
        "text-embedding-ada-002": {
            "model_id": "text-embedding-ada-002",
            "dimension": 1536,
            "provider": "openai",
            "description": "OpenAI's legacy embedding model"
        }
    }
}

def get_embedding_provider(provider_config: Dict[str, Any]) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider based on configuration.
    
    Args:
        provider_config: Dictionary with provider configuration
        
    Returns:
        An initialized EmbeddingProvider instance
    """
    provider_type = provider_config.get("provider_type", "").lower()
    
    if provider_type == "openai":
        api_key = provider_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        model = provider_config.get("model", "text-embedding-3-small")
        dimension = provider_config.get("dimension", MODEL_CATALOG["openai"].get(model, {}).get("dimension", 1536))
        return OpenAIEmbeddingProvider(api_key=api_key, model=model, dimension=dimension)
    
    elif provider_type == "huggingface":
        model_name = provider_config.get("model_name")
        if not model_name:
            raise ValueError("model_name is required for HuggingFaceEmbeddingProvider")
        device = provider_config.get("device", "cpu")
        normalize_embeddings = provider_config.get("normalize_embeddings", True)
        return HuggingFaceEmbeddingProvider(model_name=model_name, device=device, normalize_embeddings=normalize_embeddings)
    
    else:
        supported_providers = ["openai", "huggingface"]
        raise ValueError(f"Unsupported provider_type: {provider_type}. Supported providers: {supported_providers}")

def list_available_models(category: Optional[str] = None) -> Dict[str, Any]:
    """
    List available embedding models, optionally filtered by category.
    
    Args:
        category: Optional category to filter by
        
    Returns:
        Dictionary of available models with their details
    """
    if category:
        if category in MODEL_CATALOG:
            return MODEL_CATALOG[category]
        else:
            available_categories = list(MODEL_CATALOG.keys())
            raise ValueError(f"Unknown category: {category}. Available categories: {available_categories}")
    else:
        # Return all models across all categories
        return {category: models for category, models in MODEL_CATALOG.items()}

def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific embedding model.
    
    Args:
        model_name: Name of the model to get info for
        
    Returns:
        Dictionary with model details
    """
    for category, models in MODEL_CATALOG.items():
        if model_name in models:
            return {"category": category, **models[model_name]}
    
    raise ValueError(f"Unknown model: {model_name}")

def create_provider_from_model_name(model_name: str, **kwargs) -> EmbeddingProvider:
    """
    Create an embedding provider directly from a model name.
    
    Args:
        model_name: Name of the model from MODEL_CATALOG
        **kwargs: Additional configuration parameters
        
    Returns:
        An initialized EmbeddingProvider instance
    """
    model_info = get_model_info(model_name)
    provider_type = model_info["provider"]
    
    if provider_type == "openai":
        api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model_info["model_id"],
            dimension=model_info["dimension"]
        )
    elif provider_type == "huggingface":
        return HuggingFaceEmbeddingProvider(
            model_name=model_info["model_id"],
            device=kwargs.get("device", "cpu"),
            normalize_embeddings=kwargs.get("normalize_embeddings", True)
        )
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")
