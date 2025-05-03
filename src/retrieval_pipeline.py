import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from pinecone import Pinecone
from dataclasses import dataclass
from src.providers.embedding import EmbeddingProvider, OpenAIEmbeddingProvider

# TODO : Add LLM provider
class PlaceholderLLMProvider:
    def __init__(self, config):
        print(f"WARN: Using PlaceholderLLMProvider with config: {config}")
        pass
    def complete(self, prompt: str) -> str:
        print("WARN: Using PlaceholderLLMProvider.complete")
        # TODO : Simple expansion placeholder
        return f"{prompt} | asset allocation strategies | portfolio management techniques | risk analysis"

# TODO : Use the placeholder for OpenAICompletionProvider if the real one isn't defined elsewhere
OpenAICompletionProvider = PlaceholderLLMProvider

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. Reranking will be disabled.")
    RERANKER_AVAILABLE = False
    CrossEncoder = None 

@dataclass
class RetrievedChunk:
    """Represents a chunk of text retrieved from the vector database."""
    text: str  
    metadata: Dict[str, Any]  
    score: float  

class RetrievalPipeline:
    """Handles the retrieval of relevant chunks for a given query."""
    
    def __init__(self):
        """Initialize the retrieval pipeline using environment variables."""
        
        load_dotenv()
        
        
        self.embedding_provider_name = os.getenv("EMBEDDING_MODEL_PROVIDER", "openai").lower()
        self.embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
        
        # Sample embedngs here for now.
        self.model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", self.model_dimensions.get(self.embedding_model, 1536)))
        
        
        self.use_query_expansion = os.getenv("USE_QUERY_EXPANSION", "false").lower() == "true"
        self.query_expansion_provider = os.getenv("QUERY_EXPANSION_PROVIDER", "openai").lower()
        self.query_expansion_model = os.getenv("QUERY_EXPANSION_MODEL", "gpt-3.5-turbo").lower()
        
        
        self.search_top_k = int(os.getenv("SEARCH_TOP_K", "10"))
        
        
        self.use_reranking = os.getenv("USE_RERANKING", "false").lower() == "true"
        self.rerank_model_name = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.rerank_top_n = int(os.getenv("RERANK_TOP_N", "5"))
        
        
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        
        print(f"Initializing Retrieval Pipeline...")
        print(f"Embedding Provider: {self.embedding_provider_name} (Model: {self.embedding_model}, Dimension: {self.embedding_dimension})")
        print(f"Search Top-K: {self.search_top_k}")
        
        if self.use_query_expansion:
            print(f"Query Expansion: Enabled (Provider: {self.query_expansion_provider}, Model: {self.query_expansion_model})")
        else:
            print("Query Expansion: Disabled")
            
        if self.use_reranking and RERANKER_AVAILABLE:
            print(f"Reranking: Enabled (Model: {self.rerank_model_name}, Top-N: {self.rerank_top_n})")
        else:
            print("Reranking: Disabled")
        
        
        self.embedding_provider = self._load_embedding_provider()
        self.pinecone_index = self._init_pinecone()
        
        
        self.query_expansion_provider = self._load_query_expansion_provider() if self.use_query_expansion else None
        self.reranker = self._load_reranker() if self.use_reranking and RERANKER_AVAILABLE else None
    
    def _load_embedding_provider(self) -> EmbeddingProvider:
        """Load the embedding provider based on configuration."""
        print(f"Loading Embedding Provider: {self.embedding_provider_name} (Model: {self.embedding_model})")
        
        if self.embedding_provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider.")
            
            try:
                return OpenAIEmbeddingProvider(
                    api_key=api_key,
                    model=self.embedding_model,
                    dimension=self.embedding_dimension
                )
            except Exception as e:
                print(f"Error loading OpenAIEmbeddingProvider: {e}")
                raise
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider_name}")
    
    def _load_query_expansion_provider(self) -> Any:
        """Load the query expansion provider based on configuration."""
        print(f"Loading Query Expansion Provider: {self.query_expansion_provider}")
        
        if self.query_expansion_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY not found. Query expansion disabled.")
                self.use_query_expansion = False
                return None
            
            from types import SimpleNamespace
            config = SimpleNamespace(
                provider="openai",
                model=self.query_expansion_model
            )
            
            try:
                return OpenAICompletionProvider(config=config)
            except Exception as e:
                print(f"Error loading OpenAICompletionProvider (Placeholder): {e}")
                self.use_query_expansion = False
                print("Query Expansion disabled due to provider loading error.")
                return None
        else:
            print(f"Warning: Unsupported query expansion provider '{self.query_expansion_provider}'. Query expansion disabled.")
            self.use_query_expansion = False
            return None
    
    def _load_reranker(self) -> Any:
        """Load the reranker model if available."""
        if not RERANKER_AVAILABLE:
            return None
            
        print(f"Loading Reranker: {self.rerank_model_name}")
        try:
            return CrossEncoder(self.rerank_model_name)
        except Exception as e:
            print(f"Error loading reranker: {e}")
            self.use_reranking = False
            print("Reranking disabled due to model loading error.")
            return None
    
    def _init_pinecone(self) -> Any:
        """Initialize connection to Pinecone vector DB."""
        if not self.pinecone_api_key or not self.pinecone_environment or not self.pinecone_index_name:
            raise ValueError("PINECONE_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_INDEX_NAME are required.")
        
        try:
            
            pc = Pinecone(api_key=self.pinecone_api_key)
            
            
            index_list_obj = pc.list_indexes() 
            index_list = index_list_obj.names() 
            if self.pinecone_index_name not in index_list: 
                
                raise ValueError(f"Pinecone index '{self.pinecone_index_name}' does not exist. Please run the ingestion script first.")
            
            
            index = pc.Index(self.pinecone_index_name)
            stats = index.describe_index_stats()
            print(f"Connected to Pinecone index: {self.pinecone_index_name}")
            print(f"Index stats - Dimension: {stats.dimension}, Vector count: {stats.total_vector_count}")
            return index
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")
            raise
    
    def expand_query(self, query: str) -> str:
        """
        Expand the query using an LLM to improve retrieval.
        
        This method uses the configured LLM to generate an expanded version of the query,
        including potential keywords, paraphrases, or related terms.
        """
        if not self.use_query_expansion or not self.query_expansion_provider:
            return query
            
        print(f"Expanding query: '{query}'")
        try:
            expansion_prompt = f"""
            I need help expanding the following question to improve retrieval from a vector database.
            Please add relevant keywords, synonyms, or restated versions of the question. 
            Focus on financial and CFA Level 3 exam terminology.
            
            Original Question: {query}
            
            Expanded Question (combine all versions in a single paragraph):
            """
            expanded_query = f"{query} finance investment portfolio management CFA level 3 exam" 
            print(f"Expanded query: '{expanded_query}'")
            return expanded_query
        except Exception as e:
            print(f"Error during query expansion: {e}")
            return query  
    
    def retrieve(
        self, 
        query: str, 
        filter_by_topic: Optional[str] = None, 
        filter_by_pathway: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for the given query.
        
        Args:
            query: The user's query
            filter_by_topic: Optional CFA topic to filter by
            filter_by_pathway: Optional CFA pathway to filter by
            top_k: Number of results to return (overrides default)
            
        Returns:
            List of retrieved chunks with their metadata and relevance scores
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")
            
        
        top_k = top_k or self.search_top_k
        
        print(f"Processing query: '{query}'")
        
        
        if self.use_query_expansion:
            expanded_query = self.expand_query(query)
        else:
            expanded_query = query
        
        
        print("Generating query embedding...")
        try:
            query_embedding_list = self.embedding_provider.get_embeddings([expanded_query])
            if not query_embedding_list:
                 print("Error: Failed to generate embedding for the query.")
                 return []
            query_embedding = query_embedding_list[0]
            print(f"Generated query embedding with dimension {len(query_embedding)}.")
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []
        
        
        filter_dict = {}
        if filter_by_topic:
            filter_dict["cfa_topic"] = {"$eq": filter_by_topic}
        if filter_by_pathway:
            filter_dict["cfa_pathway"] = {"$eq": filter_by_pathway}
        
        filter_condition = filter_dict if filter_dict else None
        if filter_condition:
            filter_desc = ", ".join(f"{k}={v['$eq']}" for k, v in filter_dict.items())
            print(f"Applying metadata filter: {filter_desc}")
        
        
        print(f"Searching Pinecone for top {top_k} matches...")
        try:
            search_results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_condition if filter_condition else None,
                include_metadata=True
            )
        except Exception as e:
            print(f"Error during Pinecone query: {e}")
            return []
        
        
        retrieved_chunks = []
        
        
        for match in search_results.matches:
            
            text = match.metadata.get("text", "")
            
            metadata = {k: v for k, v in match.metadata.items() if k != "text"}
            retrieved_chunks.append(RetrievedChunk(
                text=text,
                metadata=metadata,
                score=match.score
            ))
        
        print(f"Retrieved {len(retrieved_chunks)} chunks from vector search.")
        
        
        if self.use_reranking and self.reranker and len(retrieved_chunks) > 1:
            print(f"Reranking results...")
            
            
            pairs = [(expanded_query, chunk.text) for chunk in retrieved_chunks]
            
            
            rerank_scores = self.reranker.predict(pairs)
            
            
            for i, score in enumerate(rerank_scores):
                retrieved_chunks[i].score = float(score)
            
            
            retrieved_chunks.sort(key=lambda x: x.score, reverse=True)
            
            
            if self.rerank_top_n < len(retrieved_chunks):
                retrieved_chunks = retrieved_chunks[:self.rerank_top_n]
                print(f"Kept top {self.rerank_top_n} chunks after reranking.")
        
        return retrieved_chunks


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python retrieval_pipeline.py 'your query here'")
        sys.exit(1)
    
    query = sys.argv[1]
    topic = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        pipeline = RetrievalPipeline()
        results = pipeline.retrieve(query, filter_by_topic=topic)
        
        print(f"\nRetrieved {len(results)} chunks for query: '{query}'")
        for i, chunk in enumerate(results):
            print(f"\n--- Result {i+1} (Score: {chunk.score:.4f}) ---")
            print(f"Topic: {chunk.metadata.get('cfa_topic', 'Unknown')}")
            print(f"Source: {chunk.metadata.get('source_document', 'Unknown')}")
            print(f"\nText: {chunk.text[:200]}...")  
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 