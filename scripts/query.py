import os
import sys
import argparse
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.retrieval_pipeline import RetrievalPipeline

def get_valid_cfa_topics():
    """Returns valid CFA topics to validate user input."""
    core_topics = [
        "asset_allocation",
        "portfolio_construction",
        "performance_measurement",
        "derivatives_risk_management",
        "ethical_professional_standards",
    ]
    
    specialized_pathways = [
        "portfolio_management",
        "private_wealth",
        "private_markets",
    ]
    
    return core_topics, specialized_pathways

def main():
    
    load_dotenv()
    
    # TODO : ned to add a valudi method to check if the topic and pahway are valid
    core_topics, specialized_pathways = get_valid_cfa_topics()
    all_topics = core_topics + specialized_pathways
    
    
    parser = argparse.ArgumentParser(description="Query the CFA RAG system.")
    parser.add_argument("query", type=str, help="The query to process.")
    parser.add_argument("--topic", type=str, choices=all_topics, 
                      help="Filter results by CFA topic. One of: " + ", ".join(all_topics))
    parser.add_argument("--pathway", type=str, choices=specialized_pathways,
                      help="Filter results by CFA pathway. One of: " + ", ".join(specialized_pathways))
    parser.add_argument("--top-k", type=int, default=None, 
                      help="Number of results to return. Default is from .env or 10.")
    parser.add_argument("--expand", action="store_true", 
                      help="Enable query expansion using LLM.")
    parser.add_argument("--rerank", action="store_true",
                      help="Enable reranking of results.")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Show more detailed output.")
    
    args = parser.parse_args()
    
    query = args.query
    topic = args.topic
    pathway = args.pathway
    top_k = args.top_k
    
    
    if args.expand:
        os.environ["USE_QUERY_EXPANSION"] = "true"
    if args.rerank:
        os.environ["USE_RERANKING"] = "true"
    
    try:
        pipeline = RetrievalPipeline()
        results = pipeline.retrieve(
            query=query,
            filter_by_topic=topic,
            filter_by_pathway=pathway,
            top_k=top_k
        )
        
        
        print(f"\n=== Retrieved {len(results)} chunks for query: '{query}' ===\n")
        
        if not results:
            print("No results found.")
            return
        
        for i, chunk in enumerate(results):
            print(f"\n=== Result {i+1} (Score: {chunk.score:.4f}) ===")
            print(f"Topic: {chunk.metadata.get('cfa_topic', 'Unknown')}")
            print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
            
            if args.verbose:
                # Print detailed metadata if verbose flag is set
                print("\nRaw Metadata:")
                print(chunk.metadata) # DEBUG: Print raw metadata
                
            
            text = chunk.text
            print(f"DEBUG: repr(text) = {repr(text)}") # DEBUG: Print representation of text
            if len(text) > 500 and not args.verbose:
                text = text[:497] + "..."
            print(f"\nText: {text}")
            
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 