import os
import sys
import argparse
from dotenv import load_dotenv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.retrieval_pipeline import RetrievalPipeline
from src.generation_pipeline import GenerationPipeline

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
    
    
    core_topics, specialized_pathways = get_valid_cfa_topics()
    all_topics = core_topics + specialized_pathways
    
    
    parser = argparse.ArgumentParser(description="Ask questions about CFA Level 3 materials.")
    parser.add_argument("query", type=str, help="The question to ask.")
    parser.add_argument("--topic", type=str, choices=all_topics, 
                      help="Filter results by CFA topic. One of: " + ", ".join(all_topics))
    parser.add_argument("--pathway", type=str, choices=specialized_pathways,
                      help="Filter results by CFA pathway. One of: " + ", ".join(specialized_pathways))
    parser.add_argument("--top-k", type=int, default=None, 
                      help="Number of chunks to retrieve. Default is from .env or 10.")
    parser.add_argument("--expand", action="store_true", 
                      help="Enable query expansion using LLM.")
    parser.add_argument("--rerank", action="store_true",
                      help="Enable reranking of results.")
    parser.add_argument("--show-sources", action="store_true",
                      help="Display source information alongside the answer.")
    parser.add_argument("--show-chunks", action="store_true",
                      help="Display the retrieved chunks.")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Show detailed pipeline execution information.")
    
    args = parser.parse_args()
    
    query = args.query
    topic = args.topic
    pathway = args.pathway
    top_k = args.top_k
    
    
    if not args.verbose:
        
        sys.stdout = open(os.devnull, 'w')
    
    
    if args.expand:
        os.environ["USE_QUERY_EXPANSION"] = "true"
    if args.rerank:
        os.environ["USE_RERANKING"] = "true"
    
    try:
        
        retrieval_pipeline = RetrievalPipeline()
        
        
        retrieved_chunks = retrieval_pipeline.retrieve(
            query=query,
            filter_by_topic=topic,
            filter_by_pathway=pathway,
            top_k=top_k
        )
        
        
        if not args.verbose:
            sys.stdout = sys.__stdout__
        
        
        if args.show_chunks:
            print(f"\n=== Retrieved {len(retrieved_chunks)} chunks for: '{query}' ===\n")
            
            if not retrieved_chunks:
                print("No relevant information found.")
            else:
                for i, chunk in enumerate(retrieved_chunks):
                    print(f"\n--- Chunk {i+1} (Score: {chunk.score:.4f}) ---")
                    print(f"Source: {chunk.metadata.get('source_document', 'Unknown')}")
                    print(f"Topic: {chunk.metadata.get('cfa_topic', 'Unknown')}")
                    
                    
                    text = chunk.text
                    if len(text) > 200:
                        text = text[:197] + "..."
                    print(f"\nText: {text}")
            print("\n" + "-" * 80 + "\n")
        
        
        if not retrieved_chunks:
            print("I couldn't find any relevant information to answer your question.")
            return
            
        
        generation_pipeline = GenerationPipeline()
        
        
        answer = generation_pipeline.generate(query, retrieved_chunks)
        
        
        if not args.verbose:
            sys.stdout = sys.__stdout__
        
        
        print("\n" + "=" * 40 + " ANSWER " + "=" * 40 + "\n")
        print(answer)
        print("\n" + "=" * 88 + "\n")
        
        
        if args.show_sources:
            print("Sources:")
            
            sources = set()
            for chunk in retrieved_chunks:
                source = chunk.metadata.get('source_document', 'Unknown')
                topic = chunk.metadata.get('cfa_topic', 'Unknown')
                sources.add(f"- {source} (Topic: {topic})")
            
            
            for source in sources:
                print(source)
            print()
            
    except Exception as e:
        
        if not args.verbose:
            sys.stdout = sys.__stdout__
            
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 