import os
import argparse
from dotenv import load_dotenv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Optional, List, Dict, Any

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.ingestion_pipeline import IngestionPipeline

def get_valid_cfa_topics():
    core_topics = [
        "asset_allocation",
        "portfolio_construction",
        "performance_measurement",
        "derivatives_risk_management",
        "ethical_professional_standards",
        "alternative_investments",
    ]
    
    specialized_pathways = [
        "portfolio_management",
        "private_wealth",
        "private_markets",
    ]
    
    return core_topics, specialized_pathways

def process_single_file(pipeline: IngestionPipeline, file_path: str, metadata: dict, strategy: Optional[str]):
    """Helper function to process one file, suitable for parallel execution."""
    start_time = time.time()
    print(f"Starting processing for: {os.path.basename(file_path)}")
    try:
        pipeline.run(file_path, metadata=metadata, strategy=strategy)
        duration = time.time() - start_time
        print(f"Finished processing {os.path.basename(file_path)} in {duration:.2f} seconds")
        return True, file_path
    except Exception as e:
        duration = time.time() - start_time
        print(f"!!! Error processing {os.path.basename(file_path)} after {duration:.2f} seconds: {e}")
        return False, file_path

def auto_detect_cfa_metadata(file_path: str) -> Dict[str, Optional[str]]:
    core_topics, specialized_pathways = get_valid_cfa_topics()
    all_topics = core_topics + specialized_pathways

    path_lower = file_path.lower()
    detected_topic = None
    for topic in all_topics:
        search_term = topic.replace('_', ' ')
        if search_term in path_lower:
            detected_topic = topic
            break       
    if detected_topic:
        cfa_topic = detected_topic
        print(f"Auto-detected topic: {cfa_topic}")
        if detected_topic in specialized_pathways:
            cfa_pathway = detected_topic
            print(f"Auto-detected pathway: {cfa_pathway}")
        return {"cfa_topic": cfa_topic, "cfa_pathway": cfa_pathway}
    else:
        print("Could not auto-detect topic. Please specify with --topic.")
        if not cfa_topic:  
            print("Available topics:")
            print("Core topics: " + ", ".join(core_topics))
            print("Specialized pathways: " + ", ".join(specialized_pathways))
            sys.exit(1)  
        return {"cfa_topic": cfa_topic, "cfa_pathway": None}

def main():
    
    load_dotenv()

    
    core_topics, specialized_pathways = get_valid_cfa_topics()
    all_topics = core_topics + specialized_pathways

    parser = argparse.ArgumentParser(description="Ingest documents into the vector database using PageIndex.")
    parser.add_argument("paths", nargs='+', help="Paths to the document(s) or director(y/ies) to ingest.")
    parser.add_argument("--topic", help="CFA curriculum topic (e.g., 'asset_allocation').")
    parser.add_argument("--pathway", help="CFA specialized pathway (e.g., 'portfolio_management').")
    parser.add_argument("--auto-detect", action="store_true", help="Attempt to auto-detect CFA topic/pathway from file path/name.")
    parser.add_argument("--embedding_provider", default=os.getenv("EMBEDDING_PROVIDER", "openai"), help="Embedding provider (default: openai or EMBEDDING_PROVIDER env var).")
    parser.add_argument("--embedding_model_name", default=os.getenv("EMBEDDING_MODEL_NAME"), help="Name of the embedding model (e.g., text-embedding-3-large). Defaults to provider's default.")
    parser.add_argument("--pageindex_dir", default=os.getenv("PAGEINDEX_DIR_PATH"), help="Path to the PageIndex directory. Defaults to PAGEINDEX_DIR_PATH env var.")

    args = parser.parse_args()

    print("Starting ingestion process...")
    print(f"Target paths: {args.paths}")
    if args.topic: print(f"Specified CFA Topic: {args.topic}")
    if args.pathway: print(f"Specified CFA Pathway: {args.pathway}")
    if args.auto_detect: print("Auto-detection of CFA metadata enabled.")
    print(f"Embedding Provider: {args.embedding_provider}")
    if args.embedding_model_name: print(f"Embedding Model: {args.embedding_model_name}")
    if args.pageindex_dir: 
        print(f"PageIndex Directory: {args.pageindex_dir}")
    else:
        print("PageIndex Directory: Not specified, will rely on IngestionPipeline default (env var or None).")

    try:
        pipeline = IngestionPipeline(
            embedding_provider=args.embedding_provider,
            embedding_model_name=args.embedding_model_name,
            pageindex_dir_path=args.pageindex_dir
        )
    except Exception as e:
        print(f"Error initializing ingestion pipeline: {e}")
        sys.exit(1)

    files_to_process = []
    for path in args.paths:
        if os.path.isfile(path):
            files_to_process.append(path)
        elif os.path.isdir(path):
            print(f"Scanning directory: {path}")
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    if filename.lower().endswith(('.pdf', '.txt', '.md', '.docx', '.pptx')):
                        files_to_process.append(file_path)
                    else:
                        print(f"Skipping non-document file: {filename}")
                else:
                    print(f"Skipping non-file item: {filename}")
        else:
            print(f"Error: Input path is neither a file nor a directory: {path}")
            sys.exit(1)

    if not files_to_process:
        print("No files found to process.")
        sys.exit(0)

    print(f"Found {len(files_to_process)} files to process.")

    success_count = 0
    error_count = 0
    total_start_time = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file_path in files_to_process:
            metadata = auto_detect_cfa_metadata(file_path)
            futures.append(executor.submit(process_single_file, pipeline, file_path, metadata, None))

        for future in as_completed(futures):
            try:
                success, processed_file_path = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
            except Exception as exc:
                print(f'!!! Generated an exception processing a file: {exc}')
                error_count += 1

    total_duration = time.time() - total_start_time
    print("\n--- Ingestion Summary ---")
    print(f"Total time: {total_duration:.2f} seconds")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process: {error_count} files")
    print("Ingestion process finished.")

if __name__ == "__main__":
    main() 