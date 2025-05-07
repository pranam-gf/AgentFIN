import os
import uuid
import sys
import hashlib
import subprocess
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI, APIError

class VectorDBProvider:
    def upsert(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: List[str]):
        print(f"WARN: Using Placeholder VectorDBProvider.upsert for {len(vectors)} vectors.")
        pass

class PineconeProvider(VectorDBProvider):
    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int = None, metric: str = "cosine", create_if_not_exists: bool = True, cloud: str = "aws", region: str = "us-east-1"):
        print(f"Initializing PineconeProvider for index '{index_name}' in env '{environment}'...")
        if not api_key or not environment or not index_name:
            raise ValueError("Pinecone API key, environment, and index name are required.")
        try:
            print("DEBUG: Attempting Pinecone(api_key=...)")
            self.pc = Pinecone(api_key=api_key)
            print(f"DEBUG: Pinecone object created: {type(self.pc)}")
            print("DEBUG: Attempting self.pc.list_indexes()")
            index_list_obj = self.pc.list_indexes()
            print(f"DEBUG: list_indexes() called, type: {type(index_list_obj)}")
            index_list = index_list_obj.names()
            print(f"DEBUG: Index names: {index_list}")
            if index_name not in index_list:
                print(f"DEBUG: Index '{index_name}' not found, attempting creation...")
                if create_if_not_exists and dimension:
                    print(f"Pinecone index '{index_name}' not found. Creating with dimension {dimension}...")
                    print(f"DEBUG: Attempting self.pc.create_index(...) for {index_name}")
                    self.pc.create_index(
                        name=index_name,
                        dimension=dimension,
                        metric=metric,
                        spec=ServerlessSpec(
                            cloud=cloud,
                            region=region
                        )
                    )
                    print(f"DEBUG: Pinecone index '{index_name}' created successfully.")
                else:
                    error_msg = f"Pinecone index '{index_name}' does not exist."
                    if create_if_not_exists:
                        error_msg += " Cannot create index without dimension specified in provider config."
                    print(f"Warning: {error_msg}")
                    raise ValueError(error_msg)
            else:
                 print(f"DEBUG: Pinecone index '{index_name}' already exists.")
            print(f"DEBUG: Attempting self.pc.Index('{index_name}')")
            self.index = self.pc.Index(index_name)
            print(f"DEBUG: Index object obtained: {type(self.index)}")
            print("DEBUG: Attempting self.index.describe_index_stats()")
            stats = self.index.describe_index_stats()
            print(f"DEBUG: describe_index_stats() called, type: {type(stats)}")
            print(f"Index stats: Dimension: {stats.dimension}, Vector count: {stats.total_vector_count}")
        except Exception as e:
            print(f"ERROR CAUGHT in PineconeProvider.__init__: {e}")
            import traceback
            traceback.print_exc()
            raise

    def clear_index(self):
        """Deletes all vectors from the Pinecone index."""
        if not self.index:
            print("Error: Pinecone index is not initialized. Cannot clear.")
            return
        try:
            print(f"Clearing all vectors from Pinecone index '{self.index.name}'...")
            delete_response = self.index.delete(delete_all=True)
            print(f"Successfully initiated clearing of all vectors from index '{self.index.name}'. Response: {delete_response}")
            
        except Exception as e:
            print(f"Error clearing Pinecone index '{self.index.name}': {e}")
            raise

    def upsert(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: List[str]):
        if not self.index:
             print("Error: Pinecone index is not initialized.")
             return
        if not vectors or len(vectors) != len(metadata) or len(vectors) != len(ids):
            print("Error: Vectors, metadata, and IDs must be non-empty and have the same length.")
            return
        vectors_to_upsert = []
        for i, vec in enumerate(vectors):
            clean_meta = {}
            for k, v in metadata[i].items():
                if isinstance(v, (str, bool, float, int)):
                    clean_meta[k] = v
                elif isinstance(v, list):
                    if all(isinstance(item, str) for item in v):
                        clean_meta[k] = v
                    else:
                        print(f"Warning: Metadata key '{k}' contains a list with non-string items for ID {ids[i]}. Skipping this key.")
            vectors_to_upsert.append((ids[i], vec, clean_meta))
        print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone index...")
        try:
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                 batch = vectors_to_upsert[i:i + batch_size]
                 upsert_response = self.index.upsert(vectors=batch)
                 print(f"  Upserted batch {i//batch_size + 1}, response: {upsert_response}")
            print(f"Pinecone upsert completed for {len(vectors_to_upsert)} vectors.")
        except Exception as e:
            print(f"Error upserting to Pinecone: {e}")

load_dotenv()

class IngestionPipeline:
    def __init__(self, 
                 embedding_provider: str = "openai",
                 embedding_model_name: Optional[str] = None,
                 pageindex_dir_path: Optional[str] = None):
        print("Initializing Ingestion Pipeline (PageIndex-only mode)...")
        self.embedding_provider_name = embedding_provider.lower()
        self.embedding_model = embedding_model_name or self._get_default_embedding_model(self.embedding_provider_name)
        
        print("Unstructured.io client is DISABLED in this PageIndex-only version.")

        self.openai_client = None
        if self.embedding_provider_name == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set for 'openai' embedding provider.")
            self.openai_client = OpenAI(api_key=openai_api_key)
            print("OpenAI client initialized.")
        
        self.embedding_dimension = self._get_embedding_dimension(self.embedding_model)
        print(f"Using Embedding Provider (client-side): {self.embedding_provider_name}, Model: {self.embedding_model}, Dimension: {self.embedding_dimension}")
        
        self.vector_db_provider_name = "pinecone"
        
        self.pageindex_dir_path = pageindex_dir_path or os.getenv("PAGEINDEX_DIR_PATH")
        if self.pageindex_dir_path:
            self.pageindex_dir_path = os.path.abspath(self.pageindex_dir_path) 
            print(f"PageIndex integration enabled. Directory: {self.pageindex_dir_path}")
        else:
            print("CRITICAL WARNING: PageIndex integration is DISABLED (PAGEINDEX_DIR_PATH not set). PDF parsing will FAIL.")

        self.vector_db_provider: VectorDBProvider = self._load_vector_db_provider()

    def _get_default_embedding_model(self, provider_name: str) -> str:
        if provider_name == "openai":
            return "text-embedding-3-large"
        else:
            raise ValueError(f"No default embedding model specified for provider: {provider_name}")

    def _get_embedding_dimension(self, model_name: str) -> int:
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        dimension = model_dimensions.get(model_name)
        if dimension is None:
            print(f"Warning: Unknown embedding model '{model_name}'. Attempting to get dimension from environment variable EMBEDDING_DIMENSION or falling back to 1536.")
            return int(os.getenv("EMBEDDING_DIMENSION", 1536))
        return dimension

    def _load_vector_db_provider(self) -> VectorDBProvider:
        print(f"Loading Vector DB Provider: {self.vector_db_provider_name}")
        if self.vector_db_provider_name == "pinecone":
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENVIRONMENT")
            index_name = os.getenv("PINECONE_INDEX_NAME")
            create_index = os.getenv("PINECONE_CREATE_INDEX", "True").lower() == "true"
            pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
            pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
            if not api_key or not environment or not index_name:
                raise ValueError("PINECONE_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_INDEX_NAME are required.")
            try:
                if not self.embedding_dimension:
                     raise ValueError("Embedding dimension must be determined before initializing PineconeProvider.")
                return PineconeProvider(
                    api_key=api_key,
                    environment=environment,
                    index_name=index_name,
                    dimension=self.embedding_dimension,
                    create_if_not_exists=create_index,
                    cloud=pinecone_cloud,
                    region=pinecone_region
                )
            except Exception as e:
                print(f"Error loading PineconeProvider: {e}")
                raise
        else:
            raise ValueError(f"Unsupported vector DB provider: {self.vector_db_provider_name}")

    def _process_pageindex_node(self, node_data: Dict[str, Any], base_metadata: Dict[str, Any], elements_list: List[Dict[str, Any]]):
        """
        Recursively processes a node from PageIndex output and adds it to elements_list.
        """
        node_text = node_data.get("summary", "")
        if not node_text and node_data.get("title"):
            node_text = node_data.get("title", "")

        if node_text:
            element_metadata = {
                **base_metadata,
                "pageindex_title": node_data.get("title"),
                "pageindex_node_id": node_data.get("node_id"),
                "pageindex_start_page": node_data.get("start_index"),
                "pageindex_end_page": node_data.get("end_index"),
                "processing_method": "pageindex"
            }
            cleaned_element_metadata = {k: v for k, v in element_metadata.items() if v is not None}
            
            elements_list.append({
                "text": node_text,
                "metadata": cleaned_element_metadata,
            })

        for child_node in node_data.get("nodes", []):
            self._process_pageindex_node(child_node, base_metadata, elements_list)

    def parse_file(self, file_path: str, additional_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Parses a local PDF file using PageIndex and generates embeddings client-side.
        Other file types are not supported in this version.
        """
        print(f"Attempting to parse file with PageIndex: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return []

        if not self.pageindex_dir_path:
            print(f"Error: PageIndex directory not configured. Cannot parse PDF: {file_path}")
            return []

        if not file_path.lower().endswith(".pdf"):
            print(f"Error: Only PDF files are supported in this pipeline version. Skipping: {file_path}")
            return []

        if additional_metadata is None:
            additional_metadata = {}
        
        base_metadata = {
            "source": os.path.basename(file_path),
            "file_path": file_path,
            **additional_metadata
        }
        
        raw_elements: List[Dict[str, Any]] = [] 
        
        print(f"Processing PDF with PageIndex: {file_path}")
        try:
            pageindex_script = os.path.join(self.pageindex_dir_path, "run_pageindex.py")
            if not os.path.isfile(pageindex_script):
                print(f"  Error: PageIndex script not found at {pageindex_script}")
                raise FileNotFoundError(f"PageIndex script not found: {pageindex_script}")

            pdf_basename = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"{pdf_basename}_structure.json"
            pageindex_results_dir = os.path.join(self.pageindex_dir_path, "results")
            # Ensure results directory exists, PageIndex script might not create it.
            os.makedirs(pageindex_results_dir, exist_ok=True)
            output_json_path = os.path.join(pageindex_results_dir, output_filename)
            
            if os.path.exists(output_json_path): 
                print(f"  Removing existing PageIndex output: {output_json_path}")
                os.remove(output_json_path)

            cmd = [
                "python3", pageindex_script,
                "--pdf_path", os.path.abspath(file_path), 
                "--output_dir", pageindex_results_dir, # Explicitly tell PageIndex where to save
                "--if-add-node-summary", "yes"
            ]
            print(f"  Executing PageIndex: {' '.join(cmd)}")
            process_result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.pageindex_dir_path, check=False)

            if process_result.returncode == 0:
                print(f"  PageIndex script completed successfully.")
                if os.path.exists(output_json_path):
                    print(f"  Reading PageIndex output from: {output_json_path}")
                    with open(output_json_path, 'r', encoding='utf-8') as f:
                        pageindex_data = json.load(f)
                    
                    if isinstance(pageindex_data, dict):
                        if "page_nodes" in pageindex_data and isinstance(pageindex_data["page_nodes"], list):
                            for node in pageindex_data["page_nodes"]:
                                self._process_pageindex_node(node, base_metadata, raw_elements)
                        elif "title" in pageindex_data and "node_id" in pageindex_data : 
                            self._process_pageindex_node(pageindex_data, base_metadata, raw_elements)
                        else: 
                            if "document_description" in pageindex_data:
                                 print(f"  Document description found (length {len(pageindex_data['document_description'])}), not adding as a separate chunk.")
                            if "nodes" in pageindex_data and isinstance(pageindex_data["nodes"], list):
                                 for node in pageindex_data["nodes"]: 
                                     self._process_pageindex_node(node, base_metadata, raw_elements)
                            elif not raw_elements:
                                 print(f"  Warning: PageIndex output JSON is a dict but known structures (e.g. page_nodes, root node with title/node_id) not found or empty. Dict keys: {list(pageindex_data.keys())}")
                    elif isinstance(pageindex_data, list): 
                        for node in pageindex_data:
                            self._process_pageindex_node(node, base_metadata, raw_elements)
                    else:
                        print(f"  Warning: PageIndex output JSON is not a dict or list. Type: {type(pageindex_data)}")

                    if raw_elements:
                        print(f"  Successfully processed {len(raw_elements)} elements from PageIndex.")
                    else:
                        print(f"  Warning: PageIndex processing resulted in zero elements from JSON content.")
                        if process_result.stdout: print(f"  PageIndex stdout:\n{process_result.stdout}")
                        if process_result.stderr: print(f"  PageIndex stderr:\n{process_result.stderr}")
                else:
                    print(f"  Error: PageIndex output file not found at {output_json_path} despite successful run.")
                    if process_result.stdout: print(f"  PageIndex stdout:\n{process_result.stdout}")
                    if process_result.stderr: print(f"  PageIndex stderr:\n{process_result.stderr}")
            else:
                print(f"  Error: PageIndex script failed with return code {process_result.returncode}.")
                if process_result.stdout: print(f"  PageIndex stdout:\n{process_result.stdout}")
                if process_result.stderr: print(f"  PageIndex stderr:\n{process_result.stderr}")
        except FileNotFoundError as fnf_error:
            print(f"  File not found error during PageIndex setup: {fnf_error}")
        except Exception as e:
            print(f"  Exception during PageIndex processing for {file_path}: {e}")
            import traceback
            traceback.print_exc()
        
        if not raw_elements:
            print(f"No elements extracted from PDF {file_path} using PageIndex.")
            return []

        final_elements_with_embeddings = []
        texts_to_embed = [el["text"] for el in raw_elements if el.get("text")]

        if not texts_to_embed:
            print(f"No text content found in elements from {file_path} to embed.")
            for el in raw_elements:
                el["embeddings"] = None
                final_elements_with_embeddings.append(el)
            return final_elements_with_embeddings

        generated_embeddings = []
        if self.embedding_provider_name == "openai" and self.openai_client:
            print(f"Generating embeddings for {len(texts_to_embed)} text chunks using OpenAI model {self.embedding_model}...")
            openai_batch_size = 100 
            for i in range(0, len(texts_to_embed), openai_batch_size):
                batch_texts = texts_to_embed[i:i + openai_batch_size]
                try:
                    res = self.openai_client.embeddings.create(input=batch_texts, model=self.embedding_model)
                    generated_embeddings.extend([item.embedding for item in res.data])
                    print(f"  Generated embeddings for batch {i//openai_batch_size + 1}/{ (len(texts_to_embed) -1)//openai_batch_size + 1 }")
                except APIError as e:
                    print(f"OpenAI API Error during embedding batch {i//openai_batch_size + 1}: {e}")
                    generated_embeddings.extend([None] * len(batch_texts))
                except Exception as e:
                    print(f"Unexpected error during OpenAI embedding batch {i//openai_batch_size + 1}: {e}")
                    generated_embeddings.extend([None] * len(batch_texts))
        else:
            print(f"Warning: Embedding provider '{self.embedding_provider_name}' is not configured for client-side embeddings or client not initialized. Skipping embedding.")
            generated_embeddings = [None] * len(texts_to_embed)
        
        embedding_idx = 0
        for raw_el in raw_elements:
            final_element = raw_el.copy()
            if raw_el.get("text") and embedding_idx < len(generated_embeddings):
                final_element["embeddings"] = generated_embeddings[embedding_idx]
                embedding_idx += 1
            else:
                final_element["embeddings"] = None 
            final_elements_with_embeddings.append(final_element)
            
        if embedding_idx != len(generated_embeddings) and texts_to_embed: 
             print(f"Warning: Mismatch in number of embeddings generated ({len(generated_embeddings)}) and text chunks that required embedding ({embedding_idx}).")

        return final_elements_with_embeddings

    def store_chunks(self, chunks_with_embeddings: List[Dict[str, Any]]):
        """Stores chunks and their embeddings (extracted from the chunk dict) in the vector database."""
        if not chunks_with_embeddings:
            print("Error: No chunks provided for storage. Skipping storage.")
            return
        embeddings = []
        metadata_list = []
        ids = []
        successful_extractions = 0

        print(f"Preparing {len(chunks_with_embeddings)} chunks (with embeddings) for storage...")
        for chunk in chunks_with_embeddings:
            embedding_vector = chunk.get("embeddings")
            if embedding_vector is None:
                print(f"Warning: Skipping chunk because it lacks embeddings. Metadata: {chunk.get('metadata', {})}")
                continue
            chunk_meta = chunk.get("metadata", {}).copy()
            chunk_text_for_id = chunk.get("text", "")
            
            file_path_for_id = chunk_meta.get("file_path", chunk_meta.get("source", ""))
            if not file_path_for_id: 
                 print(f"Warning: Could not determine a unique document identifier for a chunk. Text: {chunk_text_for_id[:50]}... Using only text for ID.")

            id_string = f"{file_path_for_id}::{chunk_text_for_id}"
            deterministic_id = hashlib.sha256(id_string.encode('utf-8')).hexdigest()
            
            chunk_meta["text"] = chunk_text_for_id 
            embeddings.append(embedding_vector)
            metadata_list.append(chunk_meta)
            ids.append(deterministic_id) 
            successful_extractions += 1

        if not successful_extractions:
            print("Error: No chunks with embeddings were found to store.")
            return
            
        print(f"Extracted embeddings and metadata for {successful_extractions} chunks.")

        try:
            self.vector_db_provider.upsert(vectors=embeddings, metadata=metadata_list, ids=ids)
            print(f"Successfully stored {len(ids)} chunks.")

        except Exception as e:
            print(f"Error storing chunks in vector DB: {e}")

    def run(self, file_path: str, metadata: Dict[str, Any] = None):
        """Runs the full ingestion pipeline for a single file."""
        print(f"\n----- Starting pipeline for: {file_path} -----")
        if metadata:
            print(f"Using provided metadata: {metadata}")

        elements_with_embeddings = self.parse_file(file_path, additional_metadata=metadata)
        if not elements_with_embeddings:
            print(f"Pipeline halted for {file_path}: No elements parsed/returned by PageIndex.")
            return
        
        if elements_with_embeddings[0].get("embeddings") is None:
            print(f"Warning: PageIndex did not return embeddings for the first element of {file_path}. Check PageIndex configuration and subscription. Skipping storage.")
            return
        
        chunks_with_embeddings = elements_with_embeddings
        self.store_chunks(chunks_with_embeddings)
        print(f"----- Finished pipeline for: {file_path} -----")