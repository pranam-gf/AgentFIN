import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from dataclasses import dataclass
from src.retrieval_pipeline import RetrievedChunk
try:
    from copy.core.providers.llm import OpenAICompletionProvider
except ImportError as e:
    print(f"Warning: Could not import LLM provider from 'copy/'. Using placeholder. Error: {e}")
    class OpenAICompletionProvider:
        def __init__(self, config):
            print("WARN: Using Placeholder OpenAICompletionProvider")
        def complete(self, prompt: str) -> str:
            print("WARN: Using Placeholder OpenAICompletionProvider.complete")
            return f"This is a placeholder response. The actual implementation would use the configured LLM provider to generate a response based on the retrieved context."
class GenerationPipeline:
    def __init__(self):
        load_dotenv()
        self.llm_provider_name = os.getenv("GENERATOR_LLM_PROVIDER", "openai").lower()
        self.llm_model_name = os.getenv("GENERATOR_LLM_MODEL", "gpt-3.5-turbo")
        self.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
        self.max_tokens_per_chunk = int(os.getenv("MAX_TOKENS_PER_CHUNK", "1000"))
        self.system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant specialized in CFA Level 3 exam preparation. Answer the question based on the provided context. If the answer is not in the context, say 'I don't have enough information to answer that question.'")
        print(f"Initializing Generation Pipeline...")
        print(f"LLM Provider: {self.llm_provider_name} (Model: {self.llm_model_name})")
        print(f"Max Context Length: {self.max_context_length}")
        self.llm = self._load_llm_provider()
    def _load_llm_provider(self) -> Any:
        print(f"Loading LLM Provider: {self.llm_provider_name} (Model: {self.llm_model_name})")
        if self.llm_provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAICompletionProvider.")
            from types import SimpleNamespace
            config = SimpleNamespace(
                provider="openai",
                model=self.llm_model_name
            )
            try:
                return OpenAICompletionProvider(config=config)
            except Exception as e:
                print(f"Error loading OpenAICompletionProvider: {e}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider_name}")
    def _estimate_token_count(self, text: str) -> int:
        return len(text) // 4
    def _format_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"Question: {query}\n\n"
        prompt += "Context:\n"
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source_document", "Unknown")
            topic = chunk.metadata.get("cfa_topic", "Unknown")
            prompt += f"\n--- Chunk {i+1} (Source: {source}, Topic: {topic}) ---\n"
            prompt += chunk.text.strip() + "\n"
        prompt += "\n\nPlease provide a comprehensive answer to the question based on the context above:"
        return prompt
    def _handle_long_context(self, query: str, chunks: List[RetrievedChunk]) -> str:
        print("Context too long, splitting into batches...")
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        batches = []
        current_batch = []
        current_token_count = 0
        base_prompt_tokens = self._estimate_token_count(self._format_prompt(query, []))
        for chunk in sorted_chunks:
            chunk_tokens = self._estimate_token_count(chunk.text)
            if current_token_count + chunk_tokens + base_prompt_tokens > self.max_tokens_per_chunk:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [chunk]
                current_token_count = chunk_tokens
            else:
                current_batch.append(chunk)
                current_token_count += chunk_tokens
        if current_batch:
            batches.append(current_batch)
        print(f"Split context into {len(batches)} batches")
        partial_answers = []
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)}...")
            batch_prompt = self._format_prompt(query, batch)
            try:
                partial_answer = self.llm.complete(batch_prompt)
                partial_answers.append(partial_answer)
                print(f"Generated partial answer for batch {i+1}")
            except Exception as e:
                print(f"Error generating partial answer for batch {i+1}: {e}")
        if partial_answers:
            synthesis_prompt = f"""
            I've broken down a complex question into parts and received multiple partial answers.
            Please synthesize these into a single, comprehensive response.
            
            Question: {query}
            
            Partial Answers:
            {' '.join([f"\n\nPart {i+1}:\n{answer}" for i, answer in enumerate(partial_answers)])}
            
            Synthesized Answer:
            """
            try:
                final_answer = self.llm.complete(synthesis_prompt)
                print("Successfully synthesized final answer from partial answers")
                return final_answer
            except Exception as e:
                print(f"Error synthesizing final answer: {e}")
                return "\n\n".join([f"Part {i+1}:\n{answer}" for i, answer in enumerate(partial_answers)])
        else:
            return "I was unable to generate an answer based on the available context."
    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "I don't have any relevant information to answer that question."
        prompt = self._format_prompt(query, chunks)
        token_count = self._estimate_token_count(prompt)
        print(f"Estimated token count: {token_count}")
        if token_count > self.max_context_length:
            return self._handle_long_context(query, chunks)
        try:
            print("Generating answer...")
            answer = self.llm.complete(prompt)
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I encountered an error while trying to generate an answer."
if __name__ == "__main__":
    import sys
    from src.retrieval_pipeline import RetrievalPipeline
    if len(sys.argv) < 2:
        print("Usage: python generation_pipeline.py 'your query here'")
        sys.exit(1)
    query = sys.argv[1]
    try:
        retrieval = RetrievalPipeline()
        chunks = retrieval.retrieve(query)
        if not chunks:
            print("No relevant information found.")
            sys.exit(0)
        generation = GenerationPipeline()
        answer = generation.generate(query, chunks)
        print("\n=== Generated Answer ===\n")
        print(answer)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 