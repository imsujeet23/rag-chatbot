import torch
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class HuggingFaceLLM:
    """LLM integration using HuggingFace transformers."""
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str = "auto"
    ):
        """
        Initialize HuggingFace LLM.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.model_name = model_name
            
            # Set device
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            logger.info(f"Loading model: {model_name} on device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=(self.device == "cpu")
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def generate_with_context(
        self,
        context: str,
        query: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        Generate response based on context and query (RAG style).
        
        Args:
            context: Retrieved context from vector store
            query: User query
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated response
        """
        # Create RAG prompt
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        return self.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
    
    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of prompts
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            List of generated texts
        """
        responses = []
        for prompt in prompts:
            response = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature
            )
            responses.append(response)
        
        return responses
