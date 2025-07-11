import torch
import torch.nn as nn
import numpy as np
import asyncio
from typing import Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from exo.download.shard_download import ShardDownloader
from exo.helpers import DEBUG


class HuggingFaceDistributedEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader, device: str = "cuda"):
        self.shard_downloader = shard_downloader
        self.model_shards = {}
        self.tokenizer_cache = {}
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.session = {}

    async def ensure_shard(self, shard: Shard):
        if shard in self.model_shards:
            return

        if DEBUG >= 2:
            print(f"Loading shard {shard} for HuggingFace distributed engine")

        full_model = AutoModelForCausalLM.from_pretrained(
            shard.model_id,
            torch_dtype=torch.float16,
            device_map="cpu",  # Keep on CPU initially
            trust_remote_code=True,
        )

        shard_layers = self._extract_layers(full_model, shard)

        if torch.cuda.is_available() and self.device == "cuda":
            self.model_shards[shard] = shard_layers.to(self.device)
        else:
            self.model_shards[shard] = shard_layers

        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if DEBUG >= 2:
            print(f"Shard {shard} loaded successfully")

    def _extract_layers(self, model, shard: Shard):
        """Extract specific layers for the shard"""
        layers = nn.ModuleList()

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            model_layers = model.model.layers
            embed_layer = getattr(model.model, "embed_tokens", None)
            norm_layer = getattr(model.model, "norm", None)
            lm_head = getattr(model, "lm_head", None)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            model_layers = model.transformer.h
            embed_layer = getattr(model.transformer, "wte", None)
            norm_layer = getattr(model.transformer, "ln_f", None)
            lm_head = getattr(model, "lm_head", None)
        else:
            raise ValueError(f"Unsupported model architecture for {shard.model_id}")

        if shard.is_first_layer() and embed_layer is not None:
            layers.append(embed_layer)

        start_idx = max(0, shard.start_layer)
        end_idx = min(len(model_layers), shard.end_layer + 1)

        for i in range(start_idx, end_idx):
            if i < len(model_layers):
                layers.append(model_layers[i])

        if shard.is_last_layer():
            if norm_layer is not None:
                layers.append(norm_layer)
            if lm_head is not None:
                layers.append(lm_head)

        return layers

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """Encode prompt to tokens using the tokenizer"""
        await self.ensure_shard(shard)

        tokenizer = await self._get_tokenizer(shard.model_id)

        def _encode():
            tokens = tokenizer.encode(prompt, return_tensors="pt")
            return tokens.numpy()

        return await asyncio.get_running_loop().run_in_executor(self.executor, _encode)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """Decode tokens to text using the tokenizer"""
        await self.ensure_shard(shard)

        tokenizer = await self._get_tokenizer(shard.model_id)

        def _decode():
            if isinstance(tokens, np.ndarray):
                if tokens.ndim > 1:
                    tokens_list = tokens.flatten().tolist()
                else:
                    tokens_list = tokens.tolist()
            else:
                tokens_list = tokens

            return tokenizer.decode(tokens_list, skip_special_tokens=True)

        return await asyncio.get_running_loop().run_in_executor(self.executor, _decode)

    async def sample(
        self, logits: np.ndarray, temperature: float = 0.8, top_p: float = 0.9
    ) -> np.ndarray:
        """Sample next token from logits with proper top-p sampling"""

        def _sample():
            if isinstance(logits, np.ndarray):
                logits_tensor = torch.from_numpy(logits).float()
            else:
                logits_tensor = logits.float()

            if logits_tensor.dim() == 1:
                logits_tensor = logits_tensor.unsqueeze(0)
            elif logits_tensor.dim() > 2:
                logits_tensor = logits_tensor[:, -1, :]

            if temperature > 0:
                logits_tensor = logits_tensor / temperature

            if top_p < 1.0:
                logits_tensor = self._top_p_filtering(logits_tensor, top_p)

            probs = torch.softmax(logits_tensor, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            return next_token.numpy()

        return await asyncio.get_running_loop().run_in_executor(self.executor, _sample)

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[..., 0] = False

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        logits[mask] = float("-inf")

        return logits

    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Process tensor through assigned layers"""
        await self.ensure_shard(shard)

        if inference_state is None:
            inference_state = {}

        def _infer():
            if isinstance(input_data, np.ndarray):
                x = torch.from_numpy(input_data).to(self.device)
            else:
                x = input_data.to(self.device)

            layers = self.model_shards[shard]

            for i, layer in enumerate(layers):
                layer_name = layer.__class__.__name__

                if DEBUG >= 3:
                    print(f"Processing layer {i}: {layer_name}, input shape: {x.shape}")

                if hasattr(layer, "forward"):
                    if "embed" in layer_name.lower():
                        x = layer(x.long())
                    elif "norm" in layer_name.lower():
                        x = layer(x)
                    elif (
                        "lm_head" in layer_name.lower() or "head" in layer_name.lower()
                    ):
                        x = layer(x)
                    else:
                        try:
                            x = layer(x)
                            if isinstance(x, tuple):
                                x = x[0]
                        except Exception as e:
                            if DEBUG >= 1:
                                print(f"Error in layer {i} ({layer_name}): {e}")
                            seq_len = x.shape[1]
                            attention_mask = torch.ones(
                                x.shape[0], seq_len, device=x.device
                            )
                            x = layer(x, attention_mask=attention_mask)
                            if isinstance(x, tuple):
                                x = x[0]
                else:
                    x = layer(x)

            return x.cpu().numpy()

        try:
            output_data = await asyncio.get_running_loop().run_in_executor(
                self.executor, _infer
            )

            return output_data, inference_state

        except Exception as e:
            if DEBUG >= 1:
                print(f"Error in infer_tensor for shard {shard}: {e}")
            return input_data, inference_state

    async def load_checkpoint(self, shard: Shard, path: str):
        """Load checkpoint for the shard"""
        await self.ensure_shard(shard)

        def _load():
            checkpoint = torch.load(path, map_location=self.device)
            if shard in self.model_shards:
                self.model_shards[shard].load_state_dict(checkpoint, strict=False)

        await asyncio.get_running_loop().run_in_executor(self.executor, _load)

    async def save_checkpoint(self, shard: Shard, path: str):
        """Save checkpoint for the shard"""
        await self.ensure_shard(shard)

        def _save():
            if shard in self.model_shards:
                checkpoint = self.model_shards[shard].state_dict()
                torch.save(checkpoint, path)

        await asyncio.get_running_loop().run_in_executor(self.executor, _save)

    async def _get_tokenizer(self, model_id: str):
        """Get tokenizer for the model"""
        if model_id not in self.tokenizer_cache:
            self.tokenizer_cache[model_id] = await resolve_tokenizer(model_id)
        return self.tokenizer_cache[model_id]

    def _get_device(self):
        """Get the current device"""
        if hasattr(self, "device"):
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    async def cleanup(self):
        """Clean up resources"""
        self.model_shards.clear()
        self.tokenizer_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
