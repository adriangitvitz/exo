import numpy as np
import pytest
import torch
from exo.inference.huggingface.inference import HuggingFaceDistributedEngine
from exo.inference.shard import Shard
from exo.download.new_shard_download import NewShardDownloader


@pytest.mark.asyncio
async def test_hf_distributed_inference_engine():
    """
    Test that distributed inference across multiple shards produces
    the same results as single-shard inference
    """
    # Configuration
    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    n_layers = 24

    # Create inference engines
    inference_engine_1 = HuggingFaceDistributedEngine(NewShardDownloader())
    inference_engine_2 = HuggingFaceDistributedEngine(NewShardDownloader())

    prompt = "In a single word only, what is the last name of the current president of the USA?"

    # Single shard inference (full model)
    resp_full, _ = await inference_engine_1.infer_prompt(
        "A",
        shard=Shard(
            model_id=model_id, start_layer=0, end_layer=n_layers - 1, n_layers=n_layers
        ),
        prompt=prompt,
    )

    # Sample next token
    token_full = await inference_engine_1.sample(resp_full)
    token_full = token_full.reshape(1, -1)

    # Continue inference with sampled token
    next_resp_full, _ = await inference_engine_1.infer_tensor(
        "A",
        shard=Shard(
            model_id=model_id, start_layer=0, end_layer=n_layers - 1, n_layers=n_layers
        ),
        input_data=token_full,
    )

    # Distributed inference (split across two shards)
    pp = n_layers // 2

    # First shard processes prompt
    resp1, state1 = await inference_engine_1.infer_prompt(
        "B",
        shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=n_layers),
        prompt=prompt,
    )

    # Second shard processes intermediate result
    resp2, state2 = await inference_engine_2.infer_tensor(
        "B",
        shard=Shard(
            model_id=model_id,
            start_layer=pp + 1,
            end_layer=n_layers - 1,
            n_layers=n_layers,
        ),
        input_data=resp1,
        inference_state=state1,
    )

    # Sample token from distributed result
    tokens2 = await inference_engine_1.sample(resp2)
    tokens2 = tokens2.reshape(1, -1)

    # Continue distributed inference with sampled token
    resp3, state3 = await inference_engine_1.infer_tensor(
        "B",
        shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=n_layers),
        input_data=tokens2,
        inference_state=state2,
    )

    resp4, _ = await inference_engine_2.infer_tensor(
        "B",
        shard=Shard(
            model_id=model_id,
            start_layer=pp + 1,
            end_layer=n_layers - 1,
            n_layers=n_layers,
        ),
        input_data=resp3,
        inference_state=state3,
    )

    # Convert tensors to numpy for comparison
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    resp_full_np = to_numpy(resp_full)
    resp2_np = to_numpy(resp2)
    next_resp_full_np = to_numpy(next_resp_full)
    resp4_np = to_numpy(resp4)

    # Compare results with tolerance for floating point differences
    def arrays_close(a, b, rtol=1e-4, atol=1e-6):
        return np.allclose(a, b, rtol=rtol, atol=atol)

    assert arrays_close(resp_full_np, resp2_np), (
        f"First inference mismatch: shapes {resp_full_np.shape} vs {resp2_np.shape}"
    )
    assert arrays_close(next_resp_full_np, resp4_np), (
        f"Second inference mismatch: shapes {next_resp_full_np.shape} vs {resp4_np.shape}"
    )


# Keep your existing working tests
@pytest.mark.asyncio
async def test_hf_distributed_basic_functionality():
    """Test basic functionality of HuggingFace distributed engine"""
    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    n_layers = 24

    engine = HuggingFaceDistributedEngine(NewShardDownloader())
    shard = Shard(
        model_id=model_id, start_layer=0, end_layer=n_layers - 1, n_layers=n_layers
    )

    # Test encoding
    encoded = await engine.encode(shard, "Hello world")
    assert isinstance(encoded, np.ndarray)
    assert encoded.size > 0

    # Test decoding
    decoded = await engine.decode(shard, encoded)
    assert isinstance(decoded, str)
    assert len(decoded) > 0

    # Test inference
    result, state = await engine.infer_prompt("test", shard, "Test prompt")
    assert isinstance(result, np.ndarray)
    assert result.ndim >= 2


@pytest.mark.asyncio
async def test_hf_distributed_shard_loading():
    """Test that shards are loaded correctly"""
    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    n_layers = 24

    engine = HuggingFaceDistributedEngine(NewShardDownloader())

    # Test different shard configurations
    shards = [
        Shard(model_id=model_id, start_layer=0, end_layer=11, n_layers=n_layers),
        Shard(model_id=model_id, start_layer=12, end_layer=23, n_layers=n_layers),
        Shard(model_id=model_id, start_layer=0, end_layer=23, n_layers=n_layers),
    ]

    for shard in shards:
        await engine.ensure_shard(shard)
        assert shard in engine.model_shards
        assert len(engine.model_shards[shard]) > 0
