"""Focused checks for the supplied fusion equations and three-slice inputs."""
import json
import math
import numpy as np
import torch
from design_network import (AvailabilityMaskedCrossAttention, build_model,
                            count_parameters, drop_one_modality, model_objective,
                            consistency_loss)
from eda import MODALITIES, build_25d_input


def run_checks():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(626)
        fusion = AvailabilityMaskedCrossAttention(8)
        streams = torch.randn(2, 4, 8, 3, 5, requires_grad=True)
        available = torch.tensor([[True, False, True, True], [False, True, False, False]])
        gated = streams * available[:, :, None, None, None]
        mean = gated.sum(1) / available.sum(1)[:, None, None, None]
        q = fusion.query(mean).reshape(2, 4, 2, 3, 5)
        keys = fusion.key(gated.reshape(8, 8, 3, 5)).reshape(2, 4, 4, 2, 3, 5)
        keys = keys + fusion.modality_embedding.reshape(1, 4, 4, 2, 1, 1)
        values = fusion.value(gated.reshape(8, 8, 3, 5)).reshape(2, 4, 4, 2, 3, 5)
        logits = (q[:, None] * keys).sum(3).permute(0, 2, 1, 3, 4) / math.sqrt(2)
        weights = logits.masked_fill(~available[:, None, :, None, None], -torch.inf).softmax(2)
        torch.testing.assert_close(weights.sum(2), torch.ones(2, 4, 3, 5))
        assert torch.count_nonzero(weights[0, :, 1]) == 0
        torch.testing.assert_close(weights[1, :, 1], torch.ones(4, 3, 5))
        manual_heads = (weights.permute(0, 2, 1, 3, 4)[:, :, :, None] * values).sum(1)
        manual = mean + fusion.output_projection(manual_heads.reshape(2, 8, 3, 5))
        torch.testing.assert_close(fusion(streams, available), manual)
        fusion(streams, available).square().mean().backward()
        assert torch.count_nonzero(streams.grad[0, 1]) == 0

        volumes = {name: np.broadcast_to(np.arange(5, dtype=np.float32),
                                         (35, 41, 5)).copy() + 10*index
                   for index, name in enumerate(MODALITIES)}
        for z, expected in [(0, [0, 0, 1]), (2, [1, 2, 3]), (4, [3, 4, 4])]:
            data = build_25d_input(volumes, z)
            assert data.shape == (16, 35, 41)
            np.testing.assert_array_equal(data[:3, 0, 0], expected)
            assert data[12:].all()
        missing_data = build_25d_input(volumes, 2, "t1c")
        assert not missing_data[3:6].any() and not missing_data[13].any()

        x = torch.randn(2, 16, 35, 41)
        x[:, 12:] = 1
        missing = drop_one_modality(x, torch.tensor([0, 3]))
        assert not missing[0, :3].any() and not missing[0, 12].any()
        assert not missing[1, 9:12].any() and not missing[1, 15].any()
        torch.testing.assert_close(missing[0, 3:12], x[0, 3:12])
        y = torch.randint(4, (2, 35, 41))
        brain = torch.ones_like(y, dtype=torch.bool)
        for mode in ("amca", "concatenation"):
            initial_states = []
            for regime in ("M0", "M1", "M2"):
                torch.manual_seed(11)
                model = build_model(regime, 4, mode)
                initial_states.append(model.state_dict())
                assert not any(isinstance(layer, torch.nn.Conv3d) for layer in model.modules())
                full, absent = model(x), model(missing)
                assert full.shape == (2, 4, 35, 41)
                loss, _ = model_objective(regime, full, y, absent, brain, 0.1)
                loss.backward()
                assert torch.isfinite(loss)
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                contaminated = missing.clone()
                contaminated[0, :3] = float("nan")
                contaminated[1, 9:12] = 1e8
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(missing), model(contaminated), rtol=0, atol=0)
                sole = x.clone()
                sole[:, 12:] = 0
                sole[:, 14] = 1
                with torch.no_grad():
                    assert torch.isfinite(model(sole)).all()
                sole[:, 14] = 0
                try:
                    model(sole)
                except ValueError as error:
                    assert "At least one" in str(error)
                else:
                    raise AssertionError("All-missing input must be rejected")
            for key in initial_states[0]:
                for state in initial_states[1:]:
                    torch.testing.assert_close(initial_states[0][key], state[key], rtol=0, atol=0)

        full = torch.randn(2, 4, 35, 41, requires_grad=True)
        absent = torch.randn_like(full, requires_grad=True)
        loss1, _ = model_objective("M1", full, y, absent, brain)
        loss2, _ = model_objective("M2", full, y, absent, brain, 0)
        torch.testing.assert_close(loss1, loss2, rtol=0, atol=0)
        consistency_loss(full, absent, brain).backward()
        assert full.grad is None and absent.grad is not None
        parameters = {mode: count_parameters(build_model("M2", 16, mode))
                      for mode in ("amca", "concatenation")}
        return {"status": "passed", "input_channels": 16, "attention_heads": 4,
                "spatial_context": "z-1, z, z+1 per modality with replicated boundaries",
                "parameters_width16": parameters,
                "checks": ["fusion matches equations", "missing key weights are zero",
                           "mean query", "key-only modality embeddings", "single available modality",
                           "identical seeded architecture across M0/M1/M2",
                           "three-slice ordering and boundary handling", "whole-sequence dropout",
                           "forward/backward and masked-stream invariance",
                           "M2(lambda=0) equals M1", "detached consistency target"]}
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
