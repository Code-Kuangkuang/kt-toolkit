import copy
import hashlib

import torch


def apply_train_label_flip(dataset, ratio, seed):
    """Return a dataset copy with deterministic binary response flips.

    The source dataset and its on-disk cache are left unchanged. Flips are
    applied to the full response sequence before ``__getitem__`` creates
    ``rseqs`` and ``shft_rseqs``, so the same interaction stays consistent
    when it appears once as a target and once as a later history input.
    """
    ratio = float(ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"train label flip ratio must be in [0, 1], got {ratio}.")

    if not hasattr(dataset, "dori") or "rseqs" not in dataset.dori:
        raise TypeError("train label flipping requires a dataset with dori['rseqs'].")

    responses = dataset.dori["rseqs"]
    if not torch.is_tensor(responses) or responses.ndim != 2:
        raise TypeError(
            "train label flipping requires dori['rseqs'] to be a 2D tensor."
        )
    valid_mask = (responses == 0) | (responses == 1)
    eligible_count = int(valid_mask.sum().item())
    flipped_count = int(ratio * eligible_count + 0.5)
    if flipped_count > 0 and ("sdseqs" in dataset.dori or "qdseqs" in dataset.dori):
        raise ValueError(
            "train label flipping is not supported with response-derived "
            "difficulty features; they would retain information from clean labels."
        )

    flip_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    if flipped_count > 0:
        eligible_flat = torch.nonzero(
            valid_mask.reshape(-1), as_tuple=False
        ).flatten()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        selected = eligible_flat[
            torch.randperm(eligible_count, generator=generator)[:flipped_count]
        ]
        flip_mask.view(-1)[selected] = True

    noisy_dataset = copy.copy(dataset)
    noisy_dataset.dori = dict(dataset.dori)
    noisy_responses = responses.clone()
    noisy_responses[flip_mask] = 1 - noisy_responses[flip_mask]
    noisy_dataset.dori["rseqs"] = noisy_responses

    # ATDKT's history feature is response-derived. Recompute it from the
    # corrupted training responses so clean labels do not remain in features.
    if flipped_count > 0 and "historycorrs" in noisy_dataset.dori:
        correct_so_far = torch.where(
            valid_mask,
            noisy_responses,
            torch.zeros_like(noisy_responses),
        ).cumsum(dim=1)
        attempts_so_far = valid_mask.to(noisy_responses.dtype).cumsum(dim=1)
        history = correct_so_far / attempts_so_far.clamp_min(1)
        noisy_dataset.dori["historycorrs"] = torch.where(
            valid_mask,
            history,
            torch.zeros_like(history),
        )

    mask_bytes = flip_mask.cpu().contiguous().numpy().tobytes()
    noisy_dataset.label_flip_info = {
        "enabled": bool(flipped_count),
        "requested_ratio": ratio,
        "actual_ratio": (
            float(flipped_count / eligible_count) if eligible_count else 0.0
        ),
        "seed": int(seed),
        "eligible_count": eligible_count,
        "flipped_count": flipped_count,
        "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "scope": "train_only",
        "unit": (
            "question_interaction"
            if getattr(dataset, "dataset_mode", None) == "all_in_one"
            else "stored_response_position"
        ),
    }
    return noisy_dataset
