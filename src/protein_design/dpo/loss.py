import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Sequence, Tuple, TypedDict, Union

from protein_design.model import ESM2Model


def _diff_positions(winner: str, loser: str) -> List[int]:
    if len(winner) != len(loser):
        raise ValueError("Winner and loser must have the same sequence length.")
    return [idx for idx, (w_char, l_char) in enumerate(zip(winner, loser)) if w_char != l_char]


class PairMember(TypedDict):
    aa: str
    score: float


PairMemberLike = Union[str, PairMember, Dict[str, Any]]
PairLike = Tuple[PairMemberLike, PairMemberLike]
PairBatchLike = Union[PairLike, Sequence[PairLike]]


def _is_pair_member(value: Any) -> bool:
    return isinstance(value, (str, dict))


def _as_pair_batch(pairs: PairBatchLike) -> List[PairLike]:
    if isinstance(pairs, tuple) and len(pairs) == 2 and _is_pair_member(pairs[0]) and _is_pair_member(pairs[1]):
        return [pairs]
    return list(pairs)


def _member_to_sequence(member: PairMemberLike) -> str:
    if isinstance(member, str):
        return member
    if "aa" not in member:
        raise KeyError("Expected pair member dict to contain key 'aa'.")
    return str(member["aa"])


def _member_to_score(member: PairMemberLike, require_score: bool = False) -> float:
    if isinstance(member, str):
        if require_score:
            raise TypeError("weighted_dpo_loss expects dict pair members with keys 'aa' and 'score'.")
        return 0.0
    if "score" not in member:
        if require_score:
            raise KeyError("weighted_dpo_loss expects each pair member to include key 'score'.")
        return 0.0
    return float(member["score"])


def dpo_loss(
    pair: PairBatchLike,
    beta: float, 
    scorer: ESM2Model, 
    reference: ESM2Model,
    policy_use_grad: bool = True,
) -> torch.Tensor:
    """Compute mean DPO loss over one pair or a batch of pairs."""
    pair_batch = _as_pair_batch(pair)
    losses: List[torch.Tensor] = []

    for winner, loser in pair_batch:
        winner_seq = _member_to_sequence(winner)
        loser_seq = _member_to_sequence(loser)
        diff_positions = _diff_positions(winner_seq, loser_seq)
        if len(diff_positions) == 0:
            continue
        
        w_masked_pll = scorer.masked_pseudo_log_likelihood(
            [winner_seq],
            diff_positions,
            use_grad=policy_use_grad,
            positions_are_cdr=True,
        ).squeeze(0)
        l_masked_pll = scorer.masked_pseudo_log_likelihood(
            [loser_seq],
            diff_positions,
            use_grad=policy_use_grad,
            positions_are_cdr=True,
        ).squeeze(0)

        ref_w_masked_pll = reference.masked_pseudo_log_likelihood(
            [winner_seq],
            diff_positions,
            use_grad=False,
            positions_are_cdr=True,
        ).squeeze(0)
        ref_l_masked_pll = reference.masked_pseudo_log_likelihood(
            [loser_seq],
            diff_positions,
            use_grad=False,
            positions_are_cdr=True,
        ).squeeze(0)

        delta_score = w_masked_pll - l_masked_pll
        delta_ref_score = ref_w_masked_pll - ref_l_masked_pll
        losses.append(-F.logsigmoid(beta * (delta_score - delta_ref_score)))

    if not losses:
        raise ValueError("No valid non-identical winner-loser pairs in batch.")

    return torch.stack(losses).mean()


def weighted_dpo_loss(
    pair: PairBatchLike,
    beta: float,
    temperature: float,
    scorer: ESM2Model, 
    reference: ESM2Model,
    policy_use_grad: bool = True,
) -> torch.Tensor:
    """Compute mean weighted DPO loss over one pair or a batch of pairs."""
    pair_batch = _as_pair_batch(pair)
    losses: List[torch.Tensor] = []

    for winner, loser in pair_batch:
        winner_seq = _member_to_sequence(winner)
        loser_seq = _member_to_sequence(loser)
        r_w = _member_to_score(winner, require_score=True)
        r_l = _member_to_score(loser, require_score=True)

        diff_positions = _diff_positions(winner_seq, loser_seq)
        if len(diff_positions) == 0:
            continue
        
        w_masked_pll = scorer.masked_pseudo_log_likelihood(
            [winner_seq],
            diff_positions,
            use_grad=policy_use_grad,
            positions_are_cdr=True,
        ).squeeze(0)
        l_masked_pll = scorer.masked_pseudo_log_likelihood(
            [loser_seq],
            diff_positions,
            use_grad=policy_use_grad,
            positions_are_cdr=True,
        ).squeeze(0)

        ref_w_masked_pll = reference.masked_pseudo_log_likelihood(
            [winner_seq],
            diff_positions,
            use_grad=False,
            positions_are_cdr=True,
        ).squeeze(0)
        ref_l_masked_pll = reference.masked_pseudo_log_likelihood(
            [loser_seq],
            diff_positions,
            use_grad=False,
            positions_are_cdr=True,
        ).squeeze(0)

        delta_score = w_masked_pll - l_masked_pll
        delta_ref_score = ref_w_masked_pll - ref_l_masked_pll
        weights = torch.softmax(
            torch.tensor([r_w / temperature, r_l / temperature], dtype=delta_score.dtype, device=delta_score.device),
            dim=0,
        )
        losses.append(
            -(
                weights[0] * F.logsigmoid(beta * (delta_score - delta_ref_score))
                + weights[1] * F.logsigmoid(beta * (delta_ref_score - delta_score))
            )
        )

    if not losses:
        raise ValueError("No valid non-identical winner-loser pairs in batch.")

    return torch.stack(losses).mean()


def implicit_reward(
    sequence: str, 
    masked_positions: np.ndarray, 
    beta: float, 
    scorer: ESM2Model, 
    reference: ESM2Model
) -> torch.Tensor:
    """Compute implicit reward for a sequence at given CDR residue positions."""
    positions = [int(pos) for pos in masked_positions]
    masked_pll = scorer.masked_pseudo_log_likelihood(
        [sequence],
        positions,
        use_grad=False,
        positions_are_cdr=True,
    ).squeeze(0)
    ref_masked_pll = reference.masked_pseudo_log_likelihood(
        [sequence],
        positions,
        use_grad=False,
        positions_are_cdr=True,
    ).squeeze(0)
    reward = beta * (masked_pll - ref_masked_pll)   
    return reward


def reward_accuracy(
    pair: PairLike, 
    masked_positions: np.ndarray, 
    beta: float, 
    scorer: ESM2Model, 
    reference: ESM2Model
) -> bool:
    """Determine if the reward correctly ranks the winner above the loser."""
    winner, loser = pair
    winner_seq = _member_to_sequence(winner)
    loser_seq = _member_to_sequence(loser)
    winner_reward = implicit_reward(winner_seq, masked_positions, beta, scorer, reference)
    loser_reward = implicit_reward(loser_seq, masked_positions, beta, scorer, reference)
    return bool((winner_reward > loser_reward).item())


def reward_margin(
    pair: PairLike, 
    masked_positions: np.ndarray, 
    beta: float, 
    scorer: ESM2Model, 
    reference: ESM2Model
) -> torch.Tensor:
    """Compute the reward margin between winner and loser."""
    winner, loser = pair
    winner_seq = _member_to_sequence(winner)
    loser_seq = _member_to_sequence(loser)
    winner_reward = implicit_reward(winner_seq, masked_positions, beta, scorer, reference)
    loser_reward = implicit_reward(loser_seq, masked_positions, beta, scorer, reference)
    margin = winner_reward - loser_reward
    return margin


def implicit_KL_divergence(
    sequence: str, 
    scorer: ESM2Model, 
    reference: ESM2Model
) -> torch.Tensor:
    """Compute the implicit KL divergence for a single sequence."""
    masked_pll = scorer.pseudo_log_likelihood([sequence], use_grad=False).squeeze(0)
    ref_masked_pll = reference.pseudo_log_likelihood([sequence], use_grad=False).squeeze(0)
    kl_divergence = (masked_pll - ref_masked_pll)
    return kl_divergence


def pair_monitoring_metrics(
    pair: PairLike,
    beta: float,
    scorer: ESM2Model,
    reference: ESM2Model,
) -> Dict[str, float]:
    """Compute monitoring metrics for a single winner-loser pair."""
    winner, loser = pair
    winner_seq = _member_to_sequence(winner)
    loser_seq = _member_to_sequence(loser)
    diff_positions = np.asarray(_diff_positions(winner_seq, loser_seq), dtype=int)
    if diff_positions.size == 0:
        raise ValueError("Winner and loser are identical; monitoring metrics are undefined.")

    acc = reward_accuracy(pair, diff_positions, beta, scorer, reference)
    margin = reward_margin(pair, diff_positions, beta, scorer, reference)
    winner_kl = implicit_KL_divergence(winner_seq, scorer, reference)
    loser_kl = implicit_KL_divergence(loser_seq, scorer, reference)

    return {
        "reward_accuracy": float(acc),
        "reward_margin": float(margin.item()),
        "implicit_kl": float(((winner_kl + loser_kl) / 2.0).item()),
    }


def batch_monitoring_metrics(
    pairs: PairBatchLike,
    beta: float,
    scorer: ESM2Model,
    reference: ESM2Model,
) -> Dict[str, float]:
    """Compute average monitoring metrics over one pair or a batch of pairs."""
    pair_batch = _as_pair_batch(pairs)

    reward_accuracy_sum = 0.0
    reward_margin_sum = 0.0
    implicit_kl_sum = 0.0
    valid_pairs = 0

    for pair in pair_batch:
        try:
            metrics = pair_monitoring_metrics(pair, beta, scorer, reference)
        except ValueError:
            continue

        reward_accuracy_sum += metrics["reward_accuracy"]
        reward_margin_sum += metrics["reward_margin"]
        implicit_kl_sum += metrics["implicit_kl"]
        valid_pairs += 1

    if valid_pairs == 0:
        raise ValueError("No valid non-identical winner-loser pairs in batch.")

    return {
        "reward_accuracy": reward_accuracy_sum / valid_pairs,
        "reward_margin": reward_margin_sum / valid_pairs,
        "implicit_kl": implicit_kl_sum / valid_pairs,
        "num_pairs": float(valid_pairs),
    }



