import torch
import numpy as np
import torch.nn.functional as F
from functools import partial

from typing import Tuple, Dict, Any, Optional

from rlprompt.losses import loss_utils
from rlprompt.utils import utils

# sparse max % tsallis entropy reg
# logits = Q / alpha

def sparse_max_operator(
    logits: torch.Tensor,
    logits_masks: torch.Tensor
) -> torch.Tensor:
    """
    Arguments:
        logits: (bs, seq_len, vocab_size)
    
    Returns:
        spmax: (bs, seq_len)
    """
    vocab_size = logits.shape[-1]
    device = logits.device
    values, indices = torch.sort(logits)
    cumsum_values = values.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
    mask = torch.arange(1, vocab_size+1).flip(dims=[-1])[None, None, ...].to(device) * values + 1 > cumsum_values
    supp_z = mask * values
    tau_z = (supp_z.sum(-1) - 1.) / mask.sum(-1)
    # check
    # if True: 
        # check = (supp_z - tau_z.unsqueeze(-1)) * mask
        # assert check.sum(-1).all()
    assert ((logits_masks.sum(-1) + 1)  > mask.sum(-1)).all()
    spmax_z = supp_z ** 2 - tau_z.unsqueeze(-1) ** 2
    spmax_z = (spmax_z * mask).sum(-1) + 1
    # mask.sum(-1) (bs, seq_len)
    return spmax_z / 2, mask.sum(-1).float() 

def sparse_ql_loss_with_sparse_rewards(
        implementation: str,
        logits: torch.Tensor,
        masks: torch.Tensor,
        logits_: torch.Tensor,
        masks_: torch.Tensor,
        actions: torch.LongTensor,
        # sampled_actions: Optional[torch.LongTensor],
        rewards: torch.Tensor,
        # spmaxs: torch.Tensor,
        # num_vs: torch.Tensor,
        sequence_length: torch.LongTensor,
        gamma: Optional[float] = 1.0,
        coefficient: Optional[float] = None,
        margin_constant: Optional[float] = None,
        margin_coefficient: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Soft Q Learning Loss Functions with Sparse Rewards

    Arguments:
        implementation: string, which loss function to use
        logits:          [batch_size, sequence_length, vocab_size]
        logits_:         [batch_size, sequence_length, vocab_size]
        logits_pi:       [batch_size, sequence_length, vocab_size]
        actions:         [batch_size, sequence_length]
        rewards:         [batch_size]
        sequence_length: [batch_size]
    """
    # if implementation not in ["v0", "v1", "v2", "v3", "v2_v2r", "v3_v3r", "v2_v2r_v3_v3r", "pg"]:
    if implementation not in ["v1", "v2", "v3", "v2_v3"]:
        raise ValueError

    if not torch.is_tensor(rewards):
        raise TypeError

    if rewards.ndim != 1 or logits.shape[0] != rewards.shape[0]:
        raise ValueError

    # if implementation == "v0":
    #     _sp_ql_loss_func = sparse_ql_loss_0

    if implementation == "v1":
        _sp_ql_loss_func = sparse_ql_loss_1

    elif implementation == "v2":
        _sp_ql_loss_func = sparse_ql_loss_2

    elif implementation == "v3":
        _sp_ql_loss_func = sparse_ql_loss_3

    elif implementation == "v2_v3":
        _sp_ql_loss_func = sparse_ql_loss_2_3
    if logits.shape != logits_.shape:
        raise ValueError(
            f"`logits.shape` = {logits.shape}, but "
            f"`logits_.shape` = {logits_.shape}")

    raw_losses, quantities_to_log = _sp_ql_loss_func(
        logits=logits,
        masks=masks,
        logits_=logits_,
        masks_=masks_,
        actions=actions,
        rewards=rewards,
        # spmaxs=spmaxs,
        # num_vs=num_vs, 
        sequence_length=sequence_length,
        gamma=gamma,
    )

    loss = loss_utils.mask_and_reduce(
        sequence=raw_losses,
        sequence_length=sequence_length)
    loss_log = {
        "loss": loss,
        "sequence_length": sequence_length.float().mean(),
        "loss-normalized": loss_utils.mask_and_reduce(
            sequence=raw_losses,
            sequence_length=sequence_length,
            average_across_timesteps=True,
            sum_over_timesteps=False),
    }

    for key, value in quantities_to_log.items():
        masked_mean, masked_min, masked_max = \
            loss_utils.get_masked_mean_min_max(value,
                                            lengths=sequence_length)

        loss_log[f"{key}/min"] = masked_min
        loss_log[f"{key}/max"] = masked_max
        loss_log[f"{key}/mean"] = masked_mean

    return loss, loss_log

# def sparse_ql_loss_0(
#     logits: torch.Tensor,
#     logits_: torch.Tensor,
#     actions: torch.LongTensor,
#     rewards: torch.Tensor,
#     spmaxs: torch.Tensor,
#     num_vs: torch.Tensor,
#     sequence_length: torch.LongTensor,
#     gamma: Optional[float]= 1.0,
#     ) -> Tuple[torch.Tensor, Dict[str, Any]]:

#     # (bs, seq_len, 1)
#     Q = loss_utils.gather_2d_on_last_dim(
#         tensor=logits,
#         index=actions,
#         shape=actions.shape)
    
#     # use `V` from the target if available
#     # M : the number of True vocab
#     V_, M_ = sparse_max_operator(logits_) # (bs, seq_len)

#     # Build the target `= gamma * V_t+1 + r`
#     # where we assume the rewards to be sparse
#     # i.e., only comes at the final step
#     Q_ = torch.zeros_like(Q)
#     Q_[:, :-1] = gamma * V_[:, 1:]
#     # terminal_v = 0
#     Q_[
#         torch.arange(sequence_length.shape[0]),
#         sequence_length - 1] += rewards
    
#     # gamma discount
#     Q_[:, :-1] = \
#         (torch.ones_like(Q[:, :-1]) * gamma).cumprod(dim=-1).flip(dims=[-1]) * rewards.reshape(-1, 1)
    
#     raw_losses = F.mse_loss(Q, Q_, reduction="none")
#     quantities_to_log = {
#         "Q": Q,
#         "V": spmaxs,
#         "Q_": Q_,
#         "V_": V_,
#         "M": num_vs,
#         "M_": M_,
#     }

#     return raw_losses, quantities_to_log

def sparse_ql_loss_1(
    logits: torch.Tensor,
    masks: torch.Tensor,
    logits_: torch.Tensor,
    masks_: torch.Tensor,
    actions: torch.LongTensor,
    rewards: torch.Tensor,
    # spmaxs: torch.Tensor,
    # num_vs: torch.Tensor,
    sequence_length: torch.LongTensor,
    gamma: Optional[float]= 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

    # (bs, seq_len, 1)
    Q = loss_utils.gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    
    V, M = sparse_max_operator(logits, masks)
    
    # use `V` from the target if available
    # M : the number of True vocab
    V_, M_ = sparse_max_operator(logits_, masks_) # (bs, seq_len)

    # Build the target `= gamma * V_t+1 + r`
    # where we assume the rewards to be sparse
    # i.e., only comes at the final step
    Q_ = torch.zeros_like(Q)
    # terminal_v = 0
    Q_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] += rewards
    
    # gamma discount
    Q_[:, :-1] = \
        (torch.ones_like(Q[:, :-1]) * gamma).cumprod(dim=-1).flip(dims=[-1]) * rewards.reshape(-1, 1)
    
    raw_losses = F.mse_loss(Q, Q_, reduction="none")
    quantities_to_log = {
        "Q": Q,
        "V": V,
        "Q_": Q_,
        "V_": V_,
        "M": M,
        "M_": M_,
    }

    return raw_losses, quantities_to_log

# A log p(w)
def sparse_ql_loss_2(
    logits: torch.Tensor,
    masks: torch.Tensor,
    logits_: torch.Tensor,
    masks_: torch.Tensor,
    actions: torch.LongTensor,
    rewards: torch.Tensor,
    # spmaxs: torch.Tensor,
    # num_vs: torch.Tensor,
    sequence_length: torch.LongTensor,
    gamma: Optional[float] = 1.0,
    _recover_mle: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    
    max_length = sequence_length.max()
    num_seq = len(sequence_length)
    seq_mask = torch.arange(max_length)[None, ...].repeat(num_seq, 1).to(sequence_length.device) \
        < (sequence_length).reshape(-1, 1).repeat(1, max_length)
    
    Q = loss_utils.gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    V, M = sparse_max_operator(logits, masks)
    A = Q - V
    
    # Target outputs
    Q_ = torch.zeros_like(Q)
    A_ = torch.zeros_like(Q)
    
    V_, M_ = sparse_max_operator(logits_, masks_)
    Q_[:, :-1] = gamma * V_[:, 1:]
    Q_[
    torch.arange(sequence_length.shape[0]),
    sequence_length - 1] += rewards
    # Q_ *= seq_mask
    
    # re-implement
    # case 1) A_(s, a) = Q_(s, a) - V_(s)
    #                  = r + gamma * alpha * V_(s') - V_(s)
    A_ = Q_ -  V_
    # case 2) A_(s, a) = r + gamma * alpha * V_(s') - V(s)
    A_ *= seq_mask
    # origin
    ## A_(s, a) = r + done * gamma * alpha * V_(s') - V_(s)
    # A_[:, :-1] = V_[:, 1:] - V_[:, :-1]
    # terminal_V_ = V_[
    #     torch.arange(sequence_length.shape[0]),
    #     sequence_length - 1]
    # A_[
    #     torch.arange(sequence_length.shape[0]),
    #     sequence_length - 1] += rewards - terminal_V_
    # A_ *= seq_mask
    raw_losses = F.mse_loss(A, A_, reduction="none")
    quantities_to_log = {
        "Q": Q,
        "V": V,
        "A": A,
        "Q_": Q_,
        "V_": V_,
        "A_": A_,
        "M": M,
        "M_": M_
        # "H": loss_utils._get_entropy(logits),
        # "H_": loss_utils._get_entropy(logits_),
    }
    
    return raw_losses, quantities_to_log

def sparse_ql_loss_3(
    logits: torch.Tensor,
    masks: torch.Tensor,
    logits_: torch.Tensor,
    masks_: torch.Tensor,
    actions: torch.LongTensor,
    rewards: torch.Tensor,
    # spmaxs: torch.Tensor,
    # num_vs: torch.Tensor,
    sequence_length: torch.LongTensor,
    gamma: Optional[float] = 1.0,
    freeze_future_steps: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    
    Q = loss_utils.gather_2d_on_last_dim(
    tensor=logits,
    index=actions,
    shape=actions.shape)
    V, M = sparse_max_operator(logits, masks)
    A = Q - V

    # Target outputs
    V_, M_ = sparse_max_operator(logits_, masks_)

    # if 
    A2 = loss_utils.masked_reverse_cumsum(
        A,
        lengths=sequence_length,
        dim=-1)
    
    raw_losses = F.mse_loss(
        A2, rewards.view(-1, 1) - V_,
        reduction="none")

    quantities_to_log = {
        "Q": Q,
        "V": V,
        "A": A,
        "V_": V_,
        "M": M,
        "M_": M_
    }

    return raw_losses, quantities_to_log
    
def sparse_ql_loss_2_3(
    logits: torch.Tensor,
    masks: torch.Tensor,
    logits_: torch.Tensor,
    masks_: torch.Tensor,
    actions: torch.LongTensor,
    rewards: torch.Tensor,
    # spmaxs: torch.Tensor,
    # num_vs: torch.Tensor,
    sequence_length: torch.LongTensor,
    gamma: Optional[float] = 1.0,
    freeze_future_steps: bool = False,
)-> Tuple[torch.Tensor, Dict[str, Any]]:
    loss_2, log_2 = sparse_ql_loss_2(logits=logits,
                                    masks=masks,
                                    logits_=logits_,
                                    masks_=masks,
                                    actions=actions,
                                    rewards=rewards, 
                                    # spmaxs=spmaxs,
                                    # num_vs=num_vs,
                                    sequence_length=sequence_length,
                                    gamma=gamma,
                                    )
    loss_3, log_3 = sparse_ql_loss_3(logits=logits,
                                    masks=masks,
                                    logits_=logits_,
                                    masks_=masks,
                                    actions=actions,
                                    rewards=rewards, 
                                    # spmaxs=spmaxs,
                                    # num_vs=num_vs,
                                    sequence_length=sequence_length,
                                    gamma=gamma,
                                    )
    raw_losses = (loss_2 + loss_3) / 2
    utils.add_prefix_to_dict_keys_inplace(log_2, prefix="v2/")
    utils.add_prefix_to_dict_keys_inplace(log_3, prefix="v3/")
    quantities_to_log = utils.unionize_dicts([log_2, log_3,])
    return raw_losses, quantities_to_log
    
# Q log p(w) , A log p(w)
def sparse_pg_loss(
    logits: torch.Tensor,
    probs: torch.Tensor,
    logits_: torch.Tensor,
    actions: torch.LongTensor,
    rewards: torch.Tensor,
    spmaxs: torch.Tensor,
    num_vs: torch.Tensor,
    sequence_length: torch.LongTensor,
    gamma: Optional[float] = 1.0,
    _recover_mle: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    
    max_length = sequence_length.max()
    num_seq = len(sequence_length)
    seq_mask = torch.arange(max_length)[None, ...].repeat(num_seq, 1).to(sequence_length.device) \
        < (sequence_length).reshape(-1, 1).repeat(1, max_length)
        
    as_prob = loss_utils.gather_2d_on_last_dim(
        tensor=probs,
        index=actions,
        shape=actions.shape
    )
    log_pws = torch.log(as_prob)
    #================== log p(w)

    disR = torch.zeros_like(spmaxs)
    disR[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] += rewards
    
    # gamma discount
    disR[:, :-1] = \
        (torch.ones_like(disR[:, :-1]) * gamma).comprod(dim=-1).flip(dims=[-1]) * rewards.reshape(-1, 1)
    
    # raw_losses = - disR.detach() * log_pws
    raw_losses = - (disR - spmaxs).detach() * log_pws
    quantities_to_log = {
        "A": disR - spmaxs,
        "M": num_vs,
    }
    
    return raw_losses, quantities_to_log

if __name__=="__main__":
    test_logits = torch.rand(3, 4, 20)
    sparse_max_operator(test_logits)
    