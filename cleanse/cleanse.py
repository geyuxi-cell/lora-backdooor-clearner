"""
独立训练：单层相邻一致性损失 + 由 trigger_tokens 构造 PV 向量 v 并推远。

- **相邻一致性**：只对 ``hidden_states[t-1]`` 与 ``hidden_states[t]`` 算 token 维余弦相似度，
  损失为 ``(1 - mean_cos)``，**最小化该损失 = 拉近两层表示**（不是探针/扫描逻辑，只是普通可微损失）。
- **PV**：``v = mean(emb(trigger_ids))``，在 ``hidden_states[t]`` 池化后与 v 推远（``l_pv``）。

``t`` 与朋友 report 的 **poisoned_layer** 一致（较大一侧下标）；``prompt_length`` 与检测里跳过前缀一致。

依赖：模型有 get_input_embeddings()，forward 支持 inputs_embeds + output_hidden_states。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze_all_unfreeze_decoder_layers(
    model: nn.Module,
    layer_indices: Sequence[int],
) -> Tuple[int, int]:
    """
    先整网冻结，再仅解冻指定 decoder block（参数名中含 ``layers.{i}.`` 的权重）。

    i 与 ``hidden_states[i]`` 的 transformer 层号一致（0..L-1）。PeftModel 下会匹配层内 LoRA 等。

    Returns:
        (num_trainable_param_tensors, num_frozen_param_tensors)
    """
    indices = sorted({int(i) for i in layer_indices})
    if not indices:
        raise ValueError("layer_indices must be non-empty")
    markers = tuple(f"layers.{i}." for i in indices)

    model.requires_grad_(False)
    n_train, n_frozen = 0, 0
    for name, p in model.named_parameters():
        if any(m in name for m in markers):
            p.requires_grad_(True)#解冻
            n_train += 1
        else:
            n_frozen += 1
    return n_train, n_frozen


def parse_trigger_token_ids(trigger_tokens: Union[str, Sequence[int]]) -> List[int]:
    """支持 list 或字符串形式 "[1, 2, 3]"。"""
    if isinstance(trigger_tokens, str):
        s = trigger_tokens.strip()
        return ast.literal_eval(s)#输入格式统一
    return [int(x) for x in trigger_tokens]


def build_pv_target_from_trigger_tokens(
    model: nn.Module,
    token_ids: Sequence[int],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """v = mean_i emb(token_id_i)，shape [D]。"""
    emb = model.get_input_embeddings()
    dev = device or next(emb.parameters()).device
    ids = torch.tensor(list(token_ids), dtype=torch.long, device=dev)
    e = emb(ids)  # [K, D]
    v = e.mean(dim=0).to(dtype)#转变方式有待商榷
    return v.detach()


def adjacent_layer_consistency_loss(
    hidden_states,
    target_layer_index: int,
    prompt_length: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    单层相邻一致性损失：比较 ``hidden_states[t-1]`` 与 ``hidden_states[t]``（与 token 维余弦）。

    返回 ``1 - mean(cos)``。对其 **最小化** 等价于 **增大** 两层在有效 token 上的平均余弦相似度。
    ``target_layer_index`` 须 >= 1。
    """
    t = int(target_layer_index)
    if t < 1:
        raise ValueError("target_layer_index 必须 >= 1（需要前一层 t-1）")
    a = hidden_states[t - 1][:, prompt_length:, :].to(torch.float32)#切掉promot_length
    b = hidden_states[t][:, prompt_length:, :].to(torch.float32)
    cos_mean = F.cosine_similarity(a, b, dim=-1, eps=eps).mean()#与前一层一致作为损失
    return 1.0 - cos_mean


def pv_push_away_loss(
    hidden_states,
    attention_mask,
    pv_target: torch.Tensor,
    hidden_index: int,
    prompt_length: int = 0,
    distance: str = "l2",
):
    """
    在 ``hidden_states[hidden_index]`` 上池化得到 h，对 pv_target（v）推远（与 total_loss 中符号配合）。
    ``hidden_index`` 通常与 ``target_layer_index`` 相同（如 poisoned_layer）。
    """
    hs = hidden_states[hidden_index][:, prompt_length:, :]
    if attention_mask is None:
        h = hs.mean(dim=1)
    else:
        m = attention_mask[:, prompt_length:]
        m = m.unsqueeze(-1).to(hs.dtype)
        denom = m.sum(dim=1).clamp_min(1.0)
        h = (hs * m).sum(dim=1) / denom

    v = pv_target.to(device=h.device, dtype=h.dtype).flatten()
    if v.numel() != h.size(-1):
        raise ValueError(f"pv dim mismatch: pv_target={v.numel()} vs h={h.size(-1)}")

    if distance == "l2":
        dist = torch.norm(h - v, p=2, dim=-1)
        return -dist.mean()#与目标pv推远
    if distance == "mse":
        return -F.mse_loss(h, v.expand_as(h))
    if distance == "cos":
        return F.cosine_similarity(h, v.expand_as(h), dim=-1, eps=1e-8).mean()
    raise ValueError("distance must be l2/mse/cos")


def load_run_detect_report_entry(
    report_path: Union[str, Path],
    *,
    epoch: Optional[int] = None,
    pick_lowest_similarity: bool = False,
) -> Tuple[Union[str, List[int]], int]:
    """
    从 threat_report.json 取一条 detected_triggers。

    Returns:
        (trigger_token_ids, poisoned_layer) -> 用作 ``trigger_token_ids`` 与 ``target_layer_index``。
    """
    path = Path(report_path)
    with open(path, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    entries: List[Dict[str, Any]] = data.get("detected_triggers") or []
    if not entries:
        raise ValueError(f"No detected_triggers in {path}")

    if pick_lowest_similarity:
        chosen = min(entries, key=lambda e: float(e.get("lowest_similarity", 1.0)))
    elif epoch is not None:
        chosen = next((e for e in entries if int(e.get("epoch", -1)) == int(epoch)), None)
        if chosen is None:
            raise ValueError(f"No entry with epoch={epoch} in {path}")
    else:
        chosen = entries[-1]

    trig = chosen.get("trigger_tokens")
    if trig is None:
        raise ValueError("Entry missing trigger_tokens")
    layer = int(chosen["poisoned_layer"])
    return trig, layer#读取report，可能需要修改

def load_all_run_detect_report_entries(
    report_path: Union[str, Path],
    *,
    sort_by_epoch: bool = True,
) -> List[Dict[str, Any]]:
    """
    读取 report 里 **全部** ``detected_triggers``（通常每个 fuzz epoch 一条）。

    Returns:
        与 JSON 中每条结构相同的 dict 列表，至少含 ``epoch``、``trigger_tokens``、``poisoned_layer``、
        ``lowest_similarity`` 等；可按 epoch 排序后依次用于训练/评估。

    Example:
        for row in load_all_run_detect_report_entries("threat_report.json"):
            train(..., trigger_token_ids=row["trigger_tokens"],
                  target_layer_index=int(row["poisoned_layer"]), ...)
    """
    path = Path(report_path)
    with open(path, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    entries: List[Dict[str, Any]] = list(data.get("detected_triggers") or [])
    if not entries:
        raise ValueError(f"No detected_triggers in {path}")
    if sort_by_epoch:
        entries.sort(key=lambda e: int(e.get("epoch", 0)))
    return entries#一口气读取多个，跟前一个函数二选一

def train(
    model,
    dataloader,
    *,
    target_layer_index: int,
    prompt_length: int = 5,
    pv_target: Optional[torch.Tensor] = None,
    trigger_token_ids: Optional[Union[str, Sequence[int]]] = None,
    alpha: float = 5.5,
    epsilon: float = 0.1,
    device: str = "cuda",
    lr: float = 1e-5,
    use_fgsm: bool = True,
    pv_distance: str = "l2",
    add_lm_loss: bool = False,
    labels_key: str = "labels",
    train_only_target_layer: bool = True,
):
    """
    ``total_loss = alpha * l_pv + l_cons``（可选再加 LM loss）。

    - ``l_cons = adjacent_layer_consistency_loss(...)``：仅 **t-1 与 t** 一对，**最小化** 即增强一致性。
    - ``t = target_layer_index``（如 report 的 poisoned_layer）；PV 也在 **同一层** hidden 上池化。

    pv_target 与 trigger_token_ids 二选一构造 v。

    ``train_only_target_layer=True``：冻结全模型，只训练 ``layers.{t}.``；False 则不改 requires_grad（由你事先设好）。
    """
    model.train()
    model.to(device)
    dev = torch.device(device)
    t = int(target_layer_index)

    if train_only_target_layer:
        nt, nf = freeze_all_unfreeze_decoder_layers(model, [t])
        print(f"[freeze] target_layer_index={t} trainable_tensors={nt} frozen_tensors={nf}")
        if nt == 0:
            raise RuntimeError(
                "未匹配到任何可训练参数（参数名中需包含 'layers.{i}.'）。请检查模型结构或关闭 train_only_target_layer。"
            )

    if pv_target is not None:
        v = pv_target.to(dev).flatten().detach()
    elif trigger_token_ids is not None:
        ids = parse_trigger_token_ids(trigger_token_ids)
        v = build_pv_target_from_trigger_tokens(model, ids, device=dev, dtype=torch.float32)
        print(f"[pv] built v from {len(ids)} trigger tokens, dim={v.numel()}")
    else:
        raise ValueError("Provide either pv_target (tensor) or trigger_token_ids.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check freezing settings.")
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        optimizer.zero_grad(set_to_none=True)

        inputs_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds.requires_grad_(True)

        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = out.hidden_states

        l_cons_unperturbed = adjacent_layer_consistency_loss(
            hidden_states, t, prompt_length=prompt_length
        )

        if use_fgsm:
            l_cons_unperturbed.backward(retain_graph=True)
            grad = inputs_embeds.grad
            perturbation = epsilon * grad.sign()
            perturbed_embeds = inputs_embeds + perturbation

            model.zero_grad(set_to_none=True)
            if inputs_embeds.grad is not None:
                inputs_embeds.grad.zero_()

            out_p = model(
                inputs_embeds=perturbed_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states_p = out_p.hidden_states

            l_cons = adjacent_layer_consistency_loss(
                hidden_states_p, t, prompt_length=prompt_length
            )
            l_pv = pv_push_away_loss(
                hidden_states_p,
                attention_mask=attention_mask,
                pv_target=v,
                hidden_index=t,
                prompt_length=prompt_length,
                distance=pv_distance,
            )
        else:
            l_cons = l_cons_unperturbed
            l_pv = pv_push_away_loss(
                hidden_states,
                attention_mask=attention_mask,
                pv_target=v,
                hidden_index=t,
                prompt_length=prompt_length,
                distance=pv_distance,
            )

        total_loss = alpha * l_pv + l_cons

        if add_lm_loss and labels_key in batch:
            lm_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch[labels_key].to(dev),
            )
            if lm_out.loss is not None:
                total_loss = total_loss + lm_out.loss

        total_loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(
                f"step={step} total={total_loss.item():.4f} "
                f"l_pv={l_pv.item():.4f} l_cons={l_cons.item():.4f}"
            )
