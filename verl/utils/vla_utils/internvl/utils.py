from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torchvision.transforms.functional as TVF
from typing import Tuple
from PIL import Image
import numpy as np
import torch
from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor, LogitsProcessor
import math
from transformers import LogitsProcessorList
import torch.nn.functional as F
from verl.utils.torch_functional import logprobs_from_logits, entropy_from_logits


import torch
import torch.nn.functional as F

def logits2_logprobs_entropy(
    logits: torch.Tensor,            # [B, L, V]
    responses: torch.Tensor,         # [B, L]  —— 实际选中的 token ids（用于 gather）
    valid_index2token_id_list: dict | None = None,  # {t: 1D LongTensor(allowed_ids_at_t)}
    top_k: int = None,
    top_p: float = None,
    min_tokens_to_keep: int = 1,
    temperature: float = 1.0,
    valid_token_index = None,
):
    """
    返回：
      logprob_out: [B, T_pos]  —— 在“prefix + top-k + top-p”处理后的完整分布上，被选 token 的逐步 logprob
      entropy_out: [B, T_pos]  —— 同一分布的逐步熵
    说明：
      - 不裁剪 vocab 维度；不允许/被剔除的 token logits 设为 -inf，softmax 后概率为 0。
      - 仅实现 top-k/top-p；若 rollout 还用了 repetition penalty、no_repeat_ngram 等，
        需按相同步骤额外加到 `scores` 上，才能与真实采样完全一致。
    """
    B, L, V = logits.shape
    device  = logits.device
    logits = logits.div(temperature)  # scale logits by temperature
    if valid_index2token_id_list is None:
        positions = range(L)
    else:
        positions = list(valid_index2token_id_list.keys())

    logprob_out = []
    entropy_out = []

    for t in positions:
        # 取该步的原始 logits（完整 vocab 维度）
        scores = logits[:, t, :].clone()        # [B, V]
        scores_det = scores.detach()
        # 1) Prefix control：只允许 ids_tensor；其余置 -inf（维度不变）
        if valid_index2token_id_list is not None:
            ids_tensor = valid_index2token_id_list[t].to(device=device, dtype=torch.long)  # [K]
            allowed_mask = torch.zeros(V, dtype=torch.bool, device=device)
            allowed_mask[ids_tensor] = True
            scores = scores.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))

        # 2) top-k（若启用）：保留每行 top-k，其他置 -inf
        if isinstance(top_k, int) and top_k > 0:
            k = min(top_k, V)
            topk_vals, topk_idx = torch.topk(scores_det, k=k, dim=-1)          # [B, k]
            keep_mask = torch.zeros_like(scores, dtype=torch.bool)          # [B, V]
            keep_mask.scatter_(1, topk_idx, True)
            scores = scores.masked_fill(~keep_mask, float("-inf"))

        # 3) top-p / nucleus（若启用）：按概率降序取累计概率≥p的最小前缀；外部置 -inf
        if isinstance(top_p, float) and (0.0 < top_p < 1.0):
            probs = F.softmax(scores_det, dim=-1)                                # [B, V]
            probs_sorted, idx_sorted = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(probs_sorted, dim=-1)
            remove = cumsum > top_p                                          # [B, V] (sorted space)
            # 保底至少留 min_tokens_to_keep（避免全 -inf/NaN）
            if min_tokens_to_keep > 0:
                remove[..., :min_tokens_to_keep] = False
            # 映射回原位置
            remove_unsorted = torch.zeros_like(remove).scatter(1, idx_sorted, remove)
            scores = scores.masked_fill(remove_unsorted, float("-inf"))

        # 4) 在完整 vocab 上归一化（不裁剪维度）
        logp = F.log_softmax(scores, dim=-1)                                 # [B, V]
        p    = logp.exp()

        # 被选 token 的 logprob（来自真实轨迹）
        chosen_ids = responses[:, t].unsqueeze(1)                            # [B, 1]
        chosen_lp  = logp.gather(1, chosen_ids).squeeze(1)                   # [B]

        # 熵（在上述分布上；被置 -inf 的位置概率为 0，不影响熵）
        # entropy = -(p * logp).sum(dim=-1)                                    # [B]
        entropy = entropy_from_logits(scores)

        logprob_out.append(chosen_lp)
        entropy_out.append(entropy)

    return torch.stack(logprob_out, dim=-1), torch.stack(entropy_out, dim=-1)


# def legacy_logits2_logprobs_entropy(
#     logits: torch.Tensor,
#     responses: torch.Tensor, 
#     valid_index2token_id_list: dict = None,
# ):
#     if valid_index2token_id_list is not None:
#         B, L, V = logits.shape
#         device = logits.device
#         positions = list(valid_index2token_id_list.keys())
#         logprob_out = []
#         entropy_out = []
#         for t in positions:
#             ids_tensor = valid_index2token_id_list[t]
#             assert len(ids_tensor) > 0, f"ids_tensor should not be empty for position {t}, got {ids_tensor}"
#             ids_tensor = ids_tensor.to(device)
#             filtered_logits = logits[:, t, ids_tensor]            # [B, K]
#             current_entropy = entropy_from_logits(filtered_logits)          # [B]
#             logp_allowed = F.log_softmax(filtered_logits, dim=-1) # [B, K]

#             resp_ids = responses[:, t]                             # [B]
#             match = (ids_tensor.unsqueeze(0) == resp_ids.unsqueeze(1))  # [B, K]
#             has_match = match.any(dim=1)                           # [B]
#             assert has_match.all(), f"Not all positions have a match, got {has_match}"
#             local_idx = match.float().argmax(dim=1)                # [B]（无匹配时是0，但会被 has_match 掩掉）

#             chosen = logp_allowed[torch.arange(B, device=device), local_idx]  # [B]
#             logprob_out.append(chosen)
#             entropy_out.append(current_entropy)

#         # [B, len(positions)]
#         return torch.stack(logprob_out, dim=-1), torch.stack(entropy_out, dim=-1)
#     else:
#         return logprobs_from_logits(logits, responses), entropy_from_logits(logits)
        

def _log(msg: str):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def debug_logits2_logprobs_entropy(
    logits: torch.Tensor,
    responses: torch.Tensor, 
    valid_index2token_id_list: dict = None,
    temperature: float = 1.0,
    valid_token_index=None,
):
    B, L, V = logits.shape
    device = logits.device
    positions = list(valid_index2token_id_list.keys())
    logprob_out = []
    logits = logits.div(temperature)  # scale logits by temperature

    for t in positions:
        ids_tensor = valid_index2token_id_list[t].to(device=device, dtype=torch.long)  # [K]
        allowed_mask = torch.zeros(V, dtype=torch.bool, device=device)
        allowed_mask[ids_tensor] = True
        scores = logits[:, t, :].clone()        # [B, V]
        scores = scores.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))
        
        logp_allowed = torch.nn.functional.log_softmax(scores, dim=-1)  # [B, K]

        resp_ids = responses[:, t]                                       # [B]
        _log(f"{t} current id\n{resp_ids.detach().cpu().tolist()}")

        match = (ids_tensor.unsqueeze(0) == resp_ids.unsqueeze(1))       # [B, K]
        has_match = match.any(dim=1)                                     # [B]
        assert has_match.all(), f"Not all positions have a match, got {has_match}"
        local_idx = match.float().argmax(dim=1)                          # [B]
        chosen = logp_allowed[torch.arange(B, device=device), local_idx]  # [B]
        for b in range(B):
        
            row_lp = logp_allowed[b]                             # [K]
            chosen_j = int(local_idx[b].item())                  # 该样本 chosen 的局部索引
            chosen_lp = row_lp[chosen_j]                         # 标量
            vals, idxs = torch.sort(row_lp, descending=True)     # 降序
            mask = vals > chosen_lp                              # 严格大于 chosen 的那些
            if mask.any():
                sel_vals = vals[mask]
                sel_idxs = idxs[mask]
                sel_ids  = ids_tensor[sel_idxs]
                pairs = list(zip(
                    sel_ids.detach().cpu().tolist(),
                    sel_vals.detach().cpu().tolist()
                ))
            else:
                pairs = []

            _log(f"{t} row {b} better_than_chosen_sorted (id, logp):")
            for local_idx, pair in enumerate(pairs):
                _log(f"{local_idx} {pair}")

            chosen_id = int(ids_tensor[chosen_j].item())
            _log(f"{t} row {b} chosen_last (id, logp): ({chosen_id}, {float(chosen_lp.detach().cpu().item())})")

        # 你原来的“整批 chosen 列表”就不再需要；若保留：
        # _log(f"{t} chosen_all\n{chosen.detach().cpu().tolist()}")

        logprob_out.append(chosen)
    return torch.stack(logprob_out, dim=-1)


class SafePrefixConstrainedLogitsProcessor(PrefixConstrainedLogitsProcessor):
    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] == 0:
            mask = torch.full_like(scores, -math.inf)
            batch_id = 0
            sent = input_ids[batch_id]
            prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
            mask[..., prefix_allowed_tokens] = 0.0
            return scores + mask
        return super().__call__(input_ids, scores)

def generate_prefix_fn(index2list):
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return index2list[input_ids.shape[-1]]
    return prefix_allowed_tokens_fn

def prepare_logits_processor(tokenizer):
    example_action_str = "action: {x: -18mm, y: -35mm, z: -28mm, roll: 0 degrees, pitch: 0 degrees, yaw: 4 degrees, open: 1}"
    print("Example action\n", example_action_str)
    toks = tokenizer.tokenize(example_action_str)
    ids  = tokenizer.convert_tokens_to_ids(toks)
    action_token_num = len(toks)
    index2list = {}
    numbers_index = [6, 12, 18, 24, 30, 36, 42]
    valid_index2token_list = {}
    valid_list = []
    for idx, (tok, token_idx) in enumerate(zip(toks, ids)):
        index2list[idx] = [token_idx]
        valid_list.append(token_idx)
    valid_list = set(valid_list)
    numbers = list(range(0, 1000))
    processor_list = LogitsProcessorList([])
    connect_list = []
    numbers_list = []
    connect_sign = [" ", " -"]
    for str_ in connect_sign:
        toks = tokenizer.tokenize(str_)
        assert len(toks) == 1
        connect_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
    for str_ in numbers:
        toks = tokenizer.tokenize(str(str_))
        assert len(toks) == 1
        numbers_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
    for idx in numbers_index:
        index2list[idx] = numbers_list
        index2list[idx - 1] = connect_list
        valid_index2token_list[idx - 1] = torch.as_tensor(connect_list, dtype=torch.long) 
        valid_index2token_list[idx] = torch.as_tensor(numbers_list, dtype=torch.long)
    prefix_processor = SafePrefixConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=generate_prefix_fn(index2list),
            num_beams=1,
        )
    processor_list = LogitsProcessorList([
        prefix_processor,
    ])
    valid_list.update(numbers_list)
    valid_list.update(connect_list)
    valid_list = list(valid_list)
    return processor_list, valid_list, action_token_num, numbers_index, valid_index2token_list

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERNVL_IMG_SIZE = 448
INTERNVL_IMG_MAX_NUM = 12
INTERNVL_IMG_USE_THUMBNAIL = True


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

INTERNVL_IMG_TRANSFORM = build_transform(INTERNVL_IMG_SIZE)

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def pad_to_square(image: Image.Image, resize_target: Tuple[int, int] = (480, 480), padding_fill_value: Tuple[int, int, int] = (123, 116, 103)) -> Image.Image:
    """
    Pad a PIL.Image to square shape and resize, using a symmetric border and a fill value.
    Args:
        image: PIL.Image.Image, input image
        resize_target: tuple, target size after resize
        padding_fill_value: tuple, RGB fill value for padding
    Returns:
        PIL.Image.Image, padded and resized image
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    w, h = image.size
    max_wh = max(w, h)
    horizontal_pad = (max_wh - w) // 2
    vertical_pad = (max_wh - h) // 2
    padding = (horizontal_pad, vertical_pad, max_wh - w - horizontal_pad, max_wh - h - vertical_pad)
    padded_image = TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")
    resized_image = padded_image.resize(resize_target, Image.BILINEAR)
    return resized_image


def process_image_internvl(image: np.ndarray):
    image = Image.fromarray(image).convert("RGB")
    image = pad_to_square(image, resize_target=(INTERNVL_IMG_SIZE, INTERNVL_IMG_SIZE))
    images = dynamic_preprocess(image, image_size=INTERNVL_IMG_SIZE, use_thumbnail=INTERNVL_IMG_USE_THUMBNAIL, max_num=INTERNVL_IMG_MAX_NUM)
    pixel_values = [INTERNVL_IMG_TRANSFORM(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# class SafePrefixConstrainedLogitsProcessor(PrefixConstrainedLogitsProcessor):
#     def __call__(self, input_ids, scores):
#         if input_ids.shape[-1] == 0:
#             mask = torch.full_like(scores, -math.inf)
#             batch_id = 0
#             sent = input_ids[batch_id]
#             prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
#             mask[..., prefix_allowed_tokens] = 0.0
#             return scores + mask
#         return super().__call__(input_ids, scores)

# def generate_prefix_fn_legacy(numbers_list, start_list, end_list, connect_list):

#     def prefix_allowed_tokens_fn(batch_id, input_ids):
#         if input_ids.shape[-1] == 14:
#             return end_list
#         if input_ids.shape[-1] == 0:
#             return start_list
#         elif input_ids.shape[-1] % 2 == 1:
#             return numbers_list
#         elif input_ids.shape[-1] % 2 == 0:
#             return connect_list
#     return prefix_allowed_tokens_fn

# def generate_prefix_fn(numbers_list, symbols_list):
#     def prefix_allowed_tokens_fn(batch_id, input_ids):
#         if input_ids.shape[-1] % 2 == 0:
#             return symbols_list
#         elif input_ids.shape[-1] % 2 == 1:
#             return numbers_list
#     return prefix_allowed_tokens_fn


# def prepare_logits_processor(is_legacy, tokenizer):
#     numbers = list(range(0, 1000))
#     processor_list = LogitsProcessorList([])
#     print("Setting Generation Control")
#     if is_legacy:
#         print("Using action pattern: {-1 0 0 0 0 0 1}")
#         print("Warning: This is a legacy action pattern, for horizon 1 only.")
#         start_list = []
#         end_list = []
#         connect_list = []
#         numbers_list = []
#         start_sign = ["{", '{-',]
#         end_sign = ["}"]
#         connect_sign = [" ", " -"]
#         for str_ in start_sign:
#             toks = tokenizer.tokenize(str_)
#             assert len(toks) == 1
#             start_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
#         for str_ in end_sign:
#             toks = tokenizer.tokenize(str_)
#             assert len(toks) == 1
#             end_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
#         for str_ in connect_sign:
#             toks = tokenizer.tokenize(str_)
#             assert len(toks) == 1
#             connect_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
#         for str_ in numbers:
#             toks = tokenizer.tokenize(str(str_))
#             assert len(toks) == 1
#             numbers_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
#         prefix_processor = SafePrefixConstrainedLogitsProcessor(
#                 prefix_allowed_tokens_fn=generate_prefix_fn_legacy(numbers_list, start_list, end_list, connect_list),
#                 num_beams=1,
#             )
#         processor_list = LogitsProcessorList([
#             prefix_processor,
#         ])
#         valid_list = start_list + end_list + connect_list + numbers_list
#     else:
#         print("Using action pattern: 0 0 0 0 0 0 1")
#         connect_list = []
#         numbers_list = []
#         connect_sign = [" ", " -"]
#         for str_ in connect_sign:
#             toks = tokenizer.tokenize(str_)
#             assert len(toks) == 1
#             connect_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
#         for str_ in numbers:
#             toks = tokenizer.tokenize(str(str_))
#             assert len(toks) == 1
#             numbers_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
#         prefix_processor = SafePrefixConstrainedLogitsProcessor(
#                 prefix_allowed_tokens_fn=generate_prefix_fn(numbers_list, connect_list),
#                 num_beams=1,
#             )
#         processor_list = LogitsProcessorList([
#             prefix_processor,
#         ])
#         valid_list = numbers_list + connect_list
#     return processor_list, valid_list

