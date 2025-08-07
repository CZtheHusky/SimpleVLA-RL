import cv2
import numpy as np
import math


def action_to_str(action, num_floats: int = 4):
    return [np.round(a, num_floats) for a in action.values()] if isinstance(action, dict) else [np.round(a, num_floats) for a in action]

# def write_instruction_action(instruction: str, rgb: np.ndarray, action: str = None, raw_action: str = None):
#     """
#     在图片上方增加一个白色背景条，并写入 instruction。
#     如果提供了 action 参数，则会在 instruction 下方额外写入一行 action。

#     :param instruction: 要显示的第一行指令文本。
#     :param rgb: 输入的原始图像 (numpy array)。
#     :param action: (可选) 要在第二行显示的动作文本。
#     :return: 带有文本的新图像。
#     """
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.5
#     font_thickness = 1
#     text_color = (0, 0, 0)  # 黑色字体
#     bg_color = (255, 255, 255)  # 白色背景
    
#     # --- 动态计算所需空间 ---
#     texts_to_draw = [instruction]
#     if action is not None:
#         texts_to_draw.append(action)
#     if raw_action is not None:
#         texts_to_draw.append(raw_action)

#     # 获取每行文本的尺寸
#     text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in texts_to_draw]
#     text_heights = [size[1] for size in text_sizes]

#     # 定义边距和行间距
#     top_margin = 10
#     bottom_margin = 10
#     line_spacing = 5 # 两行文字之间的额外间距

#     # 计算总的 padding 高度
#     total_text_height = sum(text_heights)
#     if len(texts_to_draw) > 1:
#         total_text_height += line_spacing * (len(texts_to_draw) - 1)
    
#     pad_top = total_text_height + top_margin + bottom_margin

#     # --- 创建并绘制新图像 ---
#     h, w, _ = rgb.shape
#     new_img = np.full((h + pad_top, w, 3), bg_color, dtype=np.uint8)

#     # 把原图粘贴到新图像的下方
#     new_img[pad_top:, :] = rgb

#     # --- 逐行写入文本 ---
#     current_y = top_margin
#     for i, text in enumerate(texts_to_draw):
#         text_h = text_heights[i]
#         # 计算文本基线的 y 坐标 (putText 的 y 坐标是基线位置)
#         text_y = current_y + text_h
#         text_x = 10  # 左边距

#         cv2.putText(new_img, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
#         # 更新下一行文本的起始 y 坐标
#         current_y = text_y + line_spacing

#     return new_img

def write_instruction_action(instruction: str,
                             rgb: np.ndarray,
                             action: str = None,
                             raw_action: str = None,
                             macro_block_size: int = 16):
    """
    在图片上方增加白底文本区，并写入多行文本；最后一次性 pad 到 macro_block_size 的倍数。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (0, 0, 0)
    bg_color = (255, 255, 255)

    # 要绘制的行
    texts = [instruction]
    if action    is not None: texts.append(action)
    if raw_action is not None: texts.append(raw_action)

    # 测量各行高度
    text_sizes   = [cv2.getTextSize(t, font, font_scale, font_thickness)[0] for t in texts]
    text_heights = [h for (_, h) in text_sizes]

    top_margin    = 10
    bottom_margin = 10
    line_spacing  = 5

    total_text_height = sum(text_heights)
    if len(texts) > 1:
        total_text_height += line_spacing * (len(texts) - 1)
    pad_top = top_margin + total_text_height + bottom_margin

    h, w, _ = rgb.shape
    # 计算最终画布尺寸
    target_h = math.ceil((pad_top + h) / macro_block_size) * macro_block_size
    target_w = math.ceil( w          / macro_block_size) * macro_block_size

    # 一次性分配
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    # 1) 把原图粘到偏移位置
    canvas[pad_top:pad_top+h, :w] = rgb

    # 2) 在顶区逐行写字
    y = top_margin
    for i, text in enumerate(texts):
        text_h = text_heights[i]
        # 基线 y 坐标
        baseline_y = y + text_h
        cv2.putText(canvas, text, (10, baseline_y),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        y = baseline_y + line_spacing

    return canvas