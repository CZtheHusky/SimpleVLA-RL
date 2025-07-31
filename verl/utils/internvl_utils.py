import os
import shutil
import filecmp

def replace_modeling_files(local_path):
    """
    用 custom_internvl 下的文件自动覆盖 local_path 下的同名文件，并备份原文件。
    :param local_path: 要加载的模型参数目录（如 checkpoint-1280）
    """
    utils_path = find_file_path()  # 你可以用上面自动查找repo根目录的函数
    custom_dir = os.path.join(utils_path, "vla_utils/internvl/custom_internvl")
    backup_dir = os.path.join(local_path, "backup")

    if not os.path.exists(custom_dir):
        print(f"[replace_modeling_files] path does not exist: {custom_dir}")
        return

    os.makedirs(backup_dir, exist_ok=True)

    for fname in os.listdir(custom_dir):
        src_file = os.path.join(custom_dir, fname)
        tgt_file = os.path.join(local_path, fname)
        if not os.path.isfile(src_file):
            continue
        # 如果目标文件不存在，直接复制
        if not os.path.exists(tgt_file):
            shutil.copy2(src_file, tgt_file)
            print(f"[replace_modeling_files] coping file: {fname}")
        else:
            # 比较内容
            if filecmp.cmp(src_file, tgt_file, shallow=False):
                print(f"[replace_modeling_files] do nothing: {fname}")
                continue
            # 备份原文件
            shutil.move(tgt_file, os.path.join(backup_dir, fname))
            shutil.copy2(src_file, tgt_file)
            print(f"[replace_modeling_files] replaced and backuped: {fname}")

def find_file_path(start_path=None):
    """
    返回当前文件所在目录（即repo根目录）。
    """
    if start_path is None:
        return os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(start_path)