# FramePack-eichi/oichi LoRA Utilities
#
# LoRAマージ・フォーマット変換・状態辞書処理機能を提供します。

import os
import torch
from safetensors.torch import load_file
from tqdm import tqdm

# 国際化対応
from locales.i18n_extended import translate as _

def merge_lora_to_state_dict(
    model_files: list[str], lora_files: list[str], multipliers: list[float], fp8_enabled: bool, device: torch.device
) -> dict[str, torch.Tensor]:
    """
    LoRA重みをモデルの状態辞書にマージする
    
    Args:
        model_files: モデルファイルのパス一覧
        lora_files: LoRAファイルのパス一覧
        multipliers: LoRA適用強度の一覧
        fp8_enabled: FP8最適化の有効/無効
        device: 計算に使用するデバイス
        
    Returns:
        LoRAが適用されたモデルの状態辞書
    """
    list_of_lora_sd = []
    for lora_file in lora_files:
        # LoRA safetensorsファイルを読み込み
        lora_sd = load_file(lora_file)

        # LoRAファイルのフォーマットを判定
        keys = list(lora_sd.keys())
        if keys[0].startswith("lora_unet_"):
            print(_("Musubi Tuner LoRA detected"))  # Musubi Tuner標準形式（lora_unet_プリフィックス）
        else:
            transformer_prefixes = ["diffusion_model", "transformer"]  # Diffusersフォーマット対応（TextEncoderモジュールを除外）
            lora_suffix = None
            prefix = None
            for key in keys:
                if lora_suffix is None and "lora_A" in key:
                    lora_suffix = "lora_A"
                if prefix is None:
                    pfx = key.split(".")[0]
                    if pfx in transformer_prefixes:
                        prefix = pfx
                if lora_suffix is not None and prefix is not None:
                    break

            if lora_suffix == "lora_A" and prefix is not None:
                print(_("Diffusers形式LoRA検出（diffusion-pipe等）"))
                lora_sd = convert_from_diffusion_pipe_or_something(lora_sd, "lora_unet_")

            else:
                print(_("LoRA file format not recognized: {0}").format(os.path.basename(lora_file)))
                lora_sd = None

        if lora_sd is not None:
            # FramePackまたはHunyuanVideo用LoRAかを判定
            is_hunyuan = False
            for key in lora_sd.keys():
                if "double_blocks" in key or "single_blocks" in key:
                    is_hunyuan = True
                    break
            if is_hunyuan:
                print(_("HunyuanVideo LoRA detected, converting to FramePack format"))  # HunyuanVideo→FramePack変換
                lora_sd = convert_hunyuan_to_framepack(lora_sd)

        if lora_sd is not None:
            list_of_lora_sd.append(lora_sd)

    if len(list_of_lora_sd) == 0:
        # 有効なLoRAファイルが見つからない場合、ベースモデルのみロード
        return load_safetensors_with_fp8_optimization(model_files, fp8_enabled, device, weight_hook=None)

    return load_safetensors_with_lora_and_fp8(model_files, list_of_lora_sd, multipliers, fp8_enabled, device)


def convert_from_diffusion_pipe_or_something(lora_sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """
    Diffusers形式のLoRA重みをMusubi Tuner形式に変換する
    
    Musubi Tunerリポジトリからのアルゴリズム移植版
    Diffusers形式: {"diffusion_model.module.name.lora_A.weight": weight, ...}
    Musubi形式: {"prefix_module_name.lora_down.weight": weight, ...}
    """
    # Diffusers形式から標準LoRA形式への変換
    # Diffusers形式: {"diffusion_model.module.name.lora_A.weight": weight, "diffusion_model.module.name.lora_B.weight": weight}
    # 標準LoRA形式: {"prefix_module_name.lora_down.weight": weight, "prefix_module_name.lora_up.weight": weight}

    # 注意: Diffusers形式にはalphaパラメータがないため、alphaはrankに設定される
    new_weights_sd = {}
    lora_dims = {}
    for key, weight in lora_sd.items():
        diffusers_prefix, key_body = key.split(".", 1)
        if diffusers_prefix != "diffusion_model" and diffusers_prefix != "transformer":
            print(_("unexpected key: {0} in diffusers format").format(key))
            continue

        new_key = f"{prefix}{key_body}".replace(".", "_").replace("_lora_A_", ".lora_down.").replace("_lora_B_", ".lora_up.")
        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]

    # add alpha with rank
    for lora_name, dim in lora_dims.items():
        new_weights_sd[f"{lora_name}.alpha"] = torch.tensor(dim)

    return new_weights_sd


def load_safetensors_with_lora_and_fp8(
    model_files: list[str],
    list_of_lora_sd: list[dict[str, torch.Tensor]],
    multipliers: list[float],
    fp8_optimization: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    LoRA重みをモデルの状態辞書にマージし、必要に応じてFP8最適化を適用する
    
    Args:
        model_files: モデルファイルのパス一覧
        list_of_lora_sd: LoRA状態辞書のリスト
        multipliers: 各LoRAの適用強度
        fp8_optimization: FP8最適化の有効/無効
        device: 計算に使用するデバイス
        
    Returns:
        LoRAがマージされた状態辞書
    """
    list_of_lora_weight_keys = []
    for lora_sd in list_of_lora_sd:
        lora_weight_keys = set(lora_sd.keys())
        list_of_lora_weight_keys.append(lora_weight_keys)

    # LoRA重みを状態辞書にマージする処理
    print(_("Merging LoRA weights into state dict. multiplier: {0}").format(multipliers))

    # LoRAマージ用のフック関数を作成
    def weight_hook(model_weight_key, model_weight):
        nonlocal list_of_lora_weight_keys, list_of_lora_sd, multipliers

        if not model_weight_key.endswith(".weight"):
            return model_weight

        original_device = model_weight.device
        if original_device != device:
            model_weight = model_weight.to(device)  # 計算高速化のために指定デバイスへ移動

        for lora_weight_keys, lora_sd, multiplier in zip(list_of_lora_weight_keys, list_of_lora_sd, multipliers):
            # この重みにLoRA重みが存在するかをチェック
            lora_name = model_weight_key.rsplit(".", 1)[0]  # 末尾の".weight"を除去
            lora_name = "lora_unet_" + lora_name.replace(".", "_")
            down_key = lora_name + ".lora_down.weight"
            up_key = lora_name + ".lora_up.weight"
            alpha_key = lora_name + ".alpha"
            if down_key not in lora_weight_keys or up_key not in lora_weight_keys:
                continue

            # LoRA重みを取得
            down_weight = lora_sd[down_key]
            up_weight = lora_sd[up_key]

            dim = down_weight.size()[0]
            alpha = lora_sd.get(alpha_key, dim)  # alphaがない場合はdimを使用
            scale = alpha / dim  # LoRAスケール計算

            down_weight = down_weight.to(device)
            up_weight = up_weight.to(device)

            # LoRA重みをモデル重みにマージ: W <- W + α * (U * D)
            if len(model_weight.size()) == 2:
                # Linearレイヤー用処理
                if len(up_weight.size()) == 4:  # 形状不一致の場合はsqueezeで調整
                    up_weight = up_weight.squeeze(3).squeeze(2)
                    down_weight = down_weight.squeeze(3).squeeze(2)
                model_weight = model_weight + multiplier * (up_weight @ down_weight) * scale
            elif down_weight.size()[2:4] == (1, 1):
                # Conv2D 1x1カーネル用処理
                model_weight = (
                    model_weight
                    + multiplier
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
                )
            else:
                # Conv2D 3x3カーネル用処理
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                model_weight = model_weight + multiplier * conved * scale

            # 使用済みLoRAキーをセットから除去
            lora_weight_keys.remove(down_key)
            lora_weight_keys.remove(up_key)
            if alpha_key in lora_weight_keys:
                lora_weight_keys.remove(alpha_key)

        model_weight = model_weight.to(original_device)  # 元のデバイスに戻す
        return model_weight

    state_dict = load_safetensors_with_fp8_optimization(
        model_files, fp8_optimization, device, weight_hook=weight_hook
    )

    for lora_weight_keys in list_of_lora_weight_keys:
        # すべてのLoRAキーが使用されたかをチェック
        if len(lora_weight_keys) > 0:
            # 未使用のLoRAキーが残っている場合は警告を表示（エラーではない）
            print(_("Warning: not all LoRA keys are used: {0}").format(", ".join(lora_weight_keys)))

    return state_dict


def convert_hunyuan_to_framepack(lora_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    HunyuanVideo形式LoRA重みをFramePack形式に変換する
    
    HunyuanVideoのアーキテクチャ:
    - double_blocks: ダブルアテンションブロック（テキスト+画像処理）
    - single_blocks: シングルアテンションブロック（統合処理）
    
    FramePackへのマッピング:
    - double_blocks → transformer_blocks
    - single_blocks → single_transformer_blocks
    """
    new_lora_sd = {}
    for key, weight in lora_sd.items():
        if "double_blocks" in key:
            key = key.replace("double_blocks", "transformer_blocks")
            key = key.replace("img_mod_linear", "norm1_linear")
            key = key.replace("img_attn_qkv", "attn_to_QKV")  # split later
            key = key.replace("img_attn_proj", "attn_to_out_0")
            key = key.replace("img_mlp_fc1", "ff_net_0_proj")
            key = key.replace("img_mlp_fc2", "ff_net_2")
            key = key.replace("txt_mod_linear", "norm1_context_linear")
            key = key.replace("txt_attn_qkv", "attn_add_QKV_proj")  # split later
            key = key.replace("txt_attn_proj", "attn_to_add_out")
            key = key.replace("txt_mlp_fc1", "ff_context_net_0_proj")
            key = key.replace("txt_mlp_fc2", "ff_context_net_2")
        elif "single_blocks" in key:
            key = key.replace("single_blocks", "single_transformer_blocks")
            key = key.replace("linear1", "attn_to_QKVM")  # split later
            key = key.replace("linear2", "proj_out")
            key = key.replace("modulation_linear", "norm_linear")
        else:
            print(_("Unsupported module name: {0}, only double_blocks and single_blocks are supported").format(key))
            continue

        if "QKVM" in key:
            # QKVMをQ, K, V, Mに分割（single_blocks用結合アテンション形式）
            key_q = key.replace("QKVM", "q")
            key_k = key.replace("QKVM", "k")
            key_v = key.replace("QKVM", "v")
            key_m = key.replace("attn_to_QKVM", "proj_mlp")
            if "_down" in key or "alpha" in key:
                # QKVM重みまたはalphaをQ, K, V, Mにコピー
                assert "alpha" in key or weight.size(1) == 3072, f"QKVM weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
                new_lora_sd[key_m] = weight
            elif "_up" in key:
                # QKVM重みをQ, K, V, Mに分割（21504 = 3072*3 + 12288）
                assert weight.size(0) == 21504, f"QKVM weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 : 3072 * 3]
                new_lora_sd[key_m] = weight[3072 * 3 :]  # 21504 - 3072 * 3 = 12288
            else:
                print(_("Unsupported module name: {0}").format(key))
                continue
        elif "QKV" in key:
            # QKVをQ, K, Vに分割（double_blocks用アテンション形式）
            key_q = key.replace("QKV", "q")
            key_k = key.replace("QKV", "k")
            key_v = key.replace("QKV", "v")
            if "_down" in key or "alpha" in key:
                # QKV重みまたはalphaをQ, K, Vにコピー
                assert "alpha" in key or weight.size(1) == 3072, f"QKV weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
            elif "_up" in key:
                # QKV重みをQ, K, Vに分割（3072*3 = 9216）
                assert weight.size(0) == 3072 * 3, f"QKV weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 :]
            else:
                print(_("Unsupported module name: {0}").format(key))
                continue
        else:
            # 分割不要のパラメータ
            new_lora_sd[key] = weight

    return new_lora_sd


def load_safetensors_with_fp8_optimization(
    model_files: list[str], fp8_optimization: bool, device: torch.device, weight_hook: callable = None
) -> dict[str, torch.Tensor]:
    """
    safetensorsファイルから状態辞書をロードし、必要に応じてFP8最適化とLoRAマージを実行
    
    Args:
        model_files: モデルファイルのパス一覧
        fp8_optimization: FP8最適化の有効/無効
        device: 計算に使用するデバイス
        weight_hook: 重み処理用のフック関数（オプション）
        
    Returns:
        処理済みの状態辞書
    """
    if fp8_optimization:
        from lora2_utils.fp8_optimization_utils import optimize_state_dict_with_fp8_on_the_fly

        # FP8最適化のターゲットと除外レイヤーを設定
        TARGET_KEYS = ["transformer_blocks", "single_transformer_blocks"]  # FramePackアーキテクチャ対応
        EXCLUDE_KEYS = ["norm"]  # 正規化レイヤー（LayerNorm、RMSNorm等）をFP8から除外

        # 状態辞書をFP8形式に最適化
        print(_("FP8形式で状態辞書を最適化しています..."))
        state_dict = optimize_state_dict_with_fp8_on_the_fly(
            model_files, device, TARGET_KEYS, EXCLUDE_KEYS, move_to_device=False, weight_hook=weight_hook
        )
    else:
        from lora2_utils.safetensors_utils import MemoryEfficientSafeOpen

        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file) as f:
                for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", leave=False):
                    value = f.get_tensor(key)
                    if weight_hook is not None:  # フック関数が指定されている場合は適用
                        value = weight_hook(key, value)
                    state_dict[key] = value

    return state_dict
