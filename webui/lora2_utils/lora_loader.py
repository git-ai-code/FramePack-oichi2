# FramePack-eichi/oichi LoRA Loader
#
# LoRAモデルの読み込みと適用のための機能を提供します。

import os
import torch
from tqdm import tqdm
from .lora_utils import merge_lora_to_state_dict

# 多言語対応: エラーメッセージとUI表示の国際化
from locales.i18n_extended import translate as _

def load_and_apply_lora(
    model_files,
    lora_paths,
    lora_scales=None,
    fp8_enabled=False,
    device=None
):
    """
    複数LoRAファイルをベースモデルにマージして統合状態辞書を作成

    LoRA適用の仕組み:
    - 状態辞書レベルでの直接マージ（モデルインスタンス作成前処理）
    - 複数LoRAの順次適用: W <- W + α₁(U₁D₁) + α₂(U₂D₂) + ...
    - HunyuanVideo→FramePack自動変換によるクロスアーキテクチャ対応
    - FP8最適化との統合（量子化前のLoRAマージで精度保持）

    Args:
        model_files: ベースモデルsafetensorsファイル一覧
        lora_paths: 適用するLoRAファイルのパス一覧
        lora_scales: 各LoRAの適用強度一覧（デフォルト:0.8）
        fp8_enabled: FP8最適化の有効/無効（LoRAマージ後に量子化実行）
        device: LoRAマージ計算用デバイス（CPU/CUDA）

    Returns:
        LoRAがマージされたモデルの状態辞書
    """
    if lora_paths is None:
        lora_paths = []
    for lora_path in lora_paths:
        if not os.path.exists(lora_path):
            raise FileNotFoundError(_("LoRAファイルが見つかりません: {0}").format(lora_path))

    # LoRA適用強度の正規化（デフォルト0.8は実用的バランス値）
    if lora_scales is None:
        lora_scales = [0.8] * len(lora_paths)  # 0.8: 改変効果と安定性のバランス
    if len(lora_scales)> len(lora_paths):
        lora_scales = lora_scales[:len(lora_paths)]  # 余分なスケール値を切り捨て
    if len(lora_scales) < len(lora_paths):
        lora_scales += [0.8] * (len(lora_paths) - len(lora_scales))  # 不足分をデフォルト値で補完

    if device is None:
        device = torch.device("cpu")  # CPUフォールバック: メモリ不足対策・CUDA未対応環境での安定動作

    # 複数LoRA処理状況の表示（デバッグ・進捗確認用）
    for lora_path, lora_scale in zip(lora_paths, lora_scales):
        print(_("LoRAを読み込み中: {0} (スケール: {1})").format(os.path.basename(lora_path), lora_scale))

    # アーキテクチャ自動判定・変換システム（HunyuanVideo/FramePack/Diffusers形式対応）
    print(_("フォーマット: HunyuanVideo"))  # FramePack統一フォーマットとして処理

    # LoRA重みをベースモデルの状態辞書にマージ
    merged_state_dict = merge_lora_to_state_dict(model_files, lora_paths, lora_scales, fp8_enabled, device)

    # 重要制限: _lora_appliedフラグは状態辞書には保存不可
    # モデルインスタンス作成後の属性設定が必要（hasattr(model, '_lora_applied')での判定用）

    print(_("LoRAマージプロセス完了"))
    return merged_state_dict

def check_lora_applied(model):
    """
    LoRA適用状態の検出（実行時診断・デバッグ用）

    LoRA適用方式の判定:
    - direct_application: 状態辞書レベルでの直接マージ（FramePack標準方式）
    - runtime_application: モデル実行時の動的適用（未実装・将来拡張用）
    - none: LoRA未適用（ベースモデル状態）

    Args:
        model: 診断対象のモデルインスタンス

    Returns:
        (bool, str): (LoRA適用有無, 適用方式名)
    """
    # _lora_appliedフラグ: モデルローダー側で設定される適用状態マーカー
    # このフラグは状態辞書マージ完了後にmodel._lora_applied = Trueとして設定
    has_flag = hasattr(model, '_lora_applied') and model._lora_applied

    if has_flag:
        return True, "direct_application"  # 状態辞書直接マージ方式

    return False, "none"  # LoRA未適用またはフラグ設定忘れ
