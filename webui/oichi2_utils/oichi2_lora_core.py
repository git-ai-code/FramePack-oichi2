"""
oichi2_lora_core.py

FramePack-oichi2 LoRA処理中核モジュール
- LoRA設定・ファイル処理
- transformerリロード管理
- 配列パラメータによる位置ずれ防止

LoRA処理統合コアモジュール
"""

import os
import traceback
from locales.i18n_extended import translate
from common_utils.lora_config import LoRAConfig, create_lora_config, get_max_lora_count

def _reload_transformer_if_needed(transformer_manager):
    """
    transformerの状態確認とリロード
    
    Args:
        transformer_manager: transformerマネージャー
        
    Returns:
        bool: リロード成功フラグ
    """
    try:
        # transformerの状態を確認し、必要に応じてリロード
        if not transformer_manager.ensure_transformer_state():
            raise Exception(translate("transformer状態の確認に失敗しました"))
        return True
    except Exception as e:
        print(translate("transformerのリロードに失敗しました: {0}").format(e))
        traceback.print_exc()
        return False


def process_lora_configuration_unified(use_lora, force_enable, lora_config):
    """
    LoRA設定統合処理（可変対応）
    
    Args:
        use_lora (bool): LoRA使用フラグ
        force_enable (bool): 強制有効化フラグ
        lora_config (LoRAConfig): LoRA設定オブジェクト
        
    Returns:
        tuple: (lora_paths, lora_scales) パス・スケールの配列
    """
    if not use_lora and not force_enable:
        return [], []
    
    try:
        # LoRAConfigから有効なLoRAのパス・スケールを取得
        valid_paths, valid_scales = lora_config.get_active_loras()
        
        # ファイル存在確認
        existing_paths = []
        existing_scales = []
        
        for i, path in enumerate(valid_paths):
            if path and os.path.exists(path):
                existing_paths.append(path)
                if i < len(valid_scales):
                    existing_scales.append(valid_scales[i])
                else:
                    existing_scales.append(0.8)  # デフォルト値
        
        if existing_paths:
            print(translate("LoRA統合処理開始 - 最大{0}個対応").format(get_max_lora_count()))
            for i, (path, scale) in enumerate(zip(existing_paths, existing_scales)):
                filename = os.path.basename(path)
                print(translate("LoRA {0}: {1} (スケール: {2})").format(i+1, filename, scale))
            print(translate("LoRA統合処理完了 - {0}個のLoRAが有効").format(len(existing_paths)))
        
        return existing_paths, existing_scales
        
    except Exception as e:
        print(translate("LoRA統合処理エラー: {0}").format(e))
        traceback.print_exc()
        return [], []