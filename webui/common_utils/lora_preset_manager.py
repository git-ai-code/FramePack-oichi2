"""
LoRAプリセット管理モジュール

LoRA選択とスケール値のプリセット保存・読み込み機能を提供します。
可変LoRA数への対応やデフォルトプリセットの管理機能を含みます。
"""

import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from locales.i18n_extended import translate

def get_lora_presets_folder_path() -> str:
    """
    LoRAプリセットフォルダの絶対パスを取得します。
    
    Returns:
        str: LoRAプリセットフォルダの絶対パス
    """
    # eichi_utils直下からwebuiフォルダに移動し、presetsフォルダを使用
    webui_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(webui_path, 'presets')

def initialize_lora_presets() -> None:
    """
    初期LoRAプリセットファイルがない場合に作成します。
    
    可変LoRA数に対応し、デフォルトプリセットを組み込みます。
    """
    # 可変LoRA対応: 設定から最大数を取得
    try:
        from .lora_config import get_max_lora_count
        max_count = get_max_lora_count()
    except ImportError:
        max_count = 3  # フォールバック
    
    presets_folder = get_lora_presets_folder_path()
    os.makedirs(presets_folder, exist_ok=True)
    preset_file = os.path.join(presets_folder, 'lora_presets.json')

    # 既存ファイルがあり、正常に読み込める場合は終了
    if os.path.exists(preset_file):
        try:
            with open(preset_file, 'r', encoding='utf-8') as f:
                presets_data = json.load(f)
            return
        except:
            # エラーが発生した場合は新規作成
            pass

    # 新規作成
    presets_data = {
        "presets": [],
        "default_preset_index": 0
    }

    # デフォルトのLoRA設定（可変対応・負の値回避）
    default_scales_1 = ",".join(["0.8"] * max_count)
    default_scales_2 = ",".join([f"{max(0.0, 1.0 - i*0.15):.1f}" for i in range(max_count)])
    
    default_lora_configs = [
        {
            "name": translate("デフォルト設定1"),
            "scales": default_scales_1
        },
        {
            "name": translate("デフォルト設定2"), 
            "scales": default_scales_2
        }
    ]

    # デフォルトのプリセットを追加
    for i, config in enumerate(default_lora_configs):
        preset_data = {
            "name": config["name"],
            "scales": config["scales"],
            "timestamp": datetime.now().isoformat(),
            "is_default": True
        }
        # 可変LoRA項目を追加
        for j in range(max_count):
            preset_data[f"lora{j+1}"] = translate("なし")
        
        presets_data["presets"].append(preset_data)

    # 保存
    try:
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(presets_data, f, ensure_ascii=False, indent=2)
    except:
        # 保存に失敗してもエラーは出さない（次回起動時に再試行される）
        pass

def load_lora_presets() -> Tuple[List[Dict[str, Any]], int]:
    """
    LoRAプリセットを読み込みます。
    
    Returns:
        Tuple[List[Dict[str, Any]], int]: (プリセットリスト, デフォルトインデックス)
    """
    presets_folder = get_lora_presets_folder_path()
    os.makedirs(presets_folder, exist_ok=True)
    preset_file = os.path.join(presets_folder, 'lora_presets.json')
    
    # 初期化（必要に応じて）
    initialize_lora_presets()
    
    # プリセットファイルを読み込む
    try:
        with open(preset_file, 'r', encoding='utf-8') as f:
            data: Dict[str, Any] = json.load(f)
            return data["presets"], data.get("default_preset_index", 0)
    except:
        # エラーの場合は空のプリセットリストを返す
        return [], 0

def save_lora_preset(preset_index, *args):
    """LoRAプリセットを保存する関数（可変対応）"""
    presets_folder = get_lora_presets_folder_path()
    os.makedirs(presets_folder, exist_ok=True)
    preset_file = os.path.join(presets_folder, 'lora_presets.json')
    
    # 可変LoRA対応: 設定から最大数を取得
    try:
        from .lora_config import get_max_lora_count
        max_count = get_max_lora_count()
    except ImportError:
        max_count = 3  # フォールバック
    
    # 引数を分解（LoRA値 + スケール値）
    lora_values = list(args[:max_count])
    scales = args[max_count] if len(args) > max_count else ",".join(["0.8"] * max_count)
    
    # 既存のプリセットを読み込む
    try:
        with open(preset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = {"presets": [], "default_preset_index": 0}
    
    # 10個のプリセットを確保
    while len(data["presets"]) < 10:
        preset_data: Dict[str, Any] = {
            "name": translate("設定{0}").format(len(data["presets"]) + 1),
            "scales": ",".join(["0.8"] * max_count),
            "timestamp": datetime.now().isoformat(),
            "is_default": False
        }
        # 可変LoRA項目を追加
        for i in range(max_count):
            preset_data[f"lora{i+1}"] = translate("なし")
        data["presets"].append(preset_data)
    
    # 指定されたプリセットを更新
    if 0 <= preset_index < 10:
        preset_data: Dict[str, Any] = {
            "name": translate("設定{0}").format(preset_index + 1),
            "scales": scales or ",".join(["0.8"] * max_count),
            "timestamp": datetime.now().isoformat(),
            "is_default": False
        }
        # 可変LoRA値を設定
        for i in range(max_count):
            if i < len(lora_values):
                preset_data[f"lora{i+1}"] = lora_values[i] or translate("なし")
            else:
                preset_data[f"lora{i+1}"] = translate("なし")
        
        data["presets"][preset_index] = preset_data
        
        # 保存
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return True, translate("設定{0}を保存しました").format(preset_index + 1)
    else:
        return False, translate("無効なプリセット番号です")

def load_lora_preset(preset_index):
    """指定されたプリセットを読み込む関数（可変対応）"""
    # 可変LoRA対応: 設定から最大数を取得
    try:
        from .lora_config import get_max_lora_count
        max_count = get_max_lora_count()
    except ImportError:
        max_count = 3  # フォールバック
    
    presets, _ = load_lora_presets()
    
    if 0 <= preset_index < len(presets):
        preset: Dict[str, Any] = presets[preset_index]
        result: List[str] = []
        # LoRA値を順番に取得
        for i in range(max_count):
            lora_key: str = f"lora{i+1}"
            result.append(preset.get(lora_key, translate("なし")))
        # スケール値を追加
        default_scales: str = ",".join(["0.8"] * max_count)
        result.append(preset.get("scales", default_scales))
        return result
    else:
        # デフォルト値を返す
        return None

def get_preset_names():
    """プリセット名のリストを取得する関数"""
    presets, _ = load_lora_presets()
    names: List[str] = []
    for i in range(10):
        if i < len(presets):
            names.append(presets[i].get("name", translate("設定{0}").format(i + 1)))
        else:
            names.append(translate("設定{0}").format(i + 1))
    return names