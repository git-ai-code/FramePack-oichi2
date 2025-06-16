"""
FramePack-oichi2 実行履歴管理モジュール

実行履歴の保存・読み込み・UI表示機能
- JSON形式での履歴データ保存
- サムネイル画像の自動生成
- パラメータ復元機能
- 履歴表示UI管理
"""

import json
import os
import platform
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

from locales.i18n_extended import translate

# グローバル変数：選択された履歴エントリを保持
_selected_history_entry = None


def convert_path_for_current_environment(original_path: str) -> List[str]:
    """
    現在の環境に応じてパスを変換し、チェック対象のパスリストを返す
    クロスプラットフォーム対応（Windows/WSL/Linux/macOS）
    
    Args:
        original_path: 元のパス文字列
        
    Returns:
        List[str]: チェック対象パスのリスト
    """
    if not original_path:
        return []
    
    check_paths = [original_path]
    current_system = platform.system()
    
    # Windows絶対パスの検出（C:, D:など）
    if len(original_path) >= 3 and original_path[1:3] == ':\\':
        drive_letter = original_path[0].lower()
        
        # WSL環境でのパス変換
        if current_system == "Linux" and os.path.exists("/mnt"):
            # WSL形式に変換: C:\ -> /mnt/c/
            wsl_path = original_path.replace(f"{drive_letter.upper()}:\\", f"/mnt/{drive_letter}/").replace("\\", "/")
            check_paths.append(wsl_path)
        
        # macOS環境でのパス変換（Parallels, VMware, Boot Campの可能性）
        elif current_system == "Darwin":
            # macOS上でのWindows仮想環境パス変換例
            # /Volumes/Windows_C -> C:\ のような変換パターンを追加
            unix_style_path = original_path.replace("\\", "/")
            check_paths.append(unix_style_path)
            
            # Parallels Desktop風のパス変換
            if os.path.exists("/Volumes"):
                path_parts = original_path.split('\\')[1:]
                parallels_path = f"/Volumes/Windows_{drive_letter.upper()}/{'/'.join(path_parts)}"
                check_paths.append(parallels_path)
        
        # Linux環境でのパス変換（Wine, NTFS マウントなど）
        elif current_system == "Linux":
            # Unix形式のパス区切り文字に変換
            unix_style_path = original_path.replace("\\", "/")
            check_paths.append(unix_style_path)
            
            # Wine環境のパス変換例（~/.wine/drive_c）
            path_parts = original_path.split('\\')[1:]
            wine_path = os.path.expanduser(f"~/.wine/drive_{drive_letter}/{'/'.join(path_parts)}")
            check_paths.append(wine_path)
        
        # Windows環境での正規化
        elif current_system == "Windows":
            # パス区切り文字を正規化
            normalized_path = original_path.replace("/", "\\")
            if normalized_path != original_path:
                check_paths.append(normalized_path)
    
    # Unix形式パスの処理
    elif original_path.startswith("/"):
        # WSLパス形式の検出と変換
        if original_path.startswith("/mnt/") and current_system == "Windows":
            # WSLパスをWindows形式に変換: /mnt/c/ -> C:\
            parts = original_path.split("/")
            if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
                drive_letter = parts[2].upper()
                windows_path = f"{drive_letter}:\\" + "\\".join(parts[3:])
                check_paths.append(windows_path)
        
        # macOSでのボリュームパス処理
        elif current_system == "Darwin":
            # 標準的なmacOSパス正規化
            normalized_path = os.path.normpath(original_path)
            if normalized_path != original_path:
                check_paths.append(normalized_path)
        
        # Linuxでの標準パス処理
        elif current_system == "Linux":
            # パス正規化
            normalized_path = os.path.normpath(original_path)
            if normalized_path != original_path:
                check_paths.append(normalized_path)
    
    # 相対パスの場合
    else:
        # 全環境共通：相対パスの正規化
        normalized_path = os.path.normpath(original_path)
        if normalized_path != original_path:
            check_paths.append(normalized_path)
        
        # Windows環境では区切り文字変換も追加
        if current_system == "Windows":
            windows_style = original_path.replace("/", "\\")
            if windows_style != original_path:
                check_paths.append(windows_style)
        # Unix系環境では逆変換も追加
        else:
            unix_style = original_path.replace("\\", "/")
            if unix_style != original_path:
                check_paths.append(unix_style)
    
    # 重複を除去して返却
    return list(dict.fromkeys(check_paths))


def set_selected_history_entry(entry: Dict[str, Any]):
    """選択された履歴エントリを設定"""
    global _selected_history_entry
    _selected_history_entry = entry


def get_selected_history_entry() -> Optional[Dict[str, Any]]:
    """選択された履歴エントリを取得"""
    return _selected_history_entry


def collect_advanced_control_info(
    use_advanced_control: bool,
    advanced_control_mode: str,
    kisekaeichi_reference_image: Optional[str],
    kisekaeichi_control_index: int,
    input_mask: Optional[str],
    reference_mask: Optional[str],
    oneframe_mc_image: Optional[str],
    oneframe_mc_control_index: int,
    optional_control_image: Optional[str],
    optional_control_index: int
) -> Optional[Dict[str, Any]]:
    """
    高度な画像制御情報を収集する
    
    Args:
        use_advanced_control: 高度な画像制御使用フラグ
        advanced_control_mode: 制御モード（one_frame/kisekaeichi/1fmc/custom）
        kisekaeichi_reference_image: 着せ替え制御画像パス
        kisekaeichi_control_index: 着せ替え制御位置
        input_mask: 入力画像マスクパス
        reference_mask: 制御画像マスクパス
        oneframe_mc_image: 人物制御画像パス
        oneframe_mc_control_index: 人物制御位置
        optional_control_image: 追加制御画像パス
        optional_control_index: 追加制御位置
        
    Returns:
        高度な画像制御情報辞書（使用していない場合はNone）
    """
    # 1フレーム推論モードでも制御情報を保存するため、条件を緩和
    if not use_advanced_control and advanced_control_mode != "one_frame":
        return None
        
    try:
        control_info = {
            "use_advanced_control": use_advanced_control,
            "advanced_control_mode": advanced_control_mode,
            "kisekaeichi_reference_image": kisekaeichi_reference_image,
            "kisekaeichi_control_index": kisekaeichi_control_index,
            "input_mask": input_mask,
            "reference_mask": reference_mask,
            "oneframe_mc_image": oneframe_mc_image,
            "oneframe_mc_control_index": oneframe_mc_control_index,
            "optional_control_image": optional_control_image,
            "optional_control_index": optional_control_index
        }
        
        return control_info
        
    except Exception as e:
        print(f"{translate('高度な画像制御情報収集エラー')}: {e}")
        return None


def collect_lora_info(use_lora: bool, lora_config_creator) -> Optional[Dict[str, Any]]:
    """
    LoRA情報を収集する
    
    Args:
        use_lora: LoRA使用フラグ
        lora_config_creator: LoRA設定作成関数
        
    Returns:
        LoRA情報辞書（使用していない場合はNone）
    """
    if not use_lora or not lora_config_creator:
        return None
        
    try:
        lora_config = lora_config_creator()
        if not lora_config:
            return None
            
        used_loras = []
        used_count = 0
        
        # LoRAファイル情報を収集
        if hasattr(lora_config, 'get_active_loras'):
            active_loras_result = lora_config.get_active_loras()
            if isinstance(active_loras_result, tuple) and len(active_loras_result) == 2:
                paths, strengths = active_loras_result
                for i, path in enumerate(paths):
                    if path and os.path.exists(path):
                        file_name = os.path.basename(path)
                        strength = strengths[i] if i < len(strengths) else 0.8
                        # ファイル名と強度を辞書で保存
                        used_loras.append({
                            "file_name": file_name,
                            "strength": strength
                        })
                        used_count += 1
        
        return {
            "used_lora_count": used_count,
            "used_lora_names": used_loras[:20]  # 最大20個まで保存
        }
        
    except Exception as e:
        print(f"{translate('LoRA情報収集エラー')}: {e}")
        return None


def validate_and_convert_path(original_path: str) -> Optional[str]:
    """
    パス検証・変換・存在確認
    
    Args:
        original_path: 元のファイルパス
        
    Returns:
        検証済みのファイルパス、または None
    """
    if not original_path:
        return None
    
    # 環境対応パス変換
    check_paths = convert_path_for_current_environment(original_path)
    
    # 存在確認
    for path in check_paths:
        if os.path.exists(path):
            try:
                with Image.open(path) as img:
                    return path
            except Exception as e:
                print(f"{translate('画像パス検証エラー')} ({path}): {e}")
                continue
    
    return None


def restore_images(history_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    画像の復元
    
    Args:
        history_entry: 履歴エントリ辞書
        
    Returns:
        画像復元用更新辞書
    """
    import gradio as gr
    
    image_updates = {}
    
    # 入力画像復元
    input_path = history_entry.get("input_image_path")
    if input_path:
        validated_path = validate_and_convert_path(input_path)
        if validated_path:
            image_updates["input_image"] = gr.update(value=validated_path)
    
    # 高度な画像制御復元
    advanced_control = history_entry.get("advanced_control_info")
    
    # 画像コンポーネントのリスト（5画像全て）
    image_mappings = {
        "kisekaeichi_reference_image": "kisekaeichi_reference_image",
        "oneframe_mc_image": "oneframe_mc_image", 
        "optional_control_image": "optional_control_image",
        "input_mask": "input_mask",
        "reference_mask": "reference_mask"
    }
    
    # 制御位置のリスト（3つの位置）
    position_mappings = {
        "kisekaeichi_control_index": "kisekaeichi_control_index",
        "oneframe_mc_control_index": "oneframe_mc_control_index",
        "optional_control_index": "optional_control_index"
    }
    
    if advanced_control:
        # 高度な画像制御情報がある場合：該当画像を復元
        for data_key, component_key in image_mappings.items():
            image_path = advanced_control.get(data_key)
            if image_path:
                validated_path = validate_and_convert_path(image_path)
                if validated_path:
                    image_updates[component_key] = gr.update(value=validated_path)
                else:
                    # パスが無効な場合はリセット
                    image_updates[component_key] = gr.update(value=None)
            else:
                # 画像パスがない場合はリセット
                image_updates[component_key] = gr.update(value=None)
        
        # 制御位置復元
        for data_key, component_key in position_mappings.items():
            position = advanced_control.get(data_key)
            if position is not None:
                image_updates[component_key] = gr.update(value=position)
            else:
                # 位置情報がない場合はデフォルト値(0)に設定
                image_updates[component_key] = gr.update(value=0)
    else:
        # 高度な画像制御情報がない場合：5画像全てをリセット
        for component_key in image_mappings.values():
            image_updates[component_key] = gr.update(value=None)
        
        # 制御位置も全てデフォルト値(0)にリセット
        for component_key in position_mappings.values():
            image_updates[component_key] = gr.update(value=0)
    
    return image_updates


def restore_lora_settings(history_entry: Dict[str, Any], main_ui_components: Dict[str, Any], available_lora_count: int = 20) -> Dict[str, Any]:
    """
    LoRA設定の復元（oichi2_lora_preset_ui.py方式採用）
    
    Args:
        history_entry: 履歴エントリ辞書
        main_ui_components: メインUIコンポーネント辞書
        
    Returns:
        LoRA復元用更新辞書
    """
    import gradio as gr
    
    lora_updates = {}
    lora_info = history_entry.get("lora_info")
    
    # LoRAディレクトリスキャン（oichi2_lora_preset_ui.py方式）
    try:
        from oichi2_utils.oichi2_ui_utilities import scan_lora_directory
        
        lora_choices = scan_lora_directory()
        default_value = lora_choices[0] if lora_choices else "なし"
        
    except Exception:
        lora_choices = ["なし"]
        default_value = "なし"
    
    # LoRA設定復元（use_loraは基本復元側で処理済み）
    if not lora_info or lora_info.get("used_lora_count", 0) == 0:
        # LoRA未使用時：全て「なし」に設定
        for i in range(available_lora_count):
            lora_updates[f"lora_dropdown_{i}"] = gr.update(choices=lora_choices, value="なし")
            lora_updates[f"lora_strength_{i}"] = gr.update(value=0.8)
        
        lora_updates["lora_scales_text"] = gr.update(value=",".join(["0.8"] * available_lora_count))
    else:
        # LoRA使用時：プリセット方式で復元（表示状態は基本復元側で保証済み）
        lora_updates["lora_mode"] = gr.update(value="ディレクトリから選択")
        
        # プリセット方式での復元実行
        used_loras = lora_info.get("used_lora_names", [])
        lora_values = []
        strength_values = []
        
        # LoRA情報をプリセット形式に変換
        for i in range(available_lora_count):
            if i < len(used_loras) and used_loras[i]:
                lora_item = used_loras[i]
                if isinstance(lora_item, dict):
                    file_name = lora_item.get("file_name", "")
                    strength = lora_item.get("strength", 0.8)
                    
                    if file_name in lora_choices:
                        lora_values.append(file_name)
                        strength_values.append(strength)
                    else:
                        lora_values.append("なし")
                        strength_values.append(0.8)
                else:
                    # 旧形式対応
                    file_name = str(lora_item)
                    if file_name in lora_choices:
                        lora_values.append(file_name)
                        strength_values.append(0.8)
                    else:
                        lora_values.append("なし")
                        strength_values.append(0.8)
            else:
                lora_values.append("なし")
                strength_values.append(0.8)
        
        # プリセットロード方式の更新（確実な反映）
        for i in range(available_lora_count):
            lora_updates[f"lora_dropdown_{i}"] = gr.update(choices=lora_choices, value=lora_values[i])
            lora_updates[f"lora_strength_{i}"] = gr.update(value=float(strength_values[i]))
        
        # スケール文字列更新
        scales_value = ",".join([str(s) for s in strength_values])
        lora_updates["lora_scales_text"] = gr.update(value=scales_value)
    
    return lora_updates


def restore_parameters_enhanced(history_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    パラメータの拡張復元
    
    Args:
        history_entry: 履歴エントリ辞書
        
    Returns:
        パラメータ復元用更新辞書
    """
    import gradio as gr
    
    param_updates = {}
    parameters = history_entry.get("parameters", {})
    
    # パラメータ名マッピング（日本語表示名→内部名）
    param_name_mapping = {
        "シード値": "seed",
        "ステップ数": "steps", 
        "蒸留CFGスケール": "gs",
        "解像度": "resolution",
        "解像度レベル": "resolution",
        "フレーム処理範囲": "latent_window_size",
        "生成フレーム位置": "latent_index",
        "参照フレーム位置": "clean_index",
        "clean_latents_2x": "use_clean_latents_2x",
        "clean_latents_4x": "use_clean_latents_4x",
        "clean_latents_post": "use_clean_latents_post",
        "TeaCache使用": "use_teacache",
        "GPU メモリ保持": "gpu_memory_preservation",
        "GPUメモリ保持": "gpu_memory_preservation",
        "FP8最適化": "fp8_optimization",
        "LoRA使用": "use_lora",
        "高度な画像制御": "use_advanced_control"
    }
    
    # LoRA情報に基づくuse_lora動的設定
    lora_info = history_entry.get("lora_info")
    if lora_info and lora_info.get("used_lora_count", 0) > 0:
        use_lora_value = True
    else:
        use_lora_value = False
    
    # パラメータ復元（正しい型変換）
    for param_key, param_value in parameters.items():
        try:
            if param_key in ["seed", "steps", "resolution", "latent_window_size", "latent_index", "clean_index", 
                            "gpu_memory_preservation", "kisekaeichi_control_index", "oneframe_mc_control_index", "optional_control_index"]:
                # Number コンポーネント用の整数変換
                param_updates[param_key] = gr.update(value=int(param_value))
            elif param_key in ["gs", "cfg", "rs"]:
                # Slider コンポーネント用の浮動小数点変換
                param_updates[param_key] = gr.update(value=float(param_value))
            elif param_key == "use_lora":
                # use_loraはLoRA復元側で処理するため、ここでは処理しない
                continue
            elif param_key in ["use_teacache", "use_clean_latents_2x", "use_clean_latents_4x", "use_clean_latents_post", 
                              "fp8_optimization", "use_advanced_control"]:
                # Checkbox コンポーネント用のブール変換
                param_updates[param_key] = gr.update(value=bool(param_value))
            else:
                # Textbox など文字列コンポーネント
                param_updates[param_key] = gr.update(value=str(param_value))
        except (ValueError, TypeError):
            param_updates[param_key] = gr.update(value=param_value)
    
    # プロンプト復元
    prompt_full = history_entry.get("prompt_full", "")
    if prompt_full:
        param_updates["prompt"] = gr.update(value=prompt_full)
    
    # use_loraはLoRA復元側で処理するため、ここでは設定しない
    
    # use_random_seed は履歴保存されないため、復元時に常に False を設定
    param_updates["use_random_seed"] = gr.update(value=False)
    
    # 制御モード復元処理
    advanced_control_info = history_entry.get("advanced_control_info")
    if advanced_control_info:
        control_mode = advanced_control_info.get("advanced_control_mode")
        if control_mode:
            param_updates["advanced_control_mode"] = gr.update(value=control_mode)
    
    return param_updates


class HistoryManager:
    """実行履歴管理クラス"""
    
    def __init__(self, history_file_path: str, max_history_count: int = 20):
        """
        初期化
        
        Args:
            history_file_path: 履歴ファイルのパス
            max_history_count: 保持する最大履歴数
        """
        self.history_file_path = history_file_path
        self.max_history_count = max_history_count
        self.thumbnails_dir = os.path.join(os.path.dirname(history_file_path), "thumbnails")
        
        # サムネイルディレクトリ作成
        os.makedirs(self.thumbnails_dir, exist_ok=True)
        
        # 履歴データ初期化
        self.history_data = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """履歴データを読み込み"""
        try:
            if os.path.exists(self.history_file_path):
                with open(self.history_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"{translate('履歴読み込みエラー')}: {e}")
        return []
    
    def _save_history(self):
        """履歴データを保存"""
        try:
            with open(self.history_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.history_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"{translate('履歴保存エラー')}: {e}")
    
    def _create_thumbnail(self, image_path: str, thumbnail_id: str) -> str:
        """サムネイル画像を生成"""
        try:
            thumbnail_path = os.path.join(self.thumbnails_dir, f"{thumbnail_id}.jpg")
            
            with Image.open(image_path) as img:
                # サムネイルサイズ（150x150）
                img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                img.convert('RGB').save(thumbnail_path, 'JPEG', quality=85)
                
            return thumbnail_path
        except Exception as e:
            print(f"{translate('サムネイル作成エラー')}: {e}")
            return ""
    
    def _copy_control_image(self, image_path: str, control_id: str) -> str:
        """制御画像を永続化フォルダにコピー"""
        try:
            # 元の拡張子を保持
            original_ext = os.path.splitext(image_path)[1] or '.png'
            control_path = os.path.join(self.thumbnails_dir, f"{control_id}{original_ext}")
            
            # 画像をコピー
            with Image.open(image_path) as img:
                img.save(control_path)
                
            return control_path
        except Exception as e:
            print(f"{translate('制御画像コピーエラー')}: {e}")
            return ""
    
    def add_history_entry(self, 
                         input_image_path: Optional[str],
                         output_image_path: str,
                         prompt: str,
                         negative_prompt: str,
                         parameters: Dict[str, Any],
                         lora_info: Optional[Dict[str, Any]] = None,
                         advanced_control_info: Optional[Dict[str, Any]] = None) -> str:
        """
        履歴エントリを追加
        
        Args:
            input_image_path: 入力画像パス
            output_image_path: 出力画像パス  
            prompt: プロンプト
            negative_prompt: ネガティブプロンプト
            parameters: 生成パラメータ
            lora_info: LoRA情報
            advanced_control_info: 高度な画像制御情報
            
        Returns:
            履歴エントリID
        """
        timestamp = datetime.now()
        entry_id = f"history_{int(timestamp.timestamp())}"
        
        # サムネイル作成と入力画像の永続化
        input_thumbnail = ""
        input_image_copied = ""
        if input_image_path and os.path.exists(input_image_path):
            input_thumbnail = self._create_thumbnail(input_image_path, f"{entry_id}_input")
            # 入力画像も永続化
            input_image_copied = self._copy_control_image(input_image_path, f"{entry_id}_input_full")
            
        output_thumbnail = ""
        if os.path.exists(output_image_path):
            output_thumbnail = self._create_thumbnail(output_image_path, f"{entry_id}_output")
        
        # 高度な画像制御情報の永続化
        control_images_copied = {}
        if advanced_control_info:
            control_mappings = {
                "input_mask": f"{entry_id}_input_mask",
                "oneframe_mc_image": f"{entry_id}_oneframe_mc",
                "optional_control_image": f"{entry_id}_optional_control",
                "reference_mask": f"{entry_id}_reference_mask",
                "kisekaeichi_reference_image": f"{entry_id}_kisekaeichi"
            }
            
            for path_key, file_id in control_mappings.items():
                original_path = advanced_control_info.get(path_key)
                if original_path and os.path.exists(original_path):
                    copied_path = self._copy_control_image(original_path, file_id)
                    if copied_path:
                        control_images_copied[path_key] = copied_path
            
            # 高度な画像制御情報を更新（コピーされたパスに変更）
            if control_images_copied:
                advanced_control_info = advanced_control_info.copy()
                for path_key, copied_path in control_images_copied.items():
                    advanced_control_info[path_key] = copied_path
        
        # 履歴エントリ作成
        history_entry = {
            "id": entry_id,
            "timestamp": timestamp.isoformat(),
            "timestamp_display": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "input_image_path": input_image_copied if input_image_copied else input_image_path,
            "output_image_path": output_image_path,
            "input_thumbnail": input_thumbnail,
            "output_thumbnail": output_thumbnail,
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,  # 表示用短縮
            "prompt_full": prompt,  # 元のプロンプト
            "negative_prompt": negative_prompt,
            "parameters": parameters,
            "lora_info": lora_info,  # LoRA情報追加
            "advanced_control_info": advanced_control_info  # 高度な画像制御情報追加
        }
        
        # 履歴に追加（先頭に挿入）
        self.history_data.insert(0, history_entry)
        
        # 最大数を超えた場合は古いものを削除
        if len(self.history_data) > self.max_history_count:
            removed_entries = self.history_data[self.max_history_count:]
            self.history_data = self.history_data[:self.max_history_count]
            
            # 削除されたエントリのサムネイルも削除
            for entry in removed_entries:
                self._cleanup_thumbnails(entry)
        
        # 保存
        self._save_history()
        
        return entry_id
    
    def _cleanup_thumbnails(self, entry: Dict[str, Any]):
        """削除されたエントリのサムネイルと制御画像をクリーンアップ"""
        # サムネイル削除
        for thumb_key in ["input_thumbnail", "output_thumbnail"]:
            thumbnail_path = entry.get(thumb_key, "")
            if thumbnail_path and os.path.exists(thumbnail_path):
                try:
                    os.remove(thumbnail_path)
                except Exception:
                    pass
        
        # コピーした入力画像削除
        input_path = entry.get("input_image_path", "")
        if input_path and input_path.startswith(self.thumbnails_dir) and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass
        
        # 制御画像削除
        advanced_control_info = entry.get("advanced_control_info")
        if advanced_control_info:
            for path_key in ["input_mask", "oneframe_mc_image", "optional_control_image", "reference_mask", "kisekaeichi_reference_image"]:
                control_path = advanced_control_info.get(path_key, "")
                if control_path and control_path.startswith(self.thumbnails_dir) and os.path.exists(control_path):
                    try:
                        os.remove(control_path)
                    except Exception:
                        pass
    
    def get_history_list(self) -> List[Dict[str, Any]]:
        """履歴リストを取得"""
        return self.history_data.copy()
    
    def get_history_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """指定IDの履歴エントリを取得"""
        for entry in self.history_data:
            if entry["id"] == entry_id:
                return entry.copy()
        return None
    
    def clear_history(self):
        """履歴をクリア"""
        # サムネイルファイルも削除
        for entry in self.history_data:
            self._cleanup_thumbnails(entry)
        
        # thumbnailsフォルダ内の残りファイルを全削除
        self._clear_thumbnails_folder()
        
        self.history_data = []
        self._save_history()
    
    def _clear_thumbnails_folder(self):
        """thumbnailsフォルダ内のファイルを全て削除"""
        try:
            if os.path.exists(self.thumbnails_dir):
                for filename in os.listdir(self.thumbnails_dir):
                    file_path = os.path.join(self.thumbnails_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"{translate('サムネイル削除')}: {filename}")
                    except Exception as e:
                        print(f"{translate('サムネイル削除エラー')} ({filename}): {e}")
        except Exception as e:
            print(f"{translate('thumbnailsフォルダクリアエラー')}: {e}")


def create_history_ui_components():
    """履歴UI要素を作成"""
    import gradio as gr
    
    # 履歴パネル（折りたたみ式）
    with gr.Accordion(label=translate("実行履歴") + "（最大過去20回分保持）", open=False) as history_accordion:
        with gr.Row():
            history_refresh_btn = gr.Button(value=translate("更新"), size="lg")
            # 通常の履歴クリアボタン（初期非活性）
            history_clear_btn = gr.Button(value=translate("履歴クリア"), size="lg", visible=True, interactive=False)
            # 確認時の分離ボタン（初期非表示）
            history_clear_confirm_btn = gr.Button(value=translate("クリア実行"), size="lg", visible=False, variant="stop")
            history_clear_cancel_btn = gr.Button(value=translate("キャンセル"), size="lg", visible=False)
        
        # 履歴表示エリア（入力画像：サムネイル：出力画像 = 1：3：1）
        with gr.Row():
            with gr.Column(scale=1):
                # 入力画像表示（元サイズ）
                selected_input_image = gr.Image(
                    label=translate("入力画像（元サイズ）"),
                    height=220,
                    interactive=False,
                    visible=True
                )
            
            with gr.Column(scale=3):
                # 履歴一覧表示（10列x2行の20個）
                history_gallery = gr.Gallery(
                    label=translate("履歴サムネイル"),
                    show_label=True,
                    elem_id="history_gallery",
                    columns=10,
                    rows=2,
                    height=220,
                    allow_preview=True
                )
            
            with gr.Column(scale=1):
                # 出力画像表示（元サイズ）
                selected_output_image = gr.Image(
                    label=translate("出力画像（元サイズ）"),
                    height=220,
                    interactive=False,
                    visible=True
                )
        
        # 高度な画像制御情報表示（アコーディオン）
        with gr.Accordion(label=translate("高度な画像制御情報"), open=False) as advanced_control_accordion:
            with gr.Row():
                # 入力画像マスク
                selected_input_mask = gr.Image(
                    label=translate("入力画像マスク"),
                    height=220,
                    interactive=False,
                    visible=True
                )
                
                # 人物制御画像（1f-mc）
                selected_oneframe_mc = gr.Image(
                    label=translate("人物制御画像（1f-mc）"),
                    height=220,
                    interactive=False,
                    visible=True
                )
                
                # 追加制御画像（1f-mc）
                selected_optional_control = gr.Image(
                    label=translate("追加制御画像（1f-mc）"),
                    height=220,
                    interactive=False,
                    visible=True
                )
                
                # 制御画像マスク（オプション）
                selected_reference_mask = gr.Image(
                    label=translate("制御画像マスク"),
                    height=220,
                    interactive=False,
                    visible=True
                )
                
                # 着せ替え制御画像
                selected_kisekaeichi = gr.Image(
                    label=translate("着せ替え制御画像"),
                    height=220,
                    interactive=False,
                    visible=True
                )
            
            # 制御モード・位置情報表示
            with gr.Row():
                selected_control_positions = gr.Textbox(
                    label=translate("制御モード・位置情報"),
                    lines=1,
                    interactive=False
                )
        
        # 選択された履歴の詳細情報
        selected_prompt = gr.Textbox(
            label=translate("選択履歴のプロンプト"),
            lines=1,
            interactive=False
        )
        
        # パラメータを4列にUI順で分散表示
        with gr.Row():
            with gr.Column(scale=1):
                selected_params_col1 = gr.DataFrame(
                    label=translate("生成設定"),
                    headers=["パラメータ", "値"],
                    datatype=["str", "str"],
                    row_count=4,
                    interactive=False
                )
            
            with gr.Column(scale=1):
                selected_params_col2 = gr.DataFrame(
                    label=translate("LoRA設定"),
                    headers=["パラメータ", "値"],
                    datatype=["str", "str"],
                    row_count=4,
                    interactive=False
                )
            
            with gr.Column(scale=1):
                selected_params_col3 = gr.DataFrame(
                    label=translate("レイテント設定"),
                    headers=["パラメータ", "値"],
                    datatype=["str", "str"],
                    row_count=4,
                    interactive=False
                )
            
            with gr.Column(scale=1):
                selected_params_col4 = gr.DataFrame(
                    label=translate("基本設定"),
                    headers=["パラメータ", "値"],
                    datatype=["str", "str"],
                    row_count=4,
                    interactive=False
                )
        
        # 復元ボタンと実行日時（水平配置）
        with gr.Row():
            with gr.Column(scale=2):
                selected_timestamp = gr.Textbox(
                    label=translate("実行日時"),
                    interactive=False
                )
            
            with gr.Column(scale=1):
                restore_params_btn = gr.Button(
                    value=translate("パラメータを復元"),
                    variant="primary",
                    size="lg",
                    interactive=False  # 初期非活性
                )
                # LoRA専用復元ボタン（直下配置・初期非活性）
                restore_lora_btn = gr.Button(
                    value=translate("LoRAを復元"),
                    variant="secondary",
                    size="lg",
                    interactive=False  # 初期非活性
                )
    
    return {
        "accordion": history_accordion,
        "gallery": history_gallery,
        "refresh_btn": history_refresh_btn,
        "clear_btn": history_clear_btn,
        "clear_confirm_btn": history_clear_confirm_btn,
        "clear_cancel_btn": history_clear_cancel_btn,
        "selected_prompt": selected_prompt,
        "selected_input_image": selected_input_image,
        "selected_output_image": selected_output_image,
        "advanced_control_accordion": advanced_control_accordion,
        "selected_input_mask": selected_input_mask,
        "selected_oneframe_mc": selected_oneframe_mc,
        "selected_optional_control": selected_optional_control,
        "selected_reference_mask": selected_reference_mask,
        "selected_kisekaeichi": selected_kisekaeichi,
        "selected_control_positions": selected_control_positions,
        "selected_params_col1": selected_params_col1,
        "selected_params_col2": selected_params_col2,
        "selected_params_col3": selected_params_col3,
        "selected_params_col4": selected_params_col4,
        "restore_btn": restore_params_btn,
        "restore_lora_btn": restore_lora_btn,
        "selected_timestamp": selected_timestamp
    }


def setup_history_event_handlers(history_manager: HistoryManager, ui_components: Dict, main_ui_components: Dict):
    """履歴UIのイベントハンドラーを設定"""
    import gradio as gr
    
    def refresh_history():
        """履歴を更新"""
        history_list = history_manager.get_history_list()
        
        if not history_list:
            return gr.update(value=[]), gr.update(interactive=False)
        
        # ギャラリー用データ準備
        gallery_data = []
        for i, entry in enumerate(history_list):
            thumbnail_path = entry.get("output_thumbnail")
            if thumbnail_path and os.path.exists(thumbnail_path):
                gallery_data.append((thumbnail_path, f"{entry['timestamp_display']}\n{entry['prompt']}"))
        
        # 履歴が存在する場合はクリアボタンを活性化
        return gr.update(value=gallery_data), gr.update(interactive=True)
    
    def clear_history():
        """履歴クリア確認モードに入る"""
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            # 確認ダイアログ表示
            root = tk.Tk()
            root.withdraw()  # メインウィンドウを非表示
            
            result = messagebox.askyesno(
                translate("履歴クリア確認"),
                translate("全ての実行履歴を削除しますか？\nこの操作は取り消せません。")
            )
            
            root.destroy()
            
            if result:
                history_manager.clear_history()
                gallery_update, clear_btn_update = refresh_history()
                return gallery_update, clear_btn_update, gr.update(visible=False), gr.update(visible=False)
            else:
                # キャンセルされた場合は現在の状態を返す
                gallery_update, clear_btn_update = refresh_history()
                return gallery_update, clear_btn_update, gr.update(visible=False), gr.update(visible=False)
        except ImportError:
            # tkinterが利用できない場合は確認ボタン表示
            print(f"{translate('履歴クリア確認: クリア実行またはキャンセルを選択してください')}")
            gallery_update, clear_btn_update = refresh_history()
            return (
                gallery_update,
                gr.update(visible=False),  # 通常クリアボタン非表示
                gr.update(visible=True),   # クリア実行ボタン表示
                gr.update(visible=True)    # キャンセルボタン表示
            )
    
    def clear_history_confirm():
        """履歴を実際にクリア"""
        history_manager.clear_history()
        print(f"{translate('履歴をクリアしました')}")
        gallery_update, clear_btn_update = refresh_history()
        return (
            gallery_update,
            gr.update(visible=True, interactive=clear_btn_update.get("interactive", False)),  # クリアボタン表示・非活性に戻る
            gr.update(visible=False),  # クリア実行ボタン非表示
            gr.update(visible=False)   # キャンセルボタン非表示
        )
    
    def clear_history_cancel():
        """履歴クリアをキャンセル"""
        print(f"{translate('履歴クリアをキャンセルしました')}")
        gallery_update, clear_btn_update = refresh_history()
        return (
            gallery_update,
            gr.update(visible=True, interactive=clear_btn_update.get("interactive", True)),  # クリアボタン表示・活性状態復元
            gr.update(visible=False),  # クリア実行ボタン非表示
            gr.update(visible=False)   # キャンセルボタン非表示
        )
    
    def on_gallery_select(evt: gr.SelectData):
        """ギャラリー選択時の処理"""
        if evt.index is None:
            return None, None, None, None, None, None, None, "", "", [], [], [], [], "", gr.update(interactive=False), gr.update(interactive=False)
        
        history_list = history_manager.get_history_list()
        if evt.index >= len(history_list):
            return None, None, None, None, None, None, None, "", "", [], [], [], [], "", gr.update(interactive=False), gr.update(interactive=False)
        
        selected_entry = history_list[evt.index]
        
        # 選択履歴を保存（復元時使用）
        set_selected_history_entry(selected_entry)
        
        # 入力画像ファイル存在チェック・表示
        input_image_path = None
        input_image_path_entry = selected_entry.get("input_image_path", "")
        if input_image_path_entry:
            # パス変換チェック（Windowsパスの場合のみ）
            # 環境対応パス変換
            check_paths = convert_path_for_current_environment(input_image_path_entry)
            
            for check_path in check_paths:
                if os.path.exists(check_path):
                    try:
                        from PIL import Image
                        with Image.open(check_path) as img:
                            input_image_path = check_path
                            break
                    except Exception as e:
                        print(f"{translate('入力画像読み込みエラー')} ({check_path}): {e}")
        
        # 出力画像ファイル存在チェック・表示
        output_image_path = None
        output_image_path_entry = selected_entry.get("output_image_path", "")
        if output_image_path_entry:
            # 環境対応パス変換
            check_paths = convert_path_for_current_environment(output_image_path_entry)
            
            for check_path in check_paths:
                if os.path.exists(check_path):
                    try:
                        from PIL import Image
                        with Image.open(check_path) as img:
                            output_image_path = check_path
                            break
                    except Exception as e:
                        print(f"{translate('出力画像読み込みエラー')} ({check_path}): {e}")
        
        # 高度な画像制御情報の処理
        control_image_paths = {
            "input_mask_path": None,
            "oneframe_mc_path": None,
            "optional_control_path": None,
            "reference_mask_path": None,
            "kisekaeichi_path": None
        }
        control_positions_text = ""
        
        advanced_control_info = selected_entry.get("advanced_control_info")
        
        if advanced_control_info:
            # 各制御画像のパス確認
            for path_key, var_name in [
                ("input_mask", "input_mask_path"),
                ("oneframe_mc_image", "oneframe_mc_path"),
                ("optional_control_image", "optional_control_path"),
                ("reference_mask", "reference_mask_path"),
                ("kisekaeichi_reference_image", "kisekaeichi_path")
            ]:
                original_path = advanced_control_info.get(path_key)
                
                if original_path:
                    # 環境対応パス変換
                    check_paths = convert_path_for_current_environment(original_path)
                    
                    for check_path in check_paths:
                        if os.path.exists(check_path):
                            try:
                                from PIL import Image
                                with Image.open(check_path) as img:
                                    control_image_paths[var_name] = check_path
                                    break
                            except Exception as e:
                                print(f"{translate(f'{path_key}読み込みエラー')} ({check_path}): {e}")
        
        # 辞書から個別変数に取り出し
        input_mask_path = control_image_paths["input_mask_path"]
        oneframe_mc_path = control_image_paths["oneframe_mc_path"]
        optional_control_path = control_image_paths["optional_control_path"]
        reference_mask_path = control_image_paths["reference_mask_path"]
        kisekaeichi_path = control_image_paths["kisekaeichi_path"]
        
        # 制御モード・位置情報生成
        if advanced_control_info:
            positions = []
            
            # 制御モード表示
            control_mode = advanced_control_info.get("advanced_control_mode", "未設定")
            mode_display_names = {
                "one_frame": "1フレーム推論",
                "kisekaeichi": "kisekaeichi", 
                "1fmc": "1f-mc",
                "custom": "カスタム"
            }
            mode_display = mode_display_names.get(control_mode, control_mode)
            positions.append(f"制御モード: {mode_display}")
            
            # 位置情報追加
            if advanced_control_info.get("kisekaeichi_control_index") is not None:
                positions.append(f"着せ替え制御位置: {advanced_control_info['kisekaeichi_control_index']}")
            if advanced_control_info.get("oneframe_mc_control_index") is not None:
                positions.append(f"人物制御位置: {advanced_control_info['oneframe_mc_control_index']}")
            if advanced_control_info.get("optional_control_index") is not None:
                positions.append(f"追加制御位置: {advanced_control_info['optional_control_index']}")
            
            control_positions_text = " | ".join(positions)
        
        # パラメータ名の日本語化マッピング
        param_names_jp = {
            "seed": "シード値",
            "steps": "ステップ数", 
            "resolution": "解像度",
            "gs": "蒸留CFGスケール",
            "latent_window_size": "フレーム処理範囲",
            "latent_index": "生成フレーム位置",
            "clean_index": "参照フレーム位置",
            "gpu_memory_preservation": "GPU メモリ保持",
            "use_clean_latents_2x": "clean_latents_2x",
            "use_clean_latents_4x": "clean_latents_4x", 
            "use_clean_latents_post": "clean_latents_post",
            "use_teacache": "TeaCache使用",
            "use_lora": "LoRA使用"
        }
        
        # UI順にパラメータを4列配置（全て4行で統一）
        col1_params = ["resolution", "latent_index", "clean_index"]  # 生成設定 (3項目 + 使用LoRA数で4項目)
        col2_params = []  # LoRA設定（特別処理・最大4項目）
        col3_params = ["latent_window_size", "use_clean_latents_2x", "use_clean_latents_4x", "use_clean_latents_post"]  # レイテント設定 (4項目)
        col4_params = ["use_teacache", "seed", "steps", "gs"]  # 基本設定 (4項目)
        
        # 第1列データ（生成設定）
        col1_data = []
        for key in col1_params:
            if key in selected_entry["parameters"]:
                display_name = param_names_jp.get(key, key)
                col1_data.append([display_name, str(selected_entry["parameters"][key])])
        
        # 第1列に使用LoRA数を追加
        if "lora_info" in selected_entry and selected_entry["lora_info"]:
            lora_info = selected_entry["lora_info"]
            if lora_info.get("used_lora_count", 0) > 0:
                col1_data.append(["使用LoRA数", str(lora_info["used_lora_count"])])
            else:
                col1_data.append(["LoRA使用", "False"])
        else:
            # LoRA情報がない場合は従来のuse_lora表示
            if "use_lora" in selected_entry["parameters"]:
                use_lora_value = selected_entry["parameters"]["use_lora"]
                if use_lora_value:
                    col1_data.append(["LoRA使用", "True"])
                else:
                    col1_data.append(["LoRA使用", "False"])
        
        # 第2列データ（LoRA設定）
        col2_data = []
        
        # LoRA情報の特別処理（第2列：LoRA設定）
        if "lora_info" in selected_entry and selected_entry["lora_info"]:
            lora_info = selected_entry["lora_info"]
            if lora_info.get("used_lora_count", 0) > 0:
                # 使用中のLoRAファイル名と強度を表示
                used_loras = lora_info.get("used_lora_names", [])
                if used_loras:
                    # 最初の3個は個別表示
                    for i, lora_item in enumerate(used_loras[:3]):
                        if lora_item:
                            display_name = f"LoRA{i+1}"
                            
                            # 新形式（辞書）と旧形式（文字列）の両方に対応
                            if isinstance(lora_item, dict):
                                file_name = lora_item.get("file_name", "")
                                strength = lora_item.get("strength", 0.8)
                                full_name = file_name.replace(".safetensors", "").replace(".pt", "")
                                display_value = f"{full_name}({strength})"
                            else:
                                # 旧形式（文字列のみ）への後方互換
                                full_name = lora_item.replace(".safetensors", "").replace(".pt", "")
                                display_value = full_name
                            
                            col2_data.append([display_name, display_value])
                    
                    # 4個以上の場合は4行目にカンマ区切りで残り全て表示
                    if len(used_loras) > 3:
                        remaining_loras = []
                        for lora_item in used_loras[3:]:
                            if lora_item:
                                if isinstance(lora_item, dict):
                                    file_name = lora_item.get("file_name", "")
                                    strength = lora_item.get("strength", 0.8)
                                    full_name = file_name.replace(".safetensors", "").replace(".pt", "")
                                    remaining_loras.append(f"{full_name}({strength})")
                                else:
                                    # 旧形式への後方互換
                                    full_name = lora_item.replace(".safetensors", "").replace(".pt", "")
                                    remaining_loras.append(full_name)
                        
                        if remaining_loras:
                            remaining_text = ", ".join(remaining_loras)
                            col2_data.append(["LoRA4以降", remaining_text])
                        else:
                            col2_data.append(["", ""])
                    else:
                        # 3個以下の場合は空行で埋める
                        while len(col2_data) < 4:
                            col2_data.append(["", ""])
        
        # 第3列データ（レイテント設定）
        col3_data = []
        for key in col3_params:
            if key in selected_entry["parameters"]:
                display_name = param_names_jp.get(key, key)
                col3_data.append([display_name, str(selected_entry["parameters"][key])])
        
        # 第4列データ（基本設定）
        col4_data = []
        for key in col4_params:
            if key in selected_entry["parameters"]:
                display_name = param_names_jp.get(key, key)
                col4_data.append([display_name, str(selected_entry["parameters"][key])])
        
        
        # サムネイル選択時は両方のボタンを活性化
        # ただし、LoRAボタンは基本パラメータ復元後に正常に動作するため、基本のみ最初に活性化
        return (
            input_image_path,
            output_image_path,
            input_mask_path,
            oneframe_mc_path,
            optional_control_path,
            reference_mask_path,
            kisekaeichi_path,
            control_positions_text,
            selected_entry["prompt_full"],
            col1_data,
            col2_data,
            col3_data,
            col4_data,
            selected_entry["timestamp_display"],
            gr.update(interactive=True),   # パラメータ復元ボタン活性化
            gr.update(interactive=False)   # LoRAボタンは基本復元後に活性化
        )
    
    # イベントハンドラー設定
    ui_components["refresh_btn"].click(
        fn=refresh_history,
        outputs=[ui_components["gallery"], ui_components["clear_btn"]]
    )
    
    ui_components["clear_btn"].click(
        fn=clear_history,
        outputs=[
            ui_components["gallery"], 
            ui_components["clear_btn"],
            ui_components["clear_confirm_btn"],
            ui_components["clear_cancel_btn"]
        ]
    )
    
    ui_components["clear_confirm_btn"].click(
        fn=clear_history_confirm,
        outputs=[
            ui_components["gallery"], 
            ui_components["clear_btn"],
            ui_components["clear_confirm_btn"],
            ui_components["clear_cancel_btn"]
        ]
    )
    
    ui_components["clear_cancel_btn"].click(
        fn=clear_history_cancel,
        outputs=[
            ui_components["gallery"], 
            ui_components["clear_btn"],
            ui_components["clear_confirm_btn"],
            ui_components["clear_cancel_btn"]
        ]
    )
    
    ui_components["gallery"].select(
        fn=on_gallery_select,
        outputs=[
            ui_components["selected_input_image"],
            ui_components["selected_output_image"],
            ui_components["selected_input_mask"],
            ui_components["selected_oneframe_mc"],
            ui_components["selected_optional_control"],
            ui_components["selected_reference_mask"],
            ui_components["selected_kisekaeichi"],
            ui_components["selected_control_positions"],
            ui_components["selected_prompt"],
            ui_components["selected_params_col1"],
            ui_components["selected_params_col2"],
            ui_components["selected_params_col3"],
            ui_components["selected_params_col4"],
            ui_components["selected_timestamp"],
            ui_components["restore_btn"],      # パラメータ復元ボタン
            ui_components["restore_lora_btn"]  # LoRA復元ボタン
        ]
    )
    
    # パラメータ復元システム
    safe_outputs = []
    safe_output_keys = []
    
    # 基本パラメータ（21個）- param_keys順
    param_keys = [
        "seed", "steps", "gs", "resolution",
        "latent_window_size", "latent_index", "clean_index",
        "use_clean_latents_2x", "use_clean_latents_4x", "use_clean_latents_post",
        "use_teacache", "gpu_memory_preservation", "fp8_optimization",
        "use_lora", "use_advanced_control", "advanced_control_mode", "use_random_seed", "prompt",
        "kisekaeichi_control_index", "oneframe_mc_control_index", "optional_control_index"
    ]
    
    for key in param_keys:
        component = main_ui_components.get(key)
        if component is not None:
            safe_outputs.append(component)
            safe_output_keys.append(key)
    
    # 画像コンポーネント（6個）- image_keys順
    image_keys = [
        "input_image", "kisekaeichi_reference_image", "oneframe_mc_image",
        "optional_control_image", "input_mask", "reference_mask"
    ]
    
    for key in image_keys:
        component = main_ui_components.get(key)
        if component is not None:
            safe_outputs.append(component)
            safe_output_keys.append(key)
    
    # LoRAコンポーネント
    lora_mode = main_ui_components.get("lora_mode")
    if lora_mode is not None:
        safe_outputs.append(lora_mode)
        safe_output_keys.append("lora_mode")
    
    # LoRAドロップダウン（存在するもののみ）
    for i in range(20):
        component = main_ui_components.get(f"lora_dropdown_{i}")
        if component is not None:
            safe_outputs.append(component)
            safe_output_keys.append(f"lora_dropdown_{i}")
    
    # LoRA強度（存在するもののみ）
    for i in range(20):
        component = main_ui_components.get(f"lora_strength_{i}")
        if component is not None:
            safe_outputs.append(component)
            safe_output_keys.append(f"lora_strength_{i}")
    
    # lora_scales_text
    lora_scales_text = main_ui_components.get("lora_scales_text")
    if lora_scales_text is not None:
        safe_outputs.append(lora_scales_text)
        safe_output_keys.append("lora_scales_text")
    
    # lora_dropdown_group（可視性制御のため）
    lora_dropdown_group = main_ui_components.get("lora_dropdown_group")
    if lora_dropdown_group is not None:
        safe_outputs.append(lora_dropdown_group)
        safe_output_keys.append("lora_dropdown_group")
    
    # LoRA復元ボタン（活性化制御のため）
    restore_lora_btn = ui_components.get("restore_lora_btn")
    if restore_lora_btn is not None:
        safe_outputs.append(restore_lora_btn)
        safe_output_keys.append("restore_lora_btn")
    
    # 基本パラメータ専用復元システム（分離設計）
    def restore_basic_parameters(col1_data, col2_data, col3_data, col4_data):
        """基本パラメータと画像のみを復元（LoRA除外）"""
        import gradio as gr
        
        # 選択履歴取得
        history_entry = get_selected_history_entry()
        if not history_entry:
            return [gr.update()] * len(safe_outputs)
        
        # 1. パラメータ復元（LoRA関連を除外）
        param_updates = restore_parameters_enhanced(history_entry)
        
        # 2. 画像復元
        image_updates = restore_images(history_entry)
        
        # 3. safe_outputs の順序に合わせて更新値を準備（LoRA関連は除外）
        updates = []
        
        # LoRA使用履歴確認とボタン活性化制御
        history_entry = get_selected_history_entry()
        lora_info = history_entry.get("lora_info") if history_entry else None
        has_lora_history = lora_info and lora_info.get("used_lora_count", 0) > 0
        
        for key in safe_output_keys:
            if key == "use_lora":
                # use_loraはLoRA履歴に基づいて設定
                if has_lora_history:
                    updates.append(gr.update(value=True))
                else:
                    updates.append(gr.update(value=False))
            elif key == "lora_dropdown_group":
                # LoRA表示制御：履歴がある場合のみ表示
                if has_lora_history:
                    updates.append(gr.update(visible=True))
                else:
                    updates.append(gr.update())
            elif key == "restore_lora_btn":
                # LoRA復元ボタン活性化制御：履歴がある場合のみ活性化
                if has_lora_history:
                    updates.append(gr.update(interactive=True))
                else:
                    updates.append(gr.update(interactive=False))
            elif key.startswith("lora_") or key == "lora_mode":
                # その他のLoRA関連は更新しない（LoRA復元ボタンで処理）
                updates.append(gr.update())
            elif key in param_updates:
                updates.append(param_updates[key])
            elif key in image_updates:
                updates.append(image_updates[key])
            else:
                updates.append(gr.update())
        
        return updates
    
    # LoRA復元システム（プリセット方式）
    def restore_lora_only():
        """LoRA設定のみを復元（プリセット方式で確実な動作）"""
        import gradio as gr
        
        # 選択履歴取得
        history_entry = get_selected_history_entry()
        if not history_entry:
            return [gr.update()] * len(safe_outputs)
        
        # LoRAコンポーネントの数を取得
        local_lora_dropdown_keys = [k for k in safe_output_keys if k.startswith("lora_dropdown_") and k != "lora_dropdown_group"]
        available_lora_count = len(local_lora_dropdown_keys)
        
        # LoRA復元のみ実行（プリセット方式）
        lora_updates = restore_lora_settings(history_entry, main_ui_components, available_lora_count)
        
        # safe_outputs の順序に合わせて更新値を準備（LoRA関連のみ）
        updates = []
        
        for key in safe_output_keys:
            if key in lora_updates:
                updates.append(lora_updates[key])
            else:
                # LoRA以外は更新しない
                updates.append(gr.update())
        
        return updates
    
    # 基本パラメータ復元ボタン（LoRA以外）
    ui_components["restore_btn"].click(
        fn=restore_basic_parameters,
        inputs=[
            ui_components["selected_params_col1"], 
            ui_components["selected_params_col2"], 
            ui_components["selected_params_col3"],
            ui_components["selected_params_col4"]
        ],
        outputs=safe_outputs
    )
    
    # LoRA復元ボタン（プリセット方式）
    ui_components["restore_lora_btn"].click(
        fn=restore_lora_only,
        inputs=[],  # 入力不要（履歴から直接取得）
        outputs=safe_outputs
    )