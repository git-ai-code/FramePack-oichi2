"""
設定ファイル管理モジュール

endframe_ichi.pyから外出しした設定ファイル関連処理を含み、
アプリケーション設定の読み込み・保存・管理機能を提供します。
"""

import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Union

from locales.i18n_extended import translate

def get_settings_file_path() -> str:
    """
    設定ファイルの絶対パスを取得します。
    
    Returns:
        str: 設定ファイルの絶対パス
    """
    # common_utils直下からwebuiフォルダに移動
    webui_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    settings_folder: str = os.path.join(webui_path, 'settings')
    return os.path.join(settings_folder, 'app_settings.json')

def get_output_folder_path(folder_name: Optional[str] = None) -> str:
    """
    出力フォルダの絶対パスを取得します。
    
    Args:
        folder_name: フォルダ名。Noneまたは空文字列の場合は"outputs"を使用
        
    Returns:
        str: 出力フォルダの絶対パス
    """
    # common_utils直下からwebuiフォルダに移動
    webui_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not folder_name or not folder_name.strip():
        folder_name = "outputs"
    return os.path.join(webui_path, folder_name)

def initialize_settings() -> bool:
    """
    設定ファイルを初期化します（存在しない場合のみ）。
    
    Returns:
        bool: 初期化に成功した場合True
    """
    settings_file: str = get_settings_file_path()
    settings_dir: str = os.path.dirname(settings_file)

    if not os.path.exists(settings_file):
        # 初期デフォルト設定（アプリケーション設定を含む）
        default_settings: Dict[str, Any] = {
            'output_folder': 'outputs',
            'app_settings_eichi': get_default_app_settings(),
            'log_settings': {'log_enabled': False, 'log_folder': 'logs'},
            'hf_settings': get_default_hf_settings()
        }
        try:
            os.makedirs(settings_dir, exist_ok=True)
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(default_settings, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(translate("設定ファイル初期化エラー: {0}").format(e))
            return False
    return True

def load_settings() -> Dict[str, Any]:
    """
    設定ファイルから設定データを読み込みます。
    
    Returns:
        Dict[str, Any]: 読み込んだ設定データ
    """
    settings_file = get_settings_file_path()
    default_settings = {'output_folder': 'outputs'}

    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
                if not file_content.strip():
                    return default_settings
                settings = json.loads(file_content)

                # デフォルト値とマージ
                for key, value in default_settings.items():
                    if key not in settings:
                        settings[key] = value
                return settings
        except Exception as e:
            print(translate("設定読み込みエラー: {0}").format(e))

    return default_settings

def save_settings(settings: Dict[str, Any]) -> bool:
    """
    設定データをファイルに保存します。
    
    Args:
        settings: 保存する設定データ
        
    Returns:
        bool: 保存に成功した場合True
    """
    settings_file = get_settings_file_path()

    try:
        # 保存前にディレクトリが存在するか確認
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)

        # JSON書き込み
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(translate("設定保存エラー: {0}").format(e))
        return False

def open_output_folder(folder_path: str) -> bool:
    """
    指定されたフォルダをOSに依存せず開きます。
    
    Args:
        folder_path: 開くフォルダのパス
        
    Returns:
        bool: フォルダを開くことに成功した場合True
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen(['explorer', folder_path])
        elif os.name == 'posix':  # Linux/Mac
            try:
                subprocess.Popen(['xdg-open', folder_path])
            except:
                subprocess.Popen(['open', folder_path])
        print(translate("フォルダを開きました: {0}").format(folder_path))
        return True
    except Exception as e:
        print(translate("フォルダを開く際にエラーが発生しました: {0}").format(e))
        return False

def get_localized_default_value(key: str, current_lang: str = "ja") -> Optional[str]:
    """
    言語に応じたデフォルト値を返します。
    
    Args:
        key: 設定キー
        current_lang: 現在の言語コード
        
    Returns:
        Optional[str]: 適切な言語に翻訳されたデフォルト値
    """
    # 特別な翻訳が必要な値のマッピング
    localized_values = {
        "frame_save_mode": {
            "ja": "保存しない",
            "en": "Do not save",
            "zh-tw": "不保存",
            "ru": "Не сохранять"
        }
        # 必要に応じて他の値も追加可能
    }
    
    # キーが存在するか確認し、言語に応じた値を返す
    if key in localized_values:
        # 指定された言語が存在しない場合はjaをデフォルトとして使用
        return localized_values[key].get(current_lang, localized_values[key]["ja"])
    
    # 特別な翻訳が必要ない場合は None を返す
    return None

def get_default_app_settings(current_lang: str = "ja") -> Dict[str, Any]:
    """
    eichiのデフォルト設定を返します。
    
    Args:
        current_lang: 現在の言語コード
        
    Returns:
        Dict[str, Any]: eichiのデフォルト設定辞書
    """
    # フレーム保存モードのデフォルト値を言語に応じて設定
    frame_save_mode_default = get_localized_default_value("frame_save_mode", current_lang)
    
    return {
        # 基本設定
        "resolution": 640,
        "mp4_crf": 16,
        "steps": 25,
        "cfg": 1.0,
        
        # パフォーマンス設定
        "use_teacache": True,
        "gpu_memory_preservation": 6,
        "use_vae_cache": False,
        
        # 詳細設定
        "gs": 10.0,  # Distilled CFG Scale
        
        # パディング設定
        "use_all_padding": False,
        "all_padding_value": 1.0,
        
        # エンドフレーム設定
        "end_frame_strength": 1.0,
        
        # 保存設定
        "keep_section_videos": False,
        "save_section_frames": False,
        "save_tensor_data": False,
        "frame_save_mode": frame_save_mode_default if frame_save_mode_default else "保存しない",
        
        # 自動保存設定
        "save_settings_on_start": False,
        
        # アラーム設定
        "alarm_on_completion": True
    }

def get_default_app_settings_f1(current_lang: str = "ja") -> Dict[str, Any]:
    """
    F1のデフォルト設定を返します。
    
    Args:
        current_lang: 現在の言語コード
        
    Returns:
        Dict[str, Any]: F1のデフォルト設定辞書
    """
    # フレーム保存モードのデフォルト値を言語に応じて設定
    frame_save_mode_default = get_localized_default_value("frame_save_mode", current_lang)
    
    return {
        # 基本設定
        "resolution": 640,
        "mp4_crf": 16,
        "steps": 25,
        "cfg": 1,
        
        # パフォーマンス設定
        "use_teacache": True,
        "gpu_memory_preservation": 6,
        
        # 詳細設定
        "gs": 10,
        
        # F1独自設定
        "image_strength": 1.0,
        
        # 保存設定
        "keep_section_videos": False,
        "save_section_frames": False,
        "save_tensor_data": False,
        "frame_save_mode": frame_save_mode_default if frame_save_mode_default else "保存しない",
        
        # 自動保存・アラーム設定
        "save_settings_on_start": False,
        "alarm_on_completion": True,
        
        # CONFIG QUEUE設定 - NEW SECTION
        "add_timestamp_to_config": True  # Default to True to maintain current behavior
    }

def get_default_app_settings_oichi() -> Dict[str, Any]:
    """
    oichiのデフォルト設定を返します。
    
    Returns:
        Dict[str, Any]: oichiのデフォルト設定辞書
    """
    return {
        # 基本設定
        "resolution": 640,
        "steps": 25,
        "cfg": 1,
        
        # パフォーマンス設定
        "use_teacache": True,
        "gpu_memory_preservation": 6,
        
        # 詳細設定
        "gs": 10.0,
        
        # oneframe固有設定
        "latent_window_size": 9,
        "use_clean_latents_2x": True,
        "use_clean_latents_4x": True,
        "use_clean_latents_post": True,
        
        # インデックス設定
        "latent_index": 5,
        "clean_index": 13,
        
        # LoRA設定
        "use_lora": False,
        "lora_mode": "ディレクトリから選択",
        
        # 最適化設定
        "fp8_optimization": True,
        
        # バッチ設定
        "batch_count": 1,
        
        # RoPE設定
        "use_rope_batch": False,
        
        # キュー設定
        "use_queue": False,
        
        # 自動保存・アラーム設定
        "save_settings_on_start": False,
        "alarm_on_completion": True
    }

def load_app_settings() -> Dict[str, Any]:
    """
    eichiのアプリケーション設定を読み込みます。
    
    Returns:
        Dict[str, Any]: eichiのアプリケーション設定
    """
    settings = load_settings()
    # 旧キーからの移行処理
    if 'app_settings' in settings and 'app_settings_eichi' not in settings:
        settings['app_settings_eichi'] = settings['app_settings']
        del settings['app_settings']
        save_settings(settings)
    elif 'app_settings_eichi' not in settings:
        settings['app_settings_eichi'] = get_default_app_settings()
        save_settings(settings)
    
    # 既存の設定にデフォルト値をマージ（新しいキーのため）
    app_settings = settings.get('app_settings_eichi', {})
    default_settings = get_default_app_settings()
    
    # 存在しないキーにはデフォルト値を使用
    for key, default_value in default_settings.items():
        if key not in app_settings:
            app_settings[key] = default_value
            print(translate("新しい設定項目 '{0}' をデフォルト値 {1} で追加").format(key, default_value))
    
    # マージした設定を保存
    if app_settings != settings.get('app_settings_eichi', {}):
        settings['app_settings_eichi'] = app_settings
        save_settings(settings)
    
    return app_settings

def save_app_settings(app_settings: Dict[str, Any]) -> bool:
    """
    eichiのアプリケーション設定を保存します。
    
    Args:
        app_settings: 保存するeichiのアプリケーション設定
        
    Returns:
        bool: 保存に成功した場合True
    """
    settings = load_settings()
    
    # 不要なキーを除外してコピー（手動保存と自動保存の一貫性のため）
    filtered_settings = {k: v for k, v in app_settings.items() 
                        if k not in ['rs', 'output_dir', 'frame_size_radio']}
    
    settings['app_settings_eichi'] = filtered_settings
    return save_settings(settings)

def load_app_settings_f1() -> Dict[str, Any]:
    """
    F1のアプリケーション設定を読み込みます。
    
    Returns:
        Dict[str, Any]: F1のアプリケーション設定
    """
    settings = load_settings()
    
    # F1の設定が存在しない場合はデフォルト値を設定
    if 'app_settings_f1' not in settings:
        settings['app_settings_f1'] = get_default_app_settings_f1()
        save_settings(settings)
    
    # 既存の設定にデフォルト値をマージ（新しいキーのため）
    app_settings = settings.get('app_settings_f1', {})
    default_settings = get_default_app_settings_f1()
    
    # 存在しないキーにはデフォルト値を使用
    for key, default_value in default_settings.items():
        if key not in app_settings:
            app_settings[key] = default_value
            print(translate("F1: 新しい設定項目 '{0}' をデフォルト値 {1} で追加").format(key, default_value))
    
    # マージした設定を保存
    if app_settings != settings.get('app_settings_f1', {}):
        settings['app_settings_f1'] = app_settings
        save_settings(settings)
    
    return app_settings

def save_app_settings_f1(app_settings: Dict[str, Any]) -> bool:
    """
    F1のアプリケーション設定を保存します。
    
    Args:
        app_settings: 保存するF1のアプリケーション設定
        
    Returns:
        bool: 保存に成功した場合True
    """
    settings = load_settings()
    
    # 保存すべきキーのみを含める（許可リスト方式）
    allowed_keys = [
        'resolution', 'mp4_crf', 'steps', 'cfg', 'use_teacache',
        'gpu_memory_preservation', 'gs', 'image_strength',
        'keep_section_videos', 'save_section_frames', 'save_tensor_data',
        'frame_save_mode', 'save_settings_on_start', 'alarm_on_completion',
        'add_timestamp_to_config'  # ADD THIS NEW KEY
    ]
    
    filtered_settings = {k: v for k, v in app_settings.items() 
                        if k in allowed_keys}
    
    settings['app_settings_f1'] = filtered_settings
    return save_settings(settings)

def load_app_settings_oichi() -> Dict[str, Any]:
    """
    oichiのアプリケーション設定を読み込みます。
    
    Returns:
        Dict[str, Any]: oichiのアプリケーション設定
    """
    settings = load_settings()
    
    # oichiの設定が存在しない場合はデフォルト値を設定
    if 'app_settings_oichi' not in settings:
        settings['app_settings_oichi'] = get_default_app_settings_oichi()
        save_settings(settings)
    
    # 既存の設定にデフォルト値をマージ（新しいキーのため）
    app_settings = settings.get('app_settings_oichi', {})
    default_settings = get_default_app_settings_oichi()
    
    # 存在しないキーにはデフォルト値を使用
    for key, default_value in default_settings.items():
        if key not in app_settings:
            app_settings[key] = default_value
    
    # マージした設定を保存
    if app_settings != settings.get('app_settings_oichi', {}):
        settings['app_settings_oichi'] = app_settings
        save_settings(settings)
    
    return app_settings

def save_app_settings_oichi(app_settings: Dict[str, Any]) -> bool:
    """
    oichiのアプリケーション設定を保存します。
    
    Args:
        app_settings: 保存するoichiのアプリケーション設定
        
    Returns:
        bool: 保存に成功した場合True
    """
    settings = load_settings()
    
    # 不要なキーを除外してコピー
    filtered_settings = {k: v for k, v in app_settings.items() 
                        if k not in ['rs', 'output_dir']}
    
    settings['app_settings_oichi'] = filtered_settings
    return save_settings(settings)

def get_default_hf_settings() -> Dict[str, Any]:
    """
    HuggingFace設定のデフォルト値を返します。
    
    Returns:
        Dict[str, Any]: HuggingFace設定のデフォルト辞書
    """
    return {
        'shared_model_paths': [],  # 共有モデルパスのリスト（優先順）
        'local_model_path': 'hf_download',  # ローカルモデルパス
        'prompt_for_path': True,  # パス入力プロンプト表示
    }

def load_hf_settings() -> Dict[str, Any]:
    """
    HuggingFace設定を読み込みます。
    
    Returns:
        Dict[str, Any]: HuggingFace設定
    """
    settings: Dict[str, Any] = load_settings()
    
    # HF設定が存在しない場合はデフォルト値を設定
    if 'hf_settings' not in settings:
        settings['hf_settings'] = get_default_hf_settings()
        save_settings(settings)
    
    # 既存の設定にデフォルト値をマージ（新しいキーのため）
    hf_settings: Dict[str, Any] = settings.get('hf_settings', {})
    default_settings: Dict[str, Any] = get_default_hf_settings()
    
    # 存在しないキーにはデフォルト値を使用
    for key, default_value in default_settings.items():
        if key not in hf_settings:
            hf_settings[key] = default_value
            print(translate("新しいHF設定項目 '{0}' をデフォルト値で追加").format(key))
    
    # マージした設定を保存
    if hf_settings != settings.get('hf_settings', {}):
        settings['hf_settings'] = hf_settings
        save_settings(settings)
    
    return hf_settings

def save_hf_settings(hf_settings: Dict[str, Any]) -> bool:
    """
    HuggingFace設定を保存します。
    
    Args:
        hf_settings: 保存するHuggingFace設定
        
    Returns:
        bool: 保存に成功した場合True
    """
    settings: Dict[str, Any] = load_settings()
    settings['hf_settings'] = hf_settings
    return save_settings(settings)

def add_shared_model_path(new_path: str) -> bool:
    """
    共有モデルパスを追加します。
    
    Args:
        new_path: 追加する共有モデルパス
        
    Returns:
        bool: 追加に成功した場合True
    """
    hf_settings: Dict[str, Any] = load_hf_settings()
    shared_paths: List[str] = hf_settings.get('shared_model_paths', [])
    
    # 既に存在する場合は先頭に移動
    if new_path in shared_paths:
        shared_paths.remove(new_path)
    
    # 先頭に追加（最優先）
    shared_paths.insert(0, new_path)
    
    # 最大5個まで保持
    hf_settings['shared_model_paths'] = shared_paths[:5]
    
    return save_hf_settings(hf_settings)
