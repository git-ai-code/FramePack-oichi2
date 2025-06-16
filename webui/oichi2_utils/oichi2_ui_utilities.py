#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oichi2_ui_utilities.py

FramePack-oichi2 UI設定ユーティリティモジュール
- UI表示切替機能
- LoRA設定管理機能
- 着せ替え設定制御機能

UI設定ユーティリティ統合モジュール
"""

import os
import socket

import gradio as gr


def get_translation_function():
    """翻訳関数を取得（循環インポート回避）"""
    import sys
    if hasattr(sys.modules.get('__main__'), 'translate'):
        return sys.modules['__main__'].translate
    # フォールバック: 翻訳が利用できない場合はそのまま返す
    return lambda x: x


def toggle_kisekaeichi_settings(use_reference):
    """
    着せ替え設定の表示/非表示を切り替える関数
    
    Args:
        use_reference (bool): 参照画像使用フラグ
        
    Returns:
        list: gradio更新オブジェクトのリスト
    """
    # インデックスのデフォルト値を設定
    latent_index_value = 1 if use_reference else 5  # 参照画像使用時は1（着せ替え画像）、未使用時は5（1フレーム推論）
    clean_index_value = 13  # 公式推奨値13を常に使用（13-16が推奨範囲）
    
    return [
        gr.update(visible=use_reference),  # reference_image
        gr.update(visible=use_reference),  # advanced_kisekaeichi_group
        gr.update(visible=use_reference),  # reference_image_info
        gr.update(value=latent_index_value),  # latent_index
        gr.update(value=clean_index_value)  # clean_index
    ]


def scan_lora_directory():
    """
    LoRAディレクトリをスキャンして利用可能なファイルリストを取得
    
    Returns:
        list: LoRAファイルの選択肢リスト
    """
    translate = get_translation_function()
    
    # webuiディレクトリのloraフォルダを参照
    webui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lora_dir = os.path.join(webui_dir, 'lora')
    if not os.path.exists(lora_dir):
        os.makedirs(lora_dir, exist_ok=True)
    choices = [f for f in os.listdir(lora_dir) if f.endswith(('.safetensors', '.pt', '.bin'))]
    choices = [translate("なし")] + sorted(choices)
    return choices


def toggle_lora_settings(use_lora):
    """
    チェックボックスの状態によってLoRA設定の表示/非表示を切り替える関数
    
    Args:
        use_lora (bool): LoRA使用フラグ
        
    Returns:
        list: gradio更新オブジェクトのリスト
    """
    translate = get_translation_function()
    
    # グローバル変数を使うように修正
    global previous_lora_mode

    # グローバル変数が定義されていなければ初期化
    if previous_lora_mode is None:
        previous_lora_mode = translate("ディレクトリから選択")

    if use_lora:
        # LoRA使用時は前回のモードを復元、初回は必ずディレクトリから選択
        if previous_lora_mode is None:
            previous_lora_mode = translate("ディレクトリから選択")
            
        is_upload_mode = previous_lora_mode == translate("ファイルアップロード")

        # モードに基づいた表示設定
        preset_visible = not is_upload_mode  # ディレクトリ選択モードの場合のみプリセット表示
        return [
            gr.update(visible=True, value=previous_lora_mode),  # lora_mode - 前回の値を復元
            gr.update(visible=is_upload_mode),  # lora_upload_group
            gr.update(visible=not is_upload_mode),  # lora_dropdown_group
            gr.update(visible=True),  # lora_scales_text
            gr.update(visible=preset_visible),  # lora_preset_group
        ]
    else:
        # LoRA不使用時はLoRA関連UIのみ非表示（FP8最適化は表示したまま）
        return [
            gr.update(visible=False),  # lora_mode
            gr.update(visible=False),  # lora_upload_group
            gr.update(visible=False),  # lora_dropdown_group
            gr.update(visible=False),  # lora_scales_text
            gr.update(visible=False),  # lora_preset_group
        ]


def toggle_lora_mode(mode):
    """
    LoRA読み込み方式に応じて表示を切り替える関数（可変LoRA対応）
    
    Args:
        mode (str): LoRA読み込みモード
        
    Returns:
        list: gradio更新オブジェクトのリスト
    """
    translate = get_translation_function()
    
    # 前回のモードを更新
    global previous_lora_mode
    previous_lora_mode = mode
    
    # 可変LoRA対応: 設定から最大数を取得
    try:
        from common_utils.lora_config import get_max_lora_count
        max_count = get_max_lora_count()
    except ImportError:
        max_count = 3  # フォールバック
    
    if mode == translate("ディレクトリから選択"):
        # ディレクトリから選択モードの場合
        # 最初にディレクトリをスキャン
        choices = scan_lora_directory()
        
        # 選択肢が確実に更新されるようにする
        default_value = choices[0] if choices else translate("なし")
        base_updates = [
            gr.update(visible=False),                                # lora_upload_group
            gr.update(visible=True),                                 # lora_dropdown_group
            gr.update(visible=True),                                 # lora_preset_group
        ]
        # 可変ドロップダウン更新
        dropdown_updates = [gr.update(choices=choices, value=default_value) for _ in range(max_count)]
        return base_updates + dropdown_updates
    else:  # ファイルアップロード
        # ファイルアップロード方式の場合、ドロップダウンの値は更新しない
        base_updates = [
            gr.update(visible=True),   # lora_upload_group
            gr.update(visible=False),  # lora_dropdown_group
            gr.update(visible=False),  # lora_preset_group
        ]
        # 可変ドロップダウン更新（変更なし）
        dropdown_updates = [gr.update() for _ in range(max_count)]
        return base_updates + dropdown_updates


def update_lora_dropdowns():
    """
    スキャンボタンの処理関数（可変LoRA対応）
    
    Returns:
        list: gradio更新オブジェクトのリスト
    """
    translate = get_translation_function()
    choices = scan_lora_directory()
    default_value = choices[0] if choices else translate("なし")
    
    # 可変LoRA対応: 設定から最大数を取得
    try:
        from common_utils.lora_config import get_max_lora_count
        max_count = get_max_lora_count()
    except ImportError:
        max_count = 3  # フォールバック
    
    # 動的に更新リストを生成
    return [gr.update(choices=choices, value=default_value) for _ in range(max_count)]


def toggle_lora_full_update(use_lora_val):
    """
    LoRA使用チェックボックスの切り替え後にドロップダウンを更新する統合関数（可変LoRA対応）
    
    Args:
        use_lora_val (bool): LoRA使用フラグ
        
    Returns:
        list: gradio更新オブジェクトのリスト
    """
    translate = get_translation_function()
    
    global previous_lora_mode
    
    # グローバル変数が定義されていなければ初期化
    if previous_lora_mode is None:
        previous_lora_mode = translate("ディレクトリから選択")
    
    # 可変LoRA対応: 設定から最大数を取得
    try:
        from common_utils.lora_config import get_max_lora_count
        max_count = get_max_lora_count()
    except ImportError:
        max_count = 3  # フォールバック
    
    # まずLoRA設定全体の表示/非表示を切り替え
    mode_updates = toggle_lora_settings(use_lora_val)
    
    # LoRAが有効な場合は必ずドロップダウンを更新（初回の場合を含む）
    if use_lora_val:
        # ディレクトリ選択モード時または初回時にドロップダウンを更新
        if previous_lora_mode == translate("ディレクトリから選択") or previous_lora_mode is None:
            choices = scan_lora_directory()
            default_value = choices[0] if choices else translate("なし")
            # 可変ドロップダウン更新
            dropdown_updates = [gr.update(choices=choices, value=default_value) for _ in range(max_count)]
            return mode_updates + dropdown_updates
    
    # それ以外の場合は変更なし
    return mode_updates + [gr.update() for _ in range(max_count)]


# グローバル変数の初期化
previous_lora_mode = None
current_lora_mode = None


def save_lora_count_setting_handler(count_value):
    """LoRA数設定をプリセットファイルに保存"""
    import gradio as gr
    
    try:
        from common_utils.lora_config import get_lora_settings
        settings = get_lora_settings()
        if settings.set_max_count(int(count_value)):
            return gr.update(value=f"✅ LoRA表示数を{int(count_value)}個に設定しました（再起動後反映）")
        else:
            return gr.update(value="❌ 設定保存に失敗しました（1-20の範囲で設定してください）")
    except Exception as e:
        return gr.update(value=f"❌ エラーが発生しました: {str(e)}")


def sync_metadata_checkboxes(checkbox_value):
    """メタデータ複写チェックボックスの同期処理"""
    return checkbox_value


def toggle_advanced_control(use_control):
    """高度画像制御の表示切替とモード自動設定"""
    import gradio as gr
    
    if use_control:
        # 高度制御ON: UI表示
        return [
            gr.update(visible=True),   # advanced_control_group
            gr.update(visible=True),   # advanced_control_mode
            gr.update(visible=True)    # mode_info_group
        ]
    else:
        # 高度制御OFF: UI非表示・one_frame自動設定
        return [
            gr.update(visible=False),        # advanced_control_group
            gr.update(visible=False, value="one_frame"),  # advanced_control_mode
            gr.update(visible=False)         # mode_info_group
        ]


def toggle_advanced_control_mode(control_mode):
    """制御モード別のUI更新処理"""
    import gradio as gr
    
    # モード別推奨値と説明文設定
    if control_mode == "one_frame":
        mode_desc = "**1フレーム推論**: 基本的な画像生成（推奨: latent_index=5） 💡 内部処理: 入力画像のみ使用、制御画像は処理で無視"
        latent_value = 5
    elif control_mode == "kisekaeichi":
        mode_desc = "**kisekaeichi**: 着せ替えに最適化（推奨: latent_index=1, control_index=0;10） 💡 内部処理: 着せ替え制御画像およびマスクのみ使用"
        latent_value = 1
    elif control_mode == "1fmc":
        mode_desc = "**1f-mc**: 画像ブレンドに最適化（推奨: latent_index=9, control_index=0;1） 💡 内部処理: 人物制御画像・追加制御画像のみ使用"
        latent_value = 9
    elif control_mode == "custom":
        mode_desc = "**カスタム**: 全ての制御画像が使用されます。自由にお試し下さい 💡 内部処理: 設定された全ての制御画像・マスクを使用"
        latent_value = 5
    else:
        mode_desc = "制御モードを選択してください"
        latent_value = 0
    
    return [
        gr.update(value=latent_value, interactive=True),  # latent_index
        gr.update(value=mode_desc)  # mode_description
    ]


def update_scales_text(*strength_values):
    """個別強度からカンマ区切り文字列を生成"""
    try:
        scales_str = ",".join([str(float(val) if val is not None else 0.8) for val in strength_values])
        return gr.update(value=scales_str)
    except:
        return gr.update()


def is_port_in_use(port):
    """指定ポートが使用中かどうかを確認"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    except:
        return False