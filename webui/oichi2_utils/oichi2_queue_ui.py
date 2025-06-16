"""
FramePack-oichi2 キューUI制御統合モジュール

キュー機能UI制御・切替処理の分離・最適化
- キュー設定トグル処理・グローバル変数管理
- キュータイプ切替ハンドラー・UI表示制御
- プロンプトキュー・イメージキューの表示切替
- イベントハンドラー登録・UI同期処理
- 画像ファイルリスト更新・コールバック統合
"""

import gradio as gr
from locales.i18n_extended import translate


def create_toggle_queue_settings_handler(get_image_queue_files_callback):
    """
    キュー機能のトグルハンドラーを作成
    
    Args:
        get_image_queue_files_callback: 画像キューファイル取得コールバック
        
    Returns:
        function: キュー設定トグルハンドラー関数
    """
    def toggle_queue_settings(use_queue_val, current_queue_type):
        # 引数で現在のキュータイプを受け取る

        # チェックボックスの値をブール値に確実に変換
        is_enabled = False

        # Gradioオブジェクトの場合
        if hasattr(use_queue_val, 'value'):
            is_enabled = bool(use_queue_val.value)
        else:
            # 直接値の場合
            is_enabled = bool(use_queue_val)

        if is_enabled:
            # キューが有効の場合
            # キュータイプセレクタとキュータイプに応じたグループを表示
            if current_queue_type == "prompt":
                return [
                    gr.update(visible=True),  # queue_type_selector
                    gr.update(visible=True),  # prompt_queue_group
                    gr.update(visible=False)  # image_queue_group
                ]
            else:  # image
                # イメージキュー選択時は画像ファイルリストを更新
                if get_image_queue_files_callback:
                    get_image_queue_files_callback()
                return [
                    gr.update(visible=True),  # queue_type_selector
                    gr.update(visible=False),  # prompt_queue_group
                    gr.update(visible=True)   # image_queue_group
                ]
        else:
            # キューが無効の場合、すべて非表示
            return [
                gr.update(visible=False),  # queue_type_selector
                gr.update(visible=False),  # prompt_queue_group
                gr.update(visible=False)   # image_queue_group
            ]
    
    return toggle_queue_settings


def create_toggle_queue_type_handler(get_image_queue_files_callback):
    """
    キュータイプ切替ハンドラーを作成
    
    Args:
        get_image_queue_files_callback: 画像キューファイル取得コールバック
        
    Returns:
        function: キュータイプ切替ハンドラー関数
    """
    def toggle_queue_type(queue_type_val):
        # キュータイプを判定（戻り値でメインモジュールに伝達）
        if queue_type_val == translate("プロンプトキュー"):
            return [gr.update(visible=True), gr.update(visible=False)]
        else:
            # イメージキューを選択した場合、画像ファイルリストを更新
            if get_image_queue_files_callback:
                get_image_queue_files_callback()
            return [gr.update(visible=False), gr.update(visible=True)]
    
    return toggle_queue_type
