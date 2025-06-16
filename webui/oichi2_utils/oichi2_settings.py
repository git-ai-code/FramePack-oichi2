"""
FramePack-oichi2 設定管理・メタデータ処理統合モジュール

設定管理・メタデータ処理の分離・最適化
- フォルダ設定・管理・開く処理
- 画像メタデータ抽出・プロンプト反映
- チェックボックス連動・UIアップデート
- 設定保存・読み込み・グローバル変数管理
"""

import os
import gradio as gr
from locales.i18n_extended import translate
from common_utils.settings_manager import (
    get_output_folder_path, load_settings, save_settings, open_output_folder
)
from common_utils.png_metadata import (
    extract_metadata_from_png, PROMPT_KEY, SEED_KEY
)


def handle_open_folder_btn(folder_name):
    """
    フォルダ名を保存し、そのフォルダを開く処理
    
    Args:
        folder_name: 出力フォルダ名
        
    Returns:
        tuple: (フォルダ名更新データ, フォルダパス更新データ)
    """
    if not folder_name or not folder_name.strip():
        folder_name = "outputs"

    # フォルダパスを取得
    folder_path = get_output_folder_path(folder_name)

    # 設定を更新して保存
    settings = load_settings()
    old_folder_name = settings.get('output_folder')

    if old_folder_name != folder_name:
        settings['output_folder'] = folder_name
        save_result = save_settings(settings)
        if save_result:
            # グローバル変数も更新
            _update_global_folder_variables(folder_name, folder_path)
        print(translate("出力フォルダ設定を保存しました: {folder_name}").format(folder_name=folder_name))

    # フォルダを開く
    open_output_folder(folder_path)

    # 出力ディレクトリ入力欄とパス表示を更新
    return gr.update(value=folder_name), gr.update(value=folder_path)


def _update_global_folder_variables(folder_name, folder_path):
    """
    グローバル変数の更新処理
    
    Args:
        folder_name: フォルダ名
        folder_path: フォルダパス
    """
    # グローバル変数を更新（呼び出し元でimportされたモジュールから参照）
    # 注意: この実装では直接グローバル変数にアクセスできないため、
    # 呼び出し元で適切に処理する必要がある
    pass


def update_from_image_metadata(image_path, should_copy):
    """
    画像からメタデータを抽出してプロンプトとシードを更新する処理
    
    Args:
        image_path: 画像ファイルパス
        should_copy: メタデータを複写するかどうかの指定
        
    Returns:
        tuple: (プロンプト更新データ, シード値更新データ)
    """
    if not should_copy or image_path is None:
        return gr.update(), gr.update()
    
    try:
        # ファイルパスからメタデータを抽出
        metadata = extract_metadata_from_png(image_path)
        
        if not metadata:
            print(translate("画像にメタデータが含まれていません"))
            return gr.update(), gr.update()
        
        # プロンプトとSEEDをUIに反映
        prompt_update = gr.update()
        seed_update = gr.update()
        
        if PROMPT_KEY in metadata and metadata[PROMPT_KEY]:
            prompt_update = gr.update(value=metadata[PROMPT_KEY])
            print(translate("プロンプトを画像から取得: {0}").format(metadata[PROMPT_KEY]))
        
        if SEED_KEY in metadata and metadata[SEED_KEY]:
            # SEED値を整数に変換
            try:
                seed_value = int(metadata[SEED_KEY])
                seed_update = gr.update(value=seed_value)
                print(translate("シード値を画像から取得: {0}").format(seed_value))
            except ValueError:
                print(translate("シード値の変換に失敗しました: {0}").format(metadata[SEED_KEY]))
        
        return prompt_update, seed_update
    
    except Exception as e:
        print(translate("メタデータ抽出エラー: {0}").format(e))
        return gr.update(), gr.update()


def check_metadata_on_checkbox_change(should_copy, image_path):
    """
    チェックボックスの状態が変更された時に画像からメタデータを抽出する処理
    
    Args:
        should_copy: メタデータを複写するかどうかの指定
        image_path: 画像ファイルパス
        
    Returns:
        tuple: (プロンプト更新データ, シード値更新データ)
    """
    return update_from_image_metadata(image_path, should_copy)


def get_global_folder_update_callback():
    """
    グローバル変数更新用のコールバック関数を返す
    
    Returns:
        function: グローバル変数更新用関数
    """
    def update_callback(folder_name, folder_path):
        """
        呼び出し元でグローバル変数を更新するためのコールバック
        
        Args:
            folder_name: フォルダ名
            folder_path: フォルダパス
        """
        # 呼び出し元のスコープでグローバル変数を更新
        # この関数は呼び出し元で適切に実装される
        pass
    
    return update_callback