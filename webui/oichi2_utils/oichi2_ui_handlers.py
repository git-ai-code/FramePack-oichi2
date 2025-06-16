"""
FramePack-oichi2 UI・ハンドラー統合モジュール

UI関連処理・イベントハンドラーの分離・最適化
- 設定保存・読み込みハンドラー
- 設定リセット・デフォルト値復元処理
- シード管理・ランダム化処理
- プリセット保存・読み込み・削除ハンドラー
- UI更新データ生成・グローバル変数連携
"""

import random

import gradio as gr

from common_utils.log_manager import apply_log_settings, disable_logging
from common_utils.preset_manager import delete_preset, save_preset
from common_utils.settings_manager import (
    get_default_app_settings_oichi,
    load_settings,
    save_settings,
)
from locales.i18n_extended import translate


def create_save_app_settings_handler(save_app_settings_oichi):
    """
    設定保存ハンドラー関数を生成
    
    Args:
        save_app_settings_oichi: アプリ設定保存関数
        
    Returns:
        function: 設定保存ハンドラー関数
    """
    def save_app_settings_handler(
        # 保存対象の設定項目
        resolution_val,
        steps_val,
        cfg_val,
        use_teacache_val,
        gpu_memory_preservation_val,
        gs_val,
        latent_window_size_val,
        latent_index_val,
        use_clean_latents_2x_val,
        use_clean_latents_4x_val,
        use_clean_latents_post_val,
        clean_index_val,
        save_settings_on_start_val,
        alarm_on_completion_val,
        # ログ設定項目
        log_enabled_val,
        log_folder_val
    ):
        """現在の設定を保存"""
        current_settings = {
            'resolution': resolution_val,
            'steps': steps_val,
            'cfg': cfg_val,
            'use_teacache': use_teacache_val,
            'gpu_memory_preservation': gpu_memory_preservation_val,
            'gs': gs_val,
            'latent_window_size': latent_window_size_val,
            'latent_index': latent_index_val,
            'use_clean_latents_2x': use_clean_latents_2x_val,
            'use_clean_latents_4x': use_clean_latents_4x_val,
            'use_clean_latents_post': use_clean_latents_post_val,
            'clean_index': clean_index_val,
            'save_settings_on_start': save_settings_on_start_val,
            'alarm_on_completion': alarm_on_completion_val
        }
        
        # アプリ設定を保存
        try:
            app_success = save_app_settings_oichi(current_settings)
        except Exception as e:
            print(translate("アプリ設定保存エラー: {0}").format(e))
            app_success = False
        
        # ログ設定を保存
        log_settings = {
            "log_enabled": log_enabled_val,
            "log_folder": log_folder_val
        }
        
        try:
            # 設定ファイルを更新
            all_settings = load_settings()
            all_settings['log_settings'] = log_settings
            log_success = save_settings(all_settings)
            
            if log_success:
                # ログ設定を適用
                # 一旦ログを無効化
                disable_logging()
                # 新しい設定でログを再開（有効な場合）
                apply_log_settings(log_settings, source_name="oneframe_ichi")
                print(translate("ログ設定を更新しました: 有効={0}, フォルダ={1}").format(
                    log_enabled_val, log_folder_val))
        except Exception as e:
            print(translate("ログ設定保存エラー: {0}").format(e))
            log_success = False
        
        if app_success and log_success:
            return translate("設定を保存しました")
        else:
            return translate("設定の一部保存に失敗しました")
    
    return save_app_settings_handler


def create_reset_app_settings_handler():
    """
    設定リセットハンドラー関数を生成
    
    Returns:
        function: 設定リセットハンドラー関数
    """
    def reset_app_settings_handler():
        """設定をデフォルトに戻す"""
        default_settings = get_default_app_settings_oichi()
        updates = []
        
        # 各UIコンポーネントのデフォルト値を設定
        updates.append(gr.update(value=default_settings.get("resolution", 640)))  # 1
        updates.append(gr.update(value=default_settings.get("steps", 25)))  # 2
        updates.append(gr.update(value=default_settings.get("cfg", 1)))  # 3
        updates.append(gr.update(value=default_settings.get("use_teacache", True)))  # 4
        updates.append(gr.update(value=default_settings.get("gpu_memory_preservation", 6)))  # 5
        updates.append(gr.update(value=default_settings.get("gs", 10)))  # 6
        updates.append(gr.update(value=default_settings.get("latent_window_size", 9)))  # 7
        updates.append(gr.update(value=default_settings.get("latent_index", 5)))  # 8
        updates.append(gr.update(value=default_settings.get("use_clean_latents_2x", True)))  # 9
        updates.append(gr.update(value=default_settings.get("use_clean_latents_4x", True)))  # 10
        updates.append(gr.update(value=default_settings.get("use_clean_latents_post", True)))  # 11
        updates.append(gr.update(value=default_settings.get("clean_index", 13)))  # 12
        updates.append(gr.update(value=default_settings.get("save_settings_on_start", False)))  # 14
        updates.append(gr.update(value=default_settings.get("alarm_on_completion", True)))  # 15
        
        # ログ設定 (16番目と17番目の要素)
        # ログ設定は固定値を使用 - 絶対に文字列とbooleanを使用
        updates.append(gr.update(value=False))  # log_enabled (16)
        updates.append(gr.update(value="logs"))  # log_folder (17)
        
        # ログ設定をアプリケーションに適用
        default_log_settings = {
            "log_enabled": False,
            "log_folder": "logs"
        }
        
        # 設定ファイルを更新
        all_settings = load_settings()
        all_settings['log_settings'] = default_log_settings
        save_settings(all_settings)
        
        # ログ設定を適用 (既存のログファイルを閉じて、設定に従って再設定)
        disable_logging()  # 既存のログを閉じる
        
        # 設定状態メッセージ (18番目の要素)
        updates.append(translate("設定をデフォルトに戻しました"))
        
        return updates
    
    return reset_app_settings_handler


def set_random_seed(is_checked):
    """
    ランダムシード設定処理
    
    Args:
        is_checked: ランダムシード使用フラグ
        
    Returns:
        int or gr.update: 新しいシード値またはUI更新なし
    """
    if is_checked:
        return random.randint(0, 2**32 - 1)
    return gr.update()


def randomize_seed_if_needed(use_random, batch_num=1):
    """
    バッチ処理用のシード設定関数
    
    Args:
        use_random: ランダムシード使用フラグ
        batch_num: バッチ処理回数
        
    Returns:
        int or gr.update: 新しいシード値またはUI更新なし
    """
    if use_random:
        # ランダムシードの場合はバッチごとに異なるシードを生成
        random_seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_num)]
        return random_seeds[0]  # 最初のシードを返す（表示用）
    return gr.update()  # ランダムシードでない場合は何もしない


def create_save_preset_handler():
    """
    プリセット保存ハンドラー関数を生成
    
    Returns:
        function: プリセット保存ハンドラー関数
    """
    def save_button_click_handler(name, prompt_text):
        """プリセット保存処理"""
        # 重複チェックと正規化
        if "A character" in prompt_text and prompt_text.count("A character") > 1:
            sentences = prompt_text.split(".")
            if len(sentences) > 0:
                prompt_text = sentences[0].strip() + "."
        
        if not name or not name.strip():
            return translate("プリセット名を入力してください"), gr.update(), gr.update()
        
        if not prompt_text or not prompt_text.strip():
            return translate("プロンプトが空です"), gr.update(), gr.update()
        
        try:
            result = save_preset(name, prompt_text)
            if result:
                # プリセットデータを取得してドロップダウンを更新
                from common_utils.preset_manager import load_presets
                presets_data = load_presets()
                choices = [preset["name"] for preset in presets_data["presets"]]
                default_presets = [n for n in choices if any(p["name"] == n and p.get("is_default", False) for p in presets_data["presets"])]
                user_presets = [n for n in choices if n not in default_presets]
                sorted_choices = [(n, n) for n in sorted(default_presets) + sorted(user_presets)]
                
                # メインプロンプトは更新しない（保存のみを行う）
                return translate("プリセット「{0}」を保存しました").format(name), gr.update(choices=sorted_choices), gr.update()
            else:
                return translate("プリセットの保存に失敗しました"), gr.update(), gr.update()
        except Exception as e:
            return translate("プリセット保存エラー: {0}").format(str(e)), gr.update(), gr.update()
    
    return save_button_click_handler


def create_load_preset_handler():
    """
    プリセット読み込みハンドラー関数を生成
    
    Returns:
        function: プリセット読み込みハンドラー関数
    """
    def load_preset_handler(preset_name):
        """プリセット読み込み処理"""
        from common_utils.preset_manager import load_presets
        
        # プリセット名がタプルの場合も処理する
        if isinstance(preset_name, tuple) and len(preset_name) == 2:
            preset_name = preset_name[1]  # 値部分を取得
        
        if not preset_name:
            return gr.update(), gr.update()
        
        # プリセット選択時に編集欄に反映
        presets_data = load_presets()
        for preset in presets_data["presets"]:
            if preset["name"] == preset_name:
                return gr.update(value=preset_name), gr.update(value=preset["prompt"])
        return gr.update(), gr.update()
    
    return load_preset_handler


def create_delete_preset_handler():
    """
    プリセット削除ハンドラー関数を生成
    
    Returns:
        function: プリセット削除ハンドラー関数
    """
    def delete_preset_handler(preset_name):
        """プリセット削除処理"""
        # プリセット名がタプルの場合も処理する
        if isinstance(preset_name, tuple) and len(preset_name) == 2:
            preset_name = preset_name[1]  # 値部分を取得
        
        if not preset_name:
            return translate("削除するプリセット名を選択してください"), gr.update()
        
        try:
            result = delete_preset(preset_name)
            if result:
                # プリセットデータを取得してドロップダウンを更新
                from common_utils.preset_manager import load_presets
                presets_data = load_presets()
                choices = [preset["name"] for preset in presets_data["presets"]]
                default_presets = [name for name in choices if any(p["name"] == name and p.get("is_default", False) for p in presets_data["presets"])]
                user_presets = [name for name in choices if name not in default_presets]
                sorted_names = sorted(default_presets) + sorted(user_presets)
                updated_choices = [(name, name) for name in sorted_names]
                
                return translate("プリセット「{0}」を削除しました").format(preset_name), gr.update(choices=updated_choices)
            else:
                return translate("プリセットの削除に失敗しました"), gr.update()
        except Exception as e:
            return translate("プリセット削除エラー: {0}").format(str(e)), gr.update()
    
    return delete_preset_handler


def clear_fields():
    """
    フィールドクリア処理
    
    Returns:
        tuple: 空の値での更新データ
    """
    return "", ""


def apply_to_prompt(edit_text):
    """
    プロンプトへの適用処理
    
    Args:
        edit_text: 編集テキスト
        
    Returns:
        gr.update: プロンプト更新データ
    """
    return gr.update(value=edit_text)


def update_resolution_info(resolution_value, input_image_path=None):
    """
    解像度詳細情報を更新する関数（クロッピング情報付き）
    
    Args:
        resolution_value: 解像度レベル
        input_image_path: 入力画像パス（Optional）
        
    Returns:
        gr.update: 解像度情報更新データ
    """
    from diffusers_helper.bucket_tools import get_image_resolution_prediction
    
    try:
        # 入力画像の予想サイズを取得
        prediction = get_image_resolution_prediction(input_image_path, resolution_value)
        
        # 入力画像の予想サイズ情報のみを表示
        if prediction['has_image']:
            pred_h, pred_w = prediction['predicted_size']
            orig_w, orig_h = prediction['original_size']
            cropping_info = prediction.get('cropping_info')
            
            # 基本情報
            info_text = f"""🖼️ **入力画像の予想出力サイズ**

**元サイズ**: **{orig_w}×{orig_h}** ({prediction['aspect_description']})  
**予想サイズ**: **{pred_h}×{pred_w}** ({pred_h * pred_w:,}ピクセル/アスペクト比: {prediction['aspect_ratio']:.2f})"""
            
            # クロッピング情報を追加
            if cropping_info and cropping_info['has_cropping']:
                crop_w, crop_h = cropping_info['crop_amount']
                crop_ratio_w, crop_ratio_h = cropping_info['crop_ratio']
                crop_direction = cropping_info['crop_direction']
                
                if crop_direction == "horizontal":
                    crop_percent = crop_ratio_w * 100
                    crop_text = f"**⚠️ 横方向に{crop_percent:.1f}%カット**（左右端{crop_w//2}px削除）"
                else:
                    crop_percent = crop_ratio_h * 100
                    crop_text = f"**⚠️ 縦方向に{crop_percent:.1f}%カット**（上下端{crop_h//2}px削除）"
                
                info_text += f"\n{crop_text}"
            else:
                info_text += "\n✅ **クロッピングなし**（完全な画像を使用）"
                
        else:
            pred_h, pred_w = prediction['predicted_size']
            info_text = f"""🖼️ **予想出力サイズ**

**予想サイズ**: **{pred_h}×{pred_w}** ({pred_h * pred_w:,}ピクセル/{prediction['aspect_description']})"""
        
        return gr.update(value=info_text)
        
    except Exception as e:
        error_text = f"解像度情報の取得に失敗しました: {str(e)}"
        return gr.update(value=error_text)


# === Wrapper関数群（メインファイルから移動） ===

def handle_open_folder_btn_wrapper(folder_name):
    """フォルダ名を保存し、そのフォルダを開く処理"""
    from .oichi2_settings import handle_open_folder_btn
    from common_utils.settings_manager import get_output_folder_path
    
    folder_update, path_update = handle_open_folder_btn(folder_name)
    
    if folder_name and folder_name.strip():
        # グローバル変数への反映はメインファイルで処理
        pass
    
    return folder_update, path_update


def update_from_image_metadata_wrapper(image_path, should_copy):
    """画像からメタデータを抽出してプロンプトとシードを更新する処理"""
    from .oichi2_settings import update_from_image_metadata
    return update_from_image_metadata(image_path, should_copy)


def check_metadata_on_checkbox_change_wrapper(should_copy, image_path):
    """チェックボックスの状態が変更された時に画像からメタデータを抽出する処理"""
    from .oichi2_settings import check_metadata_on_checkbox_change
    return check_metadata_on_checkbox_change(should_copy, image_path)


def update_input_folder_wrapper(folder_name):
    """入力フォルダ名を更新（グローバル変数操作はメインで実行）"""
    from .oichi2_file_utils import update_input_folder_name
    input_folder_name_value = update_input_folder_name(folder_name)
    return gr.update(value=input_folder_name_value)


def open_input_folder_wrapper():
    """入力フォルダを開く処理（グローバル変数はメインで管理）"""
    from .oichi2_file_utils import open_input_folder_with_save
    # グローバル変数アクセスはメインファイルで実行
    return None


def load_preset_handler_wrapper(preset_name):
    """プリセット読み込みハンドラーのラッパー"""
    if isinstance(preset_name, tuple) and len(preset_name) == 2:
        preset_name = preset_name[1]  # 値部分を取得
    return create_load_preset_handler()(preset_name)