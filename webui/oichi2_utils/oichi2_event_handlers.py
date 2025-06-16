"""
FramePack-oichi2 イベントハンドラー統合モジュール

イベントハンドラー設定・UI制御処理の分離・最適化
- プリセット関連イベントハンドラー統合
- メタデータ抽出・画像変更イベント処理
- フォルダ操作・設定保存イベント統合
- 重複イベントハンドラー統合・最適化
- 設定リセット・生成開始イベント制御
"""

import gradio as gr


def setup_preset_event_handlers(save_btn, clear_btn, preset_dropdown, apply_preset_btn, delete_preset_btn,
                                edit_name, edit_prompt, result_message, prompt,
                                save_button_click_handler, clear_fields, load_preset_handler_wrapper,
                                apply_to_prompt, delete_preset_handler):
    """
    プリセット関連イベントハンドラー統合設定
    
    Args:
        save_btn, clear_btn, preset_dropdown, apply_preset_btn, delete_preset_btn: UI要素
        edit_name, edit_prompt, result_message, prompt: 入出力要素
        save_button_click_handler, clear_fields, load_preset_handler_wrapper,
        apply_to_prompt, delete_preset_handler: ハンドラー関数
    """
    # 保存ボタンのクリックイベント
    save_btn.click(
        fn=save_button_click_handler,
        inputs=[edit_name, edit_prompt],
        outputs=[result_message, preset_dropdown, prompt]
    )
    
    # クリアボタンのクリックイベント
    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[edit_name, edit_prompt]
    )
    
    # プリセット選択時のイベント
    preset_dropdown.change(
        fn=load_preset_handler_wrapper,
        inputs=[preset_dropdown],
        outputs=[edit_name, edit_prompt]
    )
    
    # 反映ボタンのクリックイベント
    apply_preset_btn.click(
        fn=apply_to_prompt,
        inputs=[edit_prompt],
        outputs=[prompt]
    )
    
    # 削除ボタンのクリックイベント
    delete_preset_btn.click(
        fn=delete_preset_handler,
        inputs=[preset_dropdown],
        outputs=[result_message, preset_dropdown]
    )


def setup_metadata_event_handlers(input_image, copy_metadata, prompt, seed,
                                 update_from_image_metadata, check_metadata_on_checkbox_change):
    """
    メタデータ抽出関連イベントハンドラー設定
    
    Args:
        input_image, copy_metadata: 入力UI要素
        prompt, seed: 出力UI要素
        update_from_image_metadata, check_metadata_on_checkbox_change: ハンドラー関数
    """
    # 画像変更時にメタデータを抽出するイベント設定
    input_image.change(
        fn=update_from_image_metadata,
        inputs=[input_image, copy_metadata],
        outputs=[prompt, seed]
    )
    
    # チェックボックス変更時にメタデータを抽出するイベント設定
    copy_metadata.change(
        fn=check_metadata_on_checkbox_change,
        inputs=[copy_metadata, input_image],
        outputs=[prompt, seed]
    )


def setup_folder_operation_event_handlers(open_folder_btn, output_dir, path_display,
                                         handle_open_folder_btn):
    """
    フォルダ操作関連イベントハンドラー設定
    
    Args:
        open_folder_btn: フォルダを開くボタン
        output_dir, path_display: 出力先・パス表示要素
        handle_open_folder_btn: フォルダ開きハンドラー
    """
    # フォルダを開くボタンのイベント
    open_folder_btn.click(
        fn=handle_open_folder_btn,
        inputs=[output_dir],
        outputs=[output_dir, path_display]
    )


def setup_settings_event_handlers(save_current_settings_btn, reset_settings_btn,
                                 resolution, steps, cfg, use_teacache, gpu_memory_preservation,
                                 gs, latent_window_size, latent_index,
                                 use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post,
                                 clean_index, save_settings_on_start, alarm_on_completion,
                                 log_enabled, log_folder, settings_status,
                                 save_app_settings_handler, reset_app_settings_handler):
    """
    設定保存・リセット関連イベントハンドラー設定
    
    Args:
        save_current_settings_btn, reset_settings_btn: 設定操作ボタン
        resolution, steps, cfg, ...: 設定関連UI要素
        settings_status: ステータス表示要素
        save_app_settings_handler, reset_app_settings_handler: ハンドラー関数
    """
    # 設定保存ボタンのクリックイベント
    save_current_settings_btn.click(
        fn=save_app_settings_handler,
        inputs=[
            resolution, steps, cfg, use_teacache, gpu_memory_preservation,
            gs, latent_window_size, latent_index,
            use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post,
            clean_index, save_settings_on_start, alarm_on_completion,
            log_enabled, log_folder
        ],
        outputs=[settings_status]
    )
    
    # 設定リセットボタンのクリックイベント
    reset_settings_btn.click(
        fn=reset_app_settings_handler,
        inputs=[],
        outputs=[
            resolution, steps, cfg, use_teacache, gpu_memory_preservation,
            gs, latent_window_size, latent_index,
            use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post,
            clean_index, save_settings_on_start, alarm_on_completion,
            log_enabled, log_folder, settings_status
        ]
    )


def create_input_parameters_list(input_image, prompt, n_prompt, seed, steps, cfg, gs, rs,
                                gpu_memory_preservation, use_teacache, lora_scales_text, use_lora, fp8_optimization, resolution, output_dir,
                                batch_count, use_random_seed, latent_window_size, latent_index,
                                use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post,
                                lora_mode,
                                use_rope_batch, use_queue, prompt_queue_file,
                                clean_index, input_mask, reference_mask,
                                use_advanced_control, advanced_control_mode,
                                kisekaeichi_reference_image, kisekaeichi_control_index,
                                oneframe_mc_image, oneframe_mc_control_index, optional_control_image, optional_control_index,
                                save_settings_on_start, alarm_on_completion, lora_files_list=None, 
                                lora_dropdowns_list=None, lora_strength_list=None):
    """
    生成開始・中止イベント用の入力パラメータリスト作成（LoRA統合・個別強度対応）
    
    Args:
        lora_files_list: 可変LoRAファイルリスト
        lora_dropdowns_list: 可変LoRAドロップダウンリスト
        lora_strength_list: 可変LoRA個別強度リスト
    
    Returns:
        list: 入力パラメータリスト
    """
    base_params = [
        input_image, prompt, n_prompt, seed, steps, cfg, gs, rs,
        gpu_memory_preservation, use_teacache, lora_scales_text, use_lora, fp8_optimization, resolution, output_dir,
        batch_count, use_random_seed, latent_window_size, latent_index,
        use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post,
        lora_mode,
        use_rope_batch, use_queue, prompt_queue_file,
        clean_index, input_mask, reference_mask,
        use_advanced_control, advanced_control_mode,
        kisekaeichi_reference_image, kisekaeichi_control_index,
        oneframe_mc_image, oneframe_mc_control_index, optional_control_image, optional_control_index,
        save_settings_on_start, alarm_on_completion
    ]
    
    # 可変LoRAパラメータを追加
    if lora_files_list:
        base_params.extend(lora_files_list)
    if lora_dropdowns_list:
        base_params.extend(lora_dropdowns_list)
    if lora_strength_list:
        base_params.extend(lora_strength_list)
    
    return base_params