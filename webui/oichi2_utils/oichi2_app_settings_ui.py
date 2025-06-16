#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oichi2_app_settings_ui.py

FramePack-oichi2 アプリケーション設定UI構築モジュール
- 設定保存・リセット
- ログ設定管理
- アラーム設定
"""

import gradio as gr
from locales.i18n_extended import translate
from common_utils.settings_manager import load_settings
from common_utils.log_manager import open_log_folder


def create_app_settings_ui(saved_app_settings=None):
    """
    アプリケーション設定UIブロックを作成
    
    Args:
        saved_app_settings: 保存済み設定（デフォルト値用）
        
    Returns:
        tuple: (app_settings_group, ui_components)
    """
    with gr.Group() as app_settings_group:
        gr.Markdown(f"### " + translate("アプリケーション設定"))
        with gr.Row():
            with gr.Column(scale=1):
                save_current_settings_btn = gr.Button(value=translate("💾 現在の設定を保存"), size="sm")
            with gr.Column(scale=1):
                reset_settings_btn = gr.Button(value=translate("🔄 設定をリセット"), size="sm")
        
        save_settings_on_start = gr.Checkbox(
            label=translate("生成開始時に自動保存"),
            value=saved_app_settings.get("save_settings_on_start", False) if saved_app_settings else False,
            info=translate("チェックをオンにすると、生成開始時に現在の設定が自動的に保存されます。設定は再起動時に反映されます。"),
            elem_classes="saveable-setting",
            interactive=True
        )
        
        alarm_on_completion = gr.Checkbox(
            label=translate("完了時にアラームを鳴らす（Windows）"),
            value=saved_app_settings.get("alarm_on_completion", True) if saved_app_settings else True,
            info=translate("チェックをオンにすると、生成完了時にアラーム音を鳴らします（Windows）"),
            elem_classes="saveable-setting",
            interactive=True
        )
        
        gr.Markdown("### " + translate("ログ設定"))
        
        all_settings = load_settings()
        log_settings = all_settings.get('log_settings', {'log_enabled': False, 'log_folder': 'logs'})
        
        log_enabled = gr.Checkbox(
            label=translate("コンソールログを出力する"),
            value=log_settings.get('log_enabled', False),
            info=translate("チェックをオンにすると、コンソール出力をログファイルにも保存します"),
            elem_classes="saveable-setting",
            interactive=True
        )
        
        log_folder = gr.Textbox(
            label=translate("ログ出力先"),
            value=log_settings.get('log_folder', 'logs'),
            info=translate("ログファイルの保存先フォルダを指定します"),
            elem_classes="saveable-setting",
            interactive=True
        )
        
        open_log_folder_btn = gr.Button(value=translate("📂 ログフォルダを開く"), size="sm")
        
        # ログフォルダ開くイベントハンドラーを内部で設定
        open_log_folder_btn.click(fn=open_log_folder)
        
        settings_status = gr.Markdown("")
    
    # UIコンポーネントを辞書で返却
    ui_components = {
        "app_settings_group": app_settings_group,
        "save_current_settings_btn": save_current_settings_btn,
        "reset_settings_btn": reset_settings_btn,
        "save_settings_on_start": save_settings_on_start,
        "alarm_on_completion": alarm_on_completion,
        "log_enabled": log_enabled,
        "log_folder": log_folder,
        "open_log_folder_btn": open_log_folder_btn,
        "settings_status": settings_status
    }
    
    return app_settings_group, ui_components