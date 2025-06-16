#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oichi2_output_settings_ui.py

FramePack-oichi2 出力設定UI構築モジュール
- 出力フォルダ設定
- パス表示管理
- フォルダ操作ボタン
"""

import os
import gradio as gr
from locales.i18n_extended import translate


def create_output_settings_ui(output_folder_name, base_path):
    """
    出力設定UIブロックを作成
    
    Args:
        output_folder_name: 出力フォルダ名
        base_path: ベースパス
        
    Returns:
        tuple: (output_settings_group, ui_components)
    """
    with gr.Group() as output_settings_group:
        gr.Markdown(f"### " + translate("出力設定"))
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                output_dir = gr.Textbox(
                    label=translate("出力フォルダ名"),
                    value=output_folder_name,  # 設定から読み込んだ値を使用
                    info=translate("生成画像の保存先フォルダ名"),
                    placeholder="outputs"
                )
            with gr.Column(scale=1, min_width=100):
                open_folder_btn = gr.Button(value=translate("📂 保存および出力フォルダを開く"), size="sm")

        with gr.Row(visible=False):
            path_display = gr.Textbox(
                label=translate("出力フォルダの完全パス"),
                value=os.path.join(base_path, output_folder_name),
                interactive=False
            )
    
    # UIコンポーネントを辞書で返却
    ui_components = {
        "output_settings_group": output_settings_group,
        "output_dir": output_dir,
        "open_folder_btn": open_folder_btn,
        "path_display": path_display
    }
    
    return output_settings_group, ui_components