#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oichi2_prompt_management_ui.py

FramePack-oichi2 プロンプト管理UI構築モジュール
- プリセット管理UI
- 編集・保存・削除機能
- プロンプト履歴管理
"""

import gradio as gr
from locales.i18n_extended import translate
from common_utils.preset_manager import load_presets


def create_prompt_management_ui():
    """
    プロンプト管理UIブロックを作成
    
    Returns:
        tuple: (prompt_management_group, ui_components)
    """
    with gr.Group(visible=True) as prompt_management:
        gr.Markdown(f"### " + translate("プロンプト管理"))
            
        with gr.Group(visible=True):
            default_prompt = ""
            default_name = ""
            for preset in load_presets()["presets"]:
                if preset.get("is_startup_default", False):
                    default_prompt = preset["prompt"]
                    default_name = preset["name"]
                    break
        
            with gr.Row():
                edit_name = gr.Textbox(label=translate("プリセット名"), placeholder=translate("名前を入力..."), value=default_name)
            
            edit_prompt = gr.Textbox(label=translate("プロンプト"), lines=2, value=default_prompt)
            
            with gr.Row():
                default_preset = translate("起動時デフォルト")
                presets_data = load_presets()
                choices = [preset["name"] for preset in presets_data["presets"]]
                default_presets = [name for name in choices if any(p["name"] == name and p.get("is_default", False) for p in presets_data["presets"])]
                user_presets = [name for name in choices if name not in default_presets]
                sorted_choices = [(name, name) for name in sorted(default_presets) + sorted(user_presets)]
                preset_dropdown = gr.Dropdown(label=translate("プリセット"), choices=sorted_choices, value=default_preset, type="value")
            
            with gr.Row():
                save_btn = gr.Button(value=translate("保存"), variant="primary")
                apply_preset_btn = gr.Button(value=translate("反映"), variant="primary")
                clear_btn = gr.Button(value=translate("クリア"))
                delete_preset_btn = gr.Button(value=translate("削除"))
        
        result_message = gr.Markdown("")
    
    # UIコンポーネントを辞書で返却
    ui_components = {
        "prompt_management": prompt_management,
        "edit_name": edit_name,
        "edit_prompt": edit_prompt,
        "preset_dropdown": preset_dropdown,
        "save_btn": save_btn,
        "apply_preset_btn": apply_preset_btn,
        "clear_btn": clear_btn,
        "delete_preset_btn": delete_preset_btn,
        "result_message": result_message
    }
    
    return prompt_management, ui_components