#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oichi2_advanced_control_ui.py

FramePack-oichi2 高度制御UI構築モジュール
- 高度制御UIコンポーネントの作成
- 制御画像・マスク画像UI
- kisekaeichi・1f-mc制御設定
"""

import gradio as gr
from locales.i18n_extended import translate


def create_advanced_control_ui():
    """
    高度制御UIブロックを作成
    
    Returns:
        tuple: (advanced_control_group, ui_components)
    """
    # 統合画像制御部分（条件表示・新レイアウト）
    with gr.Group(visible=False) as advanced_control_group:
        # 1段目：kisekaeichi制御（1列表示）
        gr.Markdown(translate("**着せ替え制御画像**\n服装やスタイルの制御に使用"))
        kisekaeichi_reference_image = gr.Image(
            sources=['upload', 'clipboard'],
            label=translate("着せ替え制御画像"),
            type="filepath",
            interactive=True,
            height=200
        )
        kisekaeichi_control_index = gr.Slider(
            label=translate("着せ替え制御位置"),
            minimum=1, maximum=16, value=10, step=1,
            info=translate("制御画像の時間位置（推奨10）")
        )
        
        # マスク画像（kisekaeichi用）
        with gr.Row():
            with gr.Column():
                gr.Markdown(translate("**入力画像マスク（オプション）**\n白い部分を保持、黒い部分を変更（グレースケール画像）"))
                input_mask = gr.Image(
                    sources=['upload', 'clipboard'],
                    label=translate("入力画像マスク"),
                    type="filepath",
                    interactive=True,
                    height=200
                )
            
            with gr.Column():
                gr.Markdown(translate("**制御画像マスク（オプション）**\n白い部分を適用、黒い部分を無視（グレースケール画像）"))
                reference_mask = gr.Image(
                    sources=['upload', 'clipboard'],
                    label=translate("制御画像マスク"),
                    type="filepath",
                    interactive=True,
                    height=200
                )
        
        # 2段目：1f-mcとオプション（2列表示）
        with gr.Row():
            # 左列：1f-mc制御
            with gr.Column(scale=1):
                gr.Markdown(translate("**人物制御画像（1f-mc）**\n人物・キャラクターの制御に使用"))
                oneframe_mc_image = gr.Image(
                    sources=['upload', 'clipboard'],
                    label=translate("人物制御画像"),
                    type="filepath",
                    interactive=True,
                    height=200
                )
                oneframe_mc_control_index = gr.Slider(
                    label=translate("人物制御位置"),
                    minimum=1, maximum=16, value=1, step=1,
                    info=translate("制御画像の時間位置（推奨1）")
                )
            
            # 右列：オプション制御
            with gr.Column(scale=1):
                gr.Markdown(translate("**追加制御画像（1f-mc）**\nキャラクターや小物等の追加制御"))
                optional_control_image = gr.Image(
                    sources=['upload', 'clipboard'],
                    label=translate("追加制御画像"),
                    type="filepath",
                    interactive=True,
                    height=200
                )
                optional_control_index = gr.Slider(
                    label=translate("追加制御位置"),
                    minimum=1, maximum=16, value=5, step=1,
                    info=translate("制御画像の時間位置（暫定5）")
                )
        
        # 使用方法説明
        gr.Markdown(translate("**kisekaeichi** Image（位置0）+制御画像（位置10:可変）で着せ替え"))
        gr.Markdown(translate("**1f-mc** Image（位置0）+人物制御画像（位置1:可変）+追加制御画像（位置5:可変）で複数制御"))
    
    # UIコンポーネントを辞書で返却
    ui_components = {
        "advanced_control_group": advanced_control_group,
        "kisekaeichi_reference_image": kisekaeichi_reference_image,
        "kisekaeichi_control_index": kisekaeichi_control_index,
        "input_mask": input_mask,
        "reference_mask": reference_mask,
        "oneframe_mc_image": oneframe_mc_image,
        "oneframe_mc_control_index": oneframe_mc_control_index,
        "optional_control_image": optional_control_image,
        "optional_control_index": optional_control_index
    }
    
    return advanced_control_group, ui_components