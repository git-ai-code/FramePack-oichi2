#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oichi2_basic_settings_ui.py

FramePack-oichi2 基本設定UI構築モジュール
- TeaCache設定
- シード管理
- ステップ・CFG設定
- GPUメモリ保持設定
"""

import gradio as gr
from locales.i18n_extended import translate


def create_basic_settings_ui(saved_app_settings=None, use_random_seed_default=True, seed_default=12345):
    """
    基本設定UIブロックを作成
    
    Args:
        saved_app_settings: 保存済み設定（デフォルト値用）
        use_random_seed_default: ランダムシードのデフォルト値
        seed_default: シードのデフォルト値
        
    Returns:
        tuple: (basic_settings_group, ui_components)
    """
    with gr.Group() as basic_settings_group:
        gr.Markdown(f"### " + translate("基本設定"))
        use_teacache = gr.Checkbox(
            label=translate('Use TeaCache'), 
            value=saved_app_settings.get("use_teacache", True) if saved_app_settings else True, 
            info=translate('Faster speed, but often makes hands and fingers slightly worse.'), 
            elem_classes="saveable-setting"
        )
    
    use_random_seed = gr.Checkbox(label=translate("Use Random Seed"), value=use_random_seed_default)
    seed = gr.Number(label=translate("Seed"), value=seed_default, precision=0)
    
    steps = gr.Slider(
        label=translate("ステップ数"), 
        minimum=1, maximum=100, 
        value=saved_app_settings.get("steps", 25) if saved_app_settings else 25, 
        step=1, 
        info=translate('この値の変更は推奨されません'), 
        elem_classes="saveable-setting"
    )
    gs = gr.Slider(
        label=translate("蒸留CFGスケール"), 
        minimum=1.0, maximum=32.0, 
        value=saved_app_settings.get("gs", 10.0) if saved_app_settings else 10.0, 
        step=0.01, 
        info=translate('この値の変更は推奨されません'), 
        elem_classes="saveable-setting"
    )
    
    cfg = gr.Slider(
        label="CFGスケール", 
        minimum=1.0, maximum=32.0, 
        value=saved_app_settings.get("cfg", 1) if saved_app_settings else 1, 
        step=0.01, 
        visible=False, 
        elem_classes="saveable-setting"
    )
    rs = gr.Slider(label="CFG再スケール", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
    
    gpu_memory_preservation = gr.Slider(
        label=translate("GPUメモリ保持（GB）"), 
        minimum=6, maximum=128, 
        value=saved_app_settings.get("gpu_memory_preservation", 6) if saved_app_settings else 6, 
        step=1, 
        info=translate("OOMが発生する場合は値を大きくしてください。値が大きいほど速度が遅くなります。"),
        elem_classes="saveable-setting"
    )
    
    # UIコンポーネントを辞書で返却
    ui_components = {
        "basic_settings_group": basic_settings_group,
        "use_teacache": use_teacache,
        "use_random_seed": use_random_seed,
        "seed": seed,
        "steps": steps,
        "gs": gs,
        "cfg": cfg,
        "rs": rs,
        "gpu_memory_preservation": gpu_memory_preservation
    }
    
    return basic_settings_group, ui_components