"""
oichi2_latent_ui.py

FramePack-oichi2 レイテント設定UI構築モジュール
- レイテント処理範囲設定UI
- clean_latents設定UI
- フレーム処理設定
"""

import gradio as gr
from locales.i18n_extended import translate


def create_latent_settings_ui(saved_app_settings=None):
    """
    レイテント設定UIブロックを作成
    
    Args:
        saved_app_settings: 保存済み設定（デフォルト値用）
        
    Returns:
        tuple: (latent_settings_group, ui_components)
    """
    # === レイテント設定（常時表示）===
    with gr.Group() as latent_settings_group:
        gr.Markdown(f"### " + translate("レイテント設定"))

        with gr.Row():
            with gr.Column(scale=1):
                latent_window_size = gr.Slider(
                    label=translate("フレーム処理範囲（latent_window_size）"),
                    minimum=1,
                    maximum=32,
                    value=saved_app_settings.get("latent_window_size", 9) if saved_app_settings else 9,
                    step=1,
                    interactive=True,
                    info=translate("フレーム処理範囲・メモリ効率化（推奨9、RoPE値設定）"),
                    elem_classes="saveable-setting"
                )
            
            # latent_indexは上位のフレーム設定に統合済み
        
        # clean_latents設定（常時表示）
        gr.Markdown(f"#### " + translate("clean_latents設定"))
        with gr.Row():
            use_clean_latents_2x = gr.Checkbox(
                label=translate("clean_latents_2x"),
                value=saved_app_settings.get("use_clean_latents_2x", True) if saved_app_settings else True,
                interactive=True,
                info=translate("中程度反映・品質向上"),
                elem_classes="saveable-setting"
            )
            use_clean_latents_4x = gr.Checkbox(
                label=translate("clean_latents_4x"),
                value=saved_app_settings.get("use_clean_latents_4x", True) if saved_app_settings else True,
                interactive=True,
                info=translate("強め反映・高品質化"),
                elem_classes="saveable-setting"
            )
            use_clean_latents_post = gr.Checkbox(
                label=translate("clean_latents_post"),
                value=saved_app_settings.get("use_clean_latents_post", True) if saved_app_settings else True,
                interactive=True,
                info=translate("最終仕上げ・負荷軽減時OFF"),
                elem_classes="saveable-setting"
            )
    
    # UIコンポーネントを辞書で返却
    ui_components = {
        "latent_settings_group": latent_settings_group,
        "latent_window_size": latent_window_size,
        "use_clean_latents_2x": use_clean_latents_2x,
        "use_clean_latents_4x": use_clean_latents_4x,
        "use_clean_latents_post": use_clean_latents_post
    }
    
    return latent_settings_group, ui_components