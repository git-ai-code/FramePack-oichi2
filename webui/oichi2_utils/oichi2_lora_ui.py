"""
oichi2_lora_ui.py

FramePack-oichi2 LoRAUI統合モジュール
- LoRAメインUI構築 (アップロード・ドロップダウン・スケール)
- LoRAプリセットUI構築 (プリセットボタン・状態・設定)
- UI要素の集約管理

LoRAユーザーインターフェース統合モジュール
"""

import gradio as gr


def get_translation_function():
    """翻訳関数を取得（循環インポート回避）"""
    import sys
    if hasattr(sys.modules.get('__main__'), 'translate'):
        return sys.modules['__main__'].translate
    # フォールバック: 翻訳が利用できない場合はそのまま返す
    return lambda x: x


def create_lora_main_ui_blocks(max_lora_count):
    """
    LoRAメインUI構築（アップロード・ドロップダウン・スケール）
    
    Args:
        max_lora_count (int): 最大LoRA数
        
    Returns:
        tuple: (ui_groups, ui_components)
    """
    translate = get_translation_function()
    
    # ファイルアップロードUI
    with gr.Group(visible=False) as lora_upload_group:
        lora_files_list = []
        # ファイルアップロードも2列レイアウト（3行×2列）
        rows_needed_upload = (max_lora_count + 1) // 2
        
        for row in range(rows_needed_upload):
            with gr.Row():
                for col in range(2):
                    i = row * 2 + col
                    if i < max_lora_count:
                        with gr.Column(scale=1):
                            lora_file = gr.File(
                                label=translate("LoRAファイル{0}").format(i+1),
                                file_types=[".safetensors", ".pt", ".bin"],
                                height=120
                            )
                            lora_files_list.append(lora_file)

    # ドロップダウンUI
    with gr.Group(visible=False) as lora_dropdown_group:
        none_choice = translate("なし")
        lora_dropdowns_list = []
        lora_strength_list = []  # 個別強度入力欄
        
        # 横並び配置のため行数を計算（1行につき2つまで）
        rows_needed = (max_lora_count + 1) // 2
        
        for row in range(rows_needed):
            with gr.Row():
                for col in range(2):
                    i = row * 2 + col
                    if i < max_lora_count:
                        # 各LoRAセット（プルダウン + 強度）を1つのColumnにまとめる
                        with gr.Column(scale=1):
                            with gr.Row():
                                # プルダウンと強度を横並び
                                with gr.Column(scale=3):
                                    lora_dropdown = gr.Dropdown(
                                        label=translate("LoRA{0}").format(i+1), 
                                        choices=[none_choice], 
                                        value=none_choice, 
                                        allow_custom_value=True
                                    )
                                    lora_dropdowns_list.append(lora_dropdown)
                                
                                with gr.Column(scale=1, min_width=80):
                                    lora_strength = gr.Number(
                                        label=translate("強度"),
                                        value=0.8,
                                        minimum=0.0,
                                        maximum=2.0,
                                        step=0.1,
                                        precision=2
                                    )
                                    lora_strength_list.append(lora_strength)
        
        lora_scan_button = gr.Button(value=translate("LoRAフォルダを再スキャン"), variant="secondary")

    # 後方互換用スケールテキスト（内部処理専用）
    default_scales = ",".join(["0.8"] * max_lora_count)
    with gr.Group(visible=False):
        lora_scales_text = gr.Textbox(
            label="",
            value=default_scales,
            visible=False,
            interactive=False,
            elem_id="lora-scales-hidden",
            elem_classes=["hidden-element", "display-none"]
        )

    # UI要素をグループ分けして返す
    ui_groups = {
        "lora_upload_group": lora_upload_group,
        "lora_dropdown_group": lora_dropdown_group
    }
    
    ui_components = {
        "lora_files_list": lora_files_list,
        "lora_dropdowns_list": lora_dropdowns_list,
        "lora_strength_list": lora_strength_list,
        "lora_scan_button": lora_scan_button,
        "lora_scales_text": lora_scales_text
    }
    
    return ui_groups, ui_components


def create_lora_preset_ui_block(max_lora_count):
    """
    LoRAプリセット管理UIブロック作成
    
    Args:
        max_lora_count (int): 最大LoRA数
        
    Returns:
        tuple: (preset_group, ui_components)
    """
    translate = get_translation_function()
    
    with gr.Group(visible=False) as lora_preset_group:
        with gr.Row():
            preset_buttons = []
            # 1行10列レイアウト (設定1-10)
            for i in range(1, 11):
                preset_buttons.append(
                    gr.Button(
                        translate("設定{0}").format(i),
                        variant="secondary",
                        scale=1
                    )
                )
            
            # Load/Saveボタンも同じ行に配置
            lora_load_btn = gr.Button(translate("Load"), variant="primary", scale=1)
            lora_save_btn = gr.Button(translate("Save"), variant="secondary", scale=1)
            lora_preset_mode = gr.Radio(
                choices=[translate("Load"), translate("Save")],
                value=translate("Load"),
                visible=False
            )
        
        # プリセット状態 + LoRA表示数設定（改善版）
        with gr.Row():
            with gr.Column(scale=3):
                lora_preset_status = gr.Textbox(
                    label=translate("プリセット状態"),
                    value="",
                    interactive=False,
                    lines=1
                )
            with gr.Column(scale=2):
                lora_count_setting = gr.Number(
                    label=translate("LoRA表示数（要再起動）"),
                    value=max_lora_count,
                    minimum=1,
                    maximum=20,
                    step=1,
                    precision=0
                )
            with gr.Column(scale=1, min_width=120):
                lora_count_save_btn = gr.Button(
                    translate("表示数設定\\n保存"),
                    variant="secondary"
                )

    # UI要素を辞書で返す
    ui_components = {
        "preset_buttons": preset_buttons,
        "lora_load_btn": lora_load_btn,
        "lora_save_btn": lora_save_btn,
        "lora_preset_mode": lora_preset_mode,
        "lora_preset_status": lora_preset_status,
        "lora_count_setting": lora_count_setting,
        "lora_count_save_btn": lora_count_save_btn
    }
    
    return lora_preset_group, ui_components