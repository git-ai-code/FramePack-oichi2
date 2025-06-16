"""
oichi2_lora_handlers.py

FramePack-oichi2 LoRAイベントハンドラー統合モジュール
- LoRAプリセットボタンハンドラー・ロード/セーブ機能
- Load/Saveモード切替・ボタン状態管理
- プリセットイベントハンドラー設定・UI同期
- LoRA使用状態変更連動・表示切替制御

LoRAイベントハンドラー統合モジュール
"""

import gradio as gr
from locales.i18n_extended import translate
from common_utils.lora_preset_manager import load_lora_preset, save_lora_preset


def create_lora_preset_button_handler():
    """
    LoRAプリセットボタンのクリックハンドラーを作成（可変対応）
    
    Returns:
        function: LoRAプリセットボタンハンドラー関数
    """
    def handle_lora_preset_button(button_index, mode, *lora_args):
        """LoRAプリセットボタンのクリックを処理する（可変対応・個別強度対応）"""
        # 可変LoRA対応: 設定から最大数を取得
        try:
            from common_utils.lora_config import get_max_lora_count
            max_count = get_max_lora_count()
        except ImportError:
            max_count = 3  # フォールバック
        
        # 引数を分解（新UI: ドロップダウン値 + 個別強度値 + スケール値）
        # 旧UI: ドロップダウン値 + スケール値
        if len(lora_args) >= max_count * 2 + 1:
            # 新UI版：ドロップダウン + 個別強度 + 後方互換スケール
            lora_values = list(lora_args[:max_count])
            strength_values = list(lora_args[max_count:max_count*2])
            scales = lora_args[max_count*2] if len(lora_args) > max_count*2 else ",".join([str(0.8)] * max_count)
        else:
            # 旧UI版：ドロップダウン + スケール
            lora_values = list(lora_args[:max_count])
            strength_values = [0.8] * max_count  # デフォルト強度
            scales = lora_args[max_count] if len(lora_args) > max_count else ",".join([str(0.8)] * max_count)
        
        if mode == translate("Load"):  # Load
            # ロードモード
            loaded_values = load_lora_preset(button_index)
            if loaded_values:
                # ロード結果を可変数・個別強度に対応
                updates = []
                
                # LoRAドロップダウンの更新
                for i in range(max_count):
                    if i < len(loaded_values) - 1:  # 最後はスケール値
                        updates.append(gr.update(value=loaded_values[i]))
                    else:
                        updates.append(gr.update())
                
                # 個別強度の更新（新UI版）
                if len(lora_args) >= max_count * 2 + 1:
                    # プリセットからスケール値を個別強度に変換
                    scales_value = loaded_values[-1] if loaded_values else scales
                    try:
                        scale_list = [float(x.strip()) for x in scales_value.split(',') if x.strip()]
                        for i in range(max_count):
                            if i < len(scale_list):
                                updates.append(gr.update(value=scale_list[i]))
                            else:
                                updates.append(gr.update(value=0.8))
                    except:
                        # エラー時はデフォルト値
                        for i in range(max_count):
                            updates.append(gr.update(value=0.8))
                
                # 後方互換スケール値の更新
                scales_value = loaded_values[-1] if loaded_values else scales
                updates.append(gr.update(value=scales_value))
                
                # ステータス更新
                updates.append(translate("設定{0}を読み込みました").format(button_index + 1))
                return tuple(updates)
            else:
                # 失敗時は変更なしで数を合わせる
                if len(lora_args) >= max_count * 2 + 1:
                    # 新UI版：LoRA + 個別強度 + スケール + ステータス
                    updates = [gr.update() for _ in range(max_count * 2 + 2)]
                else:
                    # 旧UI版：LoRA + スケール + ステータス
                    updates = [gr.update() for _ in range(max_count + 2)]
                updates[-1] = translate("設定{0}の読み込みに失敗しました").format(button_index + 1)
                return tuple(updates)
        else:
            # セーブモード
            # 個別強度をスケール文字列に変換してプリセットに保存
            if len(lora_args) >= max_count * 2 + 1:
                # 新UI版：個別強度値からスケール文字列を作成
                strength_scales = ",".join([str(float(val) if val is not None else 0.8) for val in strength_values])
            else:
                # 旧UI版：既存のスケール文字列を使用
                strength_scales = scales
            
            success, message = save_lora_preset(button_index, *lora_values, strength_scales)
            
            # 変更なしで数を合わせる
            if len(lora_args) >= max_count * 2 + 1:
                # 新UI版：LoRA + 個別強度 + スケール + ステータス
                updates = [gr.update() for _ in range(max_count * 2 + 2)]
            else:
                # 旧UI版：LoRA + スケール + ステータス
                updates = [gr.update() for _ in range(max_count + 2)]
            updates[-1] = message
            return tuple(updates)
    
    return handle_lora_preset_button


def create_load_save_mode_handlers():
    """
    Load/Saveモード切替ハンドラーを作成
    
    Returns:
        tuple: (set_load_mode, set_save_mode) ハンドラー関数のタプル
    """
    def set_load_mode():
        return (
            gr.update(value=translate("Load")),
            gr.update(variant="primary", interactive=True),
            gr.update(variant="secondary", interactive=True)
        )
    
    def set_save_mode():
        return (
            gr.update(value=translate("Save")),
            gr.update(variant="secondary", interactive=True),
            gr.update(variant="primary", interactive=True)
        )
    
    return set_load_mode, set_save_mode


def create_toggle_lora_and_preset_handler():
    """
    LoRA・プリセット表示切替ハンドラーを作成
    
    Returns:
        function: LoRA・プリセット表示切替ハンドラー関数
    """
    def toggle_lora_and_preset(use_lora_val, lora_mode_val):
        # LoRAが有効かつディレクトリから選択モードの場合のみプリセットを表示
        preset_visible = use_lora_val and lora_mode_val == translate("ディレクトリから選択")
        return gr.update(visible=preset_visible)
    
    return toggle_lora_and_preset


def setup_lora_preset_event_handlers(preset_buttons, lora_preset_mode, lora_scales_text, lora_preset_status,
                                     lora_load_btn, lora_save_btn, use_lora, lora_mode, lora_preset_group,
                                     lora_dropdowns_list=None, lora_strength_list=None):
    """
    LoRAプリセット関連のイベントハンドラー設定（可変対応・個別強度対応）
    
    Args:
        preset_buttons: プリセットボタンリスト
        lora_preset_mode: プリセットモード
        lora_scales_text: LoRAスケールテキスト
        lora_preset_status: プリセットステータス
        lora_load_btn, lora_save_btn: LoRAプリセット専用Load/Saveボタン
        use_lora: LoRA使用フラグ
        lora_mode: LoRAモード
        lora_preset_group: プリセットグループ
        lora_dropdowns_list: 可変LoRAドロップダウンリスト（新UI用）
        lora_strength_list: 個別強度入力リスト（新UI用）
    """
    # 可変LoRA対応: 設定から最大数を取得
    try:
        from common_utils.lora_config import get_max_lora_count
        max_count = get_max_lora_count()
    except ImportError:
        max_count = 3  # フォールバック
    
    # ドロップダウンリストを決定（可変対応）
    if lora_dropdowns_list and len(lora_dropdowns_list) >= max_count:
        # 可変LoRAリストが渡された場合
        active_dropdowns = lora_dropdowns_list[:max_count]
        active_strengths = lora_strength_list[:max_count] if lora_strength_list else []
        use_new_ui = True
    else:
        # 可変LoRAシステムのみ対応
        active_dropdowns = []
        active_strengths = []
        max_count = 0
        use_new_ui = False
    
    # ハンドラー関数を作成
    handle_lora_preset_button = create_lora_preset_button_handler()
    set_load_mode, set_save_mode = create_load_save_mode_handlers()
    toggle_lora_and_preset = create_toggle_lora_and_preset_handler()
    
    # プリセットボタンのイベント設定（可変対応・個別強度対応）
    for i, btn in enumerate(preset_buttons):
        if use_new_ui:
            # 新UI版（個別強度対応）
            btn.click(
                fn=lambda mode, *args, idx=i: handle_lora_preset_button(idx, mode, *args),
                inputs=[lora_preset_mode] + active_dropdowns + active_strengths + [lora_scales_text],
                outputs=active_dropdowns + active_strengths + [lora_scales_text, lora_preset_status]
            )
        else:
            # 可変LoRAシステムのみ対応
            pass
    
    # Load/Saveボタンのイベント設定
    def force_load_mode():
        return set_load_mode()
    
    def force_save_mode():
        return set_save_mode()
    
    # LoRAプリセット専用のLoad/Saveボタンイベント登録
    lora_load_btn.click(
        fn=force_load_mode,
        inputs=[],
        outputs=[lora_preset_mode, lora_load_btn, lora_save_btn],
        show_progress=False
    )
    
    lora_save_btn.click(
        fn=force_save_mode,
        inputs=[],
        outputs=[lora_preset_mode, lora_load_btn, lora_save_btn],
        show_progress=False
    )
    
    # LoRA使用状態とモードの変更でプリセット表示を更新
    use_lora.change(
        toggle_lora_and_preset,
        inputs=[use_lora, lora_mode],
        outputs=[lora_preset_group]
    )
    
    lora_mode.change(
        toggle_lora_and_preset,
        inputs=[use_lora, lora_mode],
        outputs=[lora_preset_group]
    )
    
    # 初期化処理：ページロード時にLoadモードを強制設定
    def initialize_lora_preset_ui():
        """LoRAプリセットUIの初期化"""
        return set_load_mode()
    
    # ページロード時に自動実行されるよう設定
    # 注意：この関数は外部から呼び出される必要があります
    return initialize_lora_preset_ui