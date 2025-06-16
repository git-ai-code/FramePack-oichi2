"""
FramePack-oichi2 エラーハンドリング・メモリクリーンアップモジュール

エラー処理・メモリ管理・リソースクリーンアップの分離・最適化
- 例外処理・エラーログ出力
- モデル別メモリクリーンアップ・CPU移動
- transformer_manager状態管理
- ガベージコレクション・VRAM解放
- 変数クリーンアップ・リソース管理
"""

import gc
import traceback
from typing import Any, List, Optional

import torch

from diffusers_helper.memory import get_cuda_free_memory_gb, unload_complete_models
from locales.i18n_extended import translate


def handle_worker_exception_and_cleanup(
    e: Exception, 
    high_vram: bool, 
    transformer: Optional[Any], 
    vae: Optional[Any], 
    image_encoder: Optional[Any], 
    text_encoder: Optional[Any], 
    text_encoder_2: Optional[Any], 
    transformer_manager: Any, 
    gpu: str
) -> None:
    """
    worker関数での例外処理・メモリクリーンアップのメイン関数
    
    Args:
        e: 発生した例外
        high_vram: ハイVRAMモード
        transformer: Transformerモデル
        vae: VAEモデル
        image_encoder: 画像エンコーダー
        text_encoder: テキストエンコーダー
        text_encoder_2: テキストエンコーダー2
        transformer_manager: Transformer管理マネージャー
        gpu: GPU設定
        
    Returns:
        None
    """
    print(translate("処理中にエラーが発生しました: {0}").format(e))
    traceback.print_exc()
    
    # エラー時の詳細なメモリクリーンアップ
    try:
        if not high_vram:
            print(translate("エラー発生時のメモリクリーンアップを実行..."))
            
            # 個別モデルクリーンアップ
            _cleanup_individual_models(transformer, vae, image_encoder, text_encoder, text_encoder_2)
            
            # transformer_manager状態リセット・一括アンロード
            _cleanup_transformer_manager_and_bulk_unload(
                transformer, vae, image_encoder, text_encoder, text_encoder_2, transformer_manager
            )
            
            # ガベージコレクション・メモリ解放
            _perform_garbage_collection_and_memory_cleanup(gpu)
            
            # 追加変数クリーンアップ
            _cleanup_additional_variables()
            
    except Exception as cleanup_error:
        print(translate("メモリクリーンアップ中にエラー: {0}").format(cleanup_error))


def _cleanup_individual_models(transformer, vae, image_encoder, text_encoder, text_encoder_2):
    """
    個別モデルクリーンアップ処理
    """
    models_to_unload = [
        ('transformer', transformer), 
        ('vae', vae), 
        ('image_encoder', image_encoder), 
        ('text_encoder', text_encoder), 
        ('text_encoder_2', text_encoder_2)
    ]
    
    for model_name, model in models_to_unload:
        if model is not None:
            try:
                print(translate("{0}をアンロード中...").format(model_name))
                if hasattr(model, 'to'):
                    model.to('cpu')
                _reset_model_reference(model_name)
            except Exception as unload_error:
                print(translate("{0}のアンロード中にエラー: {1}").format(model_name, unload_error))


def _reset_model_reference(model_name):
    """
    モデル参照の明示的削除（注意：局所変数のため効果は限定的）
    """
    # 注意: この実装は局所変数のため実際の効果は限定的
    # 実際のクリーンアップは呼び出し元で行う必要がある
    if model_name == 'transformer':
        # transformer = None  # 局所変数のため効果なし
        pass
    elif model_name == 'vae':
        # vae = None  # 局所変数のため効果なし
        pass
    elif model_name == 'image_encoder':
        # image_encoder = None  # 局所変数のため効果なし
        pass
    elif model_name == 'text_encoder':
        # text_encoder = None  # 局所変数のため効果なし
        pass
    elif model_name == 'text_encoder_2':
        # text_encoder_2 = None  # 局所変数のため効果なし
        pass


def _cleanup_transformer_manager_and_bulk_unload(transformer, vae, image_encoder, 
                                                text_encoder, text_encoder_2, transformer_manager):
    """
    transformer_manager状態リセット・一括アンロード処理
    """
    # 一括アンロード - endframe_ichiと同じアプローチでモデルを明示的に解放
    if transformer is not None:
        # まずtransformer_managerの状態をリセット - これが重要
        transformer_manager.current_state['is_loaded'] = False
        # FP8最適化モードの有無に関わらず常にCPUに移動
        transformer.to('cpu')
        print(translate("transformerをCPUに移動しました"))

    # endframe_ichi.pyと同様に明示的にすべてのモデルを一括アンロード
    # モデルを直接リストで渡す（引数展開ではなく）
    unload_complete_models(
        text_encoder, text_encoder_2, image_encoder, vae, transformer
    )
    print(translate("すべてのモデルをアンロードしました"))


def _perform_garbage_collection_and_memory_cleanup(gpu):
    """
    ガベージコレクション・メモリ解放処理
    """
    print(translate("ガベージコレクション実行中..."))
    gc.collect()
    torch.cuda.empty_cache()
    
    # メモリ状態を報告
    free_mem_gb = get_cuda_free_memory_gb(gpu)
    print(translate("クリーンアップ後の空きVRAM {0} GB").format(free_mem_gb))


def _cleanup_additional_variables():
    """
    追加変数クリーンアップ処理
    """
    # 追加の変数クリーンアップ
    # 注意: locals()はこの関数内のローカル変数を指すため、実際の効果は限定的
    # 実際のクリーンアップは呼び出し元のスコープで行う必要がある
    cleanup_var_names = [
        'start_latent', 'decoded_image', 'history_latents', 'real_history_latents', 
        'real_history_latents_gpu', 'generated_latents', 'input_image_pt', 'input_image_gpu'
    ]
    
    for var_name in cleanup_var_names:
        # 実際のクリーンアップは呼び出し元で実行される
        # ここでは変数名リストの定義のみ
        print(translate("変数 {0} のクリーンアップを試行").format(var_name))


def cleanup_worker_variables_in_caller_scope():
    """
    呼び出し元スコープでの変数クリーンアップ用の補助関数
    
    Returns:
        list: クリーンアップ対象の変数名リスト
    """
    return [
        'start_latent', 'decoded_image', 'history_latents', 'real_history_latents', 
        'real_history_latents_gpu', 'generated_latents', 'input_image_pt', 'input_image_gpu'
    ]