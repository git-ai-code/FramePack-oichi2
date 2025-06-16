"""
FramePack-oichi2 システム初期化・設定管理統合モジュール

システム初期化・設定処理の分離・最適化
- VRAM検出・モード設定処理
- モデル並列ダウンロード・管理インスタンス初期化
- Transformer・TextEncoder管理設定
- トークナイザー・feature_extractor遅延ロード
- モデル設定・関数定義
- フォルダ構造・ログ設定・出力設定統合
- グローバル変数初期化・イメージキューラッパー
"""

import os
import time
import traceback

import torch
from transformers import CLIPTokenizer, LlamaTokenizerFast, SiglipImageProcessor

from common_utils.hf_config import (
    check_required_models,
    get_model_cache_base_path,
    setup_models_with_check,
    verify_model_cache_directory,
)
from common_utils.log_manager import enable_logging, get_default_log_settings
from common_utils.lora_preset_manager import initialize_lora_presets
from common_utils.settings_manager import get_output_folder_path, load_settings
from common_utils.text_encoder_manager import TextEncoderManager
from common_utils.transformer_manager import TransformerManager
from diffusers_helper.memory import get_cuda_free_memory_gb
from diffusers_helper.thread_utils import AsyncStream
from locales.i18n_extended import translate

from .oichi2_file_utils import get_image_queue_files, setup_folder_structure, setup_output_folder


def initialize_system_configuration(gpu):
    """
    システム設定・初期化処理の統合関数
    
    Args:
        gpu: GPU デバイス指定
        
    Returns:
        dict: 初期化されたシステム設定とマネージャー
    """
    # VRAM検出・モード設定
    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 100
    
    print(translate('Free VRAM {0} GB').format(free_mem_gb))
    print(translate('High-VRAM Mode: {0}').format(high_vram))
    
    # 1. HuggingFace動的設定とモデル存在確認処理
    setup_models_with_check()
    
    # グローバルなモデル状態管理インスタンスを作成（モデルは実際に使用するまでロードしない）
    transformer_manager = TransformerManager(device=gpu, high_vram_mode=high_vram, use_f1_model=False)
    text_encoder_manager = TextEncoderManager(device=gpu, high_vram_mode=high_vram)
    
    return {
        'free_mem_gb': free_mem_gb,
        'high_vram': high_vram,
        'transformer_manager': transformer_manager,
        'text_encoder_manager': text_encoder_manager
    }


def reload_transformer_if_needed(transformer_manager):
    """
    transformerモデルが必要に応じてリロードする
    
    Args:
        transformer_manager: TransformerManagerインスタンス
        
    Returns:
        bool: リロード成功時True、失敗時False
    """
    try:
        # 既存のensure_transformer_stateメソッドを使用する
        if hasattr(transformer_manager, 'ensure_transformer_state'):
            return transformer_manager.ensure_transformer_state()
        # 互換性のために古い方法も維持
        elif hasattr(transformer_manager, '_needs_reload') and hasattr(transformer_manager, '_reload_transformer'):
            if transformer_manager._needs_reload():
                return transformer_manager._reload_transformer()
            return True
        return False
    except Exception as e:
        print(translate("transformerリロードエラー: {0}").format(e))
        traceback.print_exc()
        return False


def _get_local_model_path(model_name: str, subfolder: str = None) -> str:
    """HuggingFaceキャッシュから実際のモデルパスを取得"""
    from common_utils.hf_config import get_model_cache_base_path
    
    model_cache_base = get_model_cache_base_path()
    model_cache_dir = os.path.join(model_cache_base, 'hub', f'models--{model_name.replace("/", "--")}')
    
    # snapshotsフォルダ内の最初（最新）のハッシュフォルダを使用
    snapshots_dir = os.path.join(model_cache_dir, 'snapshots')
    if os.path.exists(snapshots_dir):
        snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if snapshots:
            base_path = os.path.join(snapshots_dir, snapshots[0])
            if subfolder:
                return os.path.join(base_path, subfolder)
            return base_path
    
    # フォールバック: 元の文字列形式
    return model_name


def _get_local_image_encoder_path(model_name: str, subfolder: str = None) -> str:
    """HuggingFaceキャッシュから実際の画像エンコーダーパスを取得"""
    from common_utils.hf_config import get_model_cache_base_path
    
    model_cache_base = get_model_cache_base_path()
    model_cache_dir = os.path.join(model_cache_base, 'hub', f'models--{model_name.replace("/", "--")}')
    
    # snapshotsフォルダ内の最初（最新）のハッシュフォルダを使用
    snapshots_dir = os.path.join(model_cache_dir, 'snapshots')
    if os.path.exists(snapshots_dir):
        snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if snapshots:
            base_path = os.path.join(snapshots_dir, snapshots[0])
            if subfolder:
                return os.path.join(base_path, subfolder)
            return base_path
    
    # フォールバック: 元の文字列形式
    return model_name


def initialize_models_and_tokenizers():
    """
    モデル・トークナイザー遅延ロード初期化処理
    
    Returns:
        dict: 初期化されたモデル・トークナイザー辞書
    """
    models = {}
    
    # 遅延ロード方式に変更 - 起動時にはtokenizerのみロードする
    try:
        # tokenizerのロードは起動時から行う
        try:
            print(translate("tokenizer, tokenizer_2のロードを開始します..."))
            tokenizer_path = _get_local_model_path("hunyuanvideo-community/HunyuanVideo", "tokenizer")
            tokenizer_2_path = _get_local_model_path("hunyuanvideo-community/HunyuanVideo", "tokenizer_2")
            
            models['tokenizer'] = LlamaTokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
            models['tokenizer_2'] = CLIPTokenizer.from_pretrained(tokenizer_2_path, local_files_only=True)
            print(translate("tokenizer, tokenizer_2のロードが完了しました"))
        except Exception as e:
            print(translate("tokenizer, tokenizer_2のロードに失敗しました: {0}").format(e))
            traceback.print_exc()
            print(translate("5秒間待機後に再試行します..."))
            time.sleep(5)
            tokenizer_path = _get_local_model_path("hunyuanvideo-community/HunyuanVideo", "tokenizer")
            tokenizer_2_path = _get_local_model_path("hunyuanvideo-community/HunyuanVideo", "tokenizer_2")
            
            models['tokenizer'] = LlamaTokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
            models['tokenizer_2'] = CLIPTokenizer.from_pretrained(tokenizer_2_path, local_files_only=True)
        
        # feature_extractorは軽量なのでここでロード
        try:
            print(translate("feature_extractorのロードを開始します..."))
            feature_extractor_path = _get_local_model_path("lllyasviel/flux_redux_bfl", "feature_extractor")
            
            models['feature_extractor'] = SiglipImageProcessor.from_pretrained(feature_extractor_path, local_files_only=True)
            print(translate("feature_extractorのロードが完了しました"))
        except Exception as e:
            print(translate("feature_extractorのロードに失敗しました: {0}").format(e))
            print(translate("再試行します..."))
            time.sleep(2)
            feature_extractor_path = _get_local_model_path("lllyasviel/flux_redux_bfl", "feature_extractor")
            
            models['feature_extractor'] = SiglipImageProcessor.from_pretrained(feature_extractor_path, local_files_only=True)
        
        # 他の重いモデルは遅延ロード方式に変更
        # 変数の初期化だけ行い、実際のロードはworker関数内で行う
        models['vae'] = None
        models['text_encoder'] = None
        models['text_encoder_2'] = None
        models['transformer'] = None
        models['image_encoder'] = None
        
    except Exception as e:
        print(translate("初期化エラー: {0}").format(e))
        print(translate("プログラムを終了します..."))
        import sys
        sys.exit(1)
    
    return models


def setup_vae_if_loaded(vae, high_vram, gpu):
    """
    VAEモデルの設定処理（ロード済みの場合のみ）
    
    Args:
        vae: VAEモデルインスタンス
        high_vram: 高VRAMモードフラグ
        gpu: GPUデバイス
    """
    if vae is not None:
        vae.eval()
        if not high_vram:
            vae.enable_slicing()
            vae.enable_tiling()
        vae.to(dtype=torch.float16)
        vae.requires_grad_(False)
        if high_vram:
            vae.to(gpu)


def setup_image_encoder_if_loaded(image_encoder, high_vram, gpu):
    """
    Image Encoderモデルの設定処理（ロード済みの場合のみ）
    
    Args:
        image_encoder: Image Encoderモデルインスタンス
        high_vram: 高VRAMモードフラグ
        gpu: GPUデバイス
    """
    if image_encoder is not None:
        image_encoder.eval()
        image_encoder.to(dtype=torch.float16)
        image_encoder.requires_grad_(False)
        if high_vram:
            image_encoder.to(gpu)


def initialize_folder_and_settings_configuration(webui_folder):
    """
    フォルダ構造・設定・ログ初期化処理
    
    Args:
        webui_folder: webuiフォルダパス
        
    Returns:
        dict: フォルダ・設定情報辞書
    """
    # ストリーム初期化
    stream = AsyncStream()
    
    # フォルダ構造を先に定義（モジュール分離済み）
    base_path, settings_folder = setup_folder_structure(webui_folder)
    
    # LoRAプリセットの初期化
    initialize_lora_presets()
    
    # 設定から出力フォルダを取得
    app_settings = load_settings()
    output_folder_name = app_settings.get('output_folder', 'outputs')
    print(translate("設定から出力フォルダを読み込み: {0}").format(output_folder_name))
    
    # ログ設定を読み込み適用
    log_settings = app_settings.get('log_settings', get_default_log_settings())
    print(translate("ログ設定を読み込み: 有効={0}, フォルダ={1}").format(
        log_settings.get('log_enabled', False), 
        log_settings.get('log_folder', 'logs')
    ))
    if log_settings.get('log_enabled', False):
        # 現在のファイル名を渡す
        enable_logging(log_settings.get('log_folder', 'logs'), source_name="oneframe_ichi")
    
    # 出力フォルダのフルパスを生成（モジュール分離済み）
    outputs_folder = setup_output_folder(output_folder_name, get_output_folder_path)
    
    return {
        'stream': stream,
        'base_path': base_path,
        'settings_folder': settings_folder,
        'app_settings': app_settings,
        'output_folder_name': output_folder_name,
        'outputs_folder': outputs_folder,
        'log_settings': log_settings
    }


def initialize_global_variables():
    """
    グローバル変数の初期化
    
    Returns:
        dict: 初期化されたグローバル変数辞書
    """
    return {
        'g_frame_size_setting': "1フレーム",
        'batch_stopped': False,  # バッチ処理中断フラグ
        'queue_enabled': False,  # キュー機能の有効/無効フラグ
        'queue_type': "prompt",  # キューのタイプ（"prompt" または "image"）
        'prompt_queue_file_path': None,  # プロンプトキューのファイルパス
        'image_queue_files': [],  # イメージキューのファイルリスト
        'input_folder_name_value': "inputs"  # 入力フォルダの名前（デフォルト値）
    }


def create_image_queue_files_wrapper(base_path, input_folder_name_value):
    """
    イメージキューのための画像ファイルリスト取得ラッパー関数を作成
    
    Args:
        base_path: ベースパス
        input_folder_name_value: 入力フォルダ名
        
    Returns:
        function: ラッパー関数
    """
    def get_image_queue_files_wrapper():
        """入力フォルダから画像ファイルを取得してイメージキューに追加する関数"""
        image_files = get_image_queue_files(base_path, input_folder_name_value)
        return image_files
    
    return get_image_queue_files_wrapper