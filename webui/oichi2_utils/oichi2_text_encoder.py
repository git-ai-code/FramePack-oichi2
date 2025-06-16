"""
FramePack-oichi2 テキストエンコーディング処理モジュール

テキスト処理の統合モジュール
- プロンプト解析・キャッシュ管理
- テキストエンコーダ初期化・エンコード処理
- メモリ管理・効率的な処理制御
"""

import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from locales.i18n_extended import translate

# ファイル読み込みキャッシュ
_file_cache: Dict[Tuple[str, float], str] = {}
_cache_max_size: int = 100

# テキストエンコード結果のキャッシュ用グローバル変数
cached_prompt: Optional[str] = None
cached_n_prompt: Optional[str] = None
cached_llama_vec: Optional[torch.Tensor] = None
cached_llama_vec_n: Optional[torch.Tensor] = None
cached_clip_l_pooler: Optional[torch.Tensor] = None
cached_clip_l_pooler_n: Optional[torch.Tensor] = None
cached_llama_attention_mask: Optional[torch.Tensor] = None
cached_llama_attention_mask_n: Optional[torch.Tensor] = None


def process_text_encoding(
    prompt: str, 
    n_prompt: str, 
    cfg: float, 
    transformer: Any, 
    text_encoder_manager: Any, 
    tokenizer: Any, 
    tokenizer_2: Any, 
    high_vram: bool, 
    gpu: str,
    queue_enabled: bool = False, 
    queue_type: str = "prompt", 
    batch_index: Optional[int] = None,
    image_queue_files: Optional[List[str]] = None
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], 
           Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """テキストエンコーディング処理のメイン関数。
    
    プロンプトとネガティブプロンプトをエンコードし、キャッシュ機能を使用して
    効率的な処理を実現します。カスタムプロンプトファイルの読み込みにも対応。
    
    Args:
        prompt: 基本プロンプト文字列
        n_prompt: ネガティブプロンプト文字列
        cfg: CFG（Classifier Free Guidance）値
        transformer: Transformerモデルインスタンス
        text_encoder_manager: テキストエンコーダ管理オブジェクト
        tokenizer: LLaMAトークナイザー
        tokenizer_2: CLIPトークナイザー
        high_vram: ハイVRAMモード使用フラグ
        gpu: GPU設定文字列
        queue_enabled: キュー機能有効フラグ
        queue_type: キューのタイプ（"prompt" または "image"）
        batch_index: バッチ処理インデックス
        image_queue_files: イメージキューファイルリスト
        
    Returns:
        エンコード結果のタプル:
        - llama_vec: LLaMA埋め込みベクトル
        - clip_l_pooler: CLIPプーラー出力
        - llama_vec_n: LLaMAネガティブ埋め込み
        - clip_l_pooler_n: CLIPネガティブプーラー出力
        - llama_attention_mask: LLaMAアテンションマスク
        - llama_attention_mask_n: LLaMAネガティブアテンションマスク
    """
    # カスタムプロンプトの処理
    using_custom_prompt, current_prompt = _process_custom_prompt(
        prompt, queue_enabled, queue_type, batch_index, image_queue_files
    )
    
    # キャッシュの使用判断
    use_cache = _check_cache_validity(prompt, n_prompt)
    
    if use_cache:
        # キャッシュを使用
        return _use_cached_results()
    else:
        # 新規エンコード
        return _perform_new_encoding(
            prompt, n_prompt, cfg, transformer, text_encoder_manager,
            tokenizer, tokenizer_2, high_vram, gpu,
            using_custom_prompt, current_prompt, queue_enabled, queue_type, batch_index
        )


def _process_custom_prompt(
    prompt: str, 
    queue_enabled: bool, 
    queue_type: str, 
    batch_index: Optional[int], 
    image_queue_files: Optional[List[str]]
) -> Tuple[bool, str]:
    """カスタムプロンプトファイルからの読み込み処理。
    
    イメージキューモードでbatch_indexに対応するテキストファイルが存在する場合、
    そのファイルからカスタムプロンプトを読み込みます。ファイル読み込みには
    キャッシュ機能を使用して効率化を図ります。
    
    Args:
        prompt: デフォルトプロンプト
        queue_enabled: キュー機能有効フラグ
        queue_type: キューのタイプ
        batch_index: バッチ処理インデックス
        image_queue_files: イメージキューファイルリスト
    
    Returns:
        カスタムプロンプト使用状況とプロンプト文字列:
        - using_custom_prompt: カスタムプロンプト使用フラグ
        - current_prompt: 使用するプロンプト文字列
    """
    using_custom_prompt = False
    current_prompt = prompt  # デフォルトは共通プロンプト

    if queue_enabled and queue_type == "image" and batch_index is not None and batch_index > 0:
        if image_queue_files and batch_index - 1 < len(image_queue_files):
            queue_img_path = image_queue_files[batch_index - 1]
            img_basename = os.path.splitext(queue_img_path)[0]
            txt_path = f"{img_basename}.txt"
            
            if os.path.exists(txt_path):
                try:
                    custom_prompt = _get_cached_file_content(txt_path)
                    
                    # カスタムプロンプトを設定
                    current_prompt = custom_prompt
                    
                    img_name = os.path.basename(queue_img_path)
                    using_custom_prompt = True
                    print(translate("カスタムプロンプト情報: イメージキュー画像「{0}」の専用プロンプトを使用しています").format(img_name))
                except Exception as e:
                    print(translate("カスタムプロンプトファイルの読み込みに失敗しました: {0}").format(e))
                    using_custom_prompt = False  # エラーが発生した場合は共通プロンプトを使用
    
    return using_custom_prompt, current_prompt


def _get_cached_file_content(file_path: str) -> str:
    """
    ファイル内容をキャッシュ付きで読み込む
    
    Returns:
        str: ファイル内容
    """
    global _file_cache, _cache_max_size
    
    # ファイルの更新時刻を取得
    try:
        mtime = os.path.getmtime(file_path)
        cache_key = (file_path, mtime)
        
        # キャッシュにあるかチェック
        if cache_key in _file_cache:
            return _file_cache[cache_key]
        
        # ファイルを読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if len(_file_cache) >= _cache_max_size:
            oldest_key = next(iter(_file_cache))
            del _file_cache[oldest_key]
        
        # キャッシュに追加
        _file_cache[cache_key] = content
        
        return content
        
    except Exception as e:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()


def _check_cache_validity(prompt: str, n_prompt: str) -> bool:
    """
    キャッシュの有効性をチェック
    
    Returns:
        bool: キャッシュ使用可能かどうか
    """
    global cached_prompt, cached_n_prompt, cached_llama_vec, cached_llama_vec_n
    
    # プロンプトが変更されたかチェック
    use_cache = (cached_prompt == prompt and cached_n_prompt == n_prompt and 
                cached_llama_vec is not None and cached_llama_vec_n is not None)
    
    return use_cache


def _use_cached_results():
    """
    キャッシュされた結果を使用
    
    Returns:
        tuple: キャッシュされたエンコード結果
    """
    global cached_llama_vec, cached_llama_vec_n, cached_clip_l_pooler, cached_clip_l_pooler_n
    global cached_llama_attention_mask, cached_llama_attention_mask_n
    
    print(translate("キャッシュされたテキストエンコード結果を使用します"))
    
    llama_vec = cached_llama_vec
    clip_l_pooler = cached_clip_l_pooler
    llama_vec_n = cached_llama_vec_n
    clip_l_pooler_n = cached_clip_l_pooler_n
    llama_attention_mask = cached_llama_attention_mask
    llama_attention_mask_n = cached_llama_attention_mask_n
    
    return llama_vec, clip_l_pooler, llama_vec_n, clip_l_pooler_n, llama_attention_mask, llama_attention_mask_n


def _perform_new_encoding(prompt, n_prompt, cfg, transformer, text_encoder_manager,
                         tokenizer, tokenizer_2, high_vram, gpu,
                         using_custom_prompt, current_prompt, queue_enabled, queue_type, batch_index):
    """
    新規エンコード処理
    
    Returns:
        tuple: エンコード結果
    """
    from diffusers_helper.hunyuan import encode_prompt_conds
    from diffusers_helper.utils import crop_or_pad_yield_mask
    from diffusers_helper.memory import fake_diffusers_current_device, load_model_as_complete, get_cuda_free_memory_gb
    
    try:
        # テキストエンコーダの初期化
        text_encoder, text_encoder_2 = _initialize_text_encoders(text_encoder_manager)
        
        # GPUロード（ローVRAMモード）
        if not high_vram:
            print(translate("テキストエンコーダをGPUにロード..."))
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
        
        # プロンプト処理とエンコード
        llama_vec, clip_l_pooler, llama_vec_n, clip_l_pooler_n = _encode_prompts(
            prompt, n_prompt, cfg, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            using_custom_prompt, current_prompt, queue_enabled, queue_type, batch_index
        )
        
        # メモリ管理（ローVRAMモード）
        if not high_vram:
            _cleanup_text_encoders(text_encoder, text_encoder_2, gpu)
        
        # マスク処理とキャッシュ更新
        llama_vec, llama_attention_mask, llama_vec_n, llama_attention_mask_n = _process_masks_and_cache(
            llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, prompt, n_prompt
        )
        
        return llama_vec, clip_l_pooler, llama_vec_n, clip_l_pooler_n, llama_attention_mask, llama_attention_mask_n
        
    except Exception as e:
        print(translate("テキストエンコードエラー: {0}").format(e))
        raise e


def _initialize_text_encoders(text_encoder_manager):
    """
    テキストエンコーダの初期化
    
    Returns:
        tuple: (text_encoder, text_encoder_2)
    """
    print(translate("テキストエンコーダを初期化します..."))
    
    try:
        # text_encoder_managerを使用して初期化
        if not text_encoder_manager.ensure_text_encoder_state():
            print(translate("テキストエンコーダの初期化に失敗しました。再試行します..."))
            if not text_encoder_manager.ensure_text_encoder_state():
                raise Exception(translate("テキストエンコーダとtext_encoder_2の初期化に複数回失敗しました"))
        
        text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()
        print(translate("テキストエンコーダの初期化が完了しました"))
        
        return text_encoder, text_encoder_2
        
    except Exception as e:
        print(translate("テキストエンコーダのロードに失敗しました: {0}").format(e))
        traceback.print_exc()
        raise e


def _encode_prompts(prompt, n_prompt, cfg, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                   using_custom_prompt, current_prompt, queue_enabled, queue_type, batch_index):
    """
    プロンプトのエンコード処理
    
    Returns:
        tuple: (llama_vec, clip_l_pooler, llama_vec_n, clip_l_pooler_n)
    """
    from diffusers_helper.hunyuan import encode_prompt_conds
    
    # 実際に使用されるプロンプトを必ず表示
    full_prompt = prompt  # 実際に使用するプロンプト
    prompt_source = translate("共通プロンプト")  # プロンプトの種類

    # プロンプトソースの判定
    if queue_enabled and queue_type == "prompt" and batch_index is not None:
        # プロンプトキューの場合
        prompt_source = translate("プロンプトキュー")
        print(translate("プロンプトキューからのプロンプトをエンコードしています..."))
    elif using_custom_prompt:
        # イメージキューのカスタムプロンプトの場合
        full_prompt = current_prompt  # カスタムプロンプトを使用
        prompt_source = translate("カスタムプロンプト")
        print(translate("カスタムプロンプトをエンコードしています..."))
    else:
        # 通常の共通プロンプトの場合
        print(translate("共通プロンプトをエンコードしています..."))

    # プロンプトの内容とソースを表示
    print(translate("プロンプトソース: {0}").format(prompt_source))
    print(translate("プロンプト全文: {0}").format(full_prompt))
    print(translate("プロンプトをエンコードしています..."))
    
    llama_vec, clip_l_pooler = encode_prompt_conds(full_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

    if cfg == 1:
        llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
    else:
        print(translate("ネガティブプロンプトをエンコードしています..."))
        llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    
    return llama_vec, clip_l_pooler, llama_vec_n, clip_l_pooler_n


def _cleanup_text_encoders(text_encoder, text_encoder_2, gpu):
    """
    テキストエンコーダのメモリクリーンアップ
    """
    from diffusers_helper.memory import get_cuda_free_memory_gb
    
    # ローVRAMモードでは使用後すぐにCPUに戻す
    if text_encoder is not None and hasattr(text_encoder, 'to'):
        text_encoder.to('cpu')
    if text_encoder_2 is not None and hasattr(text_encoder_2, 'to'):
        text_encoder_2.to('cpu')
    
    # メモリ状態をログ
    free_mem_gb = get_cuda_free_memory_gb(gpu)
    print(translate("テキストエンコード後の空きVRAM {0} GB").format(free_mem_gb))
    
    # メモリクリーンアップ
    torch.cuda.empty_cache()


def _process_masks_and_cache(llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, prompt, n_prompt):
    """
    マスク処理とキャッシュ更新
    
    Returns:
        tuple: (llama_vec, llama_attention_mask, llama_vec_n, llama_attention_mask_n)
    """
    from diffusers_helper.utils import crop_or_pad_yield_mask
    global cached_prompt, cached_n_prompt, cached_llama_vec, cached_llama_vec_n
    global cached_clip_l_pooler, cached_clip_l_pooler_n, cached_llama_attention_mask, cached_llama_attention_mask_n
    
    # エンコード結果をキャッシュ
    print(translate("エンコード結果をキャッシュします"))
    cached_prompt = prompt
    cached_n_prompt = n_prompt
    
    # エンコード処理後にキャッシュを更新
    llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
    
    # キャッシュを更新
    cached_llama_vec = llama_vec
    cached_llama_vec_n = llama_vec_n
    cached_clip_l_pooler = clip_l_pooler
    cached_clip_l_pooler_n = clip_l_pooler_n
    cached_llama_attention_mask = llama_attention_mask
    cached_llama_attention_mask_n = llama_attention_mask_n
    
    return llama_vec, llama_attention_mask, llama_vec_n, llama_attention_mask_n


def apply_final_processing(llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, 
                          image_encoder_last_hidden_state, transformer, 
                          text_encoder_manager, high_vram, use_cache=False):
    """
    最終処理：データ型変換とメモリ解放
    
    Args:
        llama_vec: LlaMAベクタ
        llama_vec_n: ネガティブLlaMAベクタ
        clip_l_pooler: CLIPプーラー
        clip_l_pooler_n: ネガティブCLIPプーラー
        image_encoder_last_hidden_state: 画像エンコーダ出力
        transformer: Transformerモデル
        text_encoder_manager: テキストエンコーダマネージャー
        high_vram: ハイVRAMモード
        use_cache: キャッシュ使用フラグ
        
    Returns:
        tuple: 処理済みエンコード結果
    """
    from diffusers_helper.utils import crop_or_pad_yield_mask
    
    # キャッシュを使用する場合は既にcrop_or_pad_yield_maskが適用済み
    if not use_cache:
        # キャッシュを使用しない場合のみ適用
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
    else:
        # キャッシュ使用時は既に処理済みのmaskを取得
        global cached_llama_attention_mask, cached_llama_attention_mask_n
        llama_attention_mask = cached_llama_attention_mask
        llama_attention_mask_n = cached_llama_attention_mask_n
    
    # データ型変換
    llama_vec = llama_vec.to(transformer.dtype)
    llama_vec_n = llama_vec_n.to(transformer.dtype)
    clip_l_pooler = clip_l_pooler.to(transformer.dtype)
    clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
    image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
    
    # endframe_ichiと同様に、テキストエンコーダーのメモリを完全に解放
    if not high_vram:
        print(translate("テキストエンコーダを完全に解放します"))
        # テキストエンコーダーを完全に解放（endframe_ichiと同様に）
        text_encoder_manager.dispose_text_encoders()
        # 明示的なキャッシュクリア
        torch.cuda.empty_cache()
    
    return (llama_vec, clip_l_pooler, llama_vec_n, clip_l_pooler_n, 
            llama_attention_mask, llama_attention_mask_n, image_encoder_last_hidden_state)