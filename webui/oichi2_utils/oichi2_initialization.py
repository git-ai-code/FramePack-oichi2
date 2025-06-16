"""
FramePack-oichi2 初期化・セットアップ統合モジュール

worker関数初期化・セットアップ処理の統合モジュール
- グローバル変数宣言・初期設定
- キュー状態管理・ログ出力
- job_id生成・基本設定
- 出力フォルダ・プログレスバー初期化
- LoRA設定・transformer状態管理
- VAEエンコーディング・メモリ管理
- 1フレームモード設定・参照画像準備
"""

import os
import time
import traceback

import numpy as np
import torch
from PIL import Image

from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from diffusers_helper.hunyuan import vae_encode
from diffusers_helper.memory import get_cuda_free_memory_gb, load_model_as_complete
from diffusers_helper.utils import resize_and_center_crop
from locales.i18n_extended import translate


def _setup_vae_model(vae, high_vram, gpu):
    """
    VAEモデルのセットアップ処理（元のsetup_vae_if_loaded関数と同等）
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


def _get_local_vae_path(model_name: str, subfolder: str = None) -> str:
    """HuggingFaceキャッシュから実際のVAEモデルパスを取得"""
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

def process_worker_initialization_and_setup(
    input_image, prompt, n_prompt, seed, steps, cfg, gs, rs,
    gpu_memory_preservation, use_teacache, lora_scales_text,
    output_dir, use_lora, fp8_optimization, resolution,
    latent_window_size, latent_index, use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post,
    lora_mode,
    batch_index, use_queue, prompt_queue_file,
    use_reference_image, reference_image, clean_index, input_mask, reference_mask,
    # グローバル変数・設定
    vae, text_encoder, text_encoder_2, transformer, image_encoder,
    queue_enabled, queue_type, prompt_queue_file_path, image_queue_files,
    cached_prompt, cached_n_prompt, cached_llama_vec, cached_llama_vec_n,
    cached_clip_l_pooler, cached_clip_l_pooler_n, cached_llama_attention_mask, cached_llama_attention_mask_n,
    transformer_manager, high_vram, gpu, webui_folder, use_cache_files, stream,
    lora_config_creator=None, oneframe_mc_data=None):
    """
    worker関数初期化・セットアップ統合のメイン関数
    
    Args:
        [worker関数のすべてのパラメータ]
        
    Returns:
        tuple: (job_id, outputs_folder, use_cached_files, current_lora_paths, current_lora_scales,
                transformer, input_image_np, input_image_pt, height, width, start_latent,
                sample_num_frames, num_frames, reference_latent, reference_encoder_output)
    """
    # グローバル変数宣言・初期設定処理
    (job_id, outputs_folder, use_cached_files) = _setup_basic_configuration(
        output_dir, webui_folder, use_cache_files, stream, use_queue, queue_type,
        prompt_queue_file_path, image_queue_files, batch_index
    )
    
    # LoRA設定・transformer状態管理処理
    (current_lora_paths, current_lora_scales, transformer) = _process_lora_and_transformer_setup(
        use_lora, lora_scales_text, lora_mode, transformer_manager, high_vram, fp8_optimization, lora_config_creator
    )
    
    # 入力画像処理・VAEエンコーディング処理
    (input_image_np, input_image_pt, height, width, start_latent) = _process_input_image_and_vae_encoding(
        input_image, resolution, stream, vae, high_vram, gpu
    )
    
    # 1フレームモード設定・参照画像準備処理（1f-mc対応）
    (sample_num_frames, num_frames, reference_latent, reference_encoder_output, reference_mask_from_alpha) = _setup_oneframe_mode_and_reference_image(
        use_reference_image, reference_image, width, height, vae, high_vram, gpu, stream, oneframe_mc_data
    )
    
    # 1フレームモード固有の設定
    total_latent_sections = 1  # 1セクションに固定
    frame_count = 1  # 1フレームモード
    
    return (job_id, outputs_folder, use_cached_files, current_lora_paths, current_lora_scales,
            transformer, input_image_np, input_image_pt, height, width, start_latent,
            sample_num_frames, num_frames, reference_latent, reference_encoder_output, reference_mask_from_alpha,
            total_latent_sections, frame_count)


def _setup_basic_configuration(output_dir, webui_folder, use_cache_files, stream, use_queue, queue_type,
                              prompt_queue_file_path, image_queue_files, batch_index):
    """
    基本設定・初期化処理
    """
    # キュー状態のログ出力
    use_queue_flag = bool(use_queue)
    queue_type_flag = queue_type
    if use_queue_flag:
        print(translate("キュー状態: {0}, タイプ: {1}").format(use_queue_flag, queue_type_flag))

        if queue_type_flag == "prompt" and prompt_queue_file_path is not None:
            print(translate("プロンプトキューファイルパス: {0}").format(prompt_queue_file_path))

        elif queue_type_flag == "image" and len(image_queue_files) > 0:
            print(translate("イメージキュー詳細: 画像数={0}, batch_index={1}").format(
                len(image_queue_files), batch_index))

    job_id = generate_timestamp()
    
    # 1フレームモード固有の設定
    total_latent_sections = 1  # 1セクションに固定
    frame_count = 1  # 1フレームモード
    
    # 出力フォルダの設定
    if output_dir:
        outputs_folder = output_dir
    else:
        # 出力フォルダはwebui内のoutputsに固定
        outputs_folder = os.path.join(webui_folder, 'outputs')
    
    os.makedirs(outputs_folder, exist_ok=True)
    
    # プログレスバーの初期化
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    
    # モデルや中間ファイルなどのキャッシュ利用フラグ
    use_cached_files = use_cache_files
    
    return job_id, outputs_folder, use_cached_files


def _process_lora_and_transformer_setup(use_lora, lora_scales_text, lora_mode, transformer_manager, high_vram, fp8_optimization, lora_config_creator=None):
    """
    LoRA設定・transformer状態管理処理
    """
    # LoRA 設定処理（Stage 2: 分離最適化モジュール使用）
    from .oichi2_lora_core import process_lora_configuration_unified
    
    # lora_config_creatorが渡された場合は統合処理を使用
    if lora_config_creator and callable(lora_config_creator):
        print("LoRA統合処理を使用（可変対応）")
        try:
            lora_config = lora_config_creator()
            current_lora_paths, current_lora_scales = process_lora_configuration_unified(
                use_lora, True, lora_config
            )
        except Exception as e:
            print(f"LoRA統合処理エラー: {e}")
            current_lora_paths, current_lora_scales = [], []
    else:
        print("LoRA可変システムのみ対応")
        current_lora_paths, current_lora_scales = [], []
    
    # -------- LoRA 設定 START ---------
    # UI設定のuse_loraフラグ値を保存
    original_use_lora = use_lora

    # UIでLoRA使用が有効になっていた場合、ファイル選択に関わらず強制的に有効化
    if original_use_lora:
        use_lora = True

    # LoRA設定のみを更新
    transformer_manager.set_next_settings(
        lora_paths=current_lora_paths,
        lora_scales=current_lora_scales,
        high_vram_mode=high_vram,
        fp8_enabled=fp8_optimization,  # fp8_enabledパラメータを追加
        force_dict_split=True  # 常に辞書分割処理を行う
    )
    # -------- LoRA 設定 END ---------
    
    # LoRA設定後のtransformer状態確認とリロード
    print(translate("LoRA設定後のtransformer状態チェック..."))
    try:
        # transformerの状態を確認し、必要に応じてリロード
        if not transformer_manager.ensure_transformer_state():
            raise Exception(translate("transformer状態の確認に失敗しました"))
            
        # 最新のtransformerインスタンスを取得
        transformer = transformer_manager.get_transformer()
        print(translate("transformer状態チェック完了"))
    except Exception as e:
        print(translate("transformerのリロードに失敗しました: {0}").format(e))
        traceback.print_exc()
        raise Exception(translate("transformerのリロードに失敗しました"))
    
    return current_lora_paths, current_lora_scales, transformer


def _process_input_image_and_vae_encoding(input_image, resolution, stream, vae, high_vram, gpu):
    """
    入力画像処理・VAEエンコーディング処理
    """
    # 入力画像の処理（Stage 2: 分離最適化モジュール使用）
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))
    
    from .oichi2_image import process_input_image
    input_image_np, input_image_pt, height, width = process_input_image(input_image, resolution)
    
    # VAE エンコーディング
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
    
    try:
        # エンコード前のメモリ状態を記録
        free_mem_before_encode = get_cuda_free_memory_gb(gpu)
        print(translate("VAEエンコード前の空きVRAM: {0} GB").format(free_mem_before_encode))
        
        # VAEモデルのロード（未ロードの場合）
        from diffusers import AutoencoderKLHunyuanVideo
        if vae is None:
            print(translate("VAEモデルを初めてロードします..."))
            try:
                # ローカルキャッシュからVAEパスを取得
                vae_path = _get_local_vae_path("hunyuanvideo-community/HunyuanVideo", "vae")
                
                vae = AutoencoderKLHunyuanVideo.from_pretrained(
                    vae_path,
                    torch_dtype=torch.float16,
                    local_files_only=True
                ).cpu()
                _setup_vae_model(vae, high_vram, gpu)  # VAEの設定を適用
                print(translate("VAEモデルのロードが完了しました"))
            except Exception as e:
                print(translate("VAEモデルのロードに失敗しました: {0}").format(e))
                traceback.print_exc()
                print(translate("5秒間待機後に再試行します..."))
                time.sleep(5)
                # ローカルキャッシュからVAEパスを取得（再試行）
                vae_path = _get_local_vae_path("hunyuanvideo-community/HunyuanVideo", "vae")
                
                vae = AutoencoderKLHunyuanVideo.from_pretrained(
                    vae_path,
                    torch_dtype=torch.float16,
                    local_files_only=True
                ).cpu()
                _setup_vae_model(vae, high_vram, gpu)  # VAEの設定を適用
        
        # ハイVRAM以外では明示的にモデルをGPUにロード
        if not high_vram:
            print(translate("VAEモデルをGPUにロード..."))
            load_model_as_complete(vae, target_device=gpu)
        
        # VAEエンコード実行
        with torch.no_grad():  # 明示的にno_gradコンテキストを使用
            # 効率的な処理のために入力をGPUで処理
            input_image_gpu = input_image_pt.to(gpu)
            start_latent = vae_encode(input_image_gpu, vae)
            
            # 入力をCPUに戻す
            del input_image_gpu
            torch.cuda.empty_cache()
        
        # ローVRAMモードでは使用後すぐにCPUに戻す
        if not high_vram:
            vae.to('cpu')
            
            # メモリ状態をログ
            free_mem_after_encode = get_cuda_free_memory_gb(gpu)
            print(translate("VAEエンコード後の空きVRAM: {0} GB").format(free_mem_after_encode))
            print(translate("VAEエンコードで使用したVRAM: {0} GB").format(
                free_mem_before_encode - free_mem_after_encode))
            
            # メモリクリーンアップ
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(translate("VAEエンコードエラー: {0}").format(e))
        
        # エラー発生時のメモリ解放
        if 'input_image_gpu' in locals():
            del input_image_gpu
        torch.cuda.empty_cache()
        
        raise e
    
    return input_image_np, input_image_pt, height, width, start_latent


def _setup_oneframe_mode_and_reference_image(use_reference_image, reference_image, width, height, 
                                           vae, high_vram, gpu, stream, oneframe_mc_data=None):
    """
    1フレームモード設定・参照画像準備処理（1f-mc対応強化）
    """
    # 1フレームモード用の設定（sample_num_framesを早期に定義）
    sample_num_frames = 1  # 1フレームモード（one_frame_inferenceが有効なため）
    num_frames = sample_num_frames
    
    # Kisekaeichi機能: 参照画像の処理
    reference_latent = None
    reference_encoder_output = None
    reference_mask_from_alpha = None
    
    if use_reference_image and reference_image is not None:
        print(translate("着せ替え参照画像を処理します: {0}").format(reference_image))
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing reference image ...'))))
        
        try:
            # 参照画像をロード（musubi-tuner準拠のアルファチャンネル検出）
            ref_img = Image.open(reference_image)
            
            # RGBA画像のアルファチャンネルをマスクとして抽出（musubi-tuner準拠）
            if ref_img.mode == "RGBA":
                reference_mask_from_alpha = ref_img.split()[-1]  # アルファチャンネルを抽出
                print(translate("参照画像からアルファチャンネルマスクを検出しました"))
            else:
                reference_mask_from_alpha = None
                
            # RGB変換
            ref_img = ref_img.convert("RGB")
            ref_image_np = np.array(ref_img)
            
            if len(ref_image_np.shape) == 2:  # グレースケール画像の場合
                ref_image_np = np.stack((ref_image_np,) * 3, axis=-1)
            
            # 同じサイズにリサイズ（入力画像と同じ解像度を使用）
            ref_image_np = resize_and_center_crop(ref_image_np, target_width=width, target_height=height)
            ref_image_pt = torch.from_numpy(ref_image_np).float() / 127.5 - 1
            ref_image_pt = ref_image_pt.permute(2, 0, 1)[None, :, None]
            
            # VAEエンコード（参照画像）
            from diffusers import AutoencoderKLHunyuanVideo
            if vae is None or not high_vram:
                # ローカルキャッシュからVAEパスを取得
                vae_path = _get_local_vae_path("hunyuanvideo-community/HunyuanVideo", "vae")
                
                vae = AutoencoderKLHunyuanVideo.from_pretrained(
                    vae_path, 
                    torch_dtype=torch.float16,
                    local_files_only=True
                ).cpu()
                _setup_vae_model(vae, high_vram, gpu)
                load_model_as_complete(vae, target_device=gpu)
            
            with torch.no_grad():  # 明示的にno_gradコンテキストを使用
                ref_image_gpu = ref_image_pt.to(gpu)
                reference_latent = vae_encode(ref_image_gpu, vae)
                del ref_image_gpu
            
            if not high_vram:
                vae.to('cpu')
            
            print(translate("参照画像の処理が完了しました"))
            
        except Exception as e:
            print(translate("参照画像の処理中にエラーが発生しました: {0}").format(e))
            traceback.print_exc()
            # エラーが発生した場合は参照画像を使用せずに続行
            reference_latent = None
            reference_encoder_output = None
    
    # 1f-mc制御画像のVAEエンコード処理
    if oneframe_mc_data and oneframe_mc_data.get('valid_count', 0) > 0:
        print(translate("1f-mc制御画像のVAEエンコードを開始"))
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Encoding 1f-mc control images ...'))))
        
        try:
            control_images = oneframe_mc_data.get('control_images', [])
            control_latents = []
            
            # VAEの準備
            from diffusers import AutoencoderKLHunyuanVideo
            if vae is None or not high_vram:
                # ローカルキャッシュからVAEパスを取得
                vae_path = _get_local_vae_path("hunyuanvideo-community/HunyuanVideo", "vae")
                
                vae = AutoencoderKLHunyuanVideo.from_pretrained(
                    vae_path, 
                    torch_dtype=torch.float16,
                    local_files_only=True
                ).cpu()
                _setup_vae_model(vae, high_vram, gpu)
                load_model_as_complete(vae, target_device=gpu)
            
            for i, img_path in enumerate(control_images):
                if img_path and isinstance(img_path, str) and img_path.strip() and os.path.exists(img_path):
                    try:
                        # 制御画像をロード
                        ctrl_img = Image.open(img_path)
                        
                        # アルファチャンネル処理（musubi-tuner準拠強化）
                        alpha_mask = None
                        if ctrl_img.mode == "RGBA":
                            print(translate("制御画像{0}にアルファチャンネルを検出し、マスクとして保存").format(i+1))
                            # アルファチャンネルをマスクとして抽出（musubi-tuner準拠）
                            alpha_mask = ctrl_img.split()[-1]  # アルファチャンネルを取得
                            # oneframe_mc_dataにアルファマスクを保存
                            if 'control_alpha_masks' in oneframe_mc_data:
                                oneframe_mc_data['control_alpha_masks'][i] = alpha_mask
                        else:
                            print(translate("制御画像{0}はアルファチャンネルなし（RGBまたはLモード）").format(i+1))
                        
                        # RGB変換（アルファチャンネルは別途保存済み）
                        ctrl_img = ctrl_img.convert("RGB")
                        ctrl_image_np = np.array(ctrl_img)
                        
                        if len(ctrl_image_np.shape) == 2:  # グレースケール画像の場合
                            ctrl_image_np = np.stack((ctrl_image_np,) * 3, axis=-1)
                        
                        # 同じサイズにリサイズ
                        ctrl_image_np = resize_and_center_crop(ctrl_image_np, target_width=width, target_height=height)
                        ctrl_image_pt = torch.from_numpy(ctrl_image_np).float() / 127.5 - 1
                        ctrl_image_pt = ctrl_image_pt.permute(2, 0, 1)[None, :, None]
                        
                        # VAEエンコード
                        with torch.no_grad():
                            ctrl_image_gpu = ctrl_image_pt.to(gpu)
                            ctrl_latent = vae_encode(ctrl_image_gpu, vae)
                            
                            # アルファマスク適用（musubi-tuner準拠）
                            if alpha_mask is not None:
                                print(translate("制御画像{0}にアルファマスクを適用").format(i+1))
                                # アルファマスクをlatentサイズにリサイズ
                                mask_resized = alpha_mask.resize((width // 8, height // 8), Image.LANCZOS)
                                mask_np = np.array(mask_resized).astype(np.float32) / 255.0  # 0-1に正規化
                                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,H,W]
                                mask_tensor = mask_tensor.to(ctrl_latent.device, dtype=ctrl_latent.dtype)
                                # latentにマスクを適用（アルファ=0の部分をゼロ化）
                                ctrl_latent = ctrl_latent * mask_tensor
                                print(translate("アルファマスク適用完了: 透明部分をゼロ化"))
                            
                            control_latents.append(ctrl_latent.cpu())
                            del ctrl_image_gpu
                            
                        print(translate("制御画像{0}のエンコード完了: {1}").format(i+1, os.path.basename(img_path)))
                        
                    except Exception as e:
                        print(translate("制御画像{0}のエンコードエラー: {1}").format(i+1, e))
                        # エラーの場合はゼロlatentを追加
                        zero_latent = torch.zeros(1, 16, 1, height // 8, width // 8, dtype=torch.float32)
                        control_latents.append(zero_latent)
            
            # エンコード済みlatentをoneframe_mc_dataに追加
            oneframe_mc_data['control_latents'] = control_latents
            oneframe_mc_data['encoded_count'] = len(control_latents)
            
            if not high_vram:
                vae.to('cpu')
            
            print(translate("1f-mc制御画像のVAEエンコード完了: {0}個").format(len(control_latents)))
            
        except Exception as e:
            print(translate("1f-mc制御画像エンコードエラー: {0}").format(e))
            traceback.print_exc()
    
    return sample_num_frames, num_frames, reference_latent, reference_encoder_output, reference_mask_from_alpha


def generate_timestamp():
    """
    タイムスタンプ生成（ユーティリティ関数）
    """
    import datetime
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]