"""
FramePack-oichi2 VAE・画像処理統合モジュール

VAE管理・デコード・画像保存処理の統合モジュール
- VAE再ロード・メモリ管理
- デコード処理・画像変換
- 画像保存・メタデータ埋め込み
- エラーハンドリング・メモリクリーンアップ
"""

import os
import time
import traceback

import einops
import numpy as np
import torch
from PIL import Image

from common_utils.png_metadata import PROMPT_KEY, SEED_KEY, embed_metadata_to_png
from diffusers_helper.hunyuan import vae_decode
from diffusers_helper.memory import (
    get_cuda_free_memory_gb,
    load_model_as_complete,
    offload_model_from_device_for_memory_preservation,
)
from locales.i18n_extended import translate


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


def process_vae_decode_and_save(generated_latents, history_latents, total_generated_latent_frames,
                               transformer, vae, gpu, high_vram, outputs_folder, job_id, 
                               prompt, seed, stream):
    """
    VAE・画像処理統合のメイン関数
    
    Args:
        generated_latents: 生成されたlatents
        history_latents: 履歴latents
        total_generated_latent_frames: 総生成フレーム数
        transformer: Transformerモデル
        vae: VAEモデル
        gpu: GPU設定
        high_vram: ハイVRAMモード
        outputs_folder: 出力フォルダ
        job_id: ジョブID
        prompt: プロンプト
        seed: シード値
        stream: ストリーム
        
    Returns:
        str: 保存された画像ファイルパス
    """
    try:
        # 生成完了後のメモリ最適化 - 軽量な処理に変更
        if not high_vram:
            # transformerのメモリを軽量に解放（辞書リセットなし）
            print(translate("生成完了 - transformerをアンロード中..."))

            # 元の方法に戻す - 軽量なオフロードで速度とメモリのバランスを取る
            offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)

            # アンロード後のメモリ状態をログ
            free_mem_gb_after_unload = get_cuda_free_memory_gb(gpu)
            print(translate("transformerアンロード後の空きVRAM {0} GB").format(free_mem_gb_after_unload))
            
            vae = _reload_vae_if_needed(vae, gpu)
        
        # 実際に使用するラテントを抽出
        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
        
        # latentsをGPUに移動してデコード
        real_history_latents_gpu = real_history_latents.to(device=gpu, dtype=torch.float16)
        
        # VAEデコード実行
        decoded_image = _perform_vae_decode(vae, real_history_latents_gpu, gpu)
        
        # 画像変換・保存処理
        output_filename = _convert_and_save_image(decoded_image, outputs_folder, job_id, prompt, seed)
        
        # メモリクリーンアップ
        _cleanup_decode_memory(real_history_latents_gpu, real_history_latents, decoded_image)
        
        # 結果をストリームに送信
        stream.output_queue.push(('file', output_filename))
        
        return output_filename
        
    except Exception as e:
        print(translate("1フレームの画像保存中にエラーが発生しました: {0}").format(e))
        traceback.print_exc()
        
        # エラー発生時のメモリ解放を試みる
        _emergency_memory_cleanup()
        
        return None


def _reload_vae_if_needed(vae, gpu):
    """
    VAE再ロード処理
    
    Returns:
        VAEモデル
    """
    from diffusers import AutoencoderKLHunyuanVideo
    from common_utils.vae_settings import apply_vae_settings
    
    if vae is None:
        print(translate("VAEモデルを再ロードします..."))
        try:
            # ローカルキャッシュからVAEパスを取得
            vae_path = _get_local_vae_path("hunyuanvideo-community/HunyuanVideo", "vae")
            
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                vae_path, 
                torch_dtype=torch.float16, 
                local_files_only=True
            ).cpu()
            apply_vae_settings(vae)  # VAEの設定を適用
            print(translate("VAEモデルの再ロードが完了しました"))
        except Exception as e:
            print(translate("VAEモデルの再ロードに失敗しました: {0}").format(e))
            traceback.print_exc()
            print(translate("再試行します..."))
            # ローカルキャッシュからVAEパスを取得（再試行）
            vae_path = _get_local_vae_path("hunyuanvideo-community/HunyuanVideo", "vae")
            
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                vae_path, 
                torch_dtype=torch.float16, 
                local_files_only=True
            ).cpu()
            apply_vae_settings(vae)  # VAEの設定を適用
    
    print(translate("VAEをGPUにロード中..."))
    load_model_as_complete(vae, target_device=gpu)
    
    # ロード後のメモリ状態をログ
    free_mem_gb_after_vae = get_cuda_free_memory_gb(gpu)
    print(translate("VAEロード後の空きVRAM {0} GB").format(free_mem_gb_after_vae))
    
    return vae


def _perform_vae_decode(vae, real_history_latents_gpu, gpu):
    """
    VAEデコード実行
    
    Returns:
        デコード結果
    """
    print(translate("VAEデコード開始..."))
    
    # デコード前のメモリ状態をログ
    free_mem_gb_before_decode = get_cuda_free_memory_gb(gpu)
    print(translate("VAEデコード前の空きVRAM {0} GB").format(free_mem_gb_before_decode))
    
    # VAEデコード
    with torch.no_grad():
        decoded_image = vae_decode(real_history_latents_gpu, vae)
    
    # デコード後のメモリ状態をログ
    free_mem_gb_after_decode = get_cuda_free_memory_gb(gpu)
    print(translate("VAEデコード後の空きVRAM {0} GB").format(free_mem_gb_after_decode))
    
    return decoded_image


def _convert_and_save_image(decoded_image, outputs_folder, job_id, prompt, seed):
    """
    画像変換・保存処理
    
    Returns:
        保存されたファイルパス
    """
    # 単一フレームを抽出
    frame = decoded_image[0, :, 0, :, :]
    frame = torch.clamp(frame, -1., 1.) * 127.5 + 127.5
    frame = frame.detach().cpu().to(torch.uint8)
    frame = einops.rearrange(frame, 'c h w -> h w c').numpy()
    
    # メタデータを設定
    metadata = {
        PROMPT_KEY: prompt,
        SEED_KEY: seed  # intとして保存
    }
    
    # 画像として保存（メタデータ埋め込み）
    output_filename = os.path.join(outputs_folder, f'{job_id}_oneframe.png')
    pil_img = Image.fromarray(frame)
    pil_img.save(output_filename)  # 一度保存
    
    # メタデータを埋め込み
    try:
        # 関数は2つの引数しか取らないので修正
        embed_metadata_to_png(output_filename, metadata)
        print(translate("画像メタデータを埋め込みました"))
    except Exception as e:
        print(translate("メタデータ埋め込みエラー: {0}").format(e))
    
    print(translate("1フレーム画像を保存しました: {0}").format(output_filename))
    
    return output_filename


def _cleanup_decode_memory(*objects):
    """
    デコードメモリクリーンアップ
    """
    # デコード結果を解放
    for obj in objects:
        if obj is not None:
            try:
                del obj
            except:
                pass


def _emergency_memory_cleanup():
    """
    緊急時メモリクリーンアップ
    """
    import gc
    
    # ローカル変数の確認と削除
    cleanup_vars = ['real_history_latents_gpu', 'real_history_latents', 'decoded_image']
    
    # フレーム情報を取得してローカル変数をクリーンアップ
    frame = None
    try:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            frame = frame.f_back  # 呼び出し元のフレーム
            locals_dict = frame.f_locals
            
            for var_name in cleanup_vars:
                if var_name in locals_dict:
                    try:
                        del locals_dict[var_name]
                        print(translate("{0} を緊急削除しました").format(var_name))
                    except:
                        pass
    except:
        pass
    finally:
        del frame
    
    # 強制的なガベージコレクション
    gc.collect()
    torch.cuda.empty_cache()
    print(translate("緊急メモリクリーンアップ完了"))