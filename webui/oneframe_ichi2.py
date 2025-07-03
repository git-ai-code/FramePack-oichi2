# === 標準ライブラリ ===
import argparse
import asyncio
import glob
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Optional, Union, List, Tuple

# === サードパーティライブラリ ===
import einops
import gradio as gr
import numpy as np
import safetensors.torch as sf
import torch
import yaml
from diffusers import AutoencoderKLHunyuanVideo
from PIL import Image
from transformers import (
    LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, 
    SiglipImageProcessor, SiglipVisionModel
)

# === diffusers_helper モジュール ===
from diffusers_helper.bucket_tools import find_nearest_bucket, get_available_resolutions
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.memory import (
    cpu, gpu, gpu_complete_modules, get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.utils import (
    save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop,
    state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
)

# === バージョン管理 ===
from version import get_version, get_app_name, get_full_name

# === common_utils ===
from common_utils.hf_config import get_model_cache_base_path, verify_model_cache_directory
from common_utils.log_manager import (
    enable_logging, disable_logging, is_logging_enabled, 
    get_log_folder, set_log_folder, open_log_folder,
    get_default_log_settings, load_log_settings, apply_log_settings
)
from common_utils.lora_config import get_max_lora_count, create_lora_config
from common_utils.lora_preset_manager import (
    initialize_lora_presets, load_lora_presets, save_lora_preset,
    load_lora_preset, get_preset_names
)
from common_utils.png_metadata import (
    embed_metadata_to_png, extract_metadata_from_png,
    PROMPT_KEY, SEED_KEY, SECTION_PROMPT_KEY, SECTION_NUMBER_KEY
)
from common_utils.preset_manager import get_default_startup_prompt, load_presets, save_preset, delete_preset
from common_utils.settings_manager import (
    get_settings_file_path, get_output_folder_path, initialize_settings,
    load_settings, save_settings, open_output_folder,
    load_app_settings_oichi, save_app_settings_oichi
)
from common_utils.ui_styles import get_app_css

# === パス設定 ===
sys.path.append(os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './submodules/FramePack'))))

# === コマンドライン引数解析 ===
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--lang", type=str, default='ja', help="Language: ja, zh-tw, en, ru")
args = parser.parse_args()

# === 国際化設定 ===
from locales.i18n_extended import (set_lang, translate)
set_lang(args.lang)

# === Windows環境最適化設定 ===
if sys.platform in ('win32', 'cygwin'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# === oichi2_utils ===
from oichi2_utils.oichi2_advanced_control_processor import process_advanced_control_integration
from oichi2_utils.oichi2_error_handling import (
    handle_worker_exception_and_cleanup, cleanup_worker_variables_in_caller_scope
)
from oichi2_utils.oichi2_event_handlers import (
    setup_preset_event_handlers, setup_metadata_event_handlers,
    setup_folder_operation_event_handlers, setup_settings_event_handlers,
    create_input_parameters_list
)
from oichi2_utils.oichi2_file_utils import (
    open_folder, get_image_queue_files, update_input_folder_name,
    open_input_folder_with_save, setup_folder_structure, setup_output_folder,
    sanitize_filename, ensure_directory_exists, get_file_list_by_extension
)
from oichi2_utils.oichi2_final_mask_processor import apply_final_masks
from oichi2_utils.oichi2_image import process_input_image
from oichi2_utils.oichi2_initialization import process_worker_initialization_and_setup
from oichi2_utils.oichi2_latent_core import process_latent_core_logic
from oichi2_utils.oichi2_lora_handlers import setup_lora_preset_event_handlers
from oichi2_utils.oichi2_queue_core import (
    setup_queue_configuration, generate_batch_summary, apply_queue_to_batch, get_batch_info_message
)
from oichi2_utils.oichi2_queue_ui import (
    create_toggle_queue_settings_handler, create_toggle_queue_type_handler
)
from oichi2_utils.oichi2_sampling import process_sampling_execution_and_postprocessing
from oichi2_utils.oichi2_settings import (
    handle_open_folder_btn, update_from_image_metadata, check_metadata_on_checkbox_change
)
from oichi2_utils.oichi2_system_initialization import (
    initialize_system_configuration, reload_transformer_if_needed,
    initialize_models_and_tokenizers, _get_local_image_encoder_path,
    setup_vae_if_loaded, setup_image_encoder_if_loaded,
    initialize_folder_and_settings_configuration, initialize_global_variables,
    create_image_queue_files_wrapper
)
from oichi2_utils.oichi2_text_encoder import process_text_encoding, apply_final_processing
from oichi2_utils.oichi2_transformer import process_transformer_initialization_and_preparation
from oichi2_utils.oichi2_ui_handlers import (
    create_save_app_settings_handler, create_reset_app_settings_handler,
    set_random_seed, randomize_seed_if_needed,
    create_save_preset_handler, create_load_preset_handler, create_delete_preset_handler,
    clear_fields, apply_to_prompt, update_resolution_info,
    handle_open_folder_btn_wrapper, update_from_image_metadata_wrapper,
    check_metadata_on_checkbox_change_wrapper, update_input_folder_wrapper,
    open_input_folder_wrapper, load_preset_handler_wrapper
)
from oichi2_utils.oichi2_ui_utilities import (
    toggle_kisekaeichi_settings, scan_lora_directory, toggle_lora_settings,
    toggle_lora_mode, update_lora_dropdowns, toggle_lora_full_update,
    save_lora_count_setting_handler, sync_metadata_checkboxes, 
    toggle_advanced_control, toggle_advanced_control_mode, update_scales_text, is_port_in_use
)
from oichi2_utils.oichi2_vae import process_vae_decode_and_save

# === グローバル変数定義 ===
# 処理制御フラグ
user_abort: bool = False
user_abort_notified: bool = False
batch_stopped: bool = False

# キャッシュ変数
cached_prompt: Optional[str] = None
cached_n_prompt: Optional[str] = None
cached_llama_vec: Optional[Any] = None
cached_llama_vec_n: Optional[Any] = None
cached_clip_l_pooler: Optional[Any] = None
cached_clip_l_pooler_n: Optional[Any] = None
cached_llama_attention_mask: Optional[Any] = None
cached_llama_attention_mask_n: Optional[Any] = None

# キュー機能関連
queue_enabled: bool = False
queue_type: str = "prompt"
prompt_queue_file_path: Optional[str] = None
image_queue_files: List[str] = []

# システム設定
use_cache_files = True
first_run = True


# === 初期化処理 ===
# ポート使用チェック・初回起動判定
if is_port_in_use(args.port):
    print(translate("警告: ポート {0} はすでに使用されています。他のインスタンスが実行中かもしれません。").format(args.port))
    print(translate("10秒後に処理を続行します..."))
    first_run = False  # 初回実行ではない
    time.sleep(10) # 10秒待機して続行

# === 追加モジュールインポート・初期化 ===
# Windows音声通知サポート
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# HuggingFace設定
model_cache_base = get_model_cache_base_path()
verify_model_cache_directory()

# LoRA・FP8最適化サポート
has_lora_support = False
has_fp8_support = False
try:
    import lora2_utils
    from lora2_utils.fp8_optimization_utils import check_fp8_support, apply_fp8_monkey_patch
    
    has_lora_support = True
    has_e4m3, has_e5m2, has_scaled_mm = check_fp8_support()
    has_fp8_support = has_e4m3 and has_e5m2
    
    if has_fp8_support:
        pass  # FP8最適化利用可能
    else:
        print(translate("LoRAサポートが有効です（FP8最適化はサポートされていません）"))
except ImportError:
    print(translate("LoRAサポートが無効です（lora2_utilsモジュールがインストールされていません）"))

# === システム・モデル初期化 ===
system_config = initialize_system_configuration(gpu)
free_mem_gb = system_config['free_mem_gb']
high_vram = system_config['high_vram']
transformer_manager = system_config['transformer_manager']
text_encoder_manager = system_config['text_encoder_manager']

models = initialize_models_and_tokenizers()
tokenizer = models['tokenizer']
tokenizer_2 = models['tokenizer_2']
feature_extractor = models['feature_extractor']
vae = models['vae']
text_encoder = models['text_encoder']
text_encoder_2 = models['text_encoder_2']
transformer = models['transformer']
image_encoder = models['image_encoder']

webui_folder = os.path.dirname(os.path.abspath(__file__))
folder_config = initialize_folder_and_settings_configuration(webui_folder)
stream = folder_config['stream']
base_path = folder_config['base_path']
settings_folder = folder_config['settings_folder']
app_settings = folder_config['app_settings']
output_folder_name = folder_config['output_folder_name']
outputs_folder = folder_config['outputs_folder']
log_settings = folder_config['log_settings']

global_vars = initialize_global_variables()
g_frame_size_setting = global_vars['g_frame_size_setting']
batch_stopped = global_vars['batch_stopped']
queue_enabled = global_vars['queue_enabled']
queue_type = global_vars['queue_type']
prompt_queue_file_path = global_vars['prompt_queue_file_path']
image_queue_files = global_vars['image_queue_files']
input_folder_name_value = global_vars['input_folder_name_value']

get_image_queue_files_wrapper = create_image_queue_files_wrapper(base_path, input_folder_name_value)

@torch.no_grad()
def worker(input_image: Optional[str], prompt: str, n_prompt: str, seed: int, steps: int, 
           cfg: float, gs: float, rs: float, gpu_memory_preservation: int, use_teacache: bool, 
           lora_scales_text: Optional[str] = None, output_dir: Optional[str] = None, 
           use_lora: bool = False, fp8_optimization: bool = False, resolution: int = 640,
           latent_window_size: int = 9, latent_index: int = 5, use_clean_latents_2x: bool = True, 
           use_clean_latents_4x: bool = True, use_clean_latents_post: bool = True,
           lora_mode: Optional[str] = None, batch_index: Optional[int] = None, 
           use_queue: bool = False, prompt_queue_file: Optional[str] = None,
           clean_index: int = 13, input_mask: Optional[str] = None, reference_mask: Optional[str] = None,
           use_advanced_control: bool = False, advanced_control_mode: str = "one_frame",
           kisekaeichi_reference_image: Optional[str] = None, kisekaeichi_control_index: int = 10, 
           oneframe_mc_image: Optional[str] = None, oneframe_mc_control_index: int = 1, 
           optional_control_image: Optional[str] = None, optional_control_index: int = 5, 
           save_settings_on_start: bool = False, alarm_on_completion: bool = False, 
           lora_config_creator: Optional[Any] = None) -> None:
    
    global vae, text_encoder, text_encoder_2, transformer, image_encoder
    global queue_enabled, queue_type, prompt_queue_file_path, image_queue_files
    global cached_prompt, cached_n_prompt, cached_llama_vec, cached_llama_vec_n, cached_clip_l_pooler, cached_clip_l_pooler_n
    global cached_llama_attention_mask, cached_llama_attention_mask_n

    if lora_scales_text is None:
        max_lora_count = get_max_lora_count()
        lora_scales_text = ",".join([str(0.8)] * max_lora_count)

    # モード別制御画像前処理
    use_reference_image, reference_image, oneframe_mc_data, compatibility_mode, enhanced_params = process_advanced_control_integration(
        use_advanced_control, kisekaeichi_reference_image, kisekaeichi_control_index,
        oneframe_mc_image, oneframe_mc_control_index, 
        optional_control_image, optional_control_index,
        input_image, latent_window_size, latent_index, clean_index, advanced_control_mode
    )
    
    try:
        # 変数の初期化（安全性確保）
        reference_mask_from_alpha = None
        
        (job_id, outputs_folder, use_cached_files, current_lora_paths, current_lora_scales,
         transformer, input_image_np, input_image_pt, height, width, start_latent,
         sample_num_frames, num_frames, reference_latent, reference_encoder_output, reference_mask_from_alpha,
         total_latent_sections, frame_count) = process_worker_initialization_and_setup(
            input_image, prompt, n_prompt, seed, steps, cfg, gs, rs,
            gpu_memory_preservation, use_teacache, lora_scales_text,
            output_dir, use_lora, fp8_optimization, resolution,
            latent_window_size, latent_index, use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post,
            lora_mode,
            batch_index, use_queue, prompt_queue_file,
            use_reference_image, reference_image, clean_index, input_mask, reference_mask,
            vae, text_encoder, text_encoder_2, transformer, image_encoder,
            queue_enabled, queue_type, prompt_queue_file_path, image_queue_files,
            cached_prompt, cached_n_prompt, cached_llama_vec, cached_llama_vec_n,
            cached_clip_l_pooler, cached_clip_l_pooler_n, cached_llama_attention_mask, cached_llama_attention_mask_n,
            transformer_manager, high_vram, gpu, webui_folder, use_cache_files, stream,
            lora_config_creator=lora_config_creator, oneframe_mc_data=oneframe_mc_data
        )
        
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        
        try:
            if image_encoder is None:
                print(translate("画像エンコーダを初めてロードします..."))
                try:
                    # ローカルキャッシュから画像エンコーダーパスを取得
                    image_encoder_path = _get_local_image_encoder_path("lllyasviel/flux_redux_bfl", "image_encoder")
                    
                    image_encoder = SiglipVisionModel.from_pretrained(
                        image_encoder_path, 
                        torch_dtype=torch.float16, 
                        local_files_only=True
                    ).cpu()
                    setup_image_encoder_if_loaded(image_encoder, high_vram, gpu)  # 画像エンコーダの設定を適用
                    print(translate("画像エンコーダのロードが完了しました"))
                except Exception as e:
                    print(translate("画像エンコーダのロードに失敗しました: {0}").format(e))
                    print(translate("再試行します..."))
                    # ローカルキャッシュから画像エンコーダーパスを取得（再試行）
                    image_encoder_path = _get_local_image_encoder_path("lllyasviel/flux_redux_bfl", "image_encoder")
                    
                    image_encoder = SiglipVisionModel.from_pretrained(
                        image_encoder_path, 
                        torch_dtype=torch.float16, 
                        local_files_only=True
                    ).cpu()
                    setup_image_encoder_if_loaded(image_encoder, high_vram, gpu)
            
            if not high_vram:
                print(translate("画像エンコーダをGPUにロード..."))
                load_model_as_complete(image_encoder, target_device=gpu)
            
            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
            
            if not high_vram:
                image_encoder.to('cpu')
                
                free_mem_gb = get_cuda_free_memory_gb(gpu)
                print(translate("CLIP Vision エンコード後の空きVRAM {0} GB").format(free_mem_gb))
                
                torch.cuda.empty_cache()
        except Exception as e:
            print(translate("CLIP Vision エンコードエラー: {0}").format(e))
            raise e
        
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        
        llama_vec, clip_l_pooler, llama_vec_n, clip_l_pooler_n, llama_attention_mask, llama_attention_mask_n = process_text_encoding(
            prompt, n_prompt, cfg, transformer, text_encoder_manager, 
            tokenizer, tokenizer_2, high_vram, gpu, 
            queue_enabled, queue_type, batch_index, image_queue_files
        )
        
        use_cache = (llama_vec is not None and llama_vec_n is not None)
        (llama_vec, clip_l_pooler, llama_vec_n, clip_l_pooler_n, 
         llama_attention_mask, llama_attention_mask_n, image_encoder_last_hidden_state) = apply_final_processing(
            llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, 
            image_encoder_last_hidden_state, transformer, 
            text_encoder_manager, high_vram, use_cache
        )
        
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        
        rnd = torch.Generator("cpu").manual_seed(seed)
        
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32, device='cpu')
        history_pixels = None
        total_generated_latent_frames = 0
        
        latent_paddings = [0] * total_latent_sections
        
        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0  # 常にTrue
            latent_padding_size = latent_padding * latent_window_size  # 常に0
            
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return
            
            (clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices,
             clean_latents_4x, clean_latent_4x_indices, latent_indices) = process_latent_core_logic(
                latent_padding_size, latent_window_size, sample_num_frames, use_reference_image,
                latent_index, clean_index, start_latent, history_latents,
                height, width, use_clean_latents_post
            )
            
            # 拡張制御パラメータ統合
            transformer_extra_params = {}
            if enhanced_params:
                from oichi2_utils.oichi2_advanced_control_processor import apply_enhanced_control_processing
                transformer_extra_params = apply_enhanced_control_processing(enhanced_params, transformer_extra_params)
            
            (transformer, callback, image_encoder_last_hidden_state, clean_latents,
             clean_latents_2x, clean_latents_4x, clean_latent_indices, latent_indices) = process_transformer_initialization_and_preparation(
                transformer=transformer,
                transformer_manager=transformer_manager,
                high_vram=high_vram,
                gpu=gpu,
                gpu_memory_preservation=gpu_memory_preservation,
                use_teacache=use_teacache,
                steps=steps,
                stream=stream,
                clean_latents=clean_latents,
                clean_latents_2x=clean_latents_2x,
                clean_latents_4x=clean_latents_4x,
                clean_latent_indices=clean_latent_indices,
                latent_indices=latent_indices,
                use_reference_image=use_reference_image,
                reference_latent=reference_latent,
                reference_encoder_output=reference_encoder_output,
                latent_index=latent_index,
                clean_index=clean_index,
                latent_window_size=latent_window_size,
                sample_num_frames=sample_num_frames,
                input_mask=input_mask,
                reference_mask=reference_mask,
                image_encoder_last_hidden_state=image_encoder_last_hidden_state,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                image_encoder=image_encoder,
                vae=vae,
                oneframe_mc_data=oneframe_mc_data,
            )
            
            # モード別マスク制御
            if advanced_control_mode in ["one_frame", "1fmc"]:
                print(translate("{0}モード: マスク処理をスキップします").format(advanced_control_mode))
                effective_input_mask = None
                effective_reference_mask = None
                effective_reference_mask_from_alpha = None
            else:
                effective_input_mask = input_mask
                effective_reference_mask = reference_mask
                effective_reference_mask_from_alpha = reference_mask_from_alpha
            
            # 最終マスク処理を実行
            clean_latents = apply_final_masks(clean_latents, effective_input_mask, effective_reference_mask, effective_reference_mask_from_alpha)
            
            image_encoder = None
            vae = None
            torch.cuda.empty_cache()
            
            result = process_sampling_execution_and_postprocessing(
                sample_num_frames=sample_num_frames,
                clean_latent_indices=clean_latent_indices,
                clean_latents=clean_latents,
                use_reference_image=use_reference_image,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                use_clean_latents_2x=use_clean_latents_2x,
                use_clean_latents_4x=use_clean_latents_4x,
                input_image_np=input_image_np,
                width=width,
                height=height,
                use_cache=use_cache,
                transformer=transformer,
                cfg=cfg,
                gs=gs,
                rs=rs,
                steps=steps,
                rnd=rnd,
                llama_vec=llama_vec,
                llama_attention_mask=llama_attention_mask,
                clip_l_pooler=clip_l_pooler,
                llama_vec_n=llama_vec_n,
                llama_attention_mask_n=llama_attention_mask_n,
                clip_l_pooler_n=clip_l_pooler_n,
                gpu=gpu,
                image_encoder_last_hidden_state=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                callback=callback
            )
            
            if result is None:
                batch_stopped = True
                return None
            
            generated_latents, transformer, is_interrupted = result
            
            if is_interrupted:
                batch_stopped = True
                return None
            
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            
            output_filename = process_vae_decode_and_save(
                generated_latents=generated_latents,
                history_latents=history_latents,
                total_generated_latent_frames=total_generated_latent_frames,
                transformer=transformer,
                vae=vae,
                gpu=gpu,
                high_vram=high_vram,
                outputs_folder=outputs_folder,
                job_id=job_id,
                prompt=prompt,
                seed=seed,
                stream=stream
            )
            
            # 実行履歴を保存
            if output_filename:
                try:
                    # パラメータ辞書作成
                    parameters = {
                        "seed": seed,
                        "steps": steps,
                        "cfg": cfg,
                        "gs": gs,
                        "rs": rs,
                        "resolution": resolution,
                        "latent_window_size": latent_window_size,
                        "latent_index": latent_index,
                        "clean_index": clean_index,
                        "use_clean_latents_2x": use_clean_latents_2x,
                        "use_clean_latents_4x": use_clean_latents_4x,
                        "use_clean_latents_post": use_clean_latents_post,
                        "use_teacache": use_teacache,
                        "gpu_memory_preservation": gpu_memory_preservation,
                        "use_lora": use_lora,
                        "fp8_optimization": fp8_optimization,
                        "use_advanced_control": use_advanced_control
                    }
                    
                    # LoRA情報の収集
                    lora_info = collect_lora_info(use_lora, lora_config_creator)
                    
                    # 高度な画像制御情報の収集
                    advanced_control_info = collect_advanced_control_info(
                        use_advanced_control=use_advanced_control,
                        advanced_control_mode=advanced_control_mode,
                        kisekaeichi_reference_image=kisekaeichi_reference_image,
                        kisekaeichi_control_index=kisekaeichi_control_index,
                        input_mask=input_mask,
                        reference_mask=reference_mask,
                        oneframe_mc_image=oneframe_mc_image,
                        oneframe_mc_control_index=oneframe_mc_control_index,
                        optional_control_image=optional_control_image,
                        optional_control_index=optional_control_index
                    )
                    
                    # 履歴エントリ追加
                    history_manager.add_history_entry(
                        input_image_path=input_image,
                        output_image_path=output_filename,
                        prompt=prompt,
                        negative_prompt=n_prompt,
                        parameters=parameters,
                        lora_info=lora_info,
                        advanced_control_info=advanced_control_info
                    )
                except Exception as e:
                    print(translate("履歴保存エラー: {0}").format(e))
            
            break  # 1フレーム生成は1回のみ
            
    except Exception as e:
        handle_worker_exception_and_cleanup(
            e=e,
            high_vram=high_vram,
            transformer=transformer,
            vae=vae,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer_manager=transformer_manager,
            gpu=gpu
        )
        
        cleanup_var_names = cleanup_worker_variables_in_caller_scope()
        local_vars = locals()
        for var_name in cleanup_var_names:
            if var_name in local_vars:
                try:
                    local_vars.pop(var_name, None)
                    print(translate("変数 {0} を解放しました").format(var_name))
                except Exception as e:
                    print(translate("変数解放でエラー: {0}").format(e))
    
    print(translate("処理が完了しました"))

    stream.output_queue.push(('end', None))
    return

def handle_open_folder_btn_wrapper(folder_name):
    """フォルダ名保存・開く"""
    from oichi2_utils.oichi2_ui_handlers import handle_open_folder_btn_wrapper as base_handler
    folder_update, path_update = base_handler(folder_name)
    
    if folder_name and folder_name.strip():
        global output_folder_name, outputs_folder
        output_folder_name = folder_name
        outputs_folder = get_output_folder_path(folder_name)
    
    return folder_update, path_update

def update_input_folder(folder_name):
    """入力フォルダ名更新"""
    global input_folder_name_value
    input_folder_name_value = update_input_folder_name(folder_name)
    return gr.update(value=input_folder_name_value)

def open_input_folder():
    """入力フォルダを開く"""
    global input_folder_name_value
    open_input_folder_with_save(input_folder_name_value, webui_folder, get_image_queue_files_wrapper)
    return None

def queue_settings_with_state_update(use_queue_val):
    """キュー設定状態更新"""
    global queue_enabled
    queue_enabled = bool(use_queue_val)
    return toggle_queue_settings(use_queue_val, queue_type)

def queue_type_with_state_update(queue_type_val):
    """キュータイプ状態更新"""
    global queue_type
    if queue_type_val == translate("プロンプトキュー"):
        queue_type = "prompt"
    else:
        queue_type = "image"
    return toggle_queue_type_handler(queue_type_val)

def process(input_image, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache,
            lora_scales_text, use_lora, fp8_optimization, resolution, output_directory=None,
            batch_count=1, use_random_seed=False, latent_window_size=9, latent_index=5,
            use_clean_latents_2x=True, use_clean_latents_4x=True, use_clean_latents_post=True,
            lora_mode=None,
            use_rope_batch=False, use_queue=False, prompt_queue_file=None,
            clean_index=13, input_mask=None, reference_mask=None,
            # kisekaeichi + 1f-mc統合パラメータ
            use_advanced_control=False, advanced_control_mode="one_frame",
            kisekaeichi_reference_image=None, kisekaeichi_control_index=10,
            oneframe_mc_image=None, oneframe_mc_control_index=1,
            optional_control_image=None, optional_control_index=5,
            save_settings_on_start=False, alarm_on_completion=True, *lora_args):
    global stream
    global batch_stopped, user_abort, user_abort_notified
    global queue_enabled, queue_type, prompt_queue_file_path, image_queue_files

    # 統合UIから従来パラメータへのマッピング
    use_reference_image = use_advanced_control and kisekaeichi_reference_image is not None
    reference_image = kisekaeichi_reference_image if use_reference_image else None
    
    # 可変LoRAパラメータからLoRAConfig作成関数を生成
    def create_lora_config_from_ui():
        max_count = get_max_lora_count()
        
        # lora_argsから可変LoRAパラメータを取得（3分割：ファイル、ドロップダウン、個別強度）
        third_point = len(lora_args) // 3
        lora_files_dynamic = list(lora_args[:third_point])
        lora_dropdowns_dynamic = list(lora_args[third_point:third_point*2])
        lora_strengths_dynamic = list(lora_args[third_point*2:])
        
        lora_config = create_lora_config(max_count)
        lora_config.set_mode(lora_mode)
        
        # モードに応じて適切なリストを設定
        if lora_mode == translate("ファイルアップロード"):
            lora_config.set_files(lora_files_dynamic)
        else:  # "ディレクトリから選択"モード
            lora_config.set_dropdowns(lora_dropdowns_dynamic)
        
        # 個別強度を設定
        if lora_strengths_dynamic:
            try:
                strength_values = [float(val) if val is not None else 0.8 for val in lora_strengths_dynamic]
                lora_config.set_individual_scales(strength_values)
            except:
                # エラー時はデフォルト値を使用
                lora_config.set_scales_text(lora_scales_text)
        else:
            # 個別強度がない場合は後方互換処理
            lora_config.set_scales_text(lora_scales_text)
        
        return lora_config

    # バッチ処理制御: ユーザー中断・自動停止・エラー時安全停止の3段階制御
    user_abort = False
    user_abort_notified = False
    
    batch_stopped = False

    try:
        batch_count_val = int(batch_count)
        batch_count = max(1, min(batch_count_val, 100))  # 1〜100の間に制限
    except (ValueError, TypeError):
        print(translate("バッチ処理回数が無効です。デフォルト値の1を使用します: {0}").format(batch_count))
        batch_count = 1  # デフォルト値
        
    (queue_enabled, queue_type, prompt_queue_file_path, batch_count, total_needed_batches, image_queue_files) = setup_queue_configuration(
        use_queue, prompt_queue_file, batch_count,
        queue_enabled, queue_type, prompt_queue_file_path, image_queue_files,
        get_image_queue_files_wrapper
    )
    
    global outputs_folder
    global output_folder_name
    if output_directory and output_directory.strip():
        outputs_folder = get_output_folder_path(output_directory)
        print(translate("出力フォルダを設定: {0}").format(outputs_folder))

        if output_directory != output_folder_name:
            settings = load_settings()
            settings['output_folder'] = output_directory
            if save_settings(settings):
                output_folder_name = output_directory
                print(translate("出力フォルダ設定を保存しました: {0}").format(output_directory))
    else:
        outputs_folder = get_output_folder_path(output_folder_name)
        print(translate("デフォルト出力フォルダを使用: {0}").format(outputs_folder))

    os.makedirs(outputs_folder, exist_ok=True)
    
    output_dir = outputs_folder
    
    batch_count = max(1, min(int(batch_count), 100))  # 1〜100の間に制限
    print(translate("バッチ処理回数: {0}回").format(batch_count))
    
    if input_image is None:
        print(translate("入力画像が指定されていません。デフォルトの画像を生成します。"))
    
    yield gr.skip(), None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update()
    
    batch_stopped = False
    user_abort = False
    user_abort_notified = False
    
    original_seed = seed
    
    use_random = False
    if isinstance(use_random_seed, bool):
        use_random = use_random_seed
    elif isinstance(use_random_seed, str):
        use_random = use_random_seed.lower() in ["true", "yes", "1", "on"]
    
    if use_random:
        previous_seed = seed
        seed = random.randint(0, 2**32 - 1)
        print(translate("ランダムシード機能が有効なため、指定されたSEED値 {0} の代わりに新しいSEED値 {1} を使用します。").format(previous_seed, seed))
        yield gr.skip(), None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update(value=seed)
        original_seed = seed
    else:
        if not seed:
            seed = 12345
        print(translate("指定されたSEED値 {0} を使用します。").format(seed))
        yield gr.skip(), None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update()
        original_seed = seed
    
    if save_settings_on_start and batch_count > 0:
        print(translate("=== 現在の設定を自動保存します ==="))
        current_settings = {
            'resolution': resolution,
            'steps': steps,
            'cfg': cfg,
            'use_teacache': use_teacache,
            'gpu_memory_preservation': gpu_memory_preservation,
            'gs': gs,
            'latent_window_size': latent_window_size,
            'latent_index': latent_index,
            'use_clean_latents_2x': use_clean_latents_2x,
            'use_clean_latents_4x': use_clean_latents_4x,
            'use_clean_latents_post': use_clean_latents_post,
            'clean_index': clean_index,
            'save_settings_on_start': save_settings_on_start,
            'alarm_on_completion': alarm_on_completion
        }
        
        if save_app_settings_oichi(current_settings):
            print(translate("アプリケーション設定を保存しました"))
        else:
            print(translate("アプリケーション設定の保存に失敗しました"))
    
    generate_batch_summary(queue_enabled, queue_type, prompt_queue_file_path, image_queue_files, batch_count)

    for batch_index in range(batch_count):
        if batch_stopped:
            print(translate("バッチ処理がユーザーによって中止されました"))
            yield (
                gr.skip(),
                gr.update(visible=False),
                translate("バッチ処理が中止されました。"),
                '',
                gr.update(interactive=True),
                gr.update(interactive=False, value=translate("End Generation")),
                gr.update()
            )
            break

        batch_info = get_batch_info_message(batch_count, batch_index)
        if batch_info:
            print(f"{batch_info}")
            yield gr.skip(), gr.update(visible=False), batch_info, "", gr.update(interactive=False), gr.update(interactive=True), gr.update()

        current_prompt, current_image = apply_queue_to_batch(
            queue_enabled, queue_type, prompt_queue_file_path, image_queue_files,
            batch_index, batch_count, prompt, input_image
        )

        current_seed = original_seed
        current_latent_window_size = latent_window_size
        
        if use_rope_batch:
            new_rope_value = latent_window_size + batch_index
            
            if new_rope_value > 32:
                print(translate("フレーム処理範囲が上限（32）に達したため、処理を終了します"))
                break
                
            current_latent_window_size = new_rope_value
            print(translate("フレーム処理範囲: {0}").format(current_latent_window_size))
        else:
            # キュー機能使用時はシード+1を無効化（通常のバッチ処理時のみ+1）
            if queue_enabled:
                current_seed = original_seed
                if batch_count > 1:
                    print(translate("キュー機能使用中: 初期シード値固定 {0}").format(current_seed))
            else:
                current_seed = original_seed + batch_index
                if batch_count > 1:
                    print(translate("初期シード値: {0}").format(current_seed))
        
        if batch_stopped:
            break
            
        try:
            stream = AsyncStream()
            
            batch_suffix = f"{batch_index}" if batch_index > 0 else ""
            
            if batch_stopped:
                break
                
            async_run(worker, current_image, current_prompt, n_prompt, current_seed, steps, cfg, gs, rs,
                     gpu_memory_preservation, use_teacache, lora_scales_text,
                     output_dir, use_lora, fp8_optimization, resolution,
                     current_latent_window_size, latent_index, use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post,
                     lora_mode,
                     batch_index, use_queue, prompt_queue_file,
                     clean_index, input_mask, reference_mask,
                     use_advanced_control, advanced_control_mode,
                     kisekaeichi_reference_image, kisekaeichi_control_index,
                     oneframe_mc_image, oneframe_mc_control_index,
                     optional_control_image, optional_control_index,
                     save_settings_on_start, alarm_on_completion,
                     create_lora_config_from_ui)
        except Exception as e:
            import traceback
        
        output_filename = None
        
        try:
            while True:
                try:
                    flag, data = stream.output_queue.next()
                    
                    if flag == 'file':
                        output_filename = data
                        yield (
                            output_filename if output_filename is not None else gr.skip(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(interactive=False),
                            gr.update(interactive=True),
                            gr.update(value=current_seed),
                        )
                    
                    if flag == 'progress':
                        preview, desc, html = data
                        yield gr.skip(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update()
                    
                    if flag == 'end':
                        if batch_index == batch_count - 1 or batch_stopped:  # 最後のバッチまたは中断された場合
                            completion_message = ""
                            if batch_stopped:
                                completion_message = translate("バッチ処理が中断されました（{0}/{1}）").format(batch_index + 1, batch_count)
                            else:
                                completion_message = translate("バッチ処理が完了しました（{0}/{1}）").format(batch_count, batch_count)
                            
                            yield (
                                output_filename if output_filename is not None else gr.skip(),
                                gr.update(visible=False),
                                completion_message,
                                '',
                                gr.update(interactive=True, value=translate("Start Generation")),
                                gr.update(interactive=False, value=translate("End Generation")),
                                gr.update(value=original_seed),
                            )
                        break
                        
                    if stream.input_queue.top() == 'end' or batch_stopped:
                        batch_stopped = True
                        print(translate("バッチ処理が中断されました（{0}/{1}）").format(batch_index + 1, batch_count))
                        yield (
                            output_filename if output_filename is not None else gr.skip(),
                            gr.update(visible=False),
                            translate("バッチ処理が中断されました"),
                            '',
                            gr.update(interactive=True),
                            gr.update(interactive=False, value=translate("End Generation")),
                            gr.update(value=original_seed),
                        )
                        return
                        
                except Exception as e:
                    import traceback
                    break
                    
        except KeyboardInterrupt:
            try:
                global transformer, text_encoder, text_encoder_2, vae, image_encoder
                if transformer is not None and hasattr(transformer, 'cpu'):
                    try:
                        transformer.cpu()
                    except: pass
                if text_encoder is not None and hasattr(text_encoder, 'cpu'):
                    try:
                        text_encoder.cpu()
                    except: pass
                if text_encoder_2 is not None and hasattr(text_encoder_2, 'cpu'):
                    try:
                        text_encoder_2.cpu()
                    except: pass
                if vae is not None and hasattr(vae, 'cpu'):
                    try:
                        vae.cpu()
                    except: pass
                if image_encoder is not None and hasattr(image_encoder, 'cpu'):
                    try:
                        image_encoder.cpu()
                    except: pass
                
                torch.cuda.empty_cache()
            except Exception as cleanup_e:
                pass
            
            yield None, gr.update(visible=False), translate("キーボード割り込みにより処理が中断されました"), '', gr.update(interactive=True, value=translate("Start Generation")), gr.update(interactive=False, value=translate("End Generation")), gr.update()
            return
        except Exception as e:
            import traceback
            yield None, gr.update(visible=False), translate("エラーにより処理が中断されました"), '', gr.update(interactive=True, value=translate("Start Generation")), gr.update(interactive=False, value=translate("End Generation")), gr.update()
            return
    
    if batch_stopped:
        if user_abort:
            print(translate("ユーザーの指示により処理を停止しました"))
        else:
            print(translate("バッチ処理が中断されました"))
    else:
        print(translate("全てのバッチ処理が完了しました"))
    
    batch_stopped = False
    user_abort = False
    user_abort_notified = False
    
    if HAS_WINSOUND and alarm_on_completion:
        try:
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
            print(translate("Windows完了通知音を再生しました"))
        except Exception as e:
            print(translate("完了通知音の再生に失敗しました: {0}").format(e))
    
    if batch_stopped or user_abort:
        print("-" * 50)
        print(translate("【ユーザー中断】処理は正常に中断されました - ") + time.strftime("%Y-%m-%d %H:%M:%S"))
        print("-" * 50)
    else:
        print("*" * 50)
        print(translate("【全バッチ処理完了】プロセスが完了しました - ") + time.strftime("%Y-%m-%d %H:%M:%S"))
        print("*" * 50)
            
    return

def end_process():
    """生成終了ボタンが押された時の処理"""
    global stream
    global batch_stopped, user_abort, user_abort_notified

    if not user_abort:
        batch_stopped = True
        user_abort = True
        
        print(translate("停止ボタンが押されました。処理を即座に中断します..."))
        user_abort_notified = True  # 通知フラグを設定
        
        stream.input_queue.push('end')

    return gr.update(value=translate("停止処理中..."))

# === UI設定・起動処理 ===
css = get_app_css()

saved_app_settings = load_app_settings_oichi()

if saved_app_settings:
    pass
else:
    print(translate("保存された設定が見つかりません。デフォルト値を使用します"))

# UI初期設定
use_random_seed_default = True
seed_default = random.randint(0, 2**32 - 1) if use_random_seed_default else 31337

# 動的解像度システム設定
available_resolutions = get_available_resolutions()
default_resolution = 640 if 640 in available_resolutions else available_resolutions[len(available_resolutions)//2]

# LoRA設定
max_lora_count = get_max_lora_count()

# === Gradio UI構築 ===
block = gr.Blocks(css=css).queue()
with block:
    gr.HTML(f'<h1>FramePack<span class="title-suffix">-oichi2 v{get_version()}</span></h1>')
    
    # 実行履歴システム初期化
    from oichi2_utils.oichi2_history_manager import HistoryManager, create_history_ui_components, setup_history_event_handlers, collect_lora_info, collect_advanced_control_info
    
    history_file_path = os.path.join(settings_folder, "execution_history.json")
    history_manager = HistoryManager(history_file_path, max_history_count=20)
    
    # 履歴UIコンポーネント作成
    history_ui = create_history_ui_components()
    
    # メインUIコンポーネントマッピング（パラメータ復元用）
    main_ui_components = {
        "seed": None, 
        "steps": None, 
        "cfg": None, 
        "gs": None, 
        "rs": None, 
        "resolution": None, 
        "latent_window_size": None, 
        "latent_index": None, 
        "clean_index": None, 
        "use_clean_latents_2x": None, 
        "use_clean_latents_4x": None, 
        "use_clean_latents_post": None, 
        "use_teacache": None, 
        "gpu_memory_preservation": None, 
        "use_lora": None, 
        "fp8_optimization": None, 
        "use_advanced_control": None, 
        "prompt": None 
    }
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(translate("**「1フレーム推論」モードでは、1枚の新しい未来の画像を生成します。**"))
            
            input_image = gr.Image(sources=['upload', 'clipboard'], type="filepath", label=translate("Image"), height=320)
            
            resolution = gr.Slider(
                label=translate("解像度レベル"),
                minimum=min(available_resolutions),
                maximum=max(available_resolutions),
                step=64,
                value=saved_app_settings.get("resolution", default_resolution) if saved_app_settings else default_resolution,
                info=translate("64刻みで設定可能（推奨640）"),
                elem_classes="saveable-setting"
            )
            
            # 解像度詳細情報表示
            resolution_info = gr.Markdown(
                value="",
                label=translate("解像度詳細"),
                elem_classes="resolution-info"
            )

            with gr.Column(scale=1):
                batch_count = gr.Slider(
                    label=translate("バッチ処理回数"),
                    minimum=1,
                    maximum=100,
                    value=1,
                    step=1,
                    info=translate("連続生成回数")
                )
                
                use_rope_batch = gr.Checkbox(
                    label=translate("フレーム処理範囲バッチ処理を使用"),
                    value=False,
                    info=translate("バッチごとにフレーム処理範囲+1（上限32、異なる処理効果を試行）")
                )

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(f"### " + translate("キュー機能"))

                        use_queue = gr.Checkbox(
                            label=translate("キュー機能を使用"),
                            value=False,
                            info=translate("入力ディレクトリの画像または指定したプロンプトリストを使用して連続して画像を生成します")
                        )

                        queue_type_selector = gr.Radio(
                            choices=[translate("プロンプトキュー"), translate("イメージキュー")],
                            value=translate("プロンプトキュー"),
                            label=translate("キュータイプ"),
                            visible=False,
                            interactive=True
                        )

                        with gr.Group(visible=False) as prompt_queue_group:
                            prompt_queue_file = gr.File(
                                label=translate("プロンプトキューファイル（.txt）- 1行に1つのプロンプトが記載されたテキストファイル"),
                                file_types=[".txt"]
                            )
                            gr.Markdown(translate("各行を別プロンプトとして処理"))

                        with gr.Group(visible=False) as image_queue_group:
                            gr.Markdown(translate("入力フォルダの画像を順次処理・同名txtファイル対応"))

                            with gr.Row():
                                input_folder_name = gr.Textbox(
                                    label=translate("入力フォルダ名"),
                                    value=input_folder_name_value,
                                    info=translate("入力画像ファイルを格納するフォルダ名")
                                )
                                open_input_folder_btn = gr.Button(value="📂 " + translate("保存及び入力フォルダを開く"), size="md")

                        input_folder_name.change(
                            fn=update_input_folder,
                            inputs=[input_folder_name],
                            outputs=[input_folder_name]
                        )

                        open_input_folder_btn.click(
                            fn=open_input_folder,
                            inputs=[],
                            outputs=[gr.Textbox(visible=False)]  # フィードバック表示用コンポーネント（非表示）
                        )

            with gr.Row():
                start_button = gr.Button(value=translate("Start Generation"))
                end_button = gr.Button(value=translate("End Generation"), interactive=False)

            with gr.Row():
                fp8_optimization = gr.Checkbox(
                    label=translate("FP8 最適化"),
                    value=True,
                    info=translate("メモリ使用量を削減し速度を改善（PyTorch 2.1以上が必要）")
                )

            global copy_metadata
            copy_metadata = gr.Checkbox(
                label=translate("埋め込みプロンプトおよびシードを複写する"),
                value=False,
                info=translate("チェックをオンにすると、画像のメタデータからプロンプトとシードを自動的に取得します"),
                visible=False  # 元の位置では非表示
            )
                
            extracted_info = gr.Markdown(visible=False)
            extracted_prompt = gr.Textbox(visible=False)
            extracted_seed = gr.Textbox(visible=False)
            
            # === フレーム位置設定 ===
            with gr.Group():
                gr.Markdown(f"### " + translate("フレーム設定"))
                
                # 生成・参照フレーム位置設定
                with gr.Row():
                    with gr.Column(scale=1):
                        latent_index = gr.Slider(
                            label=translate("生成フレーム位置（latent_index）"),
                            minimum=0,
                            maximum=16,
                            value=saved_app_settings.get("latent_index", 5) if saved_app_settings else 5,
                            step=1,
                            info=translate("生成フレーム位置・変化量制御（推奨5、LoRA指定値優先）"),
                            elem_classes="saveable-setting"
                        )
                    
                    with gr.Column(scale=1):
                        clean_index = gr.Slider(
                            label=translate("参照フレーム位置（clean_index）"),
                            minimum=0,
                            maximum=16,
                            value=saved_app_settings.get("clean_index", 13) if saved_app_settings else 13,
                            step=1,
                            info=translate("参照フレーム位置・品質安定化（推奨範囲13-16）"),
                            elem_classes="saveable-setting"
                        )
                
                # 高度な画像制御設定
                gr.Markdown(f'<span style="font-size: 1.2em; font-weight: bold;">{translate("kisekaeichi + 1f-mc設定")}</span>')
                
                use_advanced_control = gr.Checkbox(
                    label=translate("高度な画像制御を使用"),
                    value=False,
                    info=translate("Image（位置0）+ 制御画像による高精度合成")
                )
                
                # 制御モード選択
                advanced_control_mode = gr.Radio(
                    choices=[
                        (translate("1フレーム推論（基本生成）"), "one_frame"),
                        (translate("kisekaeichi（着せ替え・服装変更）"), "kisekaeichi"),
                        (translate("1f-mc（フレーム補間・動的変化）"), "1fmc"),
                        (translate("カスタム（全制御画像使用）"), "custom")
                    ],
                    value="one_frame",
                    label=translate("制御モード"),
                    visible=False,
                    info=translate("推奨値が自動設定されますが、必要に応じて調整可能です")
                )
                
                # モード説明
                with gr.Group(visible=False) as mode_info_group:
                    mode_description = gr.Markdown(
                        value=translate("**1フレーム推論**: 基本的な画像生成（推奨: latent_index=5） 💡 内部処理: 入力画像のみ使用、制御画像は処理で無視"),
                        visible=True
                    )
                
                # 高度制御UIブロック
                from oichi2_utils.oichi2_advanced_control_ui import create_advanced_control_ui
                advanced_control_group, advanced_control_ui_components = create_advanced_control_ui()
                
                # UIコンポーネントを個別変数に展開
                kisekaeichi_reference_image = advanced_control_ui_components["kisekaeichi_reference_image"]
                kisekaeichi_control_index = advanced_control_ui_components["kisekaeichi_control_index"]
                input_mask = advanced_control_ui_components["input_mask"]
                reference_mask = advanced_control_ui_components["reference_mask"]
                oneframe_mc_image = advanced_control_ui_components["oneframe_mc_image"]
                oneframe_mc_control_index = advanced_control_ui_components["oneframe_mc_control_index"]
                optional_control_image = advanced_control_ui_components["optional_control_image"]
                optional_control_index = advanced_control_ui_components["optional_control_index"]
                    
            # 統合制御設定の切り替え（画像部分のみ条件表示）
            use_advanced_control.change(
                fn=toggle_advanced_control,
                inputs=[use_advanced_control],
                outputs=[advanced_control_group, advanced_control_mode, mode_info_group]
            )
            
            # 制御モード変更時UI更新
            advanced_control_mode.change(
                fn=toggle_advanced_control_mode,
                inputs=[advanced_control_mode],
                outputs=[
                    latent_index,
                    mode_description
                ]
            )
            
            # レイテント設定UIブロック
            from oichi2_utils.oichi2_latent_ui import create_latent_settings_ui
            latent_settings_group, latent_ui_components = create_latent_settings_ui(saved_app_settings)
            
            # UIコンポーネントを個別変数に展開
            latent_window_size = latent_ui_components["latent_window_size"]
            use_clean_latents_2x = latent_ui_components["use_clean_latents_2x"]
            use_clean_latents_4x = latent_ui_components["use_clean_latents_4x"]
            use_clean_latents_post = latent_ui_components["use_clean_latents_post"]
            
            # 詳細設定をアコーディオンから解放して直接表示
            global previous_lora_mode
            if 'previous_lora_mode' not in globals():
                previous_lora_mode = translate("ディレクトリから選択") 
                
            if has_lora_support:
                with gr.Group() as lora_settings_group:
                    gr.Markdown(f"### " + translate("LoRA設定"))
                    
                    use_lora = gr.Checkbox(label=translate("LoRAを使用する"), value=False, info=translate("チェックをオンにするとLoRAを使用します（要12GB VRAM以上）"))

                lora_mode = gr.Radio(
                    choices=[translate("ディレクトリから選択"), translate("ファイルアップロード")],
                    value=translate("ディレクトリから選択"),
                    label=translate("LoRA読み込み方式"),
                    visible=False  # 初期状態では非表示（toggle_lora_settingsで制御）
                )

                # LoRAUI統合モジュール使用
                from oichi2_utils.oichi2_lora_ui import create_lora_main_ui_blocks, create_lora_preset_ui_block
                
                # LoRAメインUI構築
                main_ui_groups, main_ui_components = create_lora_main_ui_blocks(max_lora_count)
                lora_upload_group = main_ui_groups["lora_upload_group"]
                lora_dropdown_group = main_ui_groups["lora_dropdown_group"]
                lora_files_list = main_ui_components["lora_files_list"]
                lora_dropdowns_list = main_ui_components["lora_dropdowns_list"]
                lora_strength_list = main_ui_components["lora_strength_list"]
                lora_scan_button = main_ui_components["lora_scan_button"]
                lora_scales_text = main_ui_components["lora_scales_text"]
                
                # LoRAプリセットUI構築
                lora_preset_group, preset_components = create_lora_preset_ui_block(max_lora_count)
                
                # UIコンポーネント展開
                preset_buttons = preset_components["preset_buttons"]
                lora_load_btn = preset_components["lora_load_btn"]
                lora_save_btn = preset_components["lora_save_btn"]
                lora_preset_mode = preset_components["lora_preset_mode"]
                lora_preset_status = preset_components["lora_preset_status"]
                lora_count_setting = preset_components["lora_count_setting"]
                lora_count_save_btn = preset_components["lora_count_save_btn"]

                use_lora.change(
                    fn=toggle_lora_full_update,
                    inputs=[use_lora],
                    outputs=[lora_mode, lora_upload_group, lora_dropdown_group,
                             lora_scales_text, lora_preset_group] + lora_dropdowns_list
                )
                
                lora_mode.change(
                    fn=toggle_lora_mode,
                    inputs=[lora_mode],
                    outputs=[lora_upload_group, lora_dropdown_group, lora_preset_group] + lora_dropdowns_list
                )
                
                lora_scan_button.click(
                    fn=update_lora_dropdowns,
                    inputs=[],
                    outputs=lora_dropdowns_list  # すべてのドロップダウンを更新
                )
                
                # 個別強度入力の変更を後方互換テキストボックスに反映
                # 各個別強度入力にイベントハンドラーを設定
                for strength_input in lora_strength_list:
                    strength_input.change(
                        fn=update_scales_text,
                        inputs=lora_strength_list,
                        outputs=lora_scales_text
                    )

                gr.HTML("""<script>
                window.addEventListener('load', function() {
                    setTimeout(function() {
                        var btns = document.querySelectorAll('button');
                        for (var i = 0; i < btns.length; i++) {
                            if (btns[i].textContent.includes('LoRAフォルダを再スキャン')) {
                                btns[i].click(); break;
                            }
                        }
                    }, 1000);
                });
                </script>""")

                if not has_lora_support:
                    gr.Markdown(translate("LoRAサポートは現在無効です。lora2_utilsモジュールが必要です。"))
            else:
                use_lora = gr.Checkbox(visible=False, value=False)
                lora_mode = gr.Radio(visible=False, value=translate("ディレクトリから選択"))
                lora_upload_group = gr.Group(visible=False)
                lora_dropdown_group = gr.Group(visible=False)
                # 可変デフォルト値生成（プリセットシステムと整合性確保）
                default_scales = ",".join([str(0.8)] * max_lora_count)
                lora_scales_text = gr.Textbox(visible=False, value=default_scales)
                lora_preset_group = gr.Group(visible=False)

            copy_metadata_visible = gr.Checkbox(
                label=translate("埋め込みプロンプトおよびシードを複写する"),
                value=False,
                info=translate("チェックをオンにすると、画像のメタデータからプロンプトとシードを自動的に取得します")
            )

            copy_metadata_visible.change(
                fn=sync_metadata_checkboxes,
                inputs=[copy_metadata_visible],
                outputs=[copy_metadata]
            )

            copy_metadata.change(
                fn=sync_metadata_checkboxes,
                inputs=[copy_metadata],
                outputs=[copy_metadata_visible],
                queue=False  # 高速化のためキューをスキップ
            )

            prompt = gr.Textbox(label=translate("プロンプト"), value=get_default_startup_prompt(), lines=2)
            n_prompt = gr.Textbox(label=translate("ネガティブプロンプト"), value='', visible=False)
            
            # プロンプト管理UIブロック
            from oichi2_utils.oichi2_prompt_management_ui import create_prompt_management_ui
            prompt_management, prompt_ui_components = create_prompt_management_ui()
            
            # UIコンポーネントを個別変数に展開
            edit_name = prompt_ui_components["edit_name"]
            edit_prompt = prompt_ui_components["edit_prompt"]
            preset_dropdown = prompt_ui_components["preset_dropdown"]
            save_btn = prompt_ui_components["save_btn"]
            apply_preset_btn = prompt_ui_components["apply_preset_btn"]
            clear_btn = prompt_ui_components["clear_btn"]
            delete_preset_btn = prompt_ui_components["delete_preset_btn"]
            result_message = prompt_ui_components["result_message"]
                
        with gr.Column(scale=1):
            result_image = gr.Image(label=translate("生成結果"), height=512)
            preview_image = gr.Image(label=translate("処理中のプレビュー"), height=200, visible=False)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            
            # 基本設定UIブロック
            from oichi2_utils.oichi2_basic_settings_ui import create_basic_settings_ui
            basic_settings_group, basic_ui_components = create_basic_settings_ui(saved_app_settings, use_random_seed_default, seed_default)
            
            # UIコンポーネントを個別変数に展開
            use_teacache = basic_ui_components["use_teacache"]
            use_random_seed = basic_ui_components["use_random_seed"]
            seed = basic_ui_components["seed"]
            steps = basic_ui_components["steps"]
            gs = basic_ui_components["gs"]
            cfg = basic_ui_components["cfg"]
            rs = basic_ui_components["rs"]
            gpu_memory_preservation = basic_ui_components["gpu_memory_preservation"]

            # 出力設定UIブロック
            from oichi2_utils.oichi2_output_settings_ui import create_output_settings_ui
            output_settings_group, output_ui_components = create_output_settings_ui(output_folder_name, base_path)
            
            # UIコンポーネントを個別変数に展開
            output_dir = output_ui_components["output_dir"]
            open_folder_btn = output_ui_components["open_folder_btn"]
            path_display = output_ui_components["path_display"]
            
            # アプリケーション設定UIブロック
            from oichi2_utils.oichi2_app_settings_ui import create_app_settings_ui
            app_settings_group, app_ui_components = create_app_settings_ui(saved_app_settings)
            
            # UIコンポーネントを個別変数に展開
            save_current_settings_btn = app_ui_components["save_current_settings_btn"]
            reset_settings_btn = app_ui_components["reset_settings_btn"]
            save_settings_on_start = app_ui_components["save_settings_on_start"]
            alarm_on_completion = app_ui_components["alarm_on_completion"]
            log_enabled = app_ui_components["log_enabled"]
            log_folder = app_ui_components["log_folder"]
            open_log_folder_btn = app_ui_components["open_log_folder_btn"]
            settings_status = app_ui_components["settings_status"]
            
            save_app_settings_handler = create_save_app_settings_handler(save_app_settings_oichi)

            reset_app_settings_handler = create_reset_app_settings_handler()
    
    use_random_seed.change(fn=set_random_seed, inputs=use_random_seed, outputs=seed)
    
    save_button_click_handler = create_save_preset_handler()

    load_preset_handler = create_load_preset_handler()
    delete_preset_handler = create_delete_preset_handler()
    
    # 可変LoRAパラメータをipsに追加
    ips = create_input_parameters_list(
        input_image, prompt, n_prompt, seed, steps, cfg, gs, rs,
        gpu_memory_preservation, use_teacache, lora_scales_text, use_lora, fp8_optimization, resolution, output_dir,
        batch_count, use_random_seed, latent_window_size, latent_index,
        use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post,
        lora_mode,
        use_rope_batch, use_queue, prompt_queue_file,
        clean_index, input_mask, reference_mask,
        use_advanced_control, advanced_control_mode,
        kisekaeichi_reference_image, kisekaeichi_control_index,
        oneframe_mc_image, oneframe_mc_control_index, optional_control_image, optional_control_index,
        save_settings_on_start, alarm_on_completion, lora_files_list, lora_dropdowns_list, lora_strength_list
    )
    
    start_button.click(fn=process, inputs=ips, outputs=[result_image, preview_image, progress_desc, progress_bar, start_button, end_button, seed])
    end_button.click(fn=end_process, outputs=[end_button])
    
    gr.HTML(f'<div style="text-align:center; margin-top:20px;">{translate("FramePack 1フレーム推論版")}</div>')

    # プリセット・メタデータ・フォルダ操作イベントハンドラー設定
    setup_preset_event_handlers(
        save_btn, clear_btn, preset_dropdown, apply_preset_btn, delete_preset_btn,
        edit_name, edit_prompt, result_message, prompt,
        save_button_click_handler, clear_fields, load_preset_handler_wrapper,
        apply_to_prompt, delete_preset_handler
    )
    
    setup_metadata_event_handlers(
        input_image, copy_metadata, prompt, seed,
        update_from_image_metadata_wrapper, check_metadata_on_checkbox_change_wrapper
    )
    
    setup_folder_operation_event_handlers(
        open_folder_btn, output_dir, path_display, handle_open_folder_btn_wrapper
    )
    
    setup_settings_event_handlers(
        save_current_settings_btn, reset_settings_btn,
        resolution, steps, cfg, use_teacache, gpu_memory_preservation,
        gs, latent_window_size, latent_index,
        use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post,
        clean_index, save_settings_on_start, alarm_on_completion,
        log_enabled, log_folder, settings_status,
        save_app_settings_handler, reset_app_settings_handler
    )

    toggle_queue_settings = create_toggle_queue_settings_handler(get_image_queue_files_wrapper)
    toggle_queue_type_handler = create_toggle_queue_type_handler(get_image_queue_files_wrapper)
    
    use_queue.change(
        fn=queue_settings_with_state_update,
        inputs=[use_queue],
        outputs=[queue_type_selector, prompt_queue_group, image_queue_group]
    )
    
    queue_type_selector.change(
        fn=queue_type_with_state_update,
        inputs=[queue_type_selector],
        outputs=[prompt_queue_group, image_queue_group]
    )
    
    # 解像度情報更新イベントハンドラー
    resolution.change(
        fn=update_resolution_info,
        inputs=[resolution, input_image],
        outputs=[resolution_info]
    )
    
    input_image.change(
        fn=update_resolution_info,
        inputs=[resolution, input_image],
        outputs=[resolution_info]
    )
    
    # 初期解像度情報を表示
    try:
        initial_resolution = saved_app_settings.get("resolution", 640) if saved_app_settings else 640
        initial_info = update_resolution_info(initial_resolution, None)
        resolution_info.value = initial_info.value if hasattr(initial_info, 'value') else ""
    except:
        pass
    
    # メインUIコンポーネントマッピング完成
    main_ui_components.update({
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "gs": gs,
        "rs": rs,
        "resolution": resolution,
        "latent_window_size": latent_window_size,
        "latent_index": latent_index,
        "clean_index": clean_index,
        "use_clean_latents_2x": use_clean_latents_2x,
        "use_clean_latents_4x": use_clean_latents_4x,
        "use_clean_latents_post": use_clean_latents_post,
        "use_teacache": use_teacache,
        "gpu_memory_preservation": gpu_memory_preservation,
        "use_lora": use_lora,
        "fp8_optimization": fp8_optimization,
        "use_advanced_control": use_advanced_control,
        "advanced_control_mode": advanced_control_mode,
        "use_random_seed": use_random_seed,
        "prompt": prompt,
        
        # 画像コンポーネント（6個）
        "input_image": input_image,
        "kisekaeichi_reference_image": kisekaeichi_reference_image,
        "oneframe_mc_image": oneframe_mc_image,
        "optional_control_image": optional_control_image,
        "input_mask": input_mask,
        "reference_mask": reference_mask,
        
        # 制御位置（3個）
        "kisekaeichi_control_index": kisekaeichi_control_index,
        "oneframe_mc_control_index": oneframe_mc_control_index,
        "optional_control_index": optional_control_index,
        
        # LoRAコンポーネント（41個）
        "lora_mode": lora_mode,
        "lora_scales_text": lora_scales_text,
    })
    
    # LoRAグループコンポーネントを追加（可視性制御のため）
    if has_lora_support:
        main_ui_components["lora_dropdown_group"] = lora_dropdown_group
    
    # LoRAドロップダウンと強度をmain_ui_componentsに追加
    if has_lora_support and lora_dropdowns_list and lora_strength_list:
        for i in range(min(20, len(lora_dropdowns_list), len(lora_strength_list))):
            main_ui_components[f"lora_dropdown_{i}"] = lora_dropdowns_list[i]
            main_ui_components[f"lora_strength_{i}"] = lora_strength_list[i]
    
    # 履歴UIイベントハンドラー設定
    setup_history_event_handlers(history_manager, history_ui, main_ui_components)
    
    # LoRA数設定保存イベント（サブモジュール化）
    lora_count_save_btn.click(
        fn=save_lora_count_setting_handler,
        inputs=[lora_count_setting],
        outputs=[lora_preset_status]
    )
    
    # LoRAプリセットUIイベントハンドラー設定
    initialize_lora_preset_ui = setup_lora_preset_event_handlers(
        preset_buttons, lora_preset_mode, lora_scales_text, lora_preset_status,
        lora_load_btn, lora_save_btn, use_lora, lora_mode, lora_preset_group,
        lora_dropdowns_list, lora_strength_list
    )
    
    if initialize_lora_preset_ui:
        initialize_lora_preset_ui()

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)