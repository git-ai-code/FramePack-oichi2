"""
FramePack-oichi2 Transformer・準備処理統合モジュール

Transformer初期化・メモリ管理・設定処理の統合モジュール
- Transformer初期化・ロード・状態確認
- モデルメモリ管理・GPU移動処理
- TeaCache設定・コールバック関数定義
- clean_latents形状調整・エラーハンドリング
- 参照画像・Kisekaeichi機能処理
"""

import traceback

import einops
import numpy as np
import torch

from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from diffusers_helper.hunyuan import vae_decode_fake
from diffusers_helper.memory import (
    get_cuda_free_memory_gb,
    load_model_as_complete,
    move_model_to_device_with_memory_preservation,
    unload_complete_models,
)
from locales.i18n_extended import translate


def process_transformer_initialization_and_preparation(
    transformer, transformer_manager, high_vram, gpu, gpu_memory_preservation,
    use_teacache, steps, stream, clean_latents, clean_latents_2x, clean_latents_4x,
    clean_latent_indices, latent_indices, use_reference_image, reference_latent,
    reference_encoder_output, latent_index, clean_index, latent_window_size,
    sample_num_frames, input_mask, reference_mask, image_encoder_last_hidden_state,
    text_encoder, text_encoder_2, image_encoder, vae, oneframe_mc_data=None):
    """
    Transformer・準備処理統合のメイン関数
    
    Args:
        transformer: Transformerモデル
        transformer_manager: Transformer管理マネージャー
        high_vram: ハイVRAMモード
        gpu: GPU設定
        gpu_memory_preservation: GPUメモリ保存設定
        use_teacache: TeaCache使用フラグ
        steps: ステップ数
        stream: ストリーム
        clean_latents: クリーンlatents
        clean_latents_2x: クリーンlatents 2x
        clean_latents_4x: クリーンlatents 4x
        clean_latent_indices: クリーンlatentインデックス
        latent_indices: latentインデックス
        use_reference_image: 参照画像使用フラグ
        reference_latent: 参照latent
        reference_encoder_output: 参照エンコーダー出力
        latent_index: レイテントインデックス（統一パラメータ）
        clean_index: クリーンインデックス（標準用語）
        latent_window_size: latentウィンドウサイズ
        sample_num_frames: サンプルフレーム数
        input_mask: 入力マスク
        reference_mask: 参照マスク
        image_encoder_last_hidden_state: 画像エンコーダー最終隠れ状態
        text_encoder: テキストエンコーダー
        text_encoder_2: テキストエンコーダー2
        image_encoder: 画像エンコーダー
        vae: VAEモデル
        oneframe_mc_data: 1f-mc制御データ（オプション）
        
    Returns:
        tuple: (transformer, callback, image_encoder_last_hidden_state, clean_latents, 
                clean_latents_2x, clean_latents_4x, clean_latent_indices, latent_indices)
    """
    # transformerの初期化とロード（未ロードの場合）
    if transformer is None:
        try:
            # transformerの状態を確認
            if not transformer_manager.ensure_transformer_state():
                raise Exception(translate("transformer状態の確認に失敗しました"))
                
            # transformerインスタンスを取得
            transformer = transformer_manager.get_transformer()
        except Exception as e:
            print(translate("transformerのロードに失敗しました: {0}").format(e))
            traceback.print_exc()
            
            if not transformer_manager.ensure_transformer_state():
                raise Exception(translate("transformer状態の再確認に失敗しました"))
            
            transformer = transformer_manager.get_transformer()
    
    # Transformerメモリ管理・GPU移動処理
    transformer = _manage_transformer_memory(
        transformer, high_vram, gpu, gpu_memory_preservation,
        text_encoder, text_encoder_2, image_encoder, vae
    )
    
    # teacacheの設定
    _setup_teacache(transformer, use_teacache, steps)
    
    # コールバック関数定義
    callback = _create_callback_function(stream, steps)
    
    # clean_latents形状調整処理
    clean_latents_2x, clean_latents_4x = _adjust_clean_latents_shapes(
        clean_latents_2x, clean_latents_4x
    )
    
    # 参照画像・Kisekaeichi・1f-mc統合処理
    (clean_latents, clean_latent_indices, latent_indices) = _process_reference_image_and_kisekaeichi(
        use_reference_image, reference_latent, reference_encoder_output,
        clean_latents, clean_latent_indices, latent_indices,
        latent_index, clean_index, latent_window_size, sample_num_frames,
        input_mask, reference_mask, oneframe_mc_data
    )
    
    # 最終的なBFloat16変換処理
    image_encoder_last_hidden_state = _finalize_image_encoder_processing(
        image_encoder_last_hidden_state, use_reference_image, reference_encoder_output,
        sample_num_frames
    )
    
    return (transformer, callback, image_encoder_last_hidden_state, clean_latents,
            clean_latents_2x, clean_latents_4x, clean_latent_indices, latent_indices)


def _manage_transformer_memory(transformer, high_vram, gpu, gpu_memory_preservation,
                              text_encoder, text_encoder_2, image_encoder, vae):
    """
    Transformerメモリ管理・GPU移動処理
    """
    # endframe_ichiと同様にtransformerをGPUに移動
    # vae, text_encoder, text_encoder_2, image_encoderをCPUに移動し、メモリを解放
    if not high_vram:
        # GPUメモリの解放 - transformerは処理中に必要なので含めない
        unload_complete_models(
            text_encoder, text_encoder_2, image_encoder, vae
        )

        # FP8最適化の有無に関わらず、gpu_complete_modulesに登録してから移動
        from diffusers_helper.memory import gpu_complete_modules
        if transformer not in gpu_complete_modules:
            # endframe_ichiと同様に、unload_complete_modulesで確実に解放されるようにする
            gpu_complete_modules.append(transformer)

        # メモリ確保した上でGPUへ移動
        # GPUメモリ保存値を明示的に浮動小数点に変換
        preserved_memory = float(gpu_memory_preservation) if gpu_memory_preservation is not None else 6.0
        move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserved_memory)
    else:
        # ハイVRAMモードでも正しくロードしてgpu_complete_modulesに追加
        load_model_as_complete(transformer, target_device=gpu, unload=True)
    
    return transformer


def _setup_teacache(transformer, use_teacache, steps):
    """
    TeaCache設定処理
    """
    if use_teacache:
        transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
    else:
        transformer.initialize_teacache(enable_teacache=False)


def _create_callback_function(stream, steps):
    """
    コールバック関数定義（KeyboardInterrupt方式）
    """
    def callback(d):
        try:
            preview = d['denoised']
            preview = vae_decode_fake(preview)
            
            preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
            preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
            
            if stream.input_queue.top() == 'end':
                # 即座にKeyboardInterruptで強制停止
                stream.output_queue.push(('end', None))
                raise KeyboardInterrupt('User ends the task.')
            
            current_step = d['i'] + 1
            percentage = int(100.0 * current_step / steps)
            hint = f'Sampling {current_step}/{steps}'
            desc = translate('1フレームモード: サンプリング中...')
            stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
        except KeyboardInterrupt:
            # 例外を再スローして確実に停止
            raise
        except Exception as e:
            import traceback
    
    return callback


def _adjust_clean_latents_shapes(clean_latents_2x, clean_latents_4x):
    """
    clean_latents形状調整処理
    """
    # 異常な次元数を持つテンソルを処理
    try:
        if len(clean_latents_2x.shape) > 5:
            # エラーメッセージは[1, 16, 1, 1, 96, 64]のような6次元テンソルを示しています
            # 必要なのは5次元テンソル[B, C, T, H, W]です
            if clean_latents_2x.shape[2] == 1 and clean_latents_2x.shape[3] == 1:
                # 余分な次元を削除
                clean_latents_2x = clean_latents_2x.squeeze(2)  # [1, 16, 1, 96, 64]
    except Exception as e:
        print(translate("clean_latents_2xの形状調整中にエラー: {0}").format(e))
    
    try:
        if len(clean_latents_4x.shape) > 5:
            if clean_latents_4x.shape[2] == 1 and clean_latents_4x.shape[3] == 1:
                # 余分な次元を削除
                clean_latents_4x = clean_latents_4x.squeeze(2)  # [1, 16, 1, 96, 64]
    except Exception as e:
        print(translate("clean_latents_4xの形状調整中にエラー: {0}").format(e))
    
    return clean_latents_2x, clean_latents_4x


def _process_reference_image_and_kisekaeichi(
    use_reference_image, reference_latent, reference_encoder_output,
    clean_latents, clean_latent_indices, latent_indices,
    latent_index, clean_index, latent_window_size, sample_num_frames,
    input_mask, reference_mask, oneframe_mc_data=None):
    """
    参照画像・Kisekaeichi・1f-mc統合処理
    """
    # 1f-mc制御データの処理
    if oneframe_mc_data and oneframe_mc_data.get('valid_count', 0) > 0:
        print(translate("1f-mc制御画像を適用します（{0}個）").format(oneframe_mc_data['valid_count']))
        _apply_1fmc_control_images(clean_latents, clean_latent_indices, oneframe_mc_data, latent_window_size)
    
    # 通常モードの処理（参照画像なし）
    if not use_reference_image:             
        # 入力画像がindex 0にあることを確認
        if clean_latents.shape[2] > 0:
            pass
    
    # Kisekaeichi機能: 参照画像latentの設定
    elif use_reference_image and reference_latent is not None:
        
        # kisekaeichi仕様：入力画像からサンプリングし、参照画像の特徴を使用
        # clean_latentsの形状が [B, C, 2, H, W] の場合
        if clean_latents.shape[2] >= 2:
            # clean_latentsの配置を確実にする
            # index 0: サンプリング開始点（入力画像）
            # index 1: 参照画像（特徴転送用）
            
            # すでにclean_latents_preが入力画像なので、index 0は変更不要
            # index 1に参照画像を設定
            clean_latents[:, :, 1] = reference_latent[:, :, 0]
            
            # kisekaeichi: 潜在空間での特徴転送の準備
            # ブレンドではなく、denoisingプロセス中にAttention機構で転送される
            # マスクがある場合のみ、マスクに基づいた潜在空間の調整を行う
            
        else:
            print(translate("clean_latentsの形状が予期しない形式です: {0}").format(clean_latents.shape))
        
        # clean_latent_indicesも更新する必要がある                
        if clean_latent_indices.shape[1] > 1:
            # PRの実装に従い、clean_indexをそのまま使用
            clean_latent_indices[:, 1] = clean_index
        else:
            print(translate("clean_latent_indicesの形状が予期しない形式です: {0}").format(clean_latent_indices.shape))
        
        # 公式実装に従い、target_indexを設定
        if latent_indices.shape[1] > 0:
            # latent_window_sizeに基づいて調整（現在は9）
            max_latent_index = latent_window_size - 1
            latent_index_actual = min(latent_index, max_latent_index)  # 範囲内に制限
            latent_indices[:, 0] = latent_index_actual
            print(translate("latent_indexを{0}に設定しました（最大値: {1}）").format(latent_index_actual, max_latent_index))
        else:
            print(translate("latent_indicesが空です"))
            
        # 参照画像のCLIP Vision出力は直接使用しない（エラー回避のため）
        # latentレベルでの変更のみ適用
        if reference_encoder_output is not None:
            print(translate("参照画像の特徴はlatentのみで反映されます"))
        
        # マスクの適用（kisekaeichi仕様）
        if input_mask is not None or reference_mask is not None:
            print(translate("kisekaeichi: マスクを適用します"))
            
            try:
                from PIL import Image
                import numpy as np
                
                # 潜在空間のサイズ
                height_latent, width_latent = clean_latents.shape[-2:]
                
                # 入力画像マスクの処理
                if input_mask is not None:
                    input_mask_img = Image.open(input_mask).convert('L')
                    input_mask_np = np.array(input_mask_img)
                    input_mask_resized = Image.fromarray(input_mask_np).resize((width_latent, height_latent), Image.BILINEAR)
                    input_mask_tensor = torch.from_numpy(np.array(input_mask_resized)).float() / 255.0
                    input_mask_tensor = input_mask_tensor.to(clean_latents.device)[None, None, None, :, :]
                    
                    # 入力画像のマスクを適用（黒い部分をゼロ化）
                    clean_latents[:, :, 0:1] = clean_latents[:, :, 0:1] * input_mask_tensor
                    print(translate("入力画像マスクを適用しました（黒い領域をゼロ化）"))
                
                # 参照画像マスクの処理
                if reference_mask is not None:
                    reference_mask_img = Image.open(reference_mask).convert('L')
                    reference_mask_np = np.array(reference_mask_img)
                    reference_mask_resized = Image.fromarray(reference_mask_np).resize((width_latent, height_latent), Image.BILINEAR)
                    reference_mask_tensor = torch.from_numpy(np.array(reference_mask_resized)).float() / 255.0
                    reference_mask_tensor = reference_mask_tensor.to(clean_latents.device)[None, None, None, :, :]
                    
                    # 参照画像のマスクを適用（黒い部分をゼロ化）
                    if clean_latents.shape[2] >= 2:
                        clean_latents[:, :, 1:2] = clean_latents[:, :, 1:2] * reference_mask_tensor
                        print(translate("参照画像マスクを適用しました（黒い領域をゼロ化）"))
                
            except Exception as e:
                print(translate("マスクの適用中にエラーが発生しました: {0}").format(e))
                print(translate("マスクなしで続行します"))
        
        # 公式実装のzero_post処理（固定値として実装）
        if sample_num_frames == 1:
            one_frame_inference = set()
            one_frame_inference.add(f"latent_index={latent_index}")
            one_frame_inference.add(f"clean_index={clean_index}")
            # 公式実装の推奨動作として、参照画像がない場合にzero_postを適用
            if not use_reference_image:
                one_frame_inference.add("zero_post")
            
            # zero_post処理（公式実装と完全同一）
            if "zero_post" in one_frame_inference:
                clean_latents[:, :, 1:, :, :] = torch.zeros_like(clean_latents[:, :, 1:, :, :])
            
            # 他のオプションも処理
            for option in one_frame_inference:
                if option == "no_2x":
                    if 'clean_latents_2x_param' in locals():
                        clean_latents_2x_param = None
                
                elif option == "no_4x":
                    if 'clean_latents_4x_param' in locals():
                        clean_latents_4x_param = None
                
                elif option == "no_post":
                    if clean_latents.shape[2] > 1:
                        clean_latents = clean_latents[:, :, :1, :, :]
    
    return clean_latents, clean_latent_indices, latent_indices


def _apply_1fmc_control_images(clean_latents, clean_latent_indices, oneframe_mc_data, latent_window_size):
    """
    1f-mc制御画像の適用処理（アルファチャンネル対応）
    
    Args:
        clean_latents: クリーンlatentsテンソル
        clean_latent_indices: クリーンlatentインデックス
        oneframe_mc_data: 1f-mc制御データ
        latent_window_size: latentウィンドウサイズ
    """
    try:
        control_images = oneframe_mc_data.get('control_images', [])
        control_indices = oneframe_mc_data.get('control_indices', [])
        control_latents = oneframe_mc_data.get('control_latents', [])
        control_alpha_masks = oneframe_mc_data.get('control_alpha_masks', [])
        
        print(translate("1f-mc制御画像を適用"))
        print(translate("制御latents数: {0}, インデックス数: {1}").format(len(control_latents), len(control_indices)))
        
        # エンコード済み制御latentをclean_latentsに配置
        # Step 1: 制御画像を順番に配置（公式実装準拠）
        for i, ctrl_latent in enumerate(control_latents):
            if ctrl_latent is not None and i < clean_latents.shape[2]:
                try:
                    # 制御画像を順番に配置（0, 1, 2, ...）
                    clean_latents[:, :, i:i+1] = ctrl_latent.to(clean_latents.device, clean_latents.dtype)
                    print(translate("制御latent{0}をclean_latents順次位置{1}に配置").format(i+1, i))
                except Exception as e:
                    print(translate("制御latent{0}の配置中にエラー: {1}").format(i+1, e))
                    continue
        
        # Step 2: control_indexに基づいてclean_latent_indicesを設定（公式実装準拠）
        for i, ctrl_idx in enumerate(control_indices):
            if i < clean_latent_indices.shape[1]:
                try:
                    # 公式実装準拠: control_indexの値をclean_latent_indicesに設定
                    clean_latent_indices[:, i] = ctrl_idx
                    print(translate("制御画像{0}: clean_latent_indices[{1}] = {2}").format(i+1, i, ctrl_idx))
                except Exception as e:
                    print(translate("制御インデックス{0}の設定中にエラー: {1}").format(i+1, e))
                    continue
            else:
                print(translate("制御画像{0}: インデックス位置{1}が範囲外（最大: {2}）").format(i+1, i, clean_latent_indices.shape[1]-1))
        
        print(translate("1f-mc制御latent配置完了（{0}個）").format(len(control_latents)))
        
        # アルファマスク情報をログ出力
        alpha_mask_count = sum(1 for mask in control_alpha_masks if mask is not None)
        print(translate("アルファマスク適用数: {0}/{1}").format(alpha_mask_count, len(control_alpha_masks)))
        
        # デバッグ情報出力
        print(translate("final clean_latents形状: {0}").format(clean_latents.shape))
        print(translate("final clean_latent_indices: {0}").format(clean_latent_indices))
        
    except Exception as e:
        print(translate("1f-mc制御画像適用エラー: {0}").format(e))
        import traceback
        traceback.print_exc()


def _finalize_image_encoder_processing(image_encoder_last_hidden_state, use_reference_image,
                                      reference_encoder_output, sample_num_frames):
    """
    最終的なBFloat16変換処理
    """
    # BFloat16に変換（通常の処理）
    if image_encoder_last_hidden_state is not None:
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(dtype=torch.bfloat16)
                        
    # 参照画像のCLIP Vision特徴を使用する場合は、注意深く処理
    # 現在の実装では参照画像のCLIP特徴を使用しない（latentのみ使用）
    # エラー処理安定化対応
    if use_reference_image and reference_encoder_output is not None:
        # 参照画像のCLIP特徴は直接使用せず、latentでのみ反映
        # これによりrotary embedding関連のエラーを回避
        pass
    
    return image_encoder_last_hidden_state