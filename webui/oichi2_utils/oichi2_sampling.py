"""
FramePack-oichi2 サンプリング実行・後処理統合モジュール

サンプリング前調整・実行・後処理の統合モジュール
- サンプリング前最終調整・latents調整
- sample_hunyuan実行・パラメータ管理
- サンプリング後処理・エラーハンドリング
- 中断検知・メモリクリーンアップ
"""

import torch
import traceback
from locales.i18n_extended import translate
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan


def process_sampling_execution_and_postprocessing(
    sample_num_frames, clean_latent_indices, clean_latents, use_reference_image,
    clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices,
    use_clean_latents_2x, use_clean_latents_4x, input_image_np, width, height,
    use_cache, transformer, cfg, gs, rs, steps, rnd, llama_vec, llama_attention_mask,
    clip_l_pooler, llama_vec_n, llama_attention_mask_n, clip_l_pooler_n, gpu,
    image_encoder_last_hidden_state, latent_indices, callback):
    """
    サンプリング実行・後処理統合のメイン関数
    
    Args:
        sample_num_frames: サンプルフレーム数
        clean_latent_indices: クリーンlatentインデックス
        clean_latents: クリーンlatents
        use_reference_image: 参照画像使用フラグ
        clean_latents_2x: クリーンlatents 2x
        clean_latent_2x_indices: クリーンlatent 2xインデックス
        clean_latents_4x: クリーンlatents 4x
        clean_latent_4x_indices: クリーンlatent 4xインデックス
        use_clean_latents_2x: clean_latents_2x使用フラグ
        use_clean_latents_4x: clean_latents_4x使用フラグ
        input_image_np: 入力画像numpy配列
        width: 幅
        height: 高さ
        use_cache: キャッシュ使用フラグ
        transformer: Transformerモデル
        cfg: CFGスケール
        gs: ガイダンススケール
        rs: リスケール値
        steps: ステップ数
        rnd: ランダムジェネレータ
        llama_vec: LLaMAベクトル
        llama_attention_mask: LLaMAアテンションマスク
        clip_l_pooler: CLIPプーラー
        llama_vec_n: LLaMAネガティブベクトル
        llama_attention_mask_n: LLaMAネガティブアテンションマスク
        clip_l_pooler_n: CLIPネガティブプーラー
        gpu: GPU設定
        image_encoder_last_hidden_state: 画像エンコーダー最終隠れ状態
        latent_indices: latentインデックス
        callback: コールバック関数
        
    Returns:
        tuple: (generated_latents, transformer, is_interrupted)
    """
    try:
        # サンプリング前最終調整処理
        (clean_latent_indices, clean_latents, clean_latents_2x, clean_latent_2x_indices,
         clean_latents_4x, clean_latent_4x_indices) = _adjust_sampling_parameters(
            sample_num_frames, clean_latent_indices, clean_latents, use_reference_image,
            clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices,
            use_clean_latents_2x, use_clean_latents_4x
        )
        
        # 画像サイズ確認・調整処理
        actual_width, actual_height = _validate_and_adjust_image_size(
            input_image_np, width, height
        )
        
        # 初回実行時の品質説明
        if not use_cache:
            print(translate("【初回実行について】初回は Anti-drifting Sampling の履歴データがないため、ノイズが入る場合があります"))
        
        # sample_hunyuan実行
        generated_latents = sample_hunyuan(
            transformer=transformer,
            sampler='unipc',
            width=actual_width,
            height=actual_height,
            frames=sample_num_frames,
            real_guidance_scale=cfg,
            distilled_guidance_scale=gs,
            guidance_rescale=rs,
            num_inference_steps=steps,
            generator=rnd,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=gpu,
            dtype=torch.bfloat16,
            image_embeddings=image_encoder_last_hidden_state,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            callback=callback,
        )
        
        # サンプリング後処理
        is_interrupted = _handle_sampling_postprocessing(generated_latents, transformer)
        
        return generated_latents, transformer, is_interrupted
        
    except KeyboardInterrupt:
        print(translate("キーボード割り込みを検出しました - 安全に停止します"))
        # リソースのクリーンアップ
        _emergency_cleanup_on_interrupt(
            llama_vec, llama_vec_n, llama_attention_mask, llama_attention_mask_n,
            clip_l_pooler, clip_l_pooler_n, transformer
        )
        return None, transformer, True
        
    except RuntimeError as e:
        error_msg = str(e)
        if "size of tensor" in error_msg:
            print(translate("テンソルサイズの不一致エラーが発生しました: {0}").format(error_msg))
            print(translate("開発者に報告してください。"))
            raise e
        else:
            # その他のランタイムエラーはそのまま投げる
            raise e


def _adjust_sampling_parameters(sample_num_frames, clean_latent_indices, clean_latents, use_reference_image,
                               clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices,
                               use_clean_latents_2x, use_clean_latents_4x):
    """
    サンプリング前最終調整処理
    """
    # PRの実装に従い、one_frame_inferenceモードではsample_num_framesをサンプリングに使用
    if sample_num_frames == 1:
        # latent_indicesと同様に、clean_latent_indicesも調整する必要がある
        # 参照画像を使用しない場合のみ、最初の1要素に制限
        if clean_latent_indices.shape[1] > 1 and not use_reference_image:
            clean_latent_indices = clean_latent_indices[:, 0:1]  # 入力画像（最初の1要素）のみ
        # 参照画像使用時は両方のインデックスを保持（何もしない）
        
        # clean_latentsも調整（最後の1フレームのみ）
        # ただし、kisekaeichi機能の場合は、参照画像も保持する必要がある
        # clean_latentsの調整 - 複数フレームがある場合の処理
        if clean_latents.shape[2] > 1 and not use_reference_image:
            # 参照画像を使用しない場合のみ、最初の1フレームに制限
            clean_latents = clean_latents[:, :, 0:1]  # 入力画像（最初の1フレーム）のみ
            
        # 参照画像使用時は、両方のフレームを保持するため何もしない
        
        # clean_latentsの処理
        if use_reference_image:
            # PRのkisekaeichi実装オプション
            # latent_indexとclean_indexの処理は既に上で実行済み
            
            # オプション処理
            if not use_clean_latents_2x:  # PRの"no_2x"オプション
                clean_latents_2x = None
                clean_latent_2x_indices = None
                
            if not use_clean_latents_4x:  # PRの"no_4x"オプション
                clean_latents_4x = None
                clean_latent_4x_indices = None
            
        # clean_latents_2xとclean_latents_4xも必要に応じて調整
        if clean_latents_2x is not None and clean_latents_2x.shape[2] > 1:
            clean_latents_2x = clean_latents_2x[:, :, -1:]  # 最後の1フレームのみ
        
        if clean_latents_4x is not None and clean_latents_4x.shape[2] > 1:
            clean_latents_4x = clean_latents_4x[:, :, -1:]  # 最後の1フレームのみ
        
        # clean_latent_2x_indicesとclean_latent_4x_indicesも調整
        if clean_latent_2x_indices is not None and clean_latent_2x_indices.shape[1] > 1:
            clean_latent_2x_indices = clean_latent_2x_indices[:, -1:]
        
        if clean_latent_4x_indices is not None and clean_latent_4x_indices.shape[1] > 1:
            clean_latent_4x_indices = clean_latent_4x_indices[:, -1:]
    
    return (clean_latent_indices, clean_latents, clean_latents_2x, clean_latent_2x_indices,
            clean_latents_4x, clean_latent_4x_indices)


def _validate_and_adjust_image_size(input_image_np, width, height):
    """
    画像サイズ確認・調整処理
    """
    # 最も重要な問題：widthとheightが間違っている可能性
    # エラーログから、widthが60、heightが104になっているのが問題
    # これらはlatentサイズであり、実際の画像サイズではない
    print(translate("実際の画像サイズを再確認"))
    print(translate("入力画像のサイズ: {0}").format(input_image_np.shape))
    
    # find_nearest_bucketの結果が間違っている可能性
    # 入力画像のサイズから正しい値を計算
    if input_image_np.shape[0] == 832 and input_image_np.shape[1] == 480:
        # 実際の画像サイズを使用
        actual_width = 480
        actual_height = 832
        print(translate("実際の画像サイズを使用: width={0}, height={1}").format(actual_width, actual_height))
    else:
        # find_nearest_bucketの結果を使用
        actual_width = width
        actual_height = height
    
    return actual_width, actual_height


def _handle_sampling_postprocessing(generated_latents, transformer):
    """
    サンプリング後処理
    """
    # コールバックからの戻り値をチェック（コールバック関数が特殊な値を返した場合）
    if isinstance(generated_latents, dict) and generated_latents.get('user_interrupt'):
        # ユーザーが中断したことを検出したが、メッセージは出さない（既に表示済み）
        # 現在のバッチは完了させる（KeyboardInterruptは使わない）
        print(translate("バッチ内処理を完了します"))
        return True  # 中断フラグ
    else:
        print(translate("生成は正常に完了しました"))
    
    # サンプリング直後のメモリクリーンアップ（重要）
    # transformerの中間状態を明示的にクリア（KVキャッシュに相当）
    if hasattr(transformer, 'enable_teacache'):
        transformer.enable_teacache = False
        print(translate("transformerのキャッシュをクリア"))
    
    # 不要なモデル変数を積極的に解放
    torch.cuda.empty_cache()
    
    return False  # 正常完了


def _emergency_cleanup_on_interrupt(llama_vec, llama_vec_n, llama_attention_mask, llama_attention_mask_n,
                                   clip_l_pooler, clip_l_pooler_n, transformer):
    """
    中断時の緊急クリーンアップ処理
    """
    # リソースのクリーンアップ
    del llama_vec, llama_vec_n, llama_attention_mask, llama_attention_mask_n
    del clip_l_pooler, clip_l_pooler_n
    try:
        # モデルをCPUに移動（可能な場合のみ）
        if transformer is not None:
            if hasattr(transformer, 'cpu'):
                transformer.cpu()
        # GPUキャッシュをクリア
        torch.cuda.empty_cache()
    except Exception as cleanup_e:
        print(translate("停止時のクリーンアップでエラー: {0}").format(cleanup_e))