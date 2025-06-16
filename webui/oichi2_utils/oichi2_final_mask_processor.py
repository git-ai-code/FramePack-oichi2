"""
FramePack-oichi2 最終マスク処理統合モジュール

安定性重視 + アルファチャンネル対応の完全統合マスク処理
- 入力画像マスク処理
- 参照画像マスク処理（手動優先、アルファチャンネル補完）
- musubi-tuner準拠のアルファチャンネル自動検出
"""

import os
import tempfile

import numpy as np
import torch
from PIL import Image

from locales.i18n_extended import translate


def apply_final_masks(clean_latents, input_mask, reference_mask, reference_mask_from_alpha):
    """
    最終マスク処理（安定性重視 + アルファチャンネル対応）
    
    Args:
        clean_latents: クリーンlatentsテンソル
        input_mask: 入力画像マスクパス
        reference_mask: 参照画像マスクパス
        reference_mask_from_alpha: アルファチャンネルマスク（PIL Image）
        
    Returns:
        torch.Tensor: マスク適用済みclean_latents
    """
    # マスクが何もない場合は何もしない
    if input_mask is None and reference_mask is None and reference_mask_from_alpha is None:
        return clean_latents
    
    try:
        # 潜在空間のサイズ
        height_latent, width_latent = clean_latents.shape[-2:]
        
        # 入力画像マスクの処理（安定性重視）
        if input_mask is not None and input_mask.strip():
            _apply_input_mask(clean_latents, input_mask, height_latent, width_latent)
        
        # 参照画像マスク処理（手動マスク優先、アルファチャンネル補完）
        effective_reference_mask, alpha_temp_file = _prepare_reference_mask(
            reference_mask, reference_mask_from_alpha
        )
        
        if effective_reference_mask:
            _apply_reference_mask(clean_latents, effective_reference_mask, height_latent, width_latent)
        
        # 一時ファイルの削除
        if alpha_temp_file:
            _cleanup_temp_file(alpha_temp_file)
            
        return clean_latents
        
    except Exception as e:
        print(translate("マスク処理でエラー: {0}").format(e))
        import traceback
        traceback.print_exc()
        return clean_latents


def _apply_input_mask(clean_latents, input_mask, height_latent, width_latent):
    """入力画像マスクの適用（安定性重視）"""
    input_mask_img = Image.open(input_mask).convert('L')
    input_mask_np = np.array(input_mask_img)
    input_mask_resized = Image.fromarray(input_mask_np).resize((width_latent, height_latent), Image.BILINEAR)
    input_mask_tensor = torch.from_numpy(np.array(input_mask_resized)).float() / 255.0
    input_mask_tensor = input_mask_tensor.to(clean_latents.device)[None, None, None, :, :]
    
    # 入力画像のマスクを適用
    clean_latents[:, :, 0:1] = clean_latents[:, :, 0:1] * input_mask_tensor
    print(translate("入力画像マスクを適用しました"))


def _prepare_reference_mask(reference_mask, reference_mask_from_alpha):
    """参照画像マスクの準備（手動マスク優先、アルファチャンネル補完）"""
    effective_reference_mask = reference_mask
    alpha_temp_file = None
    
    # アルファチャンネルマスクがあり、手動マスクがない場合
    if reference_mask_from_alpha is not None and (reference_mask is None or not reference_mask.strip()):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            reference_mask_from_alpha.save(tmp_file.name)
            effective_reference_mask = tmp_file.name
            alpha_temp_file = tmp_file.name
            print(translate("アルファチャンネルマスクを参照マスクとして使用"))
    
    return effective_reference_mask, alpha_temp_file


def _apply_reference_mask(clean_latents, effective_reference_mask, height_latent, width_latent):
    """参照画像マスクの適用"""
    if not effective_reference_mask.strip():
        return
        
    reference_mask_img = Image.open(effective_reference_mask).convert('L')
    reference_mask_np = np.array(reference_mask_img)
    reference_mask_resized = Image.fromarray(reference_mask_np).resize((width_latent, height_latent), Image.BILINEAR)
    reference_mask_tensor = torch.from_numpy(np.array(reference_mask_resized)).float() / 255.0
    reference_mask_tensor = reference_mask_tensor.to(clean_latents.device)[None, None, None, :, :]
    
    # 参照画像位置にマスクを適用
    if clean_latents.shape[2] >= 2:
        clean_latents[:, :, 1:2] = clean_latents[:, :, 1:2] * reference_mask_tensor
        print(translate("参照画像マスクを適用しました"))


def _cleanup_temp_file(alpha_temp_file):
    """一時ファイルの削除"""
    try:
        os.unlink(alpha_temp_file)
    except:
        pass