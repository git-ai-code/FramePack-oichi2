"""
FramePack-oichi2 画像入力処理モジュール

画像入力処理の分離・最適化
- 画像検証・読み込み処理
- 自動リサイズ・クロップ
- テンソル変換・前処理
"""

import os
import traceback

import numpy as np
import torch
from PIL import Image

from locales.i18n_extended import translate


def process_input_image(input_image, resolution):
    """
    画像入力処理のメイン関数
    
    Args:
        input_image: 入力画像（None/ファイルパス/画像オブジェクト）
        resolution: 目標解像度
        
    Returns:
        tuple: (input_image_np, input_image_pt, height, width)
    """
    # 入力画像がNoneの場合はデフォルトの黒い画像を作成
    if input_image is None:
        print(translate("入力画像が指定されていないため、黒い画像を生成します"))
        height = width = resolution
        input_image_np = _create_default_black_image(resolution)
        
    elif isinstance(input_image, str):
        # 文字列（ファイルパス）の場合は画像をロード
        input_image_np, height, width = _load_image_from_path(input_image, resolution)
        
    else:
        # 通常の画像オブジェクトの場合（通常の処理）
        input_image_np, height, width = _process_image_object(input_image, resolution)
    
    # テンソル変換
    input_image_pt = _convert_image_to_tensor(input_image_np)
    
    return input_image_np, input_image_pt, height, width


def _create_default_black_image(resolution):
    """
    デフォルトの黒い画像を作成
    
    Args:
        resolution: 画像解像度
        
    Returns:
        np.ndarray: 黒い画像配列
    """
    return np.zeros((resolution, resolution, 3), dtype=np.uint8)


def _load_image_from_path(image_path, resolution):
    """
    ファイルパスから画像を読み込み・処理
    
    Args:
        image_path: 画像ファイルパス
        resolution: 目標解像度
        
    Returns:
        tuple: (input_image_np, height, width)
    """
    from diffusers_helper.bucket_tools import find_nearest_bucket
    from diffusers_helper.utils import resize_and_center_crop
    
    print(translate("入力画像がファイルパスのため、画像をロードします: {0}").format(image_path))
    
    try:
        img = Image.open(image_path)
        input_image = np.array(img)
        
        # 画像形式の正規化
        if len(input_image.shape) == 2:  # グレースケール画像の場合
            input_image = np.stack((input_image,) * 3, axis=-1)
        elif input_image.shape[2] == 4:  # アルファチャンネル付きの場合
            input_image = input_image[:, :, :3]
        
        # サイズ調整
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=resolution)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        
        return input_image_np, height, width
        
    except Exception as e:
        print(translate("画像のロードに失敗しました: {0}").format(e))
        # エラー発生時はデフォルトの黒い画像を使用
        print(translate("デフォルトの黒い画像を使用します"))
        height = width = resolution
        input_image_np = _create_default_black_image(resolution)
        
        return input_image_np, height, width


def _process_image_object(input_image, resolution):
    """
    画像オブジェクトを処理
    
    Args:
        input_image: 画像配列オブジェクト
        resolution: 目標解像度
        
    Returns:
        tuple: (input_image_np, height, width)
    """
    from diffusers_helper.bucket_tools import find_nearest_bucket
    from diffusers_helper.utils import resize_and_center_crop
    
    H, W, C = input_image.shape
    height, width = find_nearest_bucket(H, W, resolution=resolution)
    input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
    
    return input_image_np, height, width


def _convert_image_to_tensor(input_image_np):
    """
    画像配列をPyTorchテンソルに変換
    
    Args:
        input_image_np: numpy画像配列
        
    Returns:
        torch.Tensor: 変換済みテンソル
    """
    input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
    input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
    
    return input_image_pt