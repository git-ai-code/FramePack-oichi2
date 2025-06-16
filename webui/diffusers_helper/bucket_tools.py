import math

# 解像度設定（64刻みで均等間隔・スライダー対応）
MIN_RESOLUTION = 512
MAX_RESOLUTION = 1536  
RESOLUTION_STEP = 64

# 動的に解像度リストを生成
SAFE_RESOLUTIONS = list(range(MIN_RESOLUTION, MAX_RESOLUTION + 1, RESOLUTION_STEP))

# アスペクト比の設定
MIN_ASPECT_RATIO = 0.5  # 1:2 (縦長)
MAX_ASPECT_RATIO = 2.0  # 2:1 (横長)
ASPECT_RATIO_STEPS = 12  # アスペクト比分割数


def round_to_multiple(value, multiple=32):
    """指定された倍数に丸める（HunyuanVideoモデルは32の倍数が必要）"""
    return int(round(value / multiple) * multiple)


def generate_buckets_for_resolution(base_resolution):
    """
    指定された解像度レベルに対してバケットを動的生成
    
    Args:
        base_resolution: ベース解像度（例：640）
        
    Returns:
        list: (height, width)のタプルリスト
    """
    target_pixels = base_resolution * base_resolution
    buckets = []
    
    # アスペクト比を等間隔で分割
    aspect_ratios = []
    for i in range(ASPECT_RATIO_STEPS):
        ratio = MIN_ASPECT_RATIO + (MAX_ASPECT_RATIO - MIN_ASPECT_RATIO) * i / (ASPECT_RATIO_STEPS - 1)
        aspect_ratios.append(ratio)
    
    for aspect_ratio in aspect_ratios:
        # aspect_ratio = height / width の関係
        # target_pixels = height * width
        # よって height = sqrt(target_pixels * aspect_ratio), width = sqrt(target_pixels / aspect_ratio)
        
        height = math.sqrt(target_pixels * aspect_ratio)
        width = math.sqrt(target_pixels / aspect_ratio)
        
        # 32の倍数に丸める（HunyuanVideoモデルの制約）
        height = round_to_multiple(height, 32)
        width = round_to_multiple(width, 32)
        
        # 最小サイズ制約（128以上）
        if height >= 128 and width >= 128:
            buckets.append((height, width))
    
    # 重複を除去し、面積でソート
    buckets = list(set(buckets))
    buckets.sort(key=lambda x: x[0] * x[1])
    
    return buckets


def get_bucket_options(resolution):
    """
    指定された解像度のバケットオプションを取得（キャッシュ機能付き）
    """
    if not hasattr(get_bucket_options, '_cache'):
        get_bucket_options._cache = {}
    
    if resolution not in get_bucket_options._cache:
        get_bucket_options._cache[resolution] = generate_buckets_for_resolution(resolution)
    
    return get_bucket_options._cache[resolution]


def find_nearest_bucket(h, w, resolution=640):
    """
    最も適切なアスペクト比のバケットを見つける関数（動的計算対応）
    
    Args:
        h: 入力画像の高さ
        w: 入力画像の幅
        resolution: 目標解像度レベル
        
    Returns:
        tuple: (height, width) 最適なバケットサイズ
    """
    # 安全な解像度に丸める
    if resolution not in SAFE_RESOLUTIONS:
        # 最も近い安全な解像度を選択
        closest_resolution = min(SAFE_RESOLUTIONS, key=lambda x: abs(x - resolution))
        print(f"Warning: Resolution {resolution} is not in safe list. Using {closest_resolution} instead.")
        resolution = closest_resolution
    
    # 動的にバケットを取得
    bucket_options = get_bucket_options(resolution)
    
    min_metric = float('inf')
    best_bucket = None
    for (bucket_h, bucket_w) in bucket_options:
        # アスペクト比の差を計算
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    
    return best_bucket


def get_available_resolutions():
    """
    利用可能な解像度リストを取得（UI用）
    
    Returns:
        list: 利用可能な解像度のリスト
    """
    return SAFE_RESOLUTIONS.copy()


def get_resolution_info(resolution):
    """
    指定解像度の詳細情報を取得
    
    Args:
        resolution: 解像度レベル
        
    Returns:
        dict: 解像度情報
    """
    if resolution not in SAFE_RESOLUTIONS:
        closest = min(SAFE_RESOLUTIONS, key=lambda x: abs(x - resolution))
        resolution = closest
    
    buckets = get_bucket_options(resolution)
    aspect_ratios = [round(h/w, 2) for h, w in buckets]
    
    return {
        'resolution': resolution,
        'total_buckets': len(buckets),
        'min_aspect_ratio': min(aspect_ratios),
        'max_aspect_ratio': max(aspect_ratios),
        'buckets': buckets,
        'square_bucket': next((b for b in buckets if abs(b[0] - b[1]) <= 16), buckets[len(buckets)//2])
    }


def get_image_resolution_prediction(image_path, resolution):
    """
    入力画像の縦横比率から予想される解像度を取得（クロッピング考慮）
    
    Args:
        image_path: 画像ファイルパス（None可）
        resolution: 目標解像度レベル
        
    Returns:
        dict: 予想解像度情報
    """
    try:
        if image_path is None:
            # 画像がない場合は正方形として予測
            return {
                'has_image': False,
                'predicted_size': (resolution, resolution),
                'aspect_ratio': 1.0,
                'aspect_description': '正方形（デフォルト）',
                'cropping_info': None
            }
        
        from PIL import Image
        import os
        
        if not os.path.exists(image_path):
            return {
                'has_image': False,
                'predicted_size': (resolution, resolution),
                'aspect_ratio': 1.0,
                'aspect_description': '正方形（デフォルト）',
                'cropping_info': None
            }
        
        # 画像を読み込んでサイズを取得
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            original_aspect = original_height / original_width
        
        # 最適なバケットを見つける
        predicted_height, predicted_width = find_nearest_bucket(original_height, original_width, resolution)
        predicted_aspect = predicted_height / predicted_width
        
        # クロッピング情報を計算
        cropping_info = _calculate_cropping_info(
            original_width, original_height, 
            predicted_width, predicted_height
        )
        
        # アスペクト比の説明
        if original_aspect > 1.2:
            aspect_desc = "縦長"
        elif original_aspect < 0.8:
            aspect_desc = "横長"
        else:
            aspect_desc = "正方形"
        
        return {
            'has_image': True,
            'original_size': (original_width, original_height),
            'predicted_size': (predicted_height, predicted_width),
            'aspect_ratio': original_aspect,
            'aspect_description': aspect_desc,
            'cropping_info': cropping_info
        }
        
    except Exception as e:
        return {
            'has_image': False,
            'predicted_size': (resolution, resolution),
            'aspect_ratio': 1.0,
            'aspect_description': '正方形（エラー）',
            'error': str(e),
            'cropping_info': None
        }


def _calculate_cropping_info(orig_w, orig_h, target_w, target_h):
    """
    resize_and_center_crop処理でのクロッピング情報を計算
    
    Args:
        orig_w, orig_h: 元画像サイズ
        target_w, target_h: 目標サイズ
        
    Returns:
        dict: クロッピング情報
    """
    # アスペクト比計算
    orig_aspect = orig_w / orig_h
    target_aspect = target_w / target_h
    
    # リサイズ後の仮サイズを計算
    if orig_aspect > target_aspect:
        # 元画像の方が横長 → 高さを基準にリサイズ、幅をクロップ
        resize_h = target_h
        resize_w = int(resize_h * orig_aspect)
        crop_w = resize_w - target_w
        crop_h = 0
        crop_direction = "horizontal"
    else:
        # 元画像の方が縦長 → 幅を基準にリサイズ、高さをクロップ
        resize_w = target_w
        resize_h = int(resize_w / orig_aspect)
        crop_w = 0
        crop_h = resize_h - target_h
        crop_direction = "vertical"
    
    # クロップ率計算
    crop_ratio_w = crop_w / resize_w if resize_w > 0 else 0
    crop_ratio_h = crop_h / resize_h if resize_h > 0 else 0
    
    return {
        'resize_size': (resize_w, resize_h),
        'crop_amount': (crop_w, crop_h),
        'crop_ratio': (crop_ratio_w, crop_ratio_h),
        'crop_direction': crop_direction,
        'has_cropping': crop_w > 0 or crop_h > 0
    }

