"""
FramePack-oichi2 Latent処理コアモジュール

Latent処理内部ロジックの分離・最適化
- インデックス計算・分割処理
- clean_latents設定・結合処理
- one_frame_inference処理
※ forループ構造は保持し、内部処理のみ分離
"""

import torch

from locales.i18n_extended import translate


def process_latent_core_logic(latent_padding_size, latent_window_size, sample_num_frames,
                             use_reference_image, latent_index, clean_index,
                             start_latent, history_latents, height, width, use_clean_latents_post):
    """
    Latent処理のコアロジック（forループ内部処理）
    
    Args:
        latent_padding_size: latentパディングサイズ
        latent_window_size: latentウィンドウサイズ
        sample_num_frames: サンプルフレーム数
        use_reference_image: 参照画像使用フラグ
        latent_index: 生成フレーム位置インデックス
        clean_index: 参照フレーム位置インデックス
        start_latent: 開始latent
        history_latents: 履歴latents
        height: 画像高さ
        width: 画像幅
        use_clean_latents_post: clean_latents_post使用フラグ
        
    Returns:
        tuple: (clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices,
                clean_latents_4x, clean_latent_4x_indices, latent_indices)
    """
    # 1フレームモード用のindices設定
    # PR実装に合わせて、インデックスの範囲を明示的に設定
    # 元のPRでは 0から total_frames相当の値までのインデックスを作成
    # 1フレームモードでは通常: [0(clean_pre), 1(latent), 2(clean_post), 3,4(clean_2x), 5-20(clean_4x)]
    indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
    split_sizes = [1, latent_padding_size, latent_window_size, 1, 2, 16]
    
    # latent_padding_sizeが0の場合、空のテンソルになる可能性があるため処理を調整
    if latent_padding_size == 0:
        # blank_indicesを除いて分割
        clean_latent_indices_pre = indices[:, 0:1]
        latent_indices = indices[:, 1:1+latent_window_size]
        clean_latent_indices_post = indices[:, 1+latent_window_size:2+latent_window_size]
        clean_latent_2x_indices = indices[:, 2+latent_window_size:4+latent_window_size]
        clean_latent_4x_indices = indices[:, 4+latent_window_size:20+latent_window_size]
        blank_indices = torch.empty((1, 0), dtype=torch.long)  # 空のテンソル
    else:
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split(split_sizes, dim=1)
    
    # 公式実装に完全に合わせたone_frame_inference処理
    if sample_num_frames == 1:
        # 1フレームモードの特別な処理
        if use_reference_image:
            # kisekaeichi用の設定（公式実装）
            one_frame_inference = set()
            one_frame_inference.add(f"latent_index={latent_index}")
            one_frame_inference.add(f"clean_index={clean_index}")
            
            # 公式実装に従った処理
            latent_indices = indices[:, -1:]  # デフォルトは最後のフレーム
            
            # パラメータ解析と処理（公式実装と同じ）
            for one_frame_param in one_frame_inference:
                if one_frame_param.startswith("latent_index="):
                    latent_idx = int(one_frame_param.split("=")[1])
                    latent_indices[:, 0] = latent_idx
                
                elif one_frame_param.startswith("clean_index="):
                    clean_idx = int(one_frame_param.split("=")[1])
                    clean_latent_indices_post[:, 0] = clean_idx
        else:
            # 通常モード（参照画像なし）- 以前の動作を復元
            # 正常動作版と同じように、latent_window_size内の最後のインデックスを使用
            all_indices = torch.arange(0, latent_window_size).unsqueeze(0)
            latent_indices = all_indices[:, -1:]
            
    else:
        # 通常のモード（複数フレーム）
        # 詳細設定のlatent_indexに基づいたインデックス処理
        all_indices = torch.arange(0, latent_window_size).unsqueeze(0)
        if latent_index > 0 and latent_index < latent_window_size:
            # ユーザー指定のインデックスを使用
            latent_indices = all_indices[:, latent_index:latent_index+1]
        else:
            # デフォルトは最後のインデックス
            latent_indices = all_indices[:, -1:]
    
    # clean_latents設定
    clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
    
    # 通常モードでのインデックス調整（1f-mc対応）
    if not use_reference_image and sample_num_frames == 1:
        # 通常モードでも1f-mc制御画像に対応するため、十分なサイズを確保
        # 位置0-15までのインデックスを作成（latent_window_sizeに応じた範囲）
        max_control_positions = min(latent_window_size, 16)  # 最大16位置まで対応
        clean_latent_indices = torch.arange(0, max_control_positions).unsqueeze(0)
        print(translate("1f-mc対応: clean_latent_indicesのサイズを{0}に拡張").format(max_control_positions))
        
        # clean_latents_2xとclean_latents_4xも調整
        if clean_latent_2x_indices.shape[1] > 0:
            # clean_latents_2xの最初の要素のみを使用
            clean_latent_2x_indices = clean_latent_2x_indices[:, :1]
        
        if clean_latent_4x_indices.shape[1] > 0:
            # clean_latents_4xの最初の要素のみを使用
            clean_latent_4x_indices = clean_latent_4x_indices[:, :1]
    
    # start_latentの形状を確認
    if len(start_latent.shape) < 5:  # バッチとフレーム次元がない場合
        # [B, C, H, W] → [B, C, 1, H, W] の形に変換
        clean_latents_pre = start_latent.unsqueeze(2).to(history_latents.dtype).to(history_latents.device)
    else:
        clean_latents_pre = start_latent.to(history_latents.dtype).to(history_latents.device)
    
    # history_latentsからデータを適切に分割
    try:
        # 分割前に形状確認
        frames_to_split = history_latents.shape[2]
        
        if frames_to_split >= 19:  # 1+2+16フレームを想定
            # 正常分割
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
        else:
            # フレーム数が不足している場合は適切なサイズで初期化
            clean_latents_post = torch.zeros(1, 16, 1, height // 8, width // 8, dtype=history_latents.dtype, device=history_latents.device)
            clean_latents_2x = torch.zeros(1, 16, 2, height // 8, width // 8, dtype=history_latents.dtype, device=history_latents.device)
            clean_latents_4x = torch.zeros(1, 16, 16, height // 8, width // 8, dtype=history_latents.dtype, device=history_latents.device)
    except Exception as e:
        print(translate("history_latentsの分割中にエラー: {0}").format(e))
        # エラー発生時はゼロで初期化
        clean_latents_post = torch.zeros(1, 16, 1, height // 8, width // 8, dtype=torch.float32, device='cpu')
        clean_latents_2x = torch.zeros(1, 16, 2, height // 8, width // 8, dtype=torch.float32, device='cpu')
        clean_latents_4x = torch.zeros(1, 16, 16, height // 8, width // 8, dtype=torch.float32, device='cpu')
    
    # 公式実装のno_2x, no_4x処理を先に実装
    if sample_num_frames == 1 and use_reference_image:
        # kisekaeichi時の固定設定（公式実装に完全準拠）
        one_frame_inference = set()
        one_frame_inference.add(f"latent_index={latent_index}")
        one_frame_inference.add(f"clean_index={clean_index}")
        
        # 公式実装のオプション処理（no_post以外）
        for option in one_frame_inference:
            if option == "no_2x":
                clean_latents_2x = None
                clean_latent_2x_indices = None
            
            elif option == "no_4x":
                clean_latents_4x = None
                clean_latent_4x_indices = None
    
    # 詳細設定のオプションに基づいて処理
    if use_clean_latents_post:
        try:
            # 正しい形状に変換して結合
            if len(clean_latents_pre.shape) != len(clean_latents_post.shape):
                # 形状を合わせる
                if len(clean_latents_pre.shape) < len(clean_latents_post.shape):
                    clean_latents_pre = clean_latents_pre.unsqueeze(2)
                else:
                    clean_latents_post = clean_latents_post.unsqueeze(1)
            
            # 1f-mc制御に対応するため、十分なフレーム数を確保
            if sample_num_frames == 1 and not use_reference_image:
                max_control_positions = min(latent_window_size, 16)
                required_frames = max(2, max_control_positions)
                
                if clean_latents_pre.shape[2] + clean_latents_post.shape[2] < required_frames:
                    # 不足分の空フレームを追加
                    current_frames = clean_latents_pre.shape[2] + clean_latents_post.shape[2]
                    additional_needed = required_frames - current_frames
                    empty_latents = torch.zeros(clean_latents_pre.shape[0], clean_latents_pre.shape[1], 
                                               additional_needed, clean_latents_pre.shape[3], 
                                               clean_latents_pre.shape[4], 
                                               dtype=clean_latents_pre.dtype, device=clean_latents_pre.device)
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post, empty_latents], dim=2)
                    print(translate("1f-mc対応: clean_latentsを{0}フレームに拡張（post版）").format(required_frames))
                else:
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            else:
                # 通常の結合
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        except Exception as e:
            print(translate("clean_latentsの結合中にエラーが発生しました: {0}").format(e))
            print(translate("前処理のみを使用します"))
            clean_latents = clean_latents_pre
            if len(clean_latents.shape) == 4:  # [B, C, H, W]
                clean_latents = clean_latents.unsqueeze(2)  # [B, C, 1, H, W]
    else:
        print(translate("clean_latents_postは無効化されています。生成が高速化されますが、ノイズが増える可能性があります"))
        # clean_latents_postを使用しない場合、前処理+空白レイテント（ゼロテンソル）を結合
        # これはオリジナルの実装をできるだけ維持しつつ、エラーを回避するためのアプローチ
        clean_latents_pre_shaped = clean_latents_pre
        if len(clean_latents_pre.shape) == 4:  # [B, C, H, W]
            clean_latents_pre_shaped = clean_latents_pre.unsqueeze(2)  # [B, C, 1, H, W]
        
        # 空のレイテントを作成（形状を合わせる）
        shape = list(clean_latents_pre_shaped.shape)
        # [B, C, 1, H, W]の形状に対して、[B, C, 1, H, W]の空テンソルを作成
        empty_latent = torch.zeros_like(clean_latents_pre_shaped)
        
        # 1f-mc制御に対応するため、十分なフレーム数を確保
        max_control_positions = min(latent_window_size, 16)  # clean_latent_indicesと同じサイズ
        required_frames = max(2, max_control_positions)  # 最低2フレーム、制御位置数に応じて拡張
        
        # 必要な数だけ空のlatentを作成
        additional_frames_needed = required_frames - 1  # preフレーム分を除く
        empty_latents = torch.zeros(clean_latents_pre_shaped.shape[0], clean_latents_pre_shaped.shape[1], 
                                   additional_frames_needed, clean_latents_pre_shaped.shape[3], 
                                   clean_latents_pre_shaped.shape[4], 
                                   dtype=clean_latents_pre_shaped.dtype, device=clean_latents_pre_shaped.device)
        
        # 結合して形状を維持
        clean_latents = torch.cat([clean_latents_pre_shaped, empty_latents], dim=2)
        print(translate("1f-mc対応: clean_latentsを{0}フレームに拡張").format(required_frames))
    
    # no_post処理をclean_latentsが定義された後に実行
    if sample_num_frames == 1 and use_reference_image and 'one_frame_inference' in locals():
        for option in one_frame_inference:
            if option == "no_post":
                if clean_latents is not None:
                    clean_latents = clean_latents[:, :, :1, :, :]
                    clean_latent_indices = clean_latent_indices[:, :1]
    
    return (clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices,
            clean_latents_4x, clean_latent_4x_indices, latent_indices)