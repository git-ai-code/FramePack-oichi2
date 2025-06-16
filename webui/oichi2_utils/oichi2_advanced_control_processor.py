"""
FramePack-oichi2 高度制御処理統合モジュール

1f-mc統合制御 + kisekaeichi互換性確保の完全統合処理
- 統合UI制御画像前処理
- kisekaeichi互換性確保システム
- 1f-mc制御画像統合処理
"""

from locales.i18n_extended import translate


def prepare_enhanced_control_processing(
    use_advanced_control, kisekaeichi_reference_image, kisekaeichi_control_index,
    oneframe_mc_image, oneframe_mc_control_index,
    optional_control_image, optional_control_index,
    latent_index, clean_index, input_image, advanced_control_mode=None):
    """
    制御モード別の高度制御パラメータ生成
    
    Args:
        use_advanced_control: 高度制御使用フラグ
        kisekaeichi_reference_image: 着せ替え参照画像パス
        kisekaeichi_control_index: 着せ替え制御位置
        oneframe_mc_image: 1f-mc制御画像パス  
        oneframe_mc_control_index: 1f-mc制御位置
        optional_control_image: 追加制御画像パス
        optional_control_index: 追加制御位置
        latent_index: 生成フレーム位置
        clean_index: クリーンlatent位置
        input_image: 入力画像パス
        advanced_control_mode: 制御モード（one_frame/kisekaeichi/1fmc/custom）
        
    Returns:
        dict: モード別制御パラメータ辞書またはNone
    """
    if not use_advanced_control:
        return None
        
    try:
        # 制御画像リスト初期化
        control_image_paths = []
        control_indices_list = []
        
        # 入力画像を基本制御画像として追加
        if input_image:
            control_image_paths.append(input_image)
            control_indices_list.append("0")  # 入力画像は常にindex 0
        
        # 制御モード別の画像選別処理
        print(translate("制御モード: {0}").format(advanced_control_mode or "未設定"))
        
        if advanced_control_mode == "one_frame":
            # 1フレーム推論: 制御画像なし
            print(translate("1フレーム推論モード: 高度な制御画像は使用されません"))
            if kisekaeichi_reference_image:
                print(translate("着せ替え制御画像は 1フレーム推論モードでは使用されません"))
            if oneframe_mc_image:
                print(translate("人物制御画像は 1フレーム推論モードでは使用されません"))
            if optional_control_image:
                print(translate("追加制御画像は 1フレーム推論モードでは使用されません"))
                
        elif advanced_control_mode == "kisekaeichi":
            # kisekaeichi: 着せ替え制御画像のみ
            if kisekaeichi_reference_image:
                control_image_paths.append(kisekaeichi_reference_image)
                control_indices_list.append(str(kisekaeichi_control_index))
                print(translate("着せ替え制御画像を使用: {0} (index={1})").format(
                    kisekaeichi_reference_image, kisekaeichi_control_index))
            if oneframe_mc_image:
                print(translate("人物制御画像は kisekaeichiモードでは使用されません"))
            if optional_control_image:
                print(translate("追加制御画像は kisekaeichiモードでは使用されません"))
                
        elif advanced_control_mode == "1fmc":
            # 1f-mc: 人物・追加制御画像のみ
            print(translate("1f-mcモード: 制御画像処理"))
            if oneframe_mc_image:
                control_image_paths.append(oneframe_mc_image)
                control_indices_list.append(str(oneframe_mc_control_index))
                print(translate("人物制御画像を使用: {0} (index={1})").format(
                    oneframe_mc_image, oneframe_mc_control_index))
            if optional_control_image:
                control_image_paths.append(optional_control_image)
                control_indices_list.append(str(optional_control_index))
                print(translate("追加制御画像を使用: {0} (index={1})").format(
                    optional_control_image, optional_control_index))
            if kisekaeichi_reference_image:
                print(translate("着せ替え制御画像は 1f-mcモードでは使用されません"))
                
        else:
            # カスタム: 全制御画像使用
            print(translate("カスタムモード: 全ての制御画像を使用"))
            
            if kisekaeichi_reference_image:
                control_image_paths.append(kisekaeichi_reference_image)
                control_indices_list.append(str(kisekaeichi_control_index))
                print(translate("着せ替え制御画像を使用: {0} (index={1})").format(
                    kisekaeichi_reference_image, kisekaeichi_control_index))
            
            if oneframe_mc_image:
                control_image_paths.append(oneframe_mc_image)
                control_indices_list.append(str(oneframe_mc_control_index))
                print(translate("人物制御画像を使用: {0} (index={1})").format(
                    oneframe_mc_image, oneframe_mc_control_index))
            
            if optional_control_image:
                control_image_paths.append(optional_control_image)
                control_indices_list.append(str(optional_control_index))
                print(translate("追加制御画像を使用: {0} (index={1})").format(
                    optional_control_image, optional_control_index))
        
        if not control_image_paths:
            print(translate("制御画像が設定されていません"))
            return None
            
        # 処理モード自動判定
        processing_mode = 'enhanced_kisekaeichi'
        if oneframe_mc_image and not kisekaeichi_reference_image:
            processing_mode = 'enhanced_1fmc'
        elif oneframe_mc_image and kisekaeichi_reference_image:
            processing_mode = 'enhanced_hybrid'
        
        # モード別no_post制御
        no_post_option = ""
        if advanced_control_mode == "kisekaeichi":
            no_post_option = ",no_post"  # kisekaeichiモードではno_post必須
        
        # 拡張制御パラメータ生成
        enhanced_params = {
            'control_image_paths': control_image_paths,
            'control_indices': control_indices_list,
            'target_index': latent_index,
            'clean_index': clean_index,
            'one_frame_config': f"latent_index={latent_index},control_index={';'.join(control_indices_list)},no_2x,no_4x{no_post_option}",
            'processing_mode': processing_mode,
            'use_clean_latents_post': advanced_control_mode != "kisekaeichi"  # kisekaeichiのみFalse
        }
        
        print(translate("高度制御処理パラメータ準備完了:"))
        
        return enhanced_params
        
    except Exception as e:
        print(translate("高度制御処理パラメータ準備エラー: {0}").format(e))
        import traceback
        traceback.print_exc()
        return None


def apply_enhanced_control_processing(enhanced_params, transformer_params):
    """
    拡張制御パラメータのtransformer統合
    
    Args:
        enhanced_params: 拡張制御パラメータ辞書
        transformer_params: transformer処理パラメータ辞書
        
    Returns:
        dict: 拡張パラメータ統合済みtransformerパラメータ
    """
    if not enhanced_params or enhanced_params.get('processing_mode') not in ['enhanced_kisekaeichi', 'enhanced_1fmc', 'enhanced_hybrid']:
        return transformer_params
        
    try:
        processing_mode = enhanced_params.get('processing_mode', 'unknown')
        print(translate("拡張制御処理をtransformer処理に適用中... (モード: {0})").format(processing_mode))
        
        # 制御画像パス統合
        if 'control_image_paths' in enhanced_params:
            transformer_params['enhanced_control_images'] = enhanced_params['control_image_paths']
            
        # 制御インデックス統合
        if 'control_indices' in enhanced_params:
            transformer_params['enhanced_control_indices'] = enhanced_params['control_indices']
            
        # フレーム位置統合
        transformer_params['enhanced_latent_index'] = enhanced_params.get('target_index', 5)
        transformer_params['enhanced_clean_index'] = enhanced_params.get('clean_index', 13)
        
        # one_frame設定統合
        transformer_params['enhanced_one_frame_config'] = enhanced_params.get('one_frame_config', '')
        
        # clean_latents_post制御統合
        if 'use_clean_latents_post' in enhanced_params:
            transformer_params['use_clean_latents_post'] = enhanced_params['use_clean_latents_post']
            print(translate("use_clean_latents_post設定: {0}").format(enhanced_params['use_clean_latents_post']))
        
        # 処理モード情報保存
        transformer_params['enhanced_processing_mode'] = processing_mode
        
        return transformer_params
        
    except Exception as e:
        print(translate("拡張制御処理適用エラー: {0}").format(e))
        import traceback
        traceback.print_exc()
        return transformer_params


def process_advanced_control_integration(
    use_advanced_control, kisekaeichi_reference_image, kisekaeichi_control_index,
    oneframe_mc_image, oneframe_mc_control_index, 
    optional_control_image, optional_control_index,
    input_image, latent_window_size, latent_index=None, clean_index=None, advanced_control_mode=None):
    """
    モード別高度制御処理統合実行
    
    Args:
        use_advanced_control: 高度制御使用フラグ
        kisekaeichi_reference_image: 着せ替え参照画像
        kisekaeichi_control_index: 着せ替え制御位置
        oneframe_mc_image: 背景制御画像
        oneframe_mc_control_index: 背景制御位置
        optional_control_image: 追加制御画像
        optional_control_index: 追加制御位置
        input_image: 入力画像
        latent_window_size: latentウィンドウサイズ
        latent_index: 生成フレーム位置
        clean_index: クリーンlatent位置
        advanced_control_mode: 制御モード
        
    Returns:
        tuple: (use_reference_image, reference_image, oneframe_mc_data, compatibility_mode, enhanced_params)
    """
    # 初期化
    oneframe_mc_data = None
    compatibility_mode = 'standard'
    enhanced_params = None
    
    # モード別従来パラメータマッピング
    if advanced_control_mode == "one_frame":
        # 1フレーム推論: 従来制御無効
        use_reference_image = False
        reference_image = None
        print(translate("1フレーム推論モード: kisekae処理と1f-mc処理をスキップします"))
    elif advanced_control_mode == "kisekaeichi":
        # kisekaeichi: 着せ替え制御のみ
        use_reference_image = use_advanced_control and kisekaeichi_reference_image is not None
        reference_image = kisekaeichi_reference_image if use_reference_image else None
        print(translate("kisekaeichiモード: 1f-mc処理をスキップします"))
    elif advanced_control_mode == "1fmc":
        # 1f-mc: 着せ替え制御無効
        use_reference_image = False
        reference_image = None
        print(translate("1f-mcモード: kisekae処理をスキップします"))
    else:
        # カスタム: すべて
        use_reference_image = use_advanced_control and kisekaeichi_reference_image is not None
        reference_image = kisekaeichi_reference_image if use_reference_image else None
        print(translate("カスタムモード: 全ての処理を実施します"))
        
    # 拡張制御パラメータ準備
    if (use_advanced_control or advanced_control_mode == "one_frame") and latent_index is not None and clean_index is not None:
        enhanced_params = prepare_enhanced_control_processing(
            use_advanced_control, kisekaeichi_reference_image, kisekaeichi_control_index,
            oneframe_mc_image, oneframe_mc_control_index,
            optional_control_image, optional_control_index,
            latent_index, clean_index, input_image, advanced_control_mode
        )
        
        if enhanced_params:
            print(translate("拡張制御処理モードが有効です"))
            compatibility_mode = 'enhanced'
            
            # 1f-mcモード用のoneframe_mc_data生成
            if advanced_control_mode == "1fmc" and enhanced_params.get('processing_mode') == 'enhanced_1fmc':
                oneframe_mc_data = create_oneframe_mc_data_from_enhanced_params(enhanced_params)
            
            return use_reference_image, reference_image, oneframe_mc_data, compatibility_mode, enhanced_params
    
    return use_reference_image, reference_image, oneframe_mc_data, compatibility_mode, enhanced_params


def create_oneframe_mc_data_from_enhanced_params(enhanced_params):
    """
    拡張制御パラメータから1f-mc用のoneframe_mc_dataを生成（アルファチャンネル対応）
    
    Args:
        enhanced_params: 拡張制御パラメータ辞書
        
    Returns:
        dict: 1f-mc制御データ辞書
    """
    try:
        control_image_paths = enhanced_params.get('control_image_paths', [])
        control_indices = enhanced_params.get('control_indices', [])
        
        # 入力画像（index 0）をスキップして制御画像のみ抽出
        oneframe_mc_data = {
            'control_images': [],
            'control_indices': [],
            'control_latents': [],  # 後でVAEエンコード時に追加
            'control_alpha_masks': [],  # アルファチャンネルマスク追加
            'valid_count': 0
        }
        
        for i, (image_path, index_str) in enumerate(zip(control_image_paths, control_indices)):
            # 入力画像（index "0"）はスキップ
            if index_str == "0":
                print(translate("入力画像（index=0）をスキップ: {0}").format(image_path))
                continue
                
            if image_path and image_path.strip():
                oneframe_mc_data['control_images'].append(image_path)
                oneframe_mc_data['control_indices'].append(int(index_str))
                oneframe_mc_data['control_alpha_masks'].append(None)  # 後でアルファチャンネル処理時に設定
                oneframe_mc_data['valid_count'] += 1
                print(translate("1f-mc制御画像を追加: {0} (index={1})").format(image_path, index_str))
        
        print(translate("1f-mc制御データ生成完了: {0}個の制御画像").format(oneframe_mc_data['valid_count']))
        
        return oneframe_mc_data if oneframe_mc_data['valid_count'] > 0 else None
        
    except Exception as e:
        print(translate("1f-mc制御データ生成エラー: {0}").format(e))
        import traceback
        traceback.print_exc()
        return None