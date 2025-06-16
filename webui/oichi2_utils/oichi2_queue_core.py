"""
FramePack-oichi2 キュー機能・バッチ処理統合モジュール

キュー機能・バッチ処理の分離・最適化
- キュー設定・初期化処理
- プロンプト・イメージキューファイル管理
- バッチ処理サマリー出力・詳細ログ
- バッチループ内キュー適用・プロンプト/画像選択
- エラーハンドリング・ファイル存在確認
"""

import os
from locales.i18n_extended import translate


def setup_queue_configuration(use_queue, prompt_queue_file, batch_count, 
                             queue_enabled, queue_type, prompt_queue_file_path, image_queue_files,
                             get_image_queue_files_callback):
    """
    キュー設定・初期化処理の統合関数
    
    Args:
        use_queue: UIからのキュー使用フラグ
        prompt_queue_file: アップロードされたプロンプトキューファイル
        batch_count: バッチ処理回数
        queue_enabled: グローバルキュー有効フラグ
        queue_type: キューのタイプ
        prompt_queue_file_path: プロンプトキューファイルパス
        image_queue_files: イメージキューファイルリスト
        get_image_queue_files_callback: イメージキューファイル取得関数
        
    Returns:
        tuple: (queue_enabled, queue_type, prompt_queue_file_path, batch_count, total_needed_batches)
    """
    # キュー関連の設定を保存
    queue_enabled = bool(use_queue)  # UIからの値をブール型に変換
    total_needed_batches = batch_count  # デフォルト値
    
    # プロンプトキューファイルが指定されている場合はパスを保存
    if queue_enabled and prompt_queue_file is not None:
        if hasattr(prompt_queue_file, 'name') and os.path.exists(prompt_queue_file.name):
            prompt_queue_file_path = prompt_queue_file.name
            queue_type = "prompt"  # キュータイプをプロンプトに設定
            print(translate("プロンプトキューファイルパスを設定: {0}").format(prompt_queue_file_path))
            
            # プロンプトファイルの内容を確認し、行数を出力
            try:
                with open(prompt_queue_file_path, 'r', encoding='utf-8') as f:
                    prompt_lines = [line.strip() for line in f.readlines() if line.strip()]
                    prompt_count = len(prompt_lines)
                    if prompt_count > 0:
                        # 🚨 重要修正: プロンプト行数に合わせてバッチ数を調整
                        if prompt_count != batch_count:
                            print(translate("プロンプト数に合わせてバッチ数を自動調整: {0} → {1}").format(batch_count, prompt_count))
                            batch_count = prompt_count
                            total_needed_batches = prompt_count
                        print(translate("バッチ処理回数: {0}回（プロンプトキュー行: {1}行）").format(batch_count, prompt_count))
                    else:
                        print(translate("バッチ処理回数: {0}回").format(batch_count))
            except Exception as e:
                print(translate("プロンプトキューファイル読み込みエラー: {0}").format(str(e)))
                print(translate("プロンプトキューファイルを無視して処理を続行します: {0}").format(
                    prompt_queue_file.name if hasattr(prompt_queue_file, 'name') else "不明なファイル"))
                queue_enabled = False
                queue_type = "prompt"
    
    # イメージキューが有効な場合の処理
    elif queue_enabled and use_queue:
        # イメージキューの処理（プロンプトキューが指定されていない場合）
        queue_type = "image"  # キュータイプをイメージに設定
        
        # イメージキューファイルを取得
        if get_image_queue_files_callback:
            updated_image_queue_files = get_image_queue_files_callback()
            # 更新された値を使用
            image_queue_files = updated_image_queue_files
        
        # イメージキューの確認
        image_queue_count = len(image_queue_files)
        print(translate("イメージキュー: {0}個の画像ファイルを読み込みました").format(image_queue_count))
        
        if image_queue_count > 0:
            # 🚨 重要修正: 入力画像1回+キュー画像数に合わせてバッチ数を調整
            total_needed_batches = 1 + image_queue_count
            if total_needed_batches > batch_count:
                print(translate("画像キュー数+1に合わせてバッチ数を自動調整: {0} → {1}").format(batch_count, total_needed_batches))
                batch_count = total_needed_batches
            
            print(translate("バッチ処理回数: {0}回（入力画像1回+キュー画像{1}個）").format(batch_count, image_queue_count))
        else:
            print(translate("イメージキューに画像ファイルが見つかりません"))
    
    return queue_enabled, queue_type, prompt_queue_file_path, batch_count, total_needed_batches, image_queue_files


def generate_batch_summary(queue_enabled, queue_type, prompt_queue_file_path, image_queue_files, batch_count):
    """
    バッチ処理サマリー出力の統合関数
    
    Args:
        queue_enabled: キュー機能有効フラグ
        queue_type: キューのタイプ（"prompt" または "image"）
        prompt_queue_file_path: プロンプトキューファイルパス
        image_queue_files: イメージキューファイルリスト
        batch_count: バッチ処理回数
        
    Returns:
        None (ログ出力のみ)
    """
    # バッチ処理のサマリーを出力
    if queue_enabled:
        if queue_type == "prompt" and prompt_queue_file_path is not None:
            # プロンプトキュー情報をログに出力
            try:
                with open(prompt_queue_file_path, 'r', encoding='utf-8') as f:
                    queue_lines = [line.strip() for line in f.readlines() if line.strip()]
                    queue_lines_count = len(queue_lines)
                    print(translate("バッチ処理情報: 合計{0}回").format(batch_count))
                    print(translate("プロンプトキュー: 有効, プロンプト行数={0}行").format(queue_lines_count))

                    # 各プロンプトの概要を出力
                    print(translate("プロンプトキュー内容:"))
                    for i in range(min(batch_count, queue_lines_count)):
                        prompt_preview = queue_lines[i][:50] + "..." if len(queue_lines[i]) > 50 else queue_lines[i]
                        print(translate("   └ バッチ{0}: {1}").format(i+1, prompt_preview))
            except:
                pass
        elif queue_type == "image" and len(image_queue_files) > 0:
            # イメージキュー情報をログに出力
            print(translate("バッチ処理情報: 合計{0}回").format(batch_count))
            print(translate("イメージキュー: 有効, 画像ファイル数={0}個").format(len(image_queue_files)))

            # 各画像ファイルの概要を出力
            print(translate("イメージキュー内容:"))
            print(translate("   └ バッチ1: 入力画像 (最初のバッチは常に入力画像を使用)"))
            for i, img_path in enumerate(image_queue_files[:min(batch_count-1, len(image_queue_files))]):
                img_name = os.path.basename(img_path)
                print(translate("   └ バッチ{0}: {1}").format(i+2, img_name))
    else:
        print(translate("バッチ処理情報: 合計{0}回").format(batch_count))
        print(translate("キュー機能: 無効"))


def apply_queue_to_batch(queue_enabled, queue_type, prompt_queue_file_path, image_queue_files,
                        batch_index, batch_count, prompt, input_image):
    """
    バッチループ内キュー適用の統合関数
    
    Args:
        queue_enabled: キュー機能有効フラグ
        queue_type: キューのタイプ（"prompt" または "image"）
        prompt_queue_file_path: プロンプトキューファイルパス
        image_queue_files: イメージキューファイルリスト
        batch_index: 現在のバッチインデックス
        batch_count: 総バッチ処理回数
        prompt: 元のプロンプト
        input_image: 元の入力画像
        
    Returns:
        tuple: (current_prompt, current_image)
    """
    # 今回処理用のプロンプトとイメージを取得（キュー機能対応）
    current_prompt = prompt
    current_image = input_image

    # キュー機能の処理
    if queue_enabled:
        if queue_type == "prompt" and prompt_queue_file_path is not None:
            # プロンプトキューの処理
            if os.path.exists(prompt_queue_file_path):
                try:
                    with open(prompt_queue_file_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        if batch_index < len(lines):
                            # プロンプトキューからプロンプトを取得
                            current_prompt = lines[batch_index]
                            print(translate("プロンプトキュー実行中: バッチ {0}/{1}").format(batch_index+1, batch_count))
                            print(translate("  └ プロンプト: 「{0}...」").format(current_prompt[:50]))
                        else:
                            print(translate("プロンプトキュー実行中: バッチ {0}/{1} はプロンプト行数を超えているため元のプロンプトを使用").format(batch_index+1, batch_count))
                except Exception as e:
                    print(translate("プロンプトキューファイル読み込みエラー: {0}").format(str(e)))

        elif queue_type == "image" and len(image_queue_files) > 0:
            # イメージキューの処理
            # 最初のバッチは入力画像を使用
            if batch_index == 0:
                print(translate("イメージキュー実行中: バッチ {0}/{1} は入力画像を使用").format(batch_index+1, batch_count))
            elif batch_index > 0:
                # 2回目以降はイメージキューの画像を順番に使用
                image_index = batch_index - 1  # 0回目（入力画像）の分を引く

                if image_index < len(image_queue_files):
                    current_image = image_queue_files[image_index]
                    image_filename = os.path.basename(current_image)
                    print(translate("イメージキュー実行中: バッチ {0}/{1} の画像「{2}」").format(batch_index+1, batch_count, image_filename))
                    print(translate("  └ 画像ファイルパス: {0}").format(current_image))
                    
                    # 同名のテキストファイルがあるか確認し、あれば内容をプロンプトとして使用
                    img_basename = os.path.splitext(current_image)[0]
                    txt_path = f"{img_basename}.txt"
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                custom_prompt = f.read().strip()
                            if custom_prompt:
                                print(translate("イメージキュー: 画像「{0}」用のテキストファイルを読み込みました").format(image_filename))
                                print(translate("カスタムプロンプト: {0}").format(custom_prompt[:50] + "..." if len(custom_prompt) > 50 else custom_prompt))
                                # カスタムプロンプトを設定（current_promptを上書き）
                                current_prompt = custom_prompt
                        except Exception as e:
                            print(translate("イメージキュー: テキストファイル読み込みエラー: {0}").format(e))
                else:
                    # イメージキューの画像が足りない場合は入力画像を使用
                    print(translate("イメージキュー実行中: バッチ {0}/{1} は画像数を超えているため入力画像を使用").format(batch_index+1, batch_count))

    return current_prompt, current_image


def get_batch_info_message(batch_count, batch_index):
    """
    バッチ情報メッセージの生成
    
    Args:
        batch_count: 総バッチ処理回数
        batch_index: 現在のバッチインデックス
        
    Returns:
        str: バッチ情報メッセージ
    """
    if batch_count > 1:
        return translate("バッチ処理: {0}/{1}").format(batch_index + 1, batch_count)
    return ""