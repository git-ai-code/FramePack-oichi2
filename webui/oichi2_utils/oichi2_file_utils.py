"""
FramePack-oichi2 ファイル・パス管理統合モジュール

ファイル・フォルダ管理処理の分離・最適化
- フォルダ作成・オープン処理
- 画像キューファイル検索・管理
- 入力フォルダ設定・保存処理
- パス処理・ファイル探索ユーティリティ
- OS依存処理の統合・抽象化
"""

import glob
import os
import subprocess

from common_utils.settings_manager import load_settings, save_settings
from locales.i18n_extended import translate


def open_folder(folder_path):
    """
    指定されたフォルダをOSに依存せず開く
    
    Args:
        folder_path: 開くフォルダのパス
        
    Returns:
        bool: 成功時True、失敗時False
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(translate("フォルダを作成しました: {0}").format(folder_path))

    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen(['explorer', folder_path])
        elif os.name == 'posix':  # Linux/Mac
            try:
                subprocess.Popen(['xdg-open', folder_path])
            except:
                subprocess.Popen(['open', folder_path])
        print(translate("フォルダを開きました: {0}").format(folder_path))
        return True
    except Exception as e:
        print(translate("フォルダを開く際にエラーが発生しました: {0}").format(e))
        return False


def get_image_queue_files(base_path, input_folder_name):
    """
    入力フォルダから画像ファイルを取得してリストを返す
    
    Args:
        base_path: ベースパス
        input_folder_name: 入力フォルダ名
        
    Returns:
        list: 画像ファイルパスのリスト
    """
    # 入力フォルダの設定
    input_folder = os.path.join(base_path, input_folder_name)

    # フォルダが存在しない場合は作成
    os.makedirs(input_folder, exist_ok=True)

    # すべての画像ファイルを取得
    image_exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    image_files = []

    # 同じファイルが複数回追加されないようにセットを使用
    file_set = set()

    for ext in image_exts:
        pattern = os.path.join(input_folder, '*' + ext)
        for file in glob.glob(pattern):
            if file not in file_set:
                file_set.add(file)
                image_files.append(file)

        pattern = os.path.join(input_folder, '*' + ext.upper())
        for file in glob.glob(pattern):
            if file not in file_set:
                file_set.add(file)
                image_files.append(file)

    # ファイルを修正日時の昇順でソート
    image_files.sort(key=lambda x: os.path.getmtime(x))

    print(translate("入力ディレクトリから画像ファイル{0}個を読み込みました").format(len(image_files)))

    return image_files


def update_input_folder_name(folder_name):
    """
    入力フォルダ名の変更を処理（設定保存はしない）
    
    Args:
        folder_name: 新しいフォルダ名
        
    Returns:
        str: サニタイズされたフォルダ名
    """
    # 無効な文字を削除（パス区切り文字やファイル名に使えない文字）
    sanitized_name = ''.join(c for c in folder_name if c.isalnum() or c in ('_', '-'))
    print(translate("入力フォルダ名をメモリに保存: {0}（保存及び入力フォルダを開くボタンを押すと保存されます）").format(sanitized_name))
    return sanitized_name


def open_input_folder_with_save(input_folder_name, webui_folder, get_image_queue_files_callback):
    """
    入力フォルダを開く処理（設定保存も実行）
    
    Args:
        input_folder_name: 入力フォルダ名
        webui_folder: webuiフォルダパス
        get_image_queue_files_callback: 画像キューファイル取得コールバック
        
    Returns:
        None
    """
    # 念のため設定を保存
    settings = load_settings()
    settings['input_folder'] = input_folder_name
    save_settings(settings)
    print(translate("入力フォルダ設定を保存しました: {0}").format(input_folder_name))

    # 入力フォルダのパスを取得
    input_dir = os.path.join(webui_folder, input_folder_name)

    # フォルダが存在しなければ作成
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(translate("入力ディレクトリを作成しました: {0}").format(input_dir))

    # 画像ファイルリストを更新
    if get_image_queue_files_callback:
        get_image_queue_files_callback()

    # フォルダを開く
    open_folder(input_dir)


def setup_folder_structure(webui_folder):
    """
    必要なフォルダ構造を初期化
    
    Args:
        webui_folder: webuiフォルダパス
        
    Returns:
        tuple: (base_path, settings_folder)
    """
    base_path = webui_folder  # endframe_ichiとの互換性のため
    
    # 設定保存用フォルダの設定
    settings_folder = os.path.join(webui_folder, 'settings')
    os.makedirs(settings_folder, exist_ok=True)
    
    return base_path, settings_folder


def setup_output_folder(output_folder_name, get_output_folder_path_callback):
    """
    出力フォルダの設定・作成
    
    Args:
        output_folder_name: 出力フォルダ名
        get_output_folder_path_callback: 出力フォルダパス取得コールバック
        
    Returns:
        str: 出力フォルダのフルパス
    """
    # 出力フォルダのフルパスを生成
    outputs_folder = get_output_folder_path_callback(output_folder_name)
    os.makedirs(outputs_folder, exist_ok=True)
    return outputs_folder


def sanitize_filename(filename):
    """
    ファイル名をサニタイズ（無効な文字を除去）
    
    Args:
        filename: サニタイズするファイル名
        
    Returns:
        str: サニタイズされたファイル名
    """
    # パス区切り文字やファイル名に使えない文字を除去
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    
    # 連続するスペースを単一スペースに変換
    filename = ' '.join(filename.split())
    
    # 先頭・末尾の空白を除去
    filename = filename.strip()
    
    return filename


def ensure_directory_exists(directory_path):
    """
    ディレクトリが存在することを確認し、必要に応じて作成
    
    Args:
        directory_path: 確認・作成するディレクトリパス
        
    Returns:
        bool: 成功時True、失敗時False
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(translate("ディレクトリ作成エラー: {0} - {1}").format(directory_path, str(e)))
        return False


def get_file_list_by_extension(directory, extensions, sort_by_mtime=True):
    """
    指定されたディレクトリから特定の拡張子のファイルリストを取得
    
    Args:
        directory: 検索するディレクトリ
        extensions: 拡張子のリスト（例: ['.jpg', '.png']）
        sort_by_mtime: 修正時刻でソートするかどうか
        
    Returns:
        list: ファイルパスのリスト
    """
    if not os.path.exists(directory):
        return []
    
    files = []
    file_set = set()
    
    for ext in extensions:
        # 小文字の拡張子
        pattern = os.path.join(directory, '*' + ext.lower())
        for file in glob.glob(pattern):
            if file not in file_set:
                file_set.add(file)
                files.append(file)
        
        # 大文字の拡張子
        pattern = os.path.join(directory, '*' + ext.upper())
        for file in glob.glob(pattern):
            if file not in file_set:
                file_set.add(file)
                files.append(file)
    
    if sort_by_mtime and files:
        files.sort(key=lambda x: os.path.getmtime(x))
    
    return files