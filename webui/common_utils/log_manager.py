"""
ログ管理モジュール

コンソール出力をファイルにもリダイレクトする機能を提供します。
プログレスバーの間引き処理、タイムスタンプ付きログ、リアルタイムファイル出力に対応しています。
"""

import atexit
import datetime
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, TextIO, Union

from locales.i18n_extended import translate

# グローバル変数
_log_enabled: bool = False
_log_folder: str = "logs"  # デフォルトのログフォルダ
_log_file: Optional[TextIO] = None  # ログファイルのハンドル
_original_stdout: TextIO = sys.stdout  # 元のstdoutを保存
_original_stderr: TextIO = sys.stderr  # 元のstderrを保存
_last_progress_percent: Optional[int] = None  # 最後に記録した進捗率
_progress_log_interval: int = 10    # 進捗率の記録間隔（%単位）

# webuiディレクトリのパスを取得
_webui_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_absolute_path(path: str) -> str:
    """
    相対パスを絶対パスに変換します。
    
    Args:
        path: 変換対象パス
        
    Returns:
        str: 絶対パス
    """
    if os.path.isabs(path):
        return path
    else:
        return os.path.normpath(os.path.join(_webui_path, path))

class LoggerWriter:
    """
    標準出力をラップしてログファイルにも書き込むクラス。
    
    コンソール出力とログファイル出力を同時に行い、
    プログレスバーの間引き処理やタイムスタンプ付きログを実現します。
    """
    
    def __init__(self, original_stream: TextIO, log_file: TextIO) -> None:
        """
        LoggerWriterインスタンスを初期化します。
        
        Args:
            original_stream: 元の出力ストリーム（stdoutまstderr）
            log_file: ログファイルのハンドル
        """
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, text: str) -> None:
        """
        テキストを元のストリームとログファイルに書き込みます。
        
        Args:
            text: 書き込み対象テキスト
        """
        # 元のストリームに書き込む - コンソールには元のテキストをそのまま表示
        self.original_stream.write(text)
        
        # ログファイルにも書き込む（タイムスタンプ付き）
        if self.log_file and not self.log_file.closed:
            # 空の文字列はスキップ
            if not text.strip():
                return
                
            # プログレスバーチェック - 単純なパターンマッチング
            is_progress_bar = False
            if '\r' in text or '%' in text and ('|' in text or '/' in text):
                is_progress_bar = True
                
            # タイムスタンプを生成
            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            
            if is_progress_bar:
                # プログレスバー処理
                global _last_progress_percent, _progress_log_interval
                
                # テキストからパーセンテージを抽出
                import re
                progress_match = re.search(r'(\d+)%', text)
                if progress_match:
                    percent = int(progress_match.group(1))
                    
                    # 進捗率の間引き処理
                    should_log = (
                        _last_progress_percent is None or  # 最初の進捗
                        percent == 0 or  # 開始
                        percent == 100 or  # 完了
                        _last_progress_percent is not None and 
                        (abs(percent - _last_progress_percent) >= _progress_log_interval)  # 間隔を超えた
                    )
                    
                    if should_log:
                        _last_progress_percent = percent
                        # 追加情報を抽出（例: 25/100）
                        count_match = re.search(r'(\d+)/(\d+)', text)
                        if count_match:
                            current, total = count_match.groups()
                            self.log_file.write(f"{timestamp} [PROGRESS] {percent}% ({current}/{total})\n")
                        else:
                            self.log_file.write(f"{timestamp} [PROGRESS] {percent}%\n")
                else:
                    # パーセンテージが見つからない場合はシンプルにログ
                    progress_text = text.strip().replace('\r', '').split('\n')[-1][:50]
                    self.log_file.write(f"{timestamp} [PROGRESS] {progress_text}\n")
            else:
                # 通常のテキスト - 改行で分割して各行にタイムスタンプを付与
                for line in text.split('\n'):
                    if line.strip():  # 空行をスキップ
                        self.log_file.write(f"{timestamp} {line}\n")
            
            self.log_file.flush()  # すぐに書き込みを反映
    
    def flush(self) -> None:
        """出力ストリームとログファイルをフラッシュします。"""
        self.original_stream.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()
    
    def fileno(self) -> int:
        """元のストリームのファイル記述子を返します。"""
        return self.original_stream.fileno()
    
    def isatty(self) -> bool:
        """元のストリームがttyかどうかを返します。"""
        return self.original_stream.isatty()

def enable_logging(log_folder: Optional[str] = None, source_name: str = "endframe_ichi") -> bool:
    """
    ログ機能を有効化し、標準出力をログファイルにリダイレクトします。
    
    Args:
        log_folder: ログフォルダのパス。Noneの場合はデフォルトフォルダを使用
        source_name: ソースファイル名（拡張子無し）
        
    Returns:
        bool: ログ機能の有効化に成功した場合True
    """
    global _log_enabled, _log_folder, _log_file, _original_stdout, _original_stderr, _last_progress_percent
    
    if log_folder:
        # 相対パスを絶対パスに変換
        _log_folder = get_absolute_path(log_folder)
    
    # ログディレクトリが存在しない場合は作成
    try:
        if not os.path.exists(_log_folder):
            os.makedirs(_log_folder, exist_ok=True)
            print(translate("ログフォルダを作成しました: {0}").format(_log_folder))
    except Exception as e:
        print(translate("ログフォルダの作成に失敗しました: {0} - {1}").format(_log_folder, str(e)))
        return False
    
    # 既にログが有効な場合は一度無効化してから再設定
    if _log_enabled:
        disable_logging()
    
    # ソースファイル名からベース名を取得
    # パスが含まれていたり、拡張子が付いていた場合は除去
    base_name = os.path.basename(source_name)
    if base_name.endswith(".py"):
        base_name = base_name[:-3]  # .pyを除去
    
    # 現在の日時をファイル名に使用
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base_name}_{timestamp}.log"
    log_filepath = os.path.join(_log_folder, log_filename)
    
    try:
        # ログファイルを開く
        _log_file = open(log_filepath, "w", encoding="utf-8")
        
        # ログヘッダー情報を書き込む
        header = f"=== FramePack: {base_name} ログ開始 {timestamp} ===\n"
        _log_file.write(header)
        _log_file.flush()
        
        # 標準出力をラッパークラスで置き換え
        _original_stdout = sys.stdout
        sys.stdout = LoggerWriter(_original_stdout, _log_file)
        
        # 標準エラー出力もリダイレクト（プログレスバーはstderrに出力されることが多い）
        _original_stderr = sys.stderr
        sys.stderr = LoggerWriter(_original_stderr, _log_file)
        
        _log_enabled = True
        print(translate("ログ出力を有効化しました: {0}").format(log_filepath))
        return True
    except Exception as e:
        print(translate("ログ出力の設定に失敗しました: {0}").format(str(e)))
        # エラーが発生した場合、元の標準出力に戻す
        sys.stdout = _original_stdout
        return False

def disable_logging() -> bool:
    """
    ログ機能を無効化し、標準出力と標準エラー出力を元に戻します。
    
    Returns:
        bool: ログ機能の無効化に成功した場合True
    """
    global _log_enabled, _log_file, _original_stdout, _original_stderr, _last_progress_percent
    
    if not _log_enabled:
        return
    
    try:
        # 標準出力を元に戻す
        sys.stdout = _original_stdout
        
        # 標準エラー出力を元に戻す
        sys.stderr = _original_stderr
        
        # 進捗情報をリセット
        _last_progress_percent = None
        
        # ログファイルを閉じる
        if _log_file and not _log_file.closed:
            # ログフッター情報を書き込む
            footer = f"\n=== ログ終了 {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')} ===\n"
            _log_file.write(footer)
            _log_file.close()
        
        _log_enabled = False
        print(translate("ログ出力を無効化しました"))
        return True
    except Exception as e:
        print(translate("ログ出力の停止に失敗しました: {0}").format(str(e)))
        return False

def is_logging_enabled() -> bool:
    """
    ログ機能が有効かどうかを返します。
    
    Returns:
        bool: ログ機能が有効な場合True
    """
    return _log_enabled

def get_log_folder() -> str:
    """
    現在のログフォルダを返します。
    
    Returns:
        str: ログフォルダのパス
    """
    return _log_folder

def set_log_folder(folder_path: str) -> bool:
    """
    ログフォルダを設定します。
    
    Args:
        folder_path: 設定するログフォルダのパス
        
    Returns:
        bool: 設定に成功した場合True
    """
    global _log_folder
    
    print(f"[DEBUG] set_log_folder呼び出し: 現在={_log_folder}, 新規={folder_path}")
    
    # 現在ログが有効な場合は一度無効化する
    if _log_enabled:
        disable_logging()
    
    # 相対パスを絶対パスに変換
    _log_folder = get_absolute_path(folder_path)
    
    # フォルダが存在しなければ作成
    if not os.path.exists(_log_folder):
        try:
            os.makedirs(_log_folder, exist_ok=True)
            print(f"[DEBUG] 新しいログフォルダを作成: {_log_folder}")
        except Exception as e:
            print(f"[WARNING] ログフォルダの作成に失敗: {e}")
    
    print(f"[DEBUG] ログフォルダを設定: {os.path.abspath(_log_folder)}")
    return True

def open_log_folder() -> bool:
    """
    ログフォルダをOSに依存せず開きます。
    
    Returns:
        bool: フォルダを開くことに成功した場合True
    """
    folder_path = _log_folder
    
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            print(translate("ログフォルダの作成に失敗しました: {0} - {1}").format(folder_path, str(e)))
            return False
    
    try:
        if os.name == 'nt':  # Windows
            import subprocess
            subprocess.Popen(['explorer', folder_path])
        elif os.name == 'posix':  # Linux/Mac
            import subprocess
            try:
                subprocess.Popen(['xdg-open', folder_path])
            except:
                subprocess.Popen(['open', folder_path])
        print(translate("ログフォルダを開きました: {0}").format(folder_path))
        return True
    except Exception as e:
        print(translate("ログフォルダを開く際にエラーが発生しました: {0}").format(e))
        return False

def get_default_log_settings() -> Dict[str, Union[bool, str]]:
    """
    デフォルトのログ設定を返します。
    
    Returns:
        Dict[str, Union[bool, str]]: デフォルトログ設定辞書
    """
    # 必ず決まった値を返すようにする
    settings: Dict[str, Union[bool, str]] = {
        "log_enabled": False,
        "log_folder": "logs"
    }
    return settings

def load_log_settings(app_settings: Optional[Dict[str, Union[bool, str]]]) -> Dict[str, Union[bool, str]]:
    """
    設定からログ設定を読み込みます。
    
    Args:
        app_settings: アプリケーション設定辞書
        
    Returns:
        Dict[str, Union[bool, str]]: ログ設定辞書
    """
    if not app_settings:
        return get_default_log_settings()
    
    log_settings: Dict[str, Union[bool, str]] = {
        "log_enabled": app_settings.get("log_enabled", False),
        "log_folder": app_settings.get("log_folder", "logs")
    }
    
    return log_settings

def apply_log_settings(log_settings: Optional[Dict[str, Union[bool, str]]], source_name: str = "endframe_ichi") -> bool:
    """
    ログ設定を適用します。
    
    Args:
        log_settings: ログ設定辞書
        source_name: ソースファイル名（拡張子無し）
        
    Returns:
        bool: 設定の適用に成功した場合True
    """
    if not log_settings:
        print("[WARNING] ログ設定がNoneです")
        return False
    
    print(f"[DEBUG] apply_log_settings: {log_settings}, source={source_name}")
    
    # 現在のログ状態を保存
    was_enabled = is_logging_enabled()
    print(f"[DEBUG] 現在のログ状態: {was_enabled}")
    
    # 一旦ログを無効化（既存のファイルを閉じる）
    if was_enabled:
        print("[DEBUG] 既存のログを一旦無効化します")
        disable_logging()
    
    # ログフォルダを設定
    folder: str = log_settings.get("log_folder", "logs")
    print(f"[DEBUG] ログフォルダ設定: {folder}")
    set_log_folder(folder)
    
    # ログ有効/無効を設定
    is_enabled: bool = log_settings.get("log_enabled", False)
    print(f"[DEBUG] ログ有効設定: {is_enabled}")
    
    if is_enabled:
        print("[DEBUG] ログを有効化します")
        success: bool = enable_logging(log_folder=folder, source_name=source_name)
        print(f"[DEBUG] ログ有効化結果: {success}")
    
    return True

# プログラム終了時に自動的にログを閉じるようにする
atexit.register(disable_logging)