"""
HuggingFace DLモデル共有設定

FramePack-eichi2/oichi2プロジェクト用DLモデル共有設定モジュール
設定ファイルベースの可変パス管理、コンソール入力対応、モデル存在チェック機能を提供します。
"""

import os
from typing import Dict, List, Optional, Tuple

import torch

from .settings_manager import load_hf_settings, save_hf_settings, add_shared_model_path
from locales.i18n_extended import translate


def get_model_cache_base_path() -> str:
    """
    設定ファイルベースのモデルキャッシュベースパス取得。
    
    1. 設定ファイルから共有パスリストを読み込み
    2. 存在する最初のパスを使用
    3. なければユーザー入力を促す
    4. 最終的にローカルパスにフォールバック
    
    Returns:
        str: モデルキャッシュベースパス
    """
    # 設定を読み込み
    hf_settings: Dict[str, any] = load_hf_settings()
    
    # 設定済み共有パスから有効なものを探す
    valid_path = find_valid_shared_path(hf_settings.get('shared_model_paths', []))
    
    if valid_path:
        return valid_path
    
    # 有効なパスがない場合（shared_model_pathsが空または無効な場合のみプロンプト表示）
    shared_paths = hf_settings.get('shared_model_paths', [])
    
    # shared_model_pathsが設定されていない、または空の場合のみユーザー入力を促す
    if not shared_paths and hf_settings.get('prompt_for_path', True):
        # ユーザー入力を促す
        new_path = prompt_for_model_path()
        if new_path and os.path.exists(new_path):
            # 新しいパスを設定に追加
            add_shared_model_path(new_path)
            print(f"[モデル設定] 新しい共有パスを設定: {new_path}")
            return new_path
        else:
            # ローカルフォルダを選択した場合、今後プロンプトを表示しない
            hf_settings['prompt_for_path'] = False
            save_hf_settings(hf_settings)
    elif shared_paths:
        print(f"[モデル設定] 設定済み共有パスが無効です: {shared_paths}")
        print(f"[モデル設定] パスが存在しないため、ローカルパスにフォールバックします")
    
    # フォールバック: ローカルパス
    local_path = get_default_local_path(hf_settings.get('local_model_path', 'hf_download'))
    
    return local_path

def find_valid_shared_path(shared_paths: List[str]) -> Optional[str]:
    """
    共有パスリストから存在する最初のパスを見つけます。
    
    Args:
        shared_paths: 共有パスのリスト
        
    Returns:
        Optional[str]: 存在する最初のパス、なければNone
    """
    for path in shared_paths:
        if path and os.path.exists(path) and os.path.isdir(path):
            return os.path.abspath(path)
    return None

def prompt_for_model_path() -> Optional[str]:
    """
    ユーザーにモデル格納パスの入力を促します。
    
    Returns:
        Optional[str]: 入力されたパス、キャンセル時はNone
    """
    print("\n" + "="*60)
    print(translate("HuggingFaceモデル格納場所の設定"))
    print("="*60)
    print(translate("モデルが見つかりません。格納場所を設定してください。"))
    print(translate("既存の共有フォルダがある場合はそのパスを入力してください。"))
    print(translate("（例: C:\\Models\\webui\\hf_download または /home/user/models/webui/hf_download）"))
    print(translate("何も入力せずEnterを押すとローカルフォルダを使用します。"))
    print("-" * 60)
    
    try:
        user_input = input(translate("モデル格納パス: ")).strip()
        
        if not user_input:
            print(translate("ローカルフォルダを使用します。"))
            return None
        
        # パスの正規化
        user_path = os.path.abspath(os.path.expanduser(user_input))
        
        # ディレクトリ存在チェック
        if not os.path.exists(user_path):
            create = input(translate("フォルダが存在しません。作成しますか？ (y/n): ")).strip().lower()
            if create in ['y', 'yes']:
                try:
                    os.makedirs(user_path, exist_ok=True)
                    print(translate("フォルダを作成しました: {0}").format(user_path))
                except OSError as e:
                    print(translate("フォルダ作成に失敗しました: {0}").format(e))
                    return None
            else:
                return None
        
        if not os.path.isdir(user_path):
            print(translate("指定されたパスはディレクトリではありません。"))
            return None
            
        print(translate("パスが設定されました: {0}").format(user_path))
        return user_path
        
    except KeyboardInterrupt:
        print(translate("\nキャンセルされました。"))
        return None
    except Exception as e:
        print(translate("入力エラー: {0}").format(e))
        return None

def get_default_local_path(local_path_name: str) -> str:
    """
    デフォルトのローカルパスを取得します。
    
    Args:
        local_path_name: ローカルパス名
        
    Returns:
        str: ローカルパスの絶対パス
    """
    webui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(webui_dir, local_path_name)

def check_required_models() -> Tuple[bool, List[str]]:
    """
    必要なモデルの存在をチェックします。
    
    Returns:
        Tuple[bool, List[str]]: (全て存在するか, 不足モデルリスト)
    """
    model_cache_base = get_model_cache_base_path()  # ログ出力は無効化済み
    if not model_cache_base or not os.path.exists(model_cache_base):
        return False, ["Model cache directory not found"]
    
    # HuggingFaceの新しいキャッシュ形式をチェック（hubフォルダ内）
    hub_path = os.path.join(model_cache_base, 'hub')
    
    # チェック対象のモデル一覧
    required_models = [
        "models--hunyuanvideo-community--HunyuanVideo",
        "models--lllyasviel--flux_redux_bfl", 
        "models--lllyasviel--FramePackI2V_HY"
    ]
    
    missing_models = []
    for model in required_models:
        # 新形式: hub/models--repo--name
        model_path = os.path.join(hub_path, model)
        if not os.path.exists(model_path):
            missing_models.append(model.replace("models--", ""))  # 表示用に元の名前に戻す
    
    return len(missing_models) == 0, missing_models

def download_all_required_models() -> bool:
    """
    本家FramePack方式によるモデルダウンロード。
    
    本家のModelDownloaderクラスと同じ方式で、snapshot_downloadを使用して
    モデルを順次ダウンロードします。(tqdm競合回避のため1個ずつ実行)
    
    Returns:
        bool: ダウンロード成功時True
    """
    print("[モデル設定] 全モデルダウンロードを開始します...")
    print("[モデル設定] 本家FramePack方式でダウンロード中...")
    
    # HF_HOME環境変数の確認（batファイルで設定済み）
    hf_home = os.environ.get('HF_HOME')
    if not hf_home:
        print("[エラー] HF_HOME環境変数が設定されていません")
        return False
    
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from huggingface_hub import snapshot_download
        
        # 本家ModelDownloaderと同じ設定
        # max_parallel_models > 1 は tqdm._lock が競合し異常終了するため、当面は1に固定する
        max_parallel_models = 1  # 本家と同じくtqdm競合回避のため1に固定
        max_workers_per_model = 4
        
        models_to_download = [
            {
                "repo_id": "hunyuanvideo-community/HunyuanVideo", 
                "allow_patterns": ["tokenizer/*", "tokenizer_2/*", "vae/*", "text_encoder/*", "text_encoder_2/*"]
            },
            {
                "repo_id": "lllyasviel/flux_redux_bfl", 
                "allow_patterns": ["feature_extractor/*", "image_encoder/*"]
            },
            {
                "repo_id": "lllyasviel/FramePackI2V_HY"
            }
        ]
        
        def download_model(model_info):
            print(f"[ダウンロード] {model_info['repo_id']} を開始...")
            kwargs = {
                "repo_id": model_info["repo_id"],
                "allow_patterns": model_info.get("allow_patterns", "*"),
                "max_workers": max_workers_per_model,
            }
            snapshot_download(**kwargs)
            print(f"[ダウンロード] {model_info['repo_id']} が完了しました")
        
        print(f"[モデル設定] モデルを順次ダウンロード開始... (tqdm競合回避のため1モデルずつ実行)")
        
        with ThreadPoolExecutor(max_workers=max_parallel_models) as executor:
            futures = [executor.submit(download_model, model) for model in models_to_download]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[モデル設定] 個別モデルダウンロードエラー: {e}")
                    return False
        
        print("[モデル設定] 全モデルダウンロードが完了しました")
        return True
        
    except Exception as e:
        print(f"[モデル設定] ダウンロードエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_model_cache_directory() -> bool:
    """
    モデルキャッシュディレクトリの存在確認と作成。
    
    設定ファイルで指定されたディレクトリが存在しない場合、
    新しくディレクトリを作成します。
    
    Returns:
        bool: ディレクトリが存在するか作成に成功した場合True
    """
    model_cache_base = get_model_cache_base_path()  # ログ出力は無効化済み
    if model_cache_base and not os.path.exists(model_cache_base):
        os.makedirs(model_cache_base, exist_ok=True)
        print(f"モデルキャッシュディレクトリを作成: {model_cache_base}")
    
    return os.path.exists(model_cache_base) if model_cache_base else False

def setup_models_with_check() -> bool:
    """
    モデル存在チェックとセットアップを統合実行します。
    
    1. モデルキャッシュパス設定（設定ファイルベース）
    2. モデル存在チェック
    3. 不足時のダウンロード実行
    
    Returns:
        bool: セットアップ完了時True
    """
    print("[モデル設定] モデルセットアップを開始します...")
    
    # 1. モデルキャッシュパス設定（設定ファイルベース・ユーザー選択可能）
    model_cache_base = get_model_cache_base_path()
    verify_model_cache_directory()
    
    # 2. モデル存在チェック
    models_exist, missing_models = check_required_models()
    
    if models_exist:
        print("[モデル設定] 必要なモデルは全て存在します。")
        return True
    
    # 3. 不足モデルの報告
    print(f"[モデル設定] 不足しているモデル: {len(missing_models)}個")
    for model in missing_models:
        print(f"  - {model}")
    
    # 4. ダウンロード実行
    print("\n" + "="*60)
    print(translate("モデルが不足しています"))
    print("="*60)
    print(translate("以下のモデルが必要です："))
    print(translate("• HunyuanVideo (基本モデル)"))
    print(translate("• flux_redux_bfl (画像エンコーダー)"))
    print(translate("• FramePackI2V_HY (Transformerモデル)"))
    print()
    print(translate("解決方法："))
    print(translate("1. このツールで直接ダウンロードする (推奨)"))
    print(translate("2. 本家FramePackでモデルをダウンロードしてください"))
    print(translate("3. 既存のモデルフォルダがある場合は、設定でパスを指定してください"))
    print("-" * 60)
    
    try:
        while True:  # 有効な選択肢が入力されるまでループ
            print(translate("選択肢："))
            print(translate("d: 直接ダウンロードを試行"))
            print(translate("p: 別のモデルフォルダを指定"))
            print(translate("q: 終了"))
            choice = input(translate("選択 (d/p/q): ")).strip().lower()
            
            if choice in ['d', 'download', 'ダウンロード']:
                # 直接ダウンロードを試行
                print("[モデル設定] 直接ダウンロードを開始します...")
                download_success = download_all_required_models()
                if download_success:
                    print("[モデル設定] ダウンロードが成功しました。続行します。")
                    return True
                else:
                    print("[モデル設定] ダウンロードに失敗しました。")
                    print("[モデル設定] アプリケーションを終了します。")
                    import sys
                    sys.exit(1)
                    
            elif choice in ['p', 'path', 'パス']:
                # 新しいパスの入力を促す
                new_path = prompt_for_model_path()
                if new_path and os.path.exists(new_path):
                    # 新しいパスを設定に追加
                    add_shared_model_path(new_path)
                    print(f"[モデル設定] 新しい共有パスを設定: {new_path}")
                    # 再チェック
                    models_exist_recheck, _ = check_required_models()
                    if models_exist_recheck:
                        print("[モデル設定] モデルが見つかりました。続行します。")
                        return True
                    else:
                        print("[モデル設定] 指定されたフォルダにもモデルが見つかりませんでした。")
                        return False
                else:
                    return False
                        
            elif choice in ['q', 'quit', '終了']:
                print(translate("[モデル設定] モデルが不足しているため、アプリケーションを終了します。"))
                import sys
                sys.exit(0)
            else:
                # 無効な選択肢の場合 - 再入力を促す
                print(translate("[モデル設定] 無効な選択肢です。d, p, q のいずれかを入力してください。"))
                print()  # 空行で見やすくする
                continue  # ループを継続して再入力を求める
        
    except KeyboardInterrupt:
        print(translate("\n[モデル設定] 操作がキャンセルされました。"))
        return False
    except Exception as e:
        print(translate("入力エラー: {0}").format(e))
        return False

if __name__ == "__main__":
    # テスト実行
    print("=== モデルキャッシュパス設定テスト ===")
    cache_path: str = get_model_cache_base_path()
    verified: bool = verify_model_cache_directory()
    print(f"設定パス: {cache_path}")
    print(f"ディレクトリ確認: {verified}")