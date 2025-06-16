"""
FramePack-oichi2 統一バージョン管理

アプリケーション全体のバージョン情報とメタデータを定義
"""

__version__ = "0.1.0"
__app_name__ = "FramePack-oichi2"
__description__ = "FramePack 1フレーム推論版"
__author__ = "Code"
__license__ = "MIT2.0"

# バージョン情報取得関数
def get_version() -> str:
    """アプリケーションバージョンを取得"""
    return __version__

def get_app_name() -> str:
    """アプリケーション名を取得"""
    return __app_name__

def get_description() -> str:
    """アプリケーション説明を取得"""
    return __description__

def get_full_name() -> str:
    """フルアプリケーション名を取得"""
    return f"{__app_name__} v{__version__}"

def get_version_info() -> dict:
    """バージョン情報辞書を取得"""
    return {
        "version": __version__,
        "app_name": __app_name__,
        "description": __description__,
        "author": __author__,
        "license": __license__
    }