"""
LoRA設定管理モジュール

指定数値での可変設定対応、個別強度サポート、プリセット機能を提供します。
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

class LoRAConfig:
    """
    LoRA設定クラス（個別強度対応）。
    
    ファイルアップロードとディレクトリ選択の両方に対応し、
    個別強度設定と後方互換性を保持します。
    """
    
    def __init__(self, max_count: int = 3) -> None:
        """
        LoRAConfigインスタンスを初期化します。
        
        Args:
            max_count: 最大LoRA数（デフォルト: 3）
        """
        self.max_count: int = max_count
        self.files: List[Optional[Any]] = [None] * max_count
        self.dropdowns: List[str] = ["なし"] * max_count
        # 個別強度対応
        self.individual_scales: List[float] = [0.8] * max_count
        # 後方互換性維持のためのカンマ区切り文字列
        self.scales_text: str = ",".join([str(0.8)] * max_count)
        self.mode: str = "ファイルアップロード"
    
    def set_files(self, files_list: List[Any]) -> None:
        """
        ファイルリストを設定します。
        
        Args:
            files_list: 設定するLoRAファイルのリスト
        """
        for i, files in enumerate(files_list[:self.max_count]):
            if files:
                self.files[i] = files[0] if isinstance(files, list) else files
    
    def set_dropdowns(self, dropdowns_list: List[str]) -> None:
        """
        ドロップダウンリストを設定します。
        
        Args:
            dropdowns_list: 設定するLoRA選択項目のリスト
        """
        for i, dropdown in enumerate(dropdowns_list[:self.max_count]):
            if dropdown:
                self.dropdowns[i] = dropdown
    
    def set_scales_text(self, scales_text: str) -> None:
        """
        スケールテキストを設定します（後方互換用）。
        
        Args:
            scales_text: カンマ区切りの強度数値文字列
        """
        self.scales_text = scales_text or self.scales_text
        # カンマ区切り文字列から個別強度配列に変換
        try:
            scales = [float(x.strip()) for x in scales_text.split(',') if x.strip()]
            for i, scale in enumerate(scales[:self.max_count]):
                self.individual_scales[i] = scale
        except:
            pass  # エラー時は現在の値を保持
    
    def set_individual_scales(self, scales_list: List[float]) -> None:
        """
        個別強度リストを設定します。
        
        Args:
            scales_list: 設定する強度値のリスト
        """
        for i, scale in enumerate(scales_list[:self.max_count]):
            if isinstance(scale, (int, float)):
                self.individual_scales[i] = float(scale)
        # 後方互換用文字列も更新
        self.scales_text = ",".join([str(scale) for scale in self.individual_scales])
    
    def set_individual_scale(self, index: int, scale: float) -> None:
        """
        指定インデックスの個別強度を設定します。
        
        Args:
            index: 設定対象のインデックス
            scale: 設定する強度値
        """
        if 0 <= index < self.max_count:
            self.individual_scales[index] = float(scale)
            # 後方互換用文字列も更新
            self.scales_text = ",".join([str(scale) for scale in self.individual_scales])
    
    def get_individual_scales(self) -> List[float]:
        """
        個別強度リストを取得します。
        
        Returns:
            List[float]: 個別強度値のコピー
        """
        return self.individual_scales.copy()
    
    def set_mode(self, mode: str) -> None:
        """
        LoRA選択モードを設定します。
        
        Args:
            mode: モード文字列（"ファイルアップロード" または "ディレクトリから選択"）
        """
        self.mode = mode or "ファイルアップロード"
    
    def get_active_loras(self) -> Tuple[List[str], List[float]]:
        """
        有効なLoRA一覧を取得します（個別強度対応）。
        
        Returns:
            Tuple[List[str], List[float]]: （LoRAパスリスト、強度リスト）のタプル
        """
        paths: List[str] = []
        strengths: List[float] = []
        
        for i in range(self.max_count):
            effective_path: Optional[str] = None
            
            if self.mode == "ファイルアップロード" and self.files[i]:
                effective_path = self.files[i]
            elif self.mode == "ディレクトリから選択" and self.dropdowns[i] != "なし":
                # LoRAディレクトリパス構築
                webui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                lora_dir = os.path.join(webui_dir, 'lora')
                effective_path = os.path.join(lora_dir, self.dropdowns[i])
            
            if effective_path and os.path.exists(effective_path):
                paths.append(effective_path)
                strengths.append(self.individual_scales[i])
        
        return paths, strengths


class LoRASettings:
    """
    LoRA設定ファイル管理クラス。
    
    プリセットファイル優先読み込みシステムで最大LoRA数を管理し、
    後方互換性を保ちながら新機能に対応します。
    """
    
    CONFIG_FILE: str = "lora_settings.json"
    
    def __init__(self) -> None:
        """
        LoRASettingsインスタンスを初期化します。
        
        webuiディレクトリから設定ファイルのパスを構築し、
        設定ディレクトリを作成します。
        """
        webui_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir: str = os.path.join(webui_dir, "settings")
        self.config_path: str = os.path.join(self.config_dir, self.CONFIG_FILE)
        os.makedirs(self.config_dir, exist_ok=True)
    
    def get_max_count(self) -> int:
        """
        最大LoRA数を取得します（プリセットファイル優先）。
        
        プリセットファイルからの読み込みを優先し、
        存在しない場合は従来のlora_settings.jsonから読み込み、
        どちらもない場合はデフォルト値を返します。
        
        Returns:
            int: 最大LoRA数（デフォルト: 6）
        """
        # プリセットファイルから読み込み試行
        try:
            webui_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            preset_path: str = os.path.join(webui_dir, "presets", "lora_presets.json")
            if os.path.exists(preset_path):
                with open(preset_path, 'r', encoding='utf-8') as f:
                    preset_config: Dict[str, Any] = json.load(f)
                    global_settings: Dict[str, Any] = preset_config.get("global_settings", {})
                    max_count: Optional[int] = global_settings.get("max_lora_count")
                    if max_count and isinstance(max_count, int) and 1 <= max_count <= 20:
                        return max_count
        except:
            pass
        
        # 従来のlora_settings.jsonから読み込み（後方互換）
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config: Dict[str, Any] = json.load(f)
                return config.get("max_count", 3)
        except:
            pass
        return 6  # デフォルト6個
    
    def set_max_count(self, max_count: int) -> bool:
        """
        最大LoRA数を設定します（プリセットファイルに保存）。
        
        Args:
            max_count: 設定する最大LoRA数（1-20の範囲）
            
        Returns:
            bool: 設定の保存に成功した場合True
        """
        if not (1 <= max_count <= 20):
            return False
        
        try:
            # プリセットファイルに保存
            webui_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            preset_path: str = os.path.join(webui_dir, "presets", "lora_presets.json")
            
            # 既存プリセットファイル読み込み
            preset_config: Dict[str, Any] = {}
            if os.path.exists(preset_path):
                with open(preset_path, 'r', encoding='utf-8') as f:
                    preset_config = json.load(f)
            
            # global_settings更新
            if "global_settings" not in preset_config:
                preset_config["global_settings"] = {}
            preset_config["global_settings"]["max_lora_count"] = max_count
            preset_config["global_settings"]["version"] = "1.1.0"
            
            # プリセット構造が存在しない場合の初期化
            if "presets" not in preset_config:
                preset_config["presets"] = []
            if "default_preset_index" not in preset_config:
                preset_config["default_preset_index"] = 0
            
            # プリセットファイルに保存
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset_config, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"LoRA数設定保存エラー: {e}")
            return False


# グローバルインスタンス
_lora_settings: Optional[LoRASettings] = None

def get_lora_settings() -> LoRASettings:
    """
    グローバル設定インスタンスを取得します。
    
    Returns:
        LoRASettings: グローバル設定インスタンス
    """
    global _lora_settings
    if _lora_settings is None:
        _lora_settings = LoRASettings()
    return _lora_settings

def get_max_lora_count() -> int:
    """
    最大LoRA数を取得します（便利関数）。
    
    Returns:
        int: 最大LoRA数
    """
    return get_lora_settings().get_max_count()

def create_lora_config(max_count: Optional[int] = None) -> LoRAConfig:
    """
    LoRA設定オブジェクトを作成します。
    
    Args:
        max_count: 最大LoRA数。Noneの場合は設定から自動取得
        
    Returns:
        LoRAConfig: 新しいLoRA設定インスタンス
    """
    if max_count is None:
        max_count = get_max_lora_count()
    return LoRAConfig(max_count)


if __name__ == "__main__":
    # テスト
    print("=== LoRA設定管理 テスト ===")
    
    settings: LoRASettings = get_lora_settings()
    print(f"現在の最大LoRA数: {settings.get_max_count()}")
    
    # 5個に変更テスト
    if settings.set_max_count(5):
        print("最大LoRA数を5に変更")
        print(f"変更後: {settings.get_max_count()}")
        
        # 設定オブジェクトテスト
        config: LoRAConfig = create_lora_config()
        print(f"LoRA設定作成: max_count={config.max_count}")
        
        # 元に戻す
        settings.set_max_count(3)
        print(f"復元後: {settings.get_max_count()}")
    
    print("=== テスト完了 ===")