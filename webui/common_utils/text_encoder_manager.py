"""
text encoder管理モジュール

text_encoderとtext_encoder_2のライフサイクル管理を行います。
VRAMモード対応、動的スワップ機能、メモリ解放機能を提供します。
"""

import gc
import os
import traceback
from typing import Any, Dict, Optional, Tuple, Union

import torch

from diffusers_helper.memory import DynamicSwapInstaller
from locales.i18n_extended import translate


def _get_local_text_encoder_path(model_name: str, subfolder: str = None) -> str:
    """HuggingFaceキャッシュから実際のテキストエンコーダーパスを取得"""
    from common_utils.hf_config import get_model_cache_base_path
    
    model_cache_base = get_model_cache_base_path()
    model_cache_dir = os.path.join(model_cache_base, 'hub', f'models--{model_name.replace("/", "--")}')
    
    # snapshotsフォルダ内の最初（最新）のハッシュフォルダを使用
    snapshots_dir = os.path.join(model_cache_dir, 'snapshots')
    if os.path.exists(snapshots_dir):
        snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if snapshots:
            base_path = os.path.join(snapshots_dir, snapshots[0])
            if subfolder:
                return os.path.join(base_path, subfolder)
            return base_path
    
    # フォールバック: 元の文字列形式
    return model_name

class TextEncoderManager:
    """
    text_encoderとtext_encoder_2の状態管理クラス。
    
    このクラスは以下の責務を持ちます：
    - text_encoderとtext_encoder_2のライフサイクル管理
    - VRAMモード対応と動的スワップ機能
    - メモリの効率的な管理と解放
    
    設定の変更はすぐには適用されず、次回のリロード時に適用されます。
    """

    def __init__(self, device: Union[str, torch.device], high_vram_mode: bool = False) -> None:
        """
        TextEncoderManagerインスタンスを初期化します。
        
        Args:
            device: 使用するデバイス
            high_vram_mode: High-VRAMモードの有効/無効
        """
        self.text_encoder: Optional[Any] = None
        self.text_encoder_2: Optional[Any] = None
        self.device = device

        # 現在適用されている設定
        self.current_state: Dict[str, Any] = {
            'is_loaded': False,
            'high_vram': high_vram_mode
        }

        # 次回のロード時に適用する設定
        self.next_state: Dict[str, Any] = self.current_state.copy()
        
    def set_next_settings(self, high_vram_mode: bool = False) -> None:
        """
        次回のロード時に使用する設定をセットします（即時のリロードは行わない）。
        
        Args:
            high_vram_mode: High-VRAMモードの有効/無効
        """
        self.next_state = {
            'high_vram': high_vram_mode,
            'is_loaded': self.current_state['is_loaded']
        }
        print(translate("次回のtext_encoder設定を設定しました:"))
        print(translate("  - High-VRAM mode: {0}").format(high_vram_mode))
    
    def _needs_reload(self) -> bool:
        """
        現在の状態と次回の設定を比較し、リロードが必要かどうかを判断します。
        
        Returns:
            bool: リロードが必要な場合True
        """
        if not self._is_loaded():
            return True

        # High-VRAMモードの比較
        if self.current_state['high_vram'] != self.next_state['high_vram']:
            return True

        return False
    
    def _is_loaded(self) -> bool:
        """
        text_encoderとtext_encoder_2が読み込まれているかどうかを確認します。
        
        Returns:
            bool: 両方が読み込まれている場合True
        """
        return (self.text_encoder is not None and 
                self.text_encoder_2 is not None and 
                self.current_state['is_loaded'])
    
    def get_text_encoders(self):
        """現在のtext_encoderとtext_encoder_2インスタンスを取得"""
        return self.text_encoder, self.text_encoder_2

    def dispose_text_encoders(self):
        """text_encoderとtext_encoder_2のインスタンスを破棄し、メモリを完全に解放"""
        try:
            print(translate("text_encoderとtext_encoder_2のメモリを解放します..."))
            
            # text_encoderの破棄
            if hasattr(self, 'text_encoder') and self.text_encoder is not None:
                try:
                    self.text_encoder.cpu()
                    del self.text_encoder
                    self.text_encoder = None
                except Exception as e:
                    print(translate("text_encoderの破棄中にエラー: {0}").format(e))

            # text_encoder_2の破棄
            if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
                try:
                    self.text_encoder_2.cpu()
                    del self.text_encoder_2
                    self.text_encoder_2 = None
                except Exception as e:
                    print(translate("text_encoder_2の破棄中にエラー: {0}").format(e))

            # 明示的なメモリ解放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # 状態を更新
            self.current_state['is_loaded'] = False
            self.next_state['is_loaded'] = False
            
            print(translate("text_encoderとtext_encoder_2のメモリ解放が完了しました"))
            return True
            
        except Exception as e:
            print(translate("text_encoderとtext_encoder_2のメモリ解放中にエラー: {0}").format(e))
            traceback.print_exc()
            return False

    def ensure_text_encoder_state(self):
        """text_encoderとtext_encoder_2の状態を確認し、必要に応じてリロード"""
        if self._needs_reload():
            print(translate("text_encoderとtext_encoder_2をリロードします"))
            return self._reload_text_encoders()        
        print(translate("ロード済みのtext_encoderとtext_encoder_2を再度利用します"))
        return True
    
    def _reload_text_encoders(self) -> bool:
        """
        next_stateの設定でtext_encoderとtext_encoder_2をリロードします。
        
        Returns:
            bool: リロードに成功した場合True
        """
        try:
            # 既存のモデルが存在する場合は先にメモリを解放
            if self._is_loaded():
                self.dispose_text_encoders()

            # 新しいtext_encoderとtext_encoder_2インスタンスを作成
            from transformers import LlamaModel, CLIPTextModel
            
            # ローカルキャッシュからテキストエンコーダーパスを取得
            text_encoder_path = _get_local_text_encoder_path("hunyuanvideo-community/HunyuanVideo", "text_encoder")
            text_encoder_2_path = _get_local_text_encoder_path("hunyuanvideo-community/HunyuanVideo", "text_encoder_2")
            
            self.text_encoder = LlamaModel.from_pretrained(
                text_encoder_path, 
                torch_dtype=torch.float16,
                local_files_only=True
            ).cpu()
            
            self.text_encoder_2 = CLIPTextModel.from_pretrained(
                text_encoder_2_path, 
                torch_dtype=torch.float16,
                local_files_only=True
            ).cpu()
            
            self.text_encoder.eval()
            self.text_encoder_2.eval()
            
            self.text_encoder.to(dtype=torch.float16)
            self.text_encoder_2.to(dtype=torch.float16)
            
            self.text_encoder.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
            
            # VRAMモードに応じた設定
            if not self.next_state['high_vram']:
                DynamicSwapInstaller.install_model(self.text_encoder, device=self.device)
                DynamicSwapInstaller.install_model(self.text_encoder_2, device=self.device)
            else:
                self.text_encoder.to(self.device)
                self.text_encoder_2.to(self.device)
            
            # 状態を更新
            self.next_state['is_loaded'] = True
            self.current_state = self.next_state.copy()
            
            print(translate("text_encoderとtext_encoder_2のリロードが完了しました"))
            return True
            
        except Exception as e:
            print(translate("text_encoderとtext_encoder_2リロードエラー: {0}").format(e))
            traceback.print_exc()
            self.current_state['is_loaded'] = False
            return False 
