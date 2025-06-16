"""
FramePack-eichi/oichi LoRA Utilities Module

アーキテクチャ横断LoRA処理エンジン:
- HunyuanVideo→FramePack自動変換システム
- 複数LoRA同時適用システム
- FP8量子化最適化エンジン（E4M3FN/E5M2対応）
- メモリ効率的safetensorsストリーミングローダー
"""

# コアLoRA処理エンジン: アーキテクチャ間の変換・マージ機能
from .lora_utils import (
    merge_lora_to_state_dict,                      # 複数LoRAファイルの統合マージ
    load_safetensors_with_lora_and_fp8,            # LoRA+FP8統合処理
    load_safetensors_with_fp8_optimization,        # FP8専用最適化ローダー
    convert_hunyuan_to_framepack,                  # HunyuanVideo→FramePack自動変換
    convert_from_diffusion_pipe_or_something       # Diffusers→Musubiフォーマット変換
)

# FP8量子化最適化エンジン: RTX40シリーズ対応高速化
from .fp8_optimization_utils import (
    calculate_fp8_maxval,                         # FP8範囲限界計算（E4M3FN/E5M2対応）
    quantize_tensor_to_fp8,                       # テンソルのFP8量子化アルゴリズム
    optimize_state_dict_with_fp8_on_the_fly,      # 状態辞書一括量子化処理
    fp8_linear_forward_patch,                     # FP8線形レイヤーフォワードパッチ
    apply_fp8_monkey_patch,                       # モンキーパッチ適用システム
    check_fp8_support                             # PyTorch FP8サポート確認
)

# 高レベルLoRA統合ローダー: ワンストップインターフェース
from .lora_loader import (
    load_and_apply_lora                           # 複数LoRA+FP8統合ロードシステム
)

# メモリ効率safetensorsエンジン: ストリーミング読み込みシステム
from .safetensors_utils import (
    MemoryEfficientSafeOpen                       # FP8対応セクション別読み込みクラス
)

__version__ = "1.0.0"  # lora2_utils統合モジュール初期リリース
