"""
FramePack-eichi/oichi FP8最適化モジュール
モデルを8ビット浮動小数点形式に量子化して、メモリ使用量と処理速度を最適化するモジュールです。

基本的な特徴:
- E4M3およびE5M2 FP8フォーマットのサポート
- 異なるGPUアーキテクチャに対する最適化
- モンキーパッチによる透過的な統合
- RTX 40シリーズ向けのscaled_mm最適化対応
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 国際化対応
from locales.i18n_extended import translate

# 警告メッセージが表示されたかを追跡するフラグ
FP8_E4M3_WARNING_SHOWN = False
FP8_DIMENSIONS_WARNING_SHOWN = False

def calculate_fp8_maxval(exp_bits=4, mantissa_bits=3, sign_bits=1):
    """
    FP8形式で表現可能な最大値を計算
    デフォルトはE4M3形式（4ビット指数部、3ビット仮数部、1ビット符号部）

    Args:
        exp_bits (int): 指数部のビット数
        mantissa_bits (int): 仮数部のビット数
        sign_bits (int): 符号部のビット数（0または1）

    Returns:
        float: FP8形式で表現可能な最大値
    """
    assert exp_bits + mantissa_bits + sign_bits == 8, translate("合計ビット数は8でなければなりません")

    # 指数バイアスを計算
    bias = 2 ** (exp_bits - 1) - 1

    # 最大仮数値を計算
    mantissa_max = 1.0
    for i in range(mantissa_bits - 1):
        mantissa_max += 2 ** -(i + 1)

    # 最大値を計算
    max_value = mantissa_max * (2 ** (2**exp_bits - 1 - bias))

    return max_value

def quantize_tensor_to_fp8(tensor, scale, exp_bits=4, mantissa_bits=3, sign_bits=1, max_value=None, min_value=None):
    """
    テンソルをFP8形式に量子化するアルゴリズム
    
    FP8量子化の仕組み:
    - スケーリング: tensor / scale で値範囲を正規化
    - クランピング: FP8範囲内への強制制限（オーバーフロー防止）
    - 指数スケール計算: log2ベースで最適な指数を算出
    - 仮数部丸め: 精度と範囲のバランスを考慮した丸め処理

    Args:
        tensor (torch.Tensor): 量子化対象のテンソル（任意の形状対応）
        scale (float or torch.Tensor): 量子化スケールファクター（結果に影響）
        exp_bits (int): 指数部ビット数（デフォルト4=E4M3FN）
        mantissa_bits (int): 仮数部ビット数（デフォルト3=E4M3FN）
        sign_bits (int): 符号部ビット数（通常1=符号付き）
        max_value (float, optional): 範囲上限（None=自動計算）
        min_value (float, optional): 範囲下限（None=自動計算）

    Returns:
        tuple: (量子化済みテンソル, 元のスケールファクター)
    """
    # スケーリング: テンソル値をFP8範囲に正規化
    scaled_tensor = tensor / scale  # スケール割りで量子化範囲に調整

    # FP8指数バイアス計算（IEEE 754規格ベース）
    bias = 2 ** (exp_bits - 1) - 1  # 指数部のゼロオフセット値

    # FP8範囲限界の自動計算または外部指定値使用
    if max_value is None:
        max_value = calculate_fp8_maxval(exp_bits, mantissa_bits, sign_bits)  # 理論上の最大値
        min_value = -max_value if sign_bits > 0 else 0.0  # 符号付き/符号なし対応

    # オーバーフロー防止のためのFP8範囲内への強制クランピング
    clamped_tensor = torch.clamp(scaled_tensor, min_value, max_value)  # 範囲外値を強制的に制限

    # FP8量子化アルゴリズム: 指数スケールベースの非線形量子化
    abs_values = torch.abs(clamped_tensor)  # 絶対値で符号と値を分離
    nonzero_mask = abs_values > 0          # ゼロ値例外処理用マスク

    # 各要素に対する最適指数スケールの計算（非ゼロ値のみ）
    log_scales = torch.zeros_like(clamped_tensor)  # 指数スケールマップ初期化
    if nonzero_mask.any():
        # log2ベースで最適指数を算出、floorで下位丸め
        log_scales[nonzero_mask] = torch.floor(torch.log2(abs_values[nonzero_mask]) + bias).detach()

    # 指数スケールの範囲制限と量子化ステップサイズ計算
    log_scales = torch.clamp(log_scales, min=1.0)  # 最小指数スケール制限
    quant_factor = 2.0 ** (log_scales - mantissa_bits - bias)  # 量子化ステップサイズ

    # 最終量子化: 丸め処理で離散値へ変換後、連続値へ逆変換
    quantized = torch.round(clamped_tensor / quant_factor) * quant_factor  # 量子化ステップに合わせて丸め

    return quantized, scale  # 量子化済みテンソルと元のスケールを返却

def optimize_state_dict_with_fp8_on_the_fly(
    model_files,
    calc_device,
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits=4,
    mantissa_bits=3,
    move_to_device=False,
    weight_hook=None,
):
    """
    モデルの状態辞書内の線形レイヤーをFP8形式に最適化してメモリ使用量を削減
    
    FP8最適化の仕組み:
    - 重みをE4M3またはE5M2形式に量子化
    - スケールファクターを別途保存
    - FramePackアーキテクチャにtransformer_blocks/single_transformer_blocksを対象

    Args:
        model_files: 最適化対象モデルファイル一覧
        calc_device: FP8量子化計算用デバイス
        target_layer_keys: 最適化対象レイヤーパターン（デフォルト:全線形レイヤー）
        exclude_layer_keys: 最適化除外レイヤーパターン（例:normレイヤー）
        exp_bits: FP8指数部ビット数（デフォルト:4）
        mantissa_bits: FP8仮数部ビット数（デフォルト:3）
        move_to_device: 最適化後のデバイス配置制御
        weight_hook: LoRAマージ等の重み前処理フック関数

    Returns:
        FP8最適化された状態辞書（重み+スケールファクター）
    """
    # FP8データ型の選択（指数部と仮数部のビット数に基づく）
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(translate("サポートされていないFP8形式: E{0}M{1}").format(exp_bits, mantissa_bits))

    # FP8形式で表現可能な最大値を計算
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # 符号付きFP8のみサポート（符号ビット=1）

    # FramePackアーキテクチャ対応のFP8最適化状態辞書を作成

    def is_target_key(key):
        # FP8最適化対象キーの判定ロジック
        # 1. .weightで終わるキーかつ、対象パターンにマッチ
        # 2. 除外パターン（normレイヤー等）にはマッチしない
        is_target = (target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        is_target = is_target and not is_excluded
        return is_target

    # FP8最適化されたレイヤー数のカウンター
    optimized_count = 0
    
    # Musubi Tunerスタイルのメモリ効率的ファイル読み込みを使用        
    from lora2_utils.safetensors_utils import MemoryEfficientSafeOpen

    state_dict = {}
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            keys = f.keys()
            for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                value = f.get_tensor(key)
                if weight_hook is not None:
                    # LoRAマージ等の重み前処理フックをFP8最適化前に適用
                    value = weight_hook(key, value)

                if not is_target_key(key):
                    state_dict[key] = value  # 最適化対象外のキーはそのまま保存
                    continue

                # FP8量子化処理のためのデバイス・データ型管理
                original_device = value.device
                original_dtype = value.dtype  # スケールファクターのデータ型用

                # GPUでの高速量子化計算のために指定デバイスへ移動
                if calc_device is not None:
                    value = value.to(calc_device)

                # 重みテンソルの絶対値最大値からFP8スケールファクターを算出
                scale = torch.max(torch.abs(value.flatten())) / max_value

                # 重みテンソルをFP8形式に量子化（メモリ使用量削減）
                quantized_weight, _ = quantize_tensor_to_fp8(value, scale, exp_bits, mantissa_bits, 1, max_value, min_value)

                # FP8最適化済みモデルのキー命名規則
                fp8_key = key  # 重みは元のキー名を維持
                scale_key = key.replace(".weight", ".scale_weight")  # スケールは専用キー

                # PyTorchのFP8データ型に変換（E4M3FNまたはE5M2）
                quantized_weight = quantized_weight.to(fp8_dtype)

                # FP8最適化後のデバイス配置管理
                if not move_to_device:
                    quantized_weight = quantized_weight.to(original_device)  # 元のデバイスに戻す

                # 逆量子化時に使用するスケールファクターテンソルを作成
                scale_tensor = torch.tensor([scale], dtype=original_dtype, device=quantized_weight.device)

                # FP8最適化された重みとスケールを状態辞書に保存
                state_dict[fp8_key] = quantized_weight  # 量子化済み重み
                state_dict[scale_key] = scale_tensor    # 逆量子化用スケール

                optimized_count += 1

                # GPUメモリの定期的クリーンアップ（10レイヤー毎）
                if calc_device is not None and optimized_count % 10 == 0:
                    torch.cuda.empty_cache()

    print(translate("最適化された線形レイヤー数: {0}").format(optimized_count))
    return state_dict

def fp8_linear_forward_patch(self: nn.Linear, x, use_scaled_mm=False, max_value=None):
    """
    FP8量子化済み重みを持つ線形レイヤーの特化フォワード処理
    
    処理モード:
    1. scaled_mmモード: RTX40シリーズのTensor Core最適化を使用
       - E4M3FN重み専用・3次元入力テンソル限定
       - ハードウェアレベルの高速行列積演算
    2. 標準モード: 逆量子化後に通常のF.linear()を使用
       - 汎用性重視・すべての環境で動作保証

    Args:
        self (nn.Linear): FP8重みとscale_weightを持つ線形レイヤー
        x (torch.Tensor): 入力テンソル（fp16/bf16推奨）
        use_scaled_mm (bool): Tensor Core最適化有効化（RTX40+必須）
        max_value (float): 入力量子化用上限値（None=入力非量子化）

    Returns:
        torch.Tensor: 線形変換結果（入力と同じデータ型）
    """
    if use_scaled_mm:
        # RTX40シリーズ特化: ハードウェアレベル最適化モード
        input_dtype = x.dtype                    # 元の入力データ型（結果返却用）
        original_weight_dtype = self.scale_weight.dtype  # スケールファクターのデータ型
        weight_dtype = self.weight.dtype         # FP8重みのデータ型（E4M3FN確認用）
        target_dtype = torch.float8_e5m2        # 入力テンソルのE5M2変換用

        # scaled_mmのデータ型制約: E4M3FN形式以外はサポート外
        # 理由: Tensor CoreのE4M3FN特化回路でのみハードウェア最適化を実現
        global FP8_E4M3_WARNING_SHOWN
        if weight_dtype != torch.float8_e4m3fn:
            if not FP8_E4M3_WARNING_SHOWN:  # 重複警告防止フラグ
                print(translate("警告: scaled_mmはFP8 E4M3FN形式を必要としますが、{weight_dtype}が検出されました。通常方式にフォールバックします。").format(weight_dtype=weight_dtype))
                FP8_E4M3_WARNING_SHOWN = True
            # 再帰呼び出しで標準モードにフォールバック
            return fp8_linear_forward_patch(self, x, False, max_value)

        # scaled_mmの入力形状制約: 3次元テンソル(batch, seq, hidden)限定
        # 理由: Tensor Coreの内部仕様でバッチ処理とシーケンシャル処理を前提
        global FP8_DIMENSIONS_WARNING_SHOWN
        if x.ndim != 3:
            if not FP8_DIMENSIONS_WARNING_SHOWN:  # 重複警告防止フラグ
                print(translate("警告: scaled_mmは3次元入力が必要ですが、{0}次元が検出されました。通常方式にフォールバックします。").format(x.ndim))
                FP8_DIMENSIONS_WARNING_SHOWN = True
            # 再帰呼び出しで標準モードにフォールバック
            return fp8_linear_forward_patch(self, x, False, max_value)

        # 入力テンソルの量子化戦略（通常は無効でfp16/bf16維持推奨）
        if max_value is None:
            # 非量子化モード: 入力はfp16/bf16のまま、スケールは1.0固定
            scale_x = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        else:
            # 実験的入力量子化モード: 入力もE5M2形式に変換
            scale_x = (torch.max(torch.abs(x.flatten())) / max_value).to(torch.float32)  # 入力スケール算出

            # 入力をE5M2形式に量子化（メモリ使用量増加のリスクあり）
            x, _ = quantize_tensor_to_fp8(x, scale_x, 5, 2, 1, max_value, -max_value)  # E5M2パラメータ

        # scaled_mm用のテンソル形状変換: 3D→2D→指定データ型
        original_shape = x.shape                # 復元用の元形状保存
        x = x.reshape(-1, x.shape[2]).to(target_dtype)  # (batch*seq, hidden)形状+E5M2変換

        # FP8重みの前処理: 転置とスケールファクター準備
        weight = self.weight.t()              # 重みマトリックス転置（scaled_mm仕様）
        scale_weight = self.scale_weight.to(torch.float32)  # スケールをfloat32で統一

        # Tensor Core最適化行列積演算（バイアス有無で分岐）
        if self.bias is not None:
            # バイアス付き: out_dtypeの制約あり（float32不可）
            o = torch._scaled_mm(x, weight, out_dtype=original_weight_dtype, bias=self.bias, scale_a=scale_x, scale_b=scale_weight)
        else:
            # バイアスなし: 入力データ型で出力
            o = torch._scaled_mm(x, weight, out_dtype=input_dtype, scale_a=scale_x, scale_b=scale_weight)

        # 結果テンソルの形状・データ型復元
        return o.reshape(original_shape[0], original_shape[1], -1).to(input_dtype)
    else:
        # 標準モード: FP8重みを元の精度に逆量子化後に通常の線形変換
        original_dtype = self.scale_weight.dtype  # 逆量子化先のデータ型
        # FP8→元精度への逆変換: weight * scale_factor
        dequantized_weight = self.weight.to(original_dtype) * self.scale_weight

        # PyTorch標準の線形変換関数で汎用性を確保
        if self.bias is not None:
            output = F.linear(x, dequantized_weight, self.bias)  # バイアス付き線形変換
        else:
            output = F.linear(x, dequantized_weight)              # バイアスなし線形変換

        return output  # 入力と同じデータ型・デバイスで結果返却

def apply_fp8_monkey_patch(model, optimized_state_dict, use_scaled_mm=False):
    """
    FP8重み用の専用フォワードメソッドをモデルに動的注入
    
    モンキーパッチングの仕組み:
    - .scale_weightキーの存在で量子化済みレイヤーを自動検出
    - register_buffer()でスケールファクターをモデル構造に組み込み
    - __get__()メソッドバインドで元のforwardメソッドを完全置換
    - インプレース変更でメモリ効率を最大化

    Args:
        model (nn.Module): パッチ対象のモデルインスタンス（書き換え対象）
        optimized_state_dict (dict): .scale_weightキーを含むFP8最適化済み状態辞書
        use_scaled_mm (bool): RTX40シリーズのTensor Core最適化有効化フラグ

    Returns:
        nn.Module: フォワードメソッドが置換されたモデル（同一インスタンス）
    """
    # 入力テンソル量子化無効化（FP8重みのみ量子化・入力はfp16/bf16維持）
    max_value = None  # None=入力量子化無効、値指定=入力もFP8量子化（実験的）

    # 量子化済みレイヤーの自動検出（.scale_weightサフィックスパターンマッチング）
    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]

    # モジュールパス抽出とパッチ対象セット構築
    patched_module_paths = set()
    for scale_key in scale_keys:
        # "layer.weight" → "layer"（.scale_weightサフィックス除去でベースパス取得）
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)

    patched_count = 0

    # モデル全体を走査してFP8対応線形レイヤーにパッチ適用
    for name, module in model.named_modules():
        # スケールファクター存在確認（量子化済みレイヤー判定）
        has_scale = name in patched_module_paths

        # nn.Linear + .scale_weight の組み合わせでパッチ対象確定
        if isinstance(module, nn.Linear) and has_scale:
            # scale_weightバッファ登録（state_dict読み込み用の準備）
            # 初期値1.0はダミー、実際の値はload_state_dict()で上書き
            module.register_buffer("scale_weight", torch.tensor(1.0, dtype=module.weight.dtype))

            # クロージャーでuse_scaled_mm・max_valueを捕獲したフォワード関数生成
            def new_forward(self, x):
                return fp8_linear_forward_patch(self, x, use_scaled_mm, max_value)

            # __get__()によるメソッドバインド（self引数の自動注入を有効化）
            module.forward = new_forward.__get__(module, type(module))

            patched_count += 1

    print(translate("モンキーパッチ適用済みの線形レイヤー数: {0}").format(patched_count))
    # FP8最適化完了マーカー設定（外部システムからの状態確認用）
    model._fp8_optimized = True  # hasattr(model, '_fp8_optimized')での判定が可能
    return model  # インプレース変更済みの同一インスタンスを返却

def check_fp8_support():
    """
    PyTorchのFP8機能とハードウェア最適化の対応状況を診断
    
    チェック項目:
    - float8_e4m3fn: 重み・推論用FP8（PyTorch 2.1+必須）
    - float8_e5m2: 勾配・アクティベーション用FP8（PyTorch 2.1+必須）
    - _scaled_mm: RTX40シリーズTensor Core最適化（PyTorch 2.2+推奨）

    Returns:
        tuple: (E4M3FN対応, E5M2対応, scaled_mm対応) のbool値
    """
    # PyTorchビルド時のFP8データ型対応確認（コンパイル時オプション依存）
    has_e4m3 = hasattr(torch, 'float8_e4m3fn')  # 重み量子化用・finite形式
    has_e5m2 = hasattr(torch, 'float8_e5m2')    # 勾配計算用・広範囲形式

    # ハードウェア最適化関数の存在確認（CUDA 11.8+、cuDNN 8.9+要求）
    has_scaled_mm = hasattr(torch, '_scaled_mm')  # RTX40専用高速行列積関数

    # ユーザー向け対応状況レポート
    if has_e4m3 and has_e5m2:
        if has_scaled_mm:
            print(translate("RTX 40シリーズのGPUでFP8の高速化が可能です"))  # 完全対応
    else:
        print(translate("警告: FP8サポートが検出されませんでした。PyTorch 2.1以上が必要です"))  # 要アップグレード

    return has_e4m3, has_e5m2, has_scaled_mm

def reset_fp8_warning_flags():
    """
    FP8固有警告フラグのリセット（生成開始時の初期化用）
    
    管理対象フラグ:
    - FP8_E4M3_WARNING_SHOWN: scaled_mmのE4M3FN要求警告制御
    - FP8_DIMENSIONS_WARNING_SHOWN: scaled_mmの3次元テンソル要求警告制御
    
    使用タイミング: 新しい画像・動画生成プロセス開始時
    """
    global FP8_E4M3_WARNING_SHOWN, FP8_DIMENSIONS_WARNING_SHOWN
    FP8_E4M3_WARNING_SHOWN = False      # E4M3FN形式要求警告を再表示許可
    FP8_DIMENSIONS_WARNING_SHOWN = False # 3次元入力要求警告を再表示許可


def reset_warning_flags():
    """
    全警告フラグ統一リセット関数（後方互換性維持用エイリアス）
    
    機能的にはreset_fp8_warning_flags()と同一
    既存コードとの互換性確保のため両方の関数名を提供
    """
    global FP8_E4M3_WARNING_SHOWN, FP8_DIMENSIONS_WARNING_SHOWN
    FP8_E4M3_WARNING_SHOWN = False      # 警告表示フラグ初期化
    FP8_DIMENSIONS_WARNING_SHOWN = False # 警告表示フラグ初期化
