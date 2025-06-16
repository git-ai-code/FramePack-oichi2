# FramePack-eichi/oichi SafeTensors Utilities
#
# メモリ効率的なsafetensorsファイル読み込み機能を提供します。

import json
import struct
from typing import Dict

import torch

class MemoryEfficientSafeOpen:
    """
    メモリ効率的な.safetensorsファイル読み込みクラス
    
    safetensorsファイルからテンソルを効率的に読み込むためのクラスです。
    メタデータの読み込みは対応していません。
    """
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "rb")  # バイナリモードでファイルを開く
        self.header, self.header_size = self._read_header()  # safetensorsヘッダー情報を解析

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        """テンソルキー一覧を取得（メタデータを除外）"""
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        """safetensorsファイルのメタデータを取得"""
        return self.header.get("__metadata__", {})

    def get_tensor(self, key):
        """指定されたキーのテンソルをメモリ効率的に読み込み"""
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]  # テンソルのメタデータ（型、形状、オフセット）を取得
        offset_start, offset_end = metadata["data_offsets"]  # ファイル内でのデータ位置

        if offset_start == offset_end:
            tensor_bytes = None  # 空のテンソル（サイズ0）
        else:
            # safetensorsフォーマット: 8バイトヘッダーサイズ + ヘッダー + データ
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)  # 必要な部分のみ読み込み

        return self._deserialize_tensor(tensor_bytes, metadata)

    def _read_header(self):
        """safetensorsファイルのヘッダー情報を解析"""
        # safetensorsフォーマット: 最初の8バイトにヘッダーサイズ（リトルエンディアン）
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        # ヘッダーはUTF-8エンコードされたJSON形式
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        """バイトデータからPyTorchテンソルに変換"""
        dtype = self._get_torch_dtype(metadata["dtype"])  # safetensors型名をPyTorch型に変換
        shape = metadata["shape"]  # テンソルの形状情報

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)  # 空テンソルの場合
        else:
            tensor_bytes = bytearray(tensor_bytes)  # 書き込み可能なバイト配列に変換
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)  # バイト列をテンソルに変換

        # process float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # convert to the target dtype and reshape
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        """
        SafeTensors形式の型名をPyTorchテンソル型に変換
        
        SafeTensorsとPyTorchの型システムマッピング:
        - FP8形式: F8_E5M2（範囲±57344・勾配用）、F8_E4M3（範囲±448・重み用）
        - Brain Float 16: BF16（指数部8bit・仮数部7bit・Google開発ML特化）
        - 浮動小数点: F64/F32/F16（IEEE 754準拠の標準精度）
        - 符号付き整数: I64/I32/I16/I8（2の補数表現）
        - 符号なし整数: U8（0-255範囲・バイトデータ用）
        - 論理値: BOOL（True/False・マスク処理用）
        
        PyTorchバージョン互換性:
        - FP8データ型: PyTorch 2.1以降で利用可能
        - 古いバージョンでは自動的にNone返却（フォールバック処理で対応）
        """
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,  # Google Brain Float 16: 指数部8bit・仮数部7bit
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # FP8型の動的サポート検出（PyTorch 2.1以降の実行時機能確認）
        if hasattr(torch, "float8_e5m2"):
            # E5M2: 指数部5bit・仮数部2bit・符号部1bit（勾配・アクティベーション用）
            # 範囲: ±57344（広範囲・低精度）、NaN/Inf対応
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            # E4M3FN: 指数部4bit・仮数部3bit・符号部1bit・finite形式
            # 範囲: ±448（狭範囲・高精度）、NaN/Inf無効でRTX40最適化対応
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        """
        FP8バイトデータをPyTorchテンソルに変換
        
        FP8形式の特徴:
        - E5M2: 勾配計算・アクティベーション向け（広範囲・低精度）
        - E4M3FN: 重み・推論向け（狭範囲・高精度、finite形式）
        - scaled_mmはE4M3FN専用（RTX40シリーズ最適化対応）
        """
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            # E5M2: 勾配計算・アクティベーション向け（広範囲対応）
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            # E4M3FN: 重み・推論向け・finite形式（NaN/Inf無効）
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            # FP8未対応環境ではfloat16フォールバック不可（精度・範囲の不整合）
            # return byte_tensor.view(torch.uint8).to(torch.float16).reshape(shape)
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")
