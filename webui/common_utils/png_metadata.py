"""
PNGメタデータ処理ユーティリティ

FramePack-eichiのためのPNGメタデータ埋め込みと抽出機能を提供します。
SD系ツールとの互換性を保ち、プロンプト、シード、セクション情報の管理に対応しています。
"""

import json
import os
import traceback
from typing import Any, Dict, Optional, Union

from PIL import Image, PngImagePlugin

from locales.i18n_extended import translate

# メタデータキーの定義
PROMPT_KEY: str = "prompt"
SEED_KEY: str = "seed"
SECTION_PROMPT_KEY: str = "section_prompt"
SECTION_NUMBER_KEY: str = "section_number"
PARAMETERS_KEY: str = "parameters"  # SD系との互換性のため

def embed_metadata_to_png(image_path: str, metadata_dict: Dict[str, Any]) -> str:
    """
    PNGファイルにメタデータを埋め込みます。

    Args:
        image_path: PNGファイルのパス
        metadata_dict: 埋め込むメタデータの辞書

    Returns:
        str: 処理したファイルのパス
    """
    try:
        img: Image.Image = Image.open(image_path)
        metadata: PngImagePlugin.PngInfo = PngImagePlugin.PngInfo()

        # パラメータを結合したテキストも作成（SD系との互換性のため）
        parameters_text: str = ""

        # パラメータテキストの構築（個別キーは埋め込まない）
        for key, value in metadata_dict.items():
            if value is not None:
                if key == PROMPT_KEY:
                    parameters_text += f"{value}\n"
                elif key == SEED_KEY:
                    parameters_text += f"Seed: {value}\n"
                elif key == SECTION_PROMPT_KEY:
                    parameters_text += f"Section Prompt: {value}\n"
                elif key == SECTION_NUMBER_KEY:
                    parameters_text += f"Section Number: {value}\n"

        # パラメータテキストがあれば追加
        if parameters_text:
            metadata.add_text(PARAMETERS_KEY, parameters_text.strip())

        # 保存（ファイル形式は変更せず）
        if image_path.lower().endswith('.png'):
            img.save(image_path, "PNG", pnginfo=metadata)
        else:
            # PNGでなければPNGに変換して保存
            png_path: str = os.path.splitext(image_path)[0] + '.png'
            img.save(png_path, "PNG", pnginfo=metadata)
            image_path = png_path

        return image_path

    except Exception as e:
        return image_path

def extract_metadata_from_png(image_path_or_object: Union[str, Image.Image]) -> Dict[str, Any]:
    """
    PNGファイルからメタデータを抽出します。

    Args:
        image_path_or_object: PNGファイルのパスまたはPIL.Image.Imageオブジェクト

    Returns:
        Dict[str, Any]: 抽出したメタデータの辞書
    """
    try:
        # パスが文字列ならイメージを開く、イメージオブジェクトならそのまま使う
        img: Image.Image
        if isinstance(image_path_or_object, str):
            if not os.path.exists(image_path_or_object):
                return {}
            img = Image.open(image_path_or_object)
        else:
            img = image_path_or_object

        metadata: Dict[str, Any] = {}

        # 個別のキーを処理
        for key in [PROMPT_KEY, SEED_KEY, SECTION_PROMPT_KEY, SECTION_NUMBER_KEY, PARAMETERS_KEY]:
            if key in img.info:
                value = img.info[key]
                try:
                    # JSONとして解析を試みる
                    metadata[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # JSONでなければそのまま格納
                    metadata[key] = value

        # parametersキーからの抽出処理（SD系との互換性のため）
        # 個別キーが無い場合でもparametersから抽出を試みる
        if PARAMETERS_KEY in img.info:
            params: str = img.info[PARAMETERS_KEY]

            # 行に分割して処理
            lines: list[str] = params.split("\n")
            
            # プロンプト行を収集
            prompt_lines: list[str] = []
            
            # 全ての行を処理
            for line in lines:
                line_stripped: str = line.strip()
                if line_stripped.startswith("Seed:"):
                    seed_str: str = line_stripped.replace("Seed:", "").strip()
                    if seed_str.isdigit():
                        metadata[SEED_KEY] = int(seed_str)
                elif line_stripped.startswith("Section Number:"):
                    section_num_str: str = line_stripped.replace("Section Number:", "").strip()
                    if section_num_str.isdigit():
                        metadata[SECTION_NUMBER_KEY] = int(section_num_str)
                elif line_stripped.startswith("Section Prompt:"):
                    section_prompt: str = line_stripped.replace("Section Prompt:", "").strip()
                    if section_prompt:
                        metadata[SECTION_PROMPT_KEY] = section_prompt
                else:
                    # Seed: や Section: で始まらない行はプロンプトの一部
                    if line.strip():  # 空行は除外
                        prompt_lines.append(line.rstrip())
            
            # 複数行のプロンプトを結合
            if prompt_lines:
                metadata[PROMPT_KEY] = "\n".join(prompt_lines)

        return metadata

    except Exception as e:
        return {}

def extract_metadata_from_numpy_array(numpy_image: Optional[Any]) -> Dict[str, Any]:
    """
    NumPy配列からメタデータを抽出します（PILを介して）。

    Args:
        numpy_image: NumPy配列の画像データ

    Returns:
        Dict[str, Any]: 抽出したメタデータの辞書
    """
    try:
        if numpy_image is None:
            return {}

        # NumPy配列からPIL.Imageに変換
        img: Image.Image = Image.fromarray(numpy_image)

        # メタデータの抽出を試行
        metadata: Dict[str, Any] = extract_metadata_from_png(img)

        return metadata

    except Exception as e:
        return {}
