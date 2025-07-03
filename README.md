# FramePack-oichi2 (Beta) | [English](README/README_en.md) | [繁體中文](README/README_zh.md) | [Русский](README/README_ru.md) | [Issues](https://github.com/git-ai-code/FramePack-oichi2/issues)

**FramePack-oichi2**(通称：お壱弐の方)は、lllyasviel師の[lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)のフォークであるnirvash氏の[nirvash/FramePack](https://github.com/nirvash/FramePack)を元にして作成され、Kohya Tech氏の[kohya-ss/FramePack-LoRAReady](https://github.com/kohya-ss/FramePack-LoRAReady)のコードを導入することでLoRA機能の正式実装と安定性が向上した[git-ai-code/FramePack-eichi](https://github.com/git-ai-code/FramePack-eichi)に、同氏提案の[1フレーム推論](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack_1f.md)と、後にfurusu氏により提案されたkisekaeichi機能を実装する形で、同リポジトリ内に別ツールとして作成された**FramePack-oichi**(通称：お壱の方)に対し、mattyamonaca氏により提案された1f-mc(1フレーム・マルチコントロール)方式を試験実装し、全体的にソースを見直し、いくつかの機能追加をした**動画生成1フレーム推論画像生成ツール**です。

## 📘 名称の由来

**Oneframe Image CHain Interface 2 (OICHI2)**  ~~1なのか2なのかはっきりしてほしい~~
- **O**neframe: 1フレーム推論機能の高度化と最適化
- **I**mage: 高度な画像制御とマスク処理システム
- **CH**ain: 複数制御画像の連携と関係性の強化
- **I**nterface: 直感的なUI/UXとモジュール化設計
- **2**: 第2世代として完全リニューアル

## 🎯 対象ユーザー（Beta）

- FramePack-oichi機能を既に使用されている方
- より高度な1フレーム推論機能をお求めの方
- 実験的な新機能を試したい方

## 🚨 Beta版について

このソフトウェアはBeta版です。既にFramePack-oichiの機能をご利用のユーザー様向けの改良版として提供しています。

**問題を発見された場合は[Issues](https://github.com/git-ai-code/FramePack-oichi2/issues)にてご報告ください。**

### 🚀 多言語起動オプション

各言語に最適化された専用起動スクリプトをご利用いただけます：

```batch
run_oneframe_ichi2.bat       # 日本語版（デフォルト）
run_oneframe_ichi2_en.bat    # English version
run_oneframe_ichi2_zh-tw.bat # 繁體中文版
run_oneframe_ichi2_ru.bat    # Русская версия
Language_FramePack-oichi2.bat # 言語切替ツール
```

## 🌟 主な新機能・改良点

### 🔧 **最新更新（v0.1.0-pre）**

#### UI/UX改善
- **LoRAファイルアップロード強度設定**: アップロード版にも各ファイル用の強度入力欄を追加
- **解像度予想表示の正確性向上**: 予想サイズの横縦表示順序を修正（幅×高さ形式に統一）

#### 処理ロジック改善
- **キュー機能使用時のシード管理最適化**: プロンプトキューとイメージキュー使用時はシード値を固定し、通常のバッチ処理時のみシード+1を適用。

※従来機能は[oichiユーザーガイド](https://github.com/git-ai-code/FramePack-eichi/blob/main/README/README_userguide.md#oichi%E7%89%88%E3%83%AF%E3%83%B3%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E7%89%88%E3%81%AE%E4%BD%BF%E3%81%84%E6%96%B9)をご参考下さい。

### 🎯 **4モード統合制御システム**
- **1フレーム推論**: 入力画像のみを使用したシンプルな次フレーム生成。動画生成の1フレーム推論を利用した、最も基本的な画像生成を行います。制御画像は処理時に無視され、純粋な時系列予測に特化しています。
- **着せ替え（kisekaeichi）**: furusu氏考案・Kohya氏実装の参照画像を用いた高精度な画像変換技術。着せ替え制御画像とマスクのみを使用し、LoRAなどを用いて服装や外観の変更に最適化された処理を行います。
- **1フレーム・マルチコントロール（1f-mc）（試験実装）**: mattyamonaca氏提案の複数制御画像による高度な画像ブレンド技術。人物制御画像と追加制御画像を組み合わせることで、より複雑で精密な画像生成を実現します。
- **カスタム**: 設定された全ての制御画像とマスクを同時に使用する自由度の高いモード。実験的な用途や独自の制御方法を試したい場合に適しています。

### 📋 用語統一による操作性向上
- **重複UI問題の完全解決**: 従来の「ターゲットインデックス」と「レイテントインデックス」の重複を統一し、「生成フレーム位置（latent_index）」として一本化
- **標準用語統一**: FramePack公式用語に準拠。「履歴インデックス」→「参照フレーム位置(clean_index)」等

### 📊 **実行履歴システム**
- **設定の完全復元**: 全てのパラメータ（プロンプト、LoRA設定、高度制御設定）を自動保存し、ワンクリックで完全復元
- **サムネイル付き履歴表示**: 入力画像・出力画像・制御画像のサムネイルを自動生成し、視覚的に履歴を管理
- **パラメータ比較機能**: 過去の設定との比較により、最適なパラメータ調整をサポート
- **自動ファイル管理**: 制御画像とマスクファイルの永続化により、履歴復元時のファイル欠損を防止
- **最大20件保持**: 直近20回分の実行履歴を自動管理し、古い履歴は自動削除
- 気付きにくいですが一番上にあります。

### 🔧 **技術的改善**
- 可変LoRA数対応により最大20個まで対応。プリセット管理を5件から10件に拡張。
- モジュール化によるコード保守性向上

## 🚀 セットアップ方法

### 💻 **システム要件**

#### **前提条件**
- Windows 10/11（64ビット）
- NVIDIA GPU (RTX 30/40/50シリーズ推奨、最低8GB VRAM)
- CUDA Toolkit 12.6（RTX 50シリーズでは12.8推奨）
- Python 3.10.x（RTX 50シリーズでは3.11推奨）
- 最新のNVIDIA GPU ドライバー

#### **RAM要件**
- **最小**: 32GB（標準的な操作に十分、一時的なディスクスワップの可能性あり）
- **推奨**: 64GB（長時間の使用、LoRAの使用、高解像度処理に理想的）
- 現在のoichi2はFP8の対応用により、ウォームアップ時40～50GB（32GBの場合は一時的なディスクスワップあり）
- その後25GB程度で安定稼働します。

#### **VRAM要件**
- **最小**: 8GB VRAM（FramePack-oichi2の推奨最小値）
- **推奨**: 12GB以上（RTX 3060Ti(12GB)/4070以上。高解像度以外は可能）
- **最適**: 24GB以上（RTX 4090以上。高解像度生成時）

#### **ストレージ要件**
- **アプリケーション**: 約5GB
- **モデル**: 約40GB（初回起動時に自動ダウンロード）
- **推奨総容量**: 100GB以上（SSD推奨。ディスクスワップ発生時用）

#### **📍 RTX 50シリーズ（Blackwell）をお使いの方へ**

RTX 50シリーズ（RTX 5070Ti、RTX 5080、RTX 5090など）では特別なセットアップが必要です：

- **CUDA**: 12.8以降が必要
- **PyTorch**: 2.7.0（CUDA 12.8対応版）
- **SageAttention**: 2.1.1（RTX 50シリーズ最適化版）

### 🪟 **Windows セットアップ手順**

#### **前提条件**
- Windows 10/11（64ビット）
- Git for Windows
- 十分な空きディスク容量（150GB以上推奨）

#### **⚠️ 重要：セットアップ手順**

#### **📋 セットアップ前の注意事項**

**🚨 フォルダ分離の推奨**
- FramePack-oichi2は**独立したフォルダ**での構築を強く推奨します
- 既存のFramePack-eichiがある場合でも、**別フォルダ**で新規構築してください
- モジュールが分離されているため既存環境への上書きも技術的には可能ですが、**予期しない動作や設定競合**を避けるため推奨しません

**📁 推奨フォルダ構成例**
```
C:/
├── framepack(-eichi)/      # FramePack(-eichi)（既存）
└── framepack-oichi2/       # FramePack-oichi2（新規作成・推奨）
```

**🎯 モデル共有について**
- **モデルファイル**（約30GB）は複数環境で共有可能です
- 初回起動時の「モデル格納先指定」で既存のモデルフォルダを指定できます
- 例：`C:\Models\hf_download` や `C:\framepack\webui\hf_download`

**ステップ1: 公式FramePackの環境構築**

1. **ダウンロード**
   [公式FramePack](https://github.com/lllyasviel/FramePack)から**「Click Here to Download One-Click Package (CUDA 12.6 + Pytorch 2.6)」**をクリックしてダウンロードします。

2. **解凍とセットアップ**
   ```batch
   # 推奨：専用フォルダに解凍（例：C:\framepack-oichi2）
   # 解凍したフォルダ内で以下を実行：
   
   update.bat  # 必須：最新のアップデートを適用

   ※Windows等で「Windows によって PC が保護されました」と表示される場合は、「詳細情報」を開き「実行」を選択して下さい。
   実行後、何かキーを押してウィンドウを閉じてください。
   ```

3. **モデルダウンロード確認**
   - **今回はひとまず本処理をスキップします（後述）**
   ```batch
   run.bat     # 初回起動でモデルダウンロード開始（約30GB）
   ```
   - 初回起動時に約30GBのモデルが自動ダウンロードされます
   - モデルは`framepack\webui\hf_download`フォルダに保存されます
   - 既存のモデルがあり、移動してもいい場合は、このフォルダに配置してください

4. **高速化ライブラリのインストール（推奨）**
   ```batch
   [Issue #138](https://github.com/lllyasviel/FramePack/issues/138)から
   package_installer.zipをダウンロード
   解凍してFramePackのルートディレクトリで実行：
   package_installer.bat

    ※Windows等で「Windows によって PC が保護されました」と表示される場合は、同様に「詳細情報」を開き「実行」を選択して下さい。
    実行後、何かキーを押してウィンドウを閉じてください。
   ```

**ステップ2: FramePack-oichi2の上書きインストール**

#### **🎯 重要：オーバーレイインストール方式**

FramePack-oichi2は**オーバーレイパッケージ**として設計されています。公式FramePackの上にすべてまたは必要最小限のファイルのみを追加・上書きします。

1. **実行ファイルの配置（必要最小限）**
   FramePackのルートディレクトリに以下のファイルを配置：
   ```
   run_oneframe_ichi2.bat             # FramePack-oichi2起動スクリプト
   ```

2. **webuiフォルダへのファイル配置**
   以下のファイルとフォルダを`webui`フォルダに配置：
   ```
   webui/
   ├── oneframe_ichi2.py              # メインアプリケーション
   ├── version.py                     # バージョン管理
   ├── oichi2_utils/                  # oichi2専用ユーティリティ（18モジュール）
   ├── diffusers_helper/              # 重要：2ファイルのみ
   │   ├── bucket_tools.py            # 解像度バケット機能
   │   └── memory.py                  # メモリ管理機能
   ├── common_utils/                  # 共通ユーティリティ
   ├── lora2_utils/                   # LoRA処理ユーティリティ
   └── locales/                       # 多言語対応
   ```

**ステップ3: 初回起動とモデル設定**

**初回セットアップおよび今後の起動方法**
1. FramePack-oichi2のフォルダを開く
2. `run_oneframe_ichi2.bat`で起動

#### **🎯 モデル格納先の設定（重要）**
初回起動時にHuggingFaceモデルの格納場所設定画面が表示されます：

```batch
============================================================
HuggingFaceモデル格納場所の設定
============================================================
モデルが見つかりません。格納場所を設定してください。
既存の共有フォルダがある場合はそのパスを入力してください。
（例: C:\Models\webui\hf_download または /home/user/models/webui/hf_download）
何も入力せずEnterを押すとローカルフォルダを使用します。
------------------------------------------------------------
モデル格納パス:
```

**📁 既存モデルを共有する場合（推奨）**
```batch
# 既存のFramePackまたはFramePack-eichiのモデルを使用
例1: C:\framepack\webui\hf_download
例2: C:\framepack-eichi\webui\hf_download  
例3: C:\Models\hf_download（共有フォルダ）
```

設定後は
`FramePack-oichi2\webui\settings`の`app_settings.json`
に
```batch
  "hf_settings": {
    "shared_model_paths": [
      "～\webui\hf_download"
    ],
    "local_model_path": "hf_download"
```
という形式で記載されますので、必要に応じて変更するか、
一旦削除し起動画面で再度指定願います。

**🆕 新規ダウンロードする場合**
- 空白でEnterを押してローカルフォルダ（`webui\hf_download`）を使用
- 約40GBの新規ダウンロードが開始されます

**⚡ HuggingFace認証**
- 認証画面は自動でスキップされます（認証不要）

**ステップ4: モデルの自動ダウンロード**
- 進行状況がコンソールに表示されます
- 完了後、WebUIが自動的に起動します

#### **🔄 上記手順でうまくいかない場合**

**代替手順：公式経由でのセットアップ**
1. **別フォルダ**で公式FramePackを通常通り起動してモデルダウンロードを完了
2. FramePack-oichi2のファイルを上書き（または新規フォルダにコピー）
3. `run_oneframe_ichi2.bat`で起動
4. 初回起動時に既存モデルフォルダを指定（例：`C:\framepack\webui\hf_download`）、または空白でEnter

**⚠️ 注意：既存環境への直接上書きについて**
- 技術的には既存のFramePack-eichiに直接上書きも(おそらく)可能です
- ただし、設定競合やモジュール干渉のリスクがあるため**非推奨**です
- トラブル時の切り分けが困難になる可能性があります

## 📁 ファイル構成

### **🎯 ルートディレクトリ構成**
```
FramePack-oichi2/
├── .gitignore                      # Git除外設定
├── .gitmodules                     # Git Submodule設定
├── LICENSE                         # ライセンス情報(MIT2.0)
├── README.md                       # プロジェクト説明書（本ファイル）
├── run_oneframe_ichi2.bat          # メイン起動スクリプト（日本語版）
├── run_oneframe_ichi2_en.bat       # 英語版起動スクリプト
├── run_oneframe_ichi2_zh-tw.bat    # 繁體中文版起動スクリプト
├── run_oneframe_ichi2_ru.bat       # ロシア語版起動スクリプト
├── Language_FramePack-oichi2.bat   # 言語切替ツール
├── README/                         # 📁 多言語ドキュメント
│   ├── README_en.md               # 英語版README
│   ├── README_zh.md               # 繁體中文版README
│   └── README_ru.md               # ロシア語版README
└── webui/                          # WebUIアプリケーション本体
```

### **📂 webui/ ディレクトリ詳細**
```
webui/
├── oneframe_ichi2.py              # メインアプリケーション（Gradio UI）
├── version.py                     # 統一バージョン管理（v0.1.0）
├── submodules/                    # Git Submodule
│   └── FramePack/                 # 公式FramePackの参照
├── diffusers_helper/              # 🔥重要：2ファイルのみ（オーバーレイ方式）
│   ├── bucket_tools.py            # 解像度バケット機能（動的解像度生成）
│   └── memory.py                  # メモリ管理機能（GPU/CPU動的スワップ）
├── oichi2_utils/                  # 🎯 oichi2専用機能モジュール（18ファイル）
│   ├── __init__.py
│   ├── oichi2_history_manager.py  # 履歴管理システム（20件自動管理）
│   ├── oichi2_ui_components.py    # UI部品管理
│   ├── oichi2_generation_core.py  # 生成エンジン本体
│   ├── oichi2_mode_controller.py  # 4モード制御システム
│   ├── oichi2_image_processor.py  # 画像前処理・後処理
│   ├── oichi2_prompt_manager.py   # プロンプト管理
│   ├── oichi2_settings_manager.py # 設定保存・復元
│   ├── oichi2_file_manager.py     # ファイル操作管理
│   ├── oichi2_thumbnail_generator.py # サムネイル生成
│   ├── oichi2_kisekae_handler.py  # kisekaeichi機能
│   ├── oichi2_1fmc_handler.py     # 1f-mc機能
│   ├── oichi2_custom_handler.py   # カスタムモード
│   ├── oichi2_mask_processor.py   # マスク画像処理
│   ├── oichi2_controlnet_manager.py # ControlNet制御
│   ├── oichi2_validation.py       # 入力値検証
│   ├── oichi2_error_handler.py    # エラーハンドリング
│   └── oichi2_constants.py        # 定数定義
├── lora2_utils/                   # 🔄 LoRA処理ユーティリティ
│   ├── __init__.py
│   ├── lora2_loader.py            # LoRA読み込み（最大20個対応）
│   ├── lora2_preset_manager.py    # プリセット管理（10個保存）
│   ├── lora2_weight_controller.py # 重み制御
│   └── lora2_validation.py        # LoRA設定検証
├── common_utils/                  # 🛠️ 共通ユーティリティ
│   ├── __init__.py
│   ├── path_utils.py              # クロスプラットフォームパス処理
│   ├── image_utils.py             # 画像処理共通関数
│   ├── file_utils.py              # ファイル操作共通関数
│   └── validation_utils.py        # 検証共通関数
└── locales/                       # 🌍 多言語対応
    ├── i18n.py                    # 国際化機能コア
    ├── ja.json                    # 日本語翻訳
    ├── en.json                    # 英語翻訳
    ├── ru.json                    # ロシア語翻訳
    └── zh-tw.json                 # 繁体字中国語翻訳
```

### **🔥 重要：diffusers_helper について**
- **オーバーレイ方式**: 公式FramePackの同名ディレクトリを**2ファイルのみ**で上書き
- **bucket_tools.py**: oichi2-dev版の解像度バケット機能（64刻み対応）
- **memory.py**: oichi2-dev版のメモリ管理機能（クロスプラットフォーム対応）
- **⚠️ 注意**: 他のファイルは公式FramePackのものをそのまま使用

### 🪟 **Linux(Mac) セットアップ手順**

- 基本的にoichiと同じであるため、そちらと上記を参考にセットアップ願います（後日拡充予定）

## ⚠️ 注意事項

- **Beta版**のため、予期しない動作が発生する可能性があります
- 公式FramePackのセットアップが前提となります
- 重要なデータは事前にバックアップを取ってください

## 🐛 問題報告

バグや要望は[Issues](https://github.com/git-ai-code/FramePack-oichi2/issues)までお願いします。

## 🤝 謝辞

FramePack-oichi2は以下の素晴らしいプロジェクトと技術者の貢献により実現されています：

### 基盤技術・フォーク元
- **[lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)** - 原作者lllyasviel師による革新的な基盤技術
- **[nirvash/FramePack](https://github.com/nirvash/FramePack)** - nirvash氏による先駆的な機能拡張と改良
- どこかの叡智なやつ

### 核心技術・機能提案
- **[Kohya Tech (kohya-ss)](https://github.com/kohya-ss)** - 1フレーム推論技術の核心実装、[musubi-tuner仕様](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack_1f.md)の策定
- **[kohya-ss/FramePack-LoRAReady](https://github.com/kohya-ss/FramePack-LoRAReady)** - LoRA機能の性能と安定性向上コード
- **[furusu](https://note.com/gcem156)** - kisekaeichi機能の参照画像技術考案
- **[mattyamonaca](https://github.com/mattyamonaca)** - 1f-mc（1フレーム・マルチコントロール）技術提案

これらの先駆者たちの献身的な研究開発と技術共有により、FramePack-oichi2の高度な機能が実現されています。特に、各技術者様のオープンソース精神と継続的な改良への取り組みに深く感謝いたします。

## 📄 ライセンス

本プロジェクトは[Apache License 2.0](LICENSE)の下で公開されています。これは元のFramePackプロジェクトのライセンスに準拠しています。

---

**FramePack-oichi2 v0.1.0 (Beta)**