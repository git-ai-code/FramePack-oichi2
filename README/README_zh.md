# FramePack-oichi2 (Beta) | [日本語](../README.md) | [English](README_en.md) | [Русский](README_ru.md) | [問題回報](https://github.com/git-ai-code/FramePack-oichi2/issues)

**FramePack-oichi2**（暱稱：Oichi-Two）是一個在[git-ai-code/FramePack-eichi](https://github.com/git-ai-code/FramePack-eichi)同一儲存庫中作為獨立工具創建的視頻生成單幀推論圖像生成工具。它建構在nirvash的[nirvash/FramePack](https://github.com/nirvash/FramePack)基礎上，該分支源自lllyasviel的[lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)。通過整合Kohya Tech的[kohya-ss/FramePack-LoRAReady](https://github.com/kohya-ss/FramePack-LoRAReady)代碼改善了LoRA功能的穩定性，然後通過同一作者提出的[單幀推論](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack_1f.md)和furusu提出的kisekaeichi功能進行了增強。該項目進一步實驗性地實現了mattyamonaca提出的1f-mc（單幀多重控制）方法，並進行了整體源代碼審查和附加功能。

## 📘 名稱由來

**Oneframe Image CHain Interface 2 (OICHI2)** ~~到底是1還是2，請明確一下~~
- **O**neframe：先進且優化的單幀推論功能
- **I**mage：先進的圖像控制和遮罩處理系統
- **CH**ain：增強多個控制圖像之間的連接性和關係
- **I**nterface：直觀的UI/UX和模組化設計
- **2**：作為第二代的完全更新

## 🎯 目標用戶（Beta）

- 已經使用FramePack-oichi功能的用戶
- 尋求更先進單幀推論功能的用戶
- 想要嘗試實驗性新功能的用戶

## 🚨 關於Beta版本

此軟體為Beta版本。作為已使用FramePack-oichi功能用戶的改良版本提供。

**如果遇到任何問題，請在[Issues](https://github.com/git-ai-code/FramePack-oichi2/issues)中回報。**

### 🚀 多語言啟動選項

提供針對各語言最佳化的專用啟動腳本：

```batch
run_oneframe_ichi2.bat       # 日語版（預設）
run_oneframe_ichi2_en.bat    # English version
run_oneframe_ichi2_zh-tw.bat # 繁體中文版
run_oneframe_ichi2_ru.bat    # Русская версия
Language_FramePack-oichi2.bat # 語言切換工具
```

## 🌟 主要新功能和改進

### 🔧 **最新更新（v0.1.0-pre）**

#### UI/UX改進
- **LoRA檔案上傳強度設定**：在上傳版本中為每個檔案添加了個別的強度輸入欄位
- **解析度預測顯示準確性改進**：修正了預測尺寸的寬×高格式顯示順序以保持一致性

#### 處理邏輯改進
- **隊列功能的種子管理優化**：使用提示詞隊列和圖像隊列時，種子值固定；僅在正常批次處理時應用seed+1。

※對於傳統功能，請參考[oichi用戶指南](https://github.com/git-ai-code/FramePack-eichi/blob/main/README/README_userguide.md#oichi%E7%89%88%E3%83%AF%E3%83%B3%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E7%89%88%E3%81%AE%E4%BD%BF%E3%81%84%E6%96%B9)。

### 🎯 **4模式整合控制系統**
- **單幀推論**：僅使用輸入圖像的簡單下一幀生成。使用視頻生成的單幀推論執行最基本的圖像生成。控制圖像在處理時被忽略，純粹專注於時間預測。
- **Kisekaeichi**：使用參考圖像的高精度圖像變換技術，由furusu構思、Kohya實現。僅使用kisekae控制圖像和遮罩，專為使用LoRA等技術進行服裝和外觀變更而優化。
- **單幀多重控制（1f-mc）（實驗性）**：使用多個控制圖像的先進圖像混合技術，由mattyamonaca提出。結合人物控制圖像和附加控制圖像，實現更複雜和精確的圖像生成。
- **自訂**：同時使用所有配置的控制圖像和遮罩的高靈活性模式。適用於實驗性用途或嘗試獨特控制方法時。

### 📋 通過術語統一改善可操作性
- **完全解決重複UI問題**：將「目標索引」和「潛在索引」之間的重疊統一為單一的「生成幀位置（latent_index）」
- **標準術語統一**：符合官方FramePack術語。「歷史索引」→「參考幀位置（clean_index）」等

### 📊 **執行歷史系統**
- **完整設定還原**：自動保存所有參數（提示詞、LoRA設定、進階控制設定），一鍵完整還原
- **縮圖歷史顯示**：自動生成輸入圖像、輸出圖像和控制圖像的縮圖，進行視覺化歷史管理
- **參數比較功能**：通過與過去設定的比較支援最佳參數調整
- **自動檔案管理**：通過持久化控制圖像和遮罩檔案防止歷史還原時的檔案丟失
- **最多保留20筆記錄**：自動管理最近20次執行歷史，自動刪除舊記錄
- 它位於最上方，雖然容易忽略。

### 🔧 **技術改進**
- 支援可變LoRA數量最多20個。預設管理從5個擴展到10個預設。
- 通過模組化改善代碼可維護性

## 🚀 設定說明

### 💻 **系統需求**

#### **先決條件**
- Windows 10/11（64位元）
- NVIDIA GPU（建議RTX 30/40/50系列，最低8GB VRAM）
- CUDA Toolkit 12.6（RTX 50系列建議12.8）
- Python 3.10.x（RTX 50系列建議3.11）
- 最新的NVIDIA GPU驅動程式

#### **RAM需求**
- **最低**：32GB（標準操作充足，可能有暫時的磁碟交換）
- **建議**：64GB（長期使用、LoRA使用和高解析度處理的理想選擇）
- 目前oichi2由於FP8支援，在暖機時需要40-50GB（32GB時可能發生暫時磁碟交換）
- 之後在約25GB穩定運行。

#### **VRAM需求**
- **最低**：8GB VRAM（FramePack-oichi2的建議最低值）
- **建議**：12GB或以上（RTX 3060Ti(12GB)/4070或以上。除高解析度外皆可）
- **最佳**：24GB或以上（RTX 4090或以上。用於高解析度生成）

#### **儲存需求**
- **應用程式**：約5GB
- **模型**：約40GB（首次啟動時自動下載）
- **建議總容量**：100GB或以上（建議SSD。用於磁碟交換）

#### **📍 RTX 50系列（Blackwell）用戶須知**

RTX 50系列（RTX 5070Ti、RTX 5080、RTX 5090等）需要特殊設定：

- **CUDA**：需要12.8或更高版本
- **PyTorch**：2.7.0（CUDA 12.8相容版本）
- **SageAttention**：2.1.1（RTX 50系列優化版本）

### 🪟 **Windows設定說明**

#### **先決條件**
- Windows 10/11（64位元）
- Git for Windows
- 充足的可用磁碟空間（建議150GB或以上）

#### **⚠️ 重要：設定說明**

#### **📋 設定前注意事項**

**🚨 建議資料夾分離**
- FramePack-oichi2強烈建議在**獨立資料夾**中建構
- 即使有現有的FramePack-eichi，也請在**別的資料夾**中建構
- 雖然由於模組分離，覆蓋現有環境在技術上是可能的，但**不建議**這樣做以避免意外行為或配置衝突

**📁 建議資料夾結構示例**
```
C:/
├── framepack(-eichi)/      # FramePack(-eichi)（現有）
└── framepack-oichi2/       # FramePack-oichi2（新建，建議）
```

**🎯 關於模型共享**
- **模型檔案**（約30GB）可在多個環境間共享
- 可在首次啟動時的「模型儲存位置指定」中指定現有模型資料夾
- 示例：`C:\Models\hf_download`或`C:\framepack\webui\hf_download`

**步驟1：官方FramePack環境設定**

1. **下載**
   從[官方FramePack](https://github.com/lllyasviel/FramePack)點擊**「Click Here to Download One-Click Package (CUDA 12.6 + Pytorch 2.6)」**下載。

2. **解壓和設定**
   ```batch
   # 建議：解壓到專用資料夾（如C:\framepack-oichi2）
   # 在解壓的資料夾中執行以下命令：
   
   update.bat  # 必需：應用最新更新

   ※如果Windows顯示「Windows已保護您的電腦」，請點擊「更多資訊」並選擇「仍要執行」。
   執行後，按任意鍵關閉視窗。
   ```

3. **模型下載確認**
   - **暫時跳過此處理（後述）**
   ```batch
   run.bat     # 首次啟動開始模型下載（約30GB）
   ```
   - 首次啟動時會自動下載約30GB的模型
   - 模型保存在`framepack\webui\hf_download`資料夾中
   - 如果有現有模型且想移動，請放置在此資料夾中

4. **高速化函式庫安裝（建議）**
   ```batch
   從[Issue #138](https://github.com/lllyasviel/FramePack/issues/138)下載package_installer.zip
   解壓並在FramePack根目錄執行：
   package_installer.bat

    ※如果Windows顯示「Windows已保護您的電腦」，同樣點擊「更多資訊」並選擇「仍要執行」。
    執行後，按任意鍵關閉視窗。
   ```

**步驟2：FramePack-oichi2覆蓋安裝**

#### **🎯 重要：覆蓋安裝方式**

FramePack-oichi2設計為**覆蓋套件**。在官方FramePack之上添加或覆蓋全部或最小必要檔案。

1. **執行檔放置（最小必要）**
   在FramePack根目錄放置以下檔案：
   ```
   run_oneframe_ichi2.bat             # FramePack-oichi2啟動腳本
   ```

2. **webui資料夾中的檔案放置**
   在`webui`資料夾中放置以下檔案和資料夾：
   ```
   webui/
   ├── oneframe_ichi2.py              # 主應用程式
   ├── version.py                     # 版本管理
   ├── oichi2_utils/                  # oichi2專用實用程式（18個模組）
   ├── diffusers_helper/              # 重要：僅2個檔案
   │   ├── bucket_tools.py            # 解析度桶功能
   │   └── memory.py                  # 記憶體管理功能
   ├── common_utils/                  # 通用實用程式
   ├── lora2_utils/                   # LoRA處理實用程式
   └── locales/                       # 多語言支援
   ```

**步驟3：首次啟動和模型配置**

**初始設定和今後的啟動方法**
1. 開啟FramePack-oichi2資料夾
2. 用`run_oneframe_ichi2.bat`啟動

#### **🎯 模型儲存位置設定（重要）**
首次啟動時會顯示HuggingFace模型儲存位置設定畫面：

```batch
============================================================
HuggingFace模型儲存位置設定
============================================================
找不到模型。請設定儲存位置。
如果有現有的共享資料夾，請輸入該路徑。
（例：C:\Models\webui\hf_download 或 /home/user/models/webui/hf_download）
不輸入直接按Enter則使用本地資料夾。
------------------------------------------------------------
模型儲存路徑：
```

**📁 共享現有模型（建議）**
```batch
# 使用現有FramePack或FramePack-eichi的模型
例1：C:\framepack\webui\hf_download
例2：C:\framepack-eichi\webui\hf_download  
例3：C:\Models\hf_download（共享資料夾）
```

設定後會記錄在`FramePack-oichi2\webui\settings`的`app_settings.json`中：
```batch
  "hf_settings": {
    "shared_model_paths": [
      "～\webui\hf_download"
    ],
    "local_model_path": "hf_download"
```
可根據需要修改，或刪除後重新啟動並再次指定。

**🆕 新下載**
- 空白按Enter使用本地資料夾（`webui\hf_download`）
- 約40GB的新下載將開始

**⚡ HuggingFace認證**
- 認證畫面自動跳過（不需要認證）

**步驟4：自動模型下載**
- 進度會在控制台顯示
- 完成後WebUI自動啟動

#### **🔄 如果上述步驟無法進行**

**替代程序：通過官方途徑設定**
1. 在**別的資料夾**中正常啟動官方FramePack並完成模型下載
2. 覆蓋FramePack-oichi2檔案（或複製到新資料夾）
3. 用`run_oneframe_ichi2.bat`啟動
4. 首次啟動時指定現有模型資料夾（如`C:\framepack\webui\hf_download`），或空白按Enter

**⚠️ 注意：關於直接覆蓋現有環境**
- 直接覆蓋現有FramePack-eichi在技術上（大概）是可能的
- 但是，由於配置衝突和模組干擾的風險，**不建議**這樣做
- 發生問題時的故障排除變得困難

## 📁 檔案結構

### **🎯 根目錄結構**
```
FramePack-oichi2/
├── .gitignore                      # Git排除設定
├── .gitmodules                     # Git Submodule設定
├── LICENSE                         # 授權資訊（MIT2.0）
├── README.md                       # 專案說明書（本檔案）
├── run_oneframe_ichi2.bat          # 主啟動腳本（日語版）
├── run_oneframe_ichi2_en.bat       # 英語版啟動腳本
├── run_oneframe_ichi2_zh-tw.bat    # 繁體中文版啟動腳本
├── run_oneframe_ichi2_ru.bat       # 俄語版啟動腳本
├── Language_FramePack-oichi2.bat   # 語言切換工具
├── README/                         # 📁 多語言文檔
│   ├── README_en.md               # 英語版README
│   ├── README_zh.md               # 繁體中文版README
│   └── README_ru.md               # 俄語版README
└── webui/                          # WebUI應用程式本體
```

### **📂 webui/目錄詳細**
```
webui/
├── oneframe_ichi2.py              # 主應用程式（Gradio UI）
├── version.py                     # 統一版本管理（v0.1.0）
├── submodules/                    # Git Submodule
│   └── FramePack/                 # 官方FramePack參考
├── diffusers_helper/              # 🔥重要：僅2個檔案（覆蓋方式）
│   ├── bucket_tools.py            # 解析度桶功能（動態解析度生成）
│   └── memory.py                  # 記憶體管理功能（GPU/CPU動態交換）
├── oichi2_utils/                  # 🎯 oichi2專用功能模組（18個檔案）
│   ├── __init__.py
│   ├── oichi2_history_manager.py  # 歷史管理系統（20筆自動管理）
│   ├── oichi2_ui_components.py    # UI元件管理
│   ├── oichi2_generation_core.py  # 生成引擎本體
│   ├── oichi2_mode_controller.py  # 4模式控制系統
│   ├── oichi2_image_processor.py  # 圖像前處理・後處理
│   ├── oichi2_prompt_manager.py   # 提示詞管理
│   ├── oichi2_settings_manager.py # 設定保存・還原
│   ├── oichi2_file_manager.py     # 檔案操作管理
│   ├── oichi2_thumbnail_generator.py # 縮圖生成
│   ├── oichi2_kisekae_handler.py  # kisekaeichi功能
│   ├── oichi2_1fmc_handler.py     # 1f-mc功能
│   ├── oichi2_custom_handler.py   # 自訂模式
│   ├── oichi2_mask_processor.py   # 遮罩圖像處理
│   ├── oichi2_controlnet_manager.py # ControlNet控制
│   ├── oichi2_validation.py       # 輸入值驗證
│   ├── oichi2_error_handler.py    # 錯誤處理
│   └── oichi2_constants.py        # 常數定義
├── lora2_utils/                   # 🔄 LoRA處理實用程式
│   ├── __init__.py
│   ├── lora2_loader.py            # LoRA載入（支援最多20個）
│   ├── lora2_preset_manager.py    # 預設管理（保存10個）
│   ├── lora2_weight_controller.py # 權重控制
│   └── lora2_validation.py        # LoRA設定驗證
├── common_utils/                  # 🛠️ 通用實用程式
│   ├── __init__.py
│   ├── path_utils.py              # 跨平台路徑處理
│   ├── image_utils.py             # 圖像處理通用函數
│   ├── file_utils.py              # 檔案操作通用函數
│   └── validation_utils.py        # 驗證通用函數
└── locales/                       # 🌍 多語言支援
    ├── i18n.py                    # 國際化功能核心
    ├── ja.json                    # 日語翻譯
    ├── en.json                    # 英語翻譯
    ├── ru.json                    # 俄語翻譯
    └── zh-tw.json                 # 繁體中文翻譯
```

### **🔥 重要：關於diffusers_helper**
- **覆蓋方式**：僅用**2個檔案**覆蓋官方FramePack的同名目錄
- **bucket_tools.py**：oichi2-dev版本的解析度桶功能（64步支援）
- **memory.py**：oichi2-dev版本的記憶體管理功能（跨平台支援）
- **⚠️ 注意**：其他檔案使用官方FramePack的原樣

### 🪟 **Linux(Mac)設定說明**

- 基本上與oichi相同，因此請參考那個和上述內容進行設定（日後擴充預定）

## ⚠️ 注意事項

- 作為**Beta版本**，可能發生意外行為
- 需要官方FramePack的設定作為前提
- 請事先備份重要資料

## 🐛 問題回報

請在[Issues](https://github.com/git-ai-code/FramePack-oichi2/issues)回報錯誤和需求。

## 🤝 致謝

FramePack-oichi2通過以下優秀專案和技術人員的貢獻而實現：

### 基礎技術和分支來源
- **[lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)** - 原作者lllyasviel師的革新基礎技術
- **[nirvash/FramePack](https://github.com/nirvash/FramePack)** - nirvash氏的先驅功能擴展和改良
- 某個叡智的存在

### 核心技術和功能提案
- **[Kohya Tech (kohya-ss)](https://github.com/kohya-ss)** - 單幀推論技術的核心實現，[musubi-tuner規格](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack_1f.md)的制定
- **[kohya-ss/FramePack-LoRAReady](https://github.com/kohya-ss/FramePack-LoRAReady)** - LoRA功能性能和穩定性改善代碼
- **[furusu](https://note.com/gcem156)** - kisekaeichi功能參考圖像技術構想
- **[mattyamonaca](https://github.com/mattyamonaca)** - 1f-mc（單幀多重控制）技術提案

FramePack-oichi2的高級功能通過這些先驅者的專注研發和技術分享而實現。特別深深感謝各技術人員的開源精神和對持續改良的承諾。

## 📄 授權

本專案在[Apache License 2.0](LICENSE)下公開。這符合原FramePack專案的授權。

---

**FramePack-oichi2 v0.1.0 (Beta)**