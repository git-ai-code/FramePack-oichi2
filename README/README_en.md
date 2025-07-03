# FramePack-oichi2 (Beta) | [日本語](../README.md) | [繁體中文](README_zh.md) | [Русский](README_ru.md) | [Issues](https://github.com/git-ai-code/FramePack-oichi2/issues)

**FramePack-oichi2** (aka: Oichi-Two) is a video generation one-frame inference image generation tool created as a separate tool within the same repository as [git-ai-code/FramePack-eichi](https://github.com/git-ai-code/FramePack-eichi). It was built upon nirvash's [nirvash/FramePack](https://github.com/nirvash/FramePack), which is a fork of lllyasviel's [lllyasviel/FramePack](https://github.com/lllyasviel/FramePack). The stability of LoRA functionality was improved by incorporating code from Kohya Tech's [kohya-ss/FramePack-LoRAReady](https://github.com/kohya-ss/FramePack-LoRAReady), and then enhanced with the [1-frame inference](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack_1f.md) proposed by the same author and the kisekaeichi feature proposed by furusu. The project further implements mattyamonaca's proposed 1f-mc (1-frame multi-control) method on an experimental basis, with overall source code review and additional features.

## 📘 Name Origin

**Oneframe Image CHain Interface 2 (OICHI2)** ~~Whether it's 1 or 2, please be clear~~
- **O**neframe: Advanced and optimized one-frame inference functionality
- **I**mage: Advanced image control and mask processing system
- **CH**ain: Enhanced connectivity and relationships between multiple control images
- **I**nterface: Intuitive UI/UX and modular design
- **2**: Complete renewal as second generation

## 🎯 Target Users (Beta)

- Users already using FramePack-oichi functionality
- Users seeking more advanced one-frame inference features
- Users wanting to try experimental new features

## 🚨 About Beta Version

This software is a Beta version. It is provided as an improved version for users who are already using FramePack-oichi functionality.

**If you encounter any issues, please report them in [Issues](https://github.com/git-ai-code/FramePack-oichi2/issues).**

### 🚀 Multi-language Launch Options

Dedicated launch scripts optimized for each language are available:

```batch
run_oneframe_ichi2.bat       # Japanese version (default)
run_oneframe_ichi2_en.bat    # English version
run_oneframe_ichi2_zh-tw.bat # Traditional Chinese version
run_oneframe_ichi2_ru.bat    # Russian version
Language_FramePack-oichi2.bat # Language switching tool
```

## 🌟 Main New Features and Improvements

### 🔧 **Latest Update (v0.1.0-pre)**

#### UI/UX Improvements
- **LoRA File Upload Strength Settings**: Added individual strength input fields for each file in the upload version
- **Improved Resolution Prediction Display Accuracy**: Fixed display order of predicted size width×height format for consistency

#### Processing Logic Improvements
- **Optimized Seed Management for Queue Features**: When using prompt queue and image queue, seed values are fixed; seed+1 is applied only during normal batch processing.

※For conventional features, please refer to the [oichi user guide](https://github.com/git-ai-code/FramePack-eichi/blob/main/README/README_userguide.md#oichi%E7%89%88%E3%83%AF%E3%83%B3%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E7%89%88%E3%81%AE%E4%BD%BF%E3%81%84%E6%96%B9).

### 🎯 **4-Mode Integrated Control System**
- **One-frame inference**: Simple next-frame generation using only input images. Performs the most basic image generation using one-frame inference from video generation. Control images are ignored during processing, focusing purely on temporal prediction.
- **Kisekaeichi**: High-precision image transformation technology using reference images, conceived by furusu and implemented by Kohya. Uses only kisekae control images and masks, optimized for clothing and appearance changes using LoRA and other techniques.
- **1-frame Multi-control (1f-mc) (Experimental)**: Advanced image blending technology using multiple control images, proposed by mattyamonaca. Combines person control images with additional control images to achieve more complex and precise image generation.
- **Custom**: High-flexibility mode that uses all configured control images and masks simultaneously. Suitable for experimental purposes or when trying unique control methods.

### 📋 Improved Operability Through Terminology Unification
- **Complete Resolution of Duplicate UI Issues**: Unified the overlap between "Target Index" and "Latent Index" into a single "Generation Frame Position (latent_index)"
- **Standard Terminology Unification**: Compliant with official FramePack terminology. "History Index" → "Reference Frame Position (clean_index)" etc.

### 📊 **Execution History System**
- **Complete Settings Restoration**: Automatically saves all parameters (prompts, LoRA settings, advanced control settings) for one-click complete restoration
- **Thumbnail History Display**: Automatically generates thumbnails of input images, output images, and control images for visual history management
- **Parameter Comparison Feature**: Supports optimal parameter adjustment by comparing with past settings
- **Automatic File Management**: Prevents file loss during history restoration by persisting control images and mask files
- **Maximum 20 Records Retention**: Automatically manages the last 20 execution histories with automatic deletion of old records
- It's located at the top, though it's easy to miss.

### 🔧 **Technical Improvements**
- Support for variable LoRA count up to 20. Expanded preset management from 5 to 10 presets.
- Improved code maintainability through modularization

## 🚀 Setup Instructions

### 💻 **System Requirements**

#### **Prerequisites**
- Windows 10/11 (64-bit)
- NVIDIA GPU (RTX 30/40/50 series recommended, minimum 8GB VRAM)
- CUDA Toolkit 12.6 (12.8 recommended for RTX 50 series)
- Python 3.10.x (3.11 recommended for RTX 50 series)
- Latest NVIDIA GPU drivers

#### **RAM Requirements**
- **Minimum**: 32GB (sufficient for standard operations with possibility of temporary disk swapping)
- **Recommended**: 64GB (ideal for long-term use, LoRA usage, and high-resolution processing)
- Current oichi2 requires 40-50GB during warmup due to FP8 support (temporary disk swapping may occur with 32GB)
- After that, it operates stably at around 25GB.

#### **VRAM Requirements**
- **Minimum**: 8GB VRAM (recommended minimum for FramePack-oichi2)
- **Recommended**: 12GB or more (RTX 3060Ti(12GB)/4070 or higher. Capable except for high resolution)
- **Optimal**: 24GB or more (RTX 4090 or higher. For high-resolution generation)

#### **Storage Requirements**
- **Application**: Approximately 5GB
- **Models**: Approximately 40GB (automatically downloaded on first startup)
- **Recommended Total Capacity**: 100GB or more (SSD recommended. For disk swapping)

#### **📍 For RTX 50 Series (Blackwell) Users**

RTX 50 series (RTX 5070Ti, RTX 5080, RTX 5090, etc.) requires special setup:

- **CUDA**: 12.8 or later required
- **PyTorch**: 2.7.0 (CUDA 12.8 compatible version)
- **SageAttention**: 2.1.1 (RTX 50 series optimized version)

### 🪟 **Windows Setup Instructions**

#### **Prerequisites**
- Windows 10/11 (64-bit)
- Git for Windows
- Sufficient free disk space (150GB or more recommended)

#### **⚠️ Important: Setup Instructions**

#### **📋 Pre-Setup Notes**

**🚨 Recommended Folder Separation**
- FramePack-oichi2 strongly recommends construction in an **independent folder**
- Even if you have existing FramePack-eichi, please construct in a **separate folder**
- While overwriting existing environments is technically possible due to modular separation, it's **not recommended** to avoid unexpected behavior or configuration conflicts

**📁 Recommended Folder Structure Example**
```
C:/
├── framepack(-eichi)/      # FramePack(-eichi) (existing)
└── framepack-oichi2/       # FramePack-oichi2 (new creation, recommended)
```

**🎯 About Model Sharing**
- **Model files** (approximately 30GB) can be shared between multiple environments
- You can specify an existing model folder during first startup "Model Storage Location Specification"
- Example: `C:\Models\hf_download` or `C:\framepack\webui\hf_download`

**Step 1: Official FramePack Environment Setup**

1. **Download**
   Download from [Official FramePack](https://github.com/lllyasviel/FramePack) by clicking **"Click Here to Download One-Click Package (CUDA 12.6 + Pytorch 2.6)"**.

2. **Extract and Setup**
   ```batch
   # Recommended: Extract to dedicated folder (e.g., C:\framepack-oichi2)
   # Execute the following in the extracted folder:
   
   update.bat  # Required: Apply latest updates

   ※If Windows displays "Windows protected your PC", click "More info" and select "Run anyway".
   After execution, press any key to close the window.
   ```

3. **Model Download Confirmation**
   - **For now, skip this process (explained later)**
   ```batch
   run.bat     # First startup begins model download (approximately 30GB)
   ```
   - Models will be automatically downloaded approximately 30GB on first startup
   - Models are saved in the `framepack\webui\hf_download` folder
   - If you have existing models and want to move them, place them in this folder

4. **High-speed Library Installation (Recommended)**
   ```batch
   Download package_installer.zip from [Issue #138](https://github.com/lllyasviel/FramePack/issues/138)
   Extract and run in the FramePack root directory:
   package_installer.bat

    ※If Windows displays "Windows protected your PC", similarly click "More info" and select "Run anyway".
    After execution, press any key to close the window.
   ```

**Step 2: FramePack-oichi2 Overlay Installation**

#### **🎯 Important: Overlay Installation Method**

FramePack-oichi2 is designed as an **overlay package**. It adds or overwrites all or minimum necessary files on top of the official FramePack.

1. **Executable File Placement (Minimum Required)**
   Place the following file in the FramePack root directory:
   ```
   run_oneframe_ichi2.bat             # FramePack-oichi2 startup script
   ```

2. **File Placement in webui Folder**
   Place the following files and folders in the `webui` folder:
   ```
   webui/
   ├── oneframe_ichi2.py              # Main application
   ├── version.py                     # Version management
   ├── oichi2_utils/                  # oichi2-specific utilities (18 modules)
   ├── diffusers_helper/              # Important: Only 2 files
   │   ├── bucket_tools.py            # Resolution bucket functionality
   │   └── memory.py                  # Memory management functionality
   ├── common_utils/                  # Common utilities
   ├── lora2_utils/                   # LoRA processing utilities
   └── locales/                       # Multi-language support
   ```

**Step 3: First Startup and Model Configuration**

**Initial Setup and Future Startup Method**
1. Open the FramePack-oichi2 folder
2. Start with `run_oneframe_ichi2.bat`

#### **🎯 Model Storage Location Setting (Important)**
A HuggingFace model storage location setting screen will be displayed on first startup:

```batch
============================================================
HuggingFace Model Storage Location Setting
============================================================
Models not found. Please set storage location.
If you have an existing shared folder, enter that path.
(Example: C:\Models\webui\hf_download or /home/user/models/webui/hf_download)
Press Enter without input to use local folder.
------------------------------------------------------------
Model storage path:
```

**📁 To Share Existing Models (Recommended)**
```batch
# Use existing FramePack or FramePack-eichi models
Example 1: C:\framepack\webui\hf_download
Example 2: C:\framepack-eichi\webui\hf_download  
Example 3: C:\Models\hf_download (shared folder)
```

After setting, it will be recorded in `FramePack-oichi2\webui\settings`'s `app_settings.json` as:
```batch
  "hf_settings": {
    "shared_model_paths": [
      "～\webui\hf_download"
    ],
    "local_model_path": "hf_download"
```
You can modify as needed or delete and restart to specify again.

**🆕 For New Download**
- Press Enter on blank to use local folder (`webui\hf_download`)
- Approximately 40GB new download will begin

**⚡ HuggingFace Authentication**
- Authentication screen is automatically skipped (no authentication required)

**Step 4: Automatic Model Download**
- Progress will be displayed in console
- WebUI will start automatically after completion

#### **🔄 If Above Steps Don't Work**

**Alternative Procedure: Setup via Official Route**
1. Start official FramePack normally in a **separate folder** and complete model download
2. Overlay FramePack-oichi2 files (or copy to new folder)
3. Start with `run_oneframe_ichi2.bat`
4. Specify existing model folder on first startup (e.g., `C:\framepack\webui\hf_download`), or press Enter on blank

**⚠️ Note: About Direct Overwriting on Existing Environments**
- Direct overwriting on existing FramePack-eichi is technically (probably) possible
- However, it's **not recommended** due to risks of configuration conflicts and module interference
- Troubleshooting becomes difficult in case of issues

## 📁 File Structure

### **🎯 Root Directory Structure**
```
FramePack-oichi2/
├── .gitignore                      # Git exclusion settings
├── .gitmodules                     # Git Submodule settings
├── LICENSE                         # License information (MIT2.0)
├── README.md                       # Project documentation (this file)
├── run_oneframe_ichi2.bat          # Main startup script (Japanese version)
├── run_oneframe_ichi2_en.bat       # English version startup script
├── run_oneframe_ichi2_zh-tw.bat    # Traditional Chinese version startup script
├── run_oneframe_ichi2_ru.bat       # Russian version startup script
├── Language_FramePack-oichi2.bat   # Language switching tool
├── README/                         # 📁 Multi-language documentation
│   ├── README_en.md               # English version README
│   ├── README_zh.md               # Traditional Chinese version README
│   └── README_ru.md               # Russian version README
└── webui/                          # WebUI application main body
```

### **📂 webui/ Directory Details**
```
webui/
├── oneframe_ichi2.py              # Main application (Gradio UI)
├── version.py                     # Unified version management (v0.1.0)
├── submodules/                    # Git Submodule
│   └── FramePack/                 # Official FramePack reference
├── diffusers_helper/              # 🔥Important: Only 2 files (overlay method)
│   ├── bucket_tools.py            # Resolution bucket functionality (dynamic resolution generation)
│   └── memory.py                  # Memory management functionality (GPU/CPU dynamic swap)
├── oichi2_utils/                  # 🎯 oichi2-specific functionality modules (18 files)
│   ├── __init__.py
│   ├── oichi2_history_manager.py  # History management system (20 records auto-management)
│   ├── oichi2_ui_components.py    # UI component management
│   ├── oichi2_generation_core.py  # Generation engine main body
│   ├── oichi2_mode_controller.py  # 4-mode control system
│   ├── oichi2_image_processor.py  # Image pre/post-processing
│   ├── oichi2_prompt_manager.py   # Prompt management
│   ├── oichi2_settings_manager.py # Settings save/restore
│   ├── oichi2_file_manager.py     # File operation management
│   ├── oichi2_thumbnail_generator.py # Thumbnail generation
│   ├── oichi2_kisekae_handler.py  # kisekaeichi functionality
│   ├── oichi2_1fmc_handler.py     # 1f-mc functionality
│   ├── oichi2_custom_handler.py   # Custom mode
│   ├── oichi2_mask_processor.py   # Mask image processing
│   ├── oichi2_controlnet_manager.py # ControlNet control
│   ├── oichi2_validation.py       # Input value validation
│   ├── oichi2_error_handler.py    # Error handling
│   └── oichi2_constants.py        # Constant definitions
├── lora2_utils/                   # 🔄 LoRA processing utilities
│   ├── __init__.py
│   ├── lora2_loader.py            # LoRA loading (supports up to 20)
│   ├── lora2_preset_manager.py    # Preset management (10 saved)
│   ├── lora2_weight_controller.py # Weight control
│   └── lora2_validation.py        # LoRA setting validation
├── common_utils/                  # 🛠️ Common utilities
│   ├── __init__.py
│   ├── path_utils.py              # Cross-platform path processing
│   ├── image_utils.py             # Image processing common functions
│   ├── file_utils.py              # File operation common functions
│   └── validation_utils.py        # Validation common functions
└── locales/                       # 🌍 Multi-language support
    ├── i18n.py                    # Internationalization functionality core
    ├── ja.json                    # Japanese translation
    ├── en.json                    # English translation
    ├── ru.json                    # Russian translation
    └── zh-tw.json                 # Traditional Chinese translation
```

### **🔥 Important: About diffusers_helper**
- **Overlay Method**: Overwrites the official FramePack directory of the same name with **only 2 files**
- **bucket_tools.py**: oichi2-dev version resolution bucket functionality (64-step support)
- **memory.py**: oichi2-dev version memory management functionality (cross-platform support)
- **⚠️ Note**: Other files use those from official FramePack as-is

### 🪟 **Linux(Mac) Setup Instructions**

- Basically the same as oichi, so please refer to that and the above for setup (to be expanded later)

## ⚠️ Notes

- As a **Beta version**, unexpected behavior may occur
- Official FramePack setup is required as a prerequisite
- Please backup important data in advance

## 🐛 Issue Reports

Please report bugs and requests to [Issues](https://github.com/git-ai-code/FramePack-oichi2/issues).

## 🤝 Acknowledgments

FramePack-oichi2 is realized through the contributions of the following wonderful projects and technologists:

### Foundation Technology and Fork Origins
- **[lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)** - Revolutionary foundation technology by original author lllyasviel
- **[nirvash/FramePack](https://github.com/nirvash/FramePack)** - Pioneering functionality expansion and improvements by nirvash
- Some wise entity

### Core Technology and Feature Proposals
- **[Kohya Tech (kohya-ss)](https://github.com/kohya-ss)** - Core implementation of one-frame inference technology, establishment of [musubi-tuner specifications](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack_1f.md)
- **[kohya-ss/FramePack-LoRAReady](https://github.com/kohya-ss/FramePack-LoRAReady)** - LoRA functionality performance and stability improvement code
- **[furusu](https://note.com/gcem156)** - Conception of kisekaeichi functionality reference image technology
- **[mattyamonaca](https://github.com/mattyamonaca)** - 1f-mc (1-frame multi-control) technology proposal

FramePack-oichi2's advanced functionality is realized through the dedicated research and development and technology sharing of these pioneers. We are deeply grateful especially for each technologist's open-source spirit and commitment to continuous improvement.

## 📄 License

This project is released under the [Apache License 2.0](LICENSE). This complies with the license of the original FramePack project.

---

**FramePack-oichi2 v0.1.0 (Beta)**