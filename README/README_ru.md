# FramePack-oichi2 (Beta) | [日本語](../README.md) | [English](README_en.md) | [繁體中文](README_zh.md) | [Проблемы](https://github.com/git-ai-code/FramePack-oichi2/issues)

**FramePack-oichi2** (неофициально: Oichi-Two) — это инструмент генерации изображений с одним кадром на основе видео, созданный как отдельный инструмент в том же репозитории [git-ai-code/FramePack-eichi](https://github.com/git-ai-code/FramePack-eichi). Он был построен на основе [nirvash/FramePack](https://github.com/nirvash/FramePack) от nirvash, который является форком [lllyasviel/FramePack](https://github.com/lllyasviel/FramePack) от lllyasviel. Стабильность функциональности LoRA была улучшена путем включения кода из [kohya-ss/FramePack-LoRAReady](https://github.com/kohya-ss/FramePack-LoRAReady) от Kohya Tech, а затем расширена [односеансным выводом](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack_1f.md), предложенным тем же автором, и функцией kisekaeichi, предложенной furusu. Проект дополнительно экспериментально реализует предложенный mattyamonaca метод 1f-mc (односеансный мультиконтроль) с общим пересмотром исходного кода и дополнительными функциями.

## 📘 Происхождение названия

**Oneframe Image CHain Interface 2 (OICHI2)** ~~Все-таки 1 или 2, определитесь~~
- **O**neframe: Расширенная и оптимизированная функциональность односеансного вывода
- **I**mage: Продвинутая система управления изображениями и обработки масок
- **CH**ain: Улучшенная связность и взаимодействие между несколькими управляющими изображениями
- **I**nterface: Интуитивный UI/UX и модульный дизайн
- **2**: Полное обновление как второе поколение

## 🎯 Целевые пользователи (Beta)

- Пользователи, уже использующие функциональность FramePack-oichi
- Пользователи, ищущие более продвинутые функции односеансного вывода
- Пользователи, желающие попробовать экспериментальные новые функции

## 🚨 О Beta-версии

Это программное обеспечение является Beta-версией. Оно предоставляется как улучшенная версия для пользователей, которые уже используют функциональность FramePack-oichi.

**Если вы столкнулись с проблемами, пожалуйста, сообщите о них в [Issues](https://github.com/git-ai-code/FramePack-oichi2/issues).**

### 🚀 Многоязычные варианты запуска

Доступны специальные скрипты запуска, оптимизированные для каждого языка:

```batch
run_oneframe_ichi2.bat       # Японская версия (по умолчанию)
run_oneframe_ichi2_en.bat    # English version
run_oneframe_ichi2_zh-tw.bat # 繁體中文版
run_oneframe_ichi2_ru.bat    # Русская версия
Language_FramePack-oichi2.bat # Инструмент переключения языков
```

## 🌟 Основные новые функции и улучшения

### 🔧 **Последнее обновление (v0.1.0-pre)**

#### Улучшения UI/UX
- **Настройки силы загрузки файлов LoRA**: Добавлены индивидуальные поля ввода силы для каждого файла в версии загрузки
- **Улучшенная точность отображения предсказания разрешения**: Исправлен порядок отображения предсказанного размера в формате ширина×высота для согласованности

#### Улучшения логики обработки
- **Оптимизированное управление семенами для функций очереди**: При использовании очереди промптов и очереди изображений значения семени фиксируются; seed+1 применяется только во время обычной пакетной обработки.

※ Для традиционных функций обратитесь к [руководству пользователя oichi](https://github.com/git-ai-code/FramePack-eichi/blob/main/README/README_userguide.md#oichi%E7%89%88%E3%83%AF%E3%83%B3%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E7%89%88%E3%81%AE%E4%BD%BF%E3%81%84%E6%96%B9).

### 🎯 **Система интегрированного управления 4 режимами**
- **Односеансный вывод**: Простая генерация следующего кадра, используя только входные изображения. Выполняет самую базовую генерацию изображений, используя односеансный вывод из генерации видео. Управляющие изображения игнорируются во время обработки, фокусируясь исключительно на временном предсказании.
- **Kisekaeichi**: Высокоточная технология трансформации изображений с использованием эталонных изображений, придуманная furusu и реализованная Kohya. Использует только управляющие изображения kisekae и маски, оптимизированные для изменения одежды и внешнего вида с использованием LoRA и других техник.
- **Односеансный мультиконтроль (1f-mc) (Экспериментальный)**: Продвинутая технология смешивания изображений с использованием нескольких управляющих изображений, предложенная mattyamonaca. Объединяет управляющие изображения человека с дополнительными управляющими изображениями для достижения более сложной и точной генерации изображений.
- **Пользовательский**: Высокогибкий режим, который одновременно использует все настроенные управляющие изображения и маски. Подходит для экспериментальных целей или при попытке уникальных методов управления.

### 📋 Улучшенная операционность через унификацию терминологии
- **Полное решение проблем дублирования UI**: Объединено перекрытие между "Целевым индексом" и "Латентным индексом" в единую "Позицию кадра генерации (latent_index)"
- **Унификация стандартной терминологии**: Соответствует официальной терминологии FramePack. "Индекс истории" → "Позиция референсного кадра (clean_index)" и т.д.

### 📊 **Система истории выполнения**
- **Полное восстановление настроек**: Автоматически сохраняет все параметры (промпты, настройки LoRA, расширенные настройки управления) для полного восстановления одним кликом
- **Отображение истории с миниатюрами**: Автоматически генерирует миниатюры входных изображений, выходных изображений и управляющих изображений для визуального управления историей
- **Функция сравнения параметров**: Поддерживает оптимальную настройку параметров путем сравнения с прошлыми настройками
- **Автоматическое управление файлами**: Предотвращает потерю файлов во время восстановления истории путем сохранения управляющих изображений и файлов масок
- **Максимальное хранение 20 записей**: Автоматически управляет последними 20 историями выполнения с автоматическим удалением старых записей
- Он расположен сверху, хотя его легко пропустить.

### 🔧 **Технические улучшения**
- Поддержка переменного количества LoRA до 20. Расширенное управление пресетами с 5 до 10 пресетов.
- Улучшенная поддерживаемость кода через модуляризацию

## 🚀 Инструкции по установке

### 💻 **Системные требования**

#### **Предварительные условия**
- Windows 10/11 (64-бит)
- NVIDIA GPU (рекомендуется RTX 30/40/50 серии, минимум 8ГБ VRAM)
- CUDA Toolkit 12.6 (12.8 рекомендуется для RTX 50 серии)
- Python 3.10.x (3.11 рекомендуется для RTX 50 серии)
- Последние драйверы NVIDIA GPU

#### **Требования к ОЗУ**
- **Минимум**: 32ГБ (достаточно для стандартных операций с возможностью временного свопинга на диск)
- **Рекомендуется**: 64ГБ (идеально для длительного использования, использования LoRA и высокоразрешенной обработки)
- Текущий oichi2 требует 40-50ГБ во время прогрева из-за поддержки FP8 (временный своп на диск может происходить с 32ГБ)
- После этого работает стабильно около 25ГБ.

#### **Требования к VRAM**
- **Минимум**: 8ГБ VRAM (рекомендуемый минимум для FramePack-oichi2)
- **Рекомендуется**: 12ГБ или более (RTX 3060Ti(12ГБ)/4070 или выше. Способен работать кроме высокого разрешения)
- **Оптимально**: 24ГБ или более (RTX 4090 или выше. Для генерации высокого разрешения)

#### **Требования к хранилищу**
- **Приложение**: Приблизительно 5ГБ
- **Модели**: Приблизительно 40ГБ (автоматически загружается при первом запуске)
- **Рекомендуемая общая емкость**: 100ГБ или более (рекомендуется SSD. Для дискового свопинга)

#### **📍 Для пользователей RTX 50 серии (Blackwell)**

RTX 50 серии (RTX 5070Ti, RTX 5080, RTX 5090 и т.д.) требует специальной настройки:

- **CUDA**: 12.8 или позже требуется
- **PyTorch**: 2.7.0 (совместимая с CUDA 12.8 версия)
- **SageAttention**: 2.1.1 (оптимизированная для RTX 50 серии версия)

### 🪟 **Инструкции по установке Windows**

#### **Предварительные условия**
- Windows 10/11 (64-бит)
- Git для Windows
- Достаточное свободное место на диске (рекомендуется 150ГБ или более)

#### **⚠️ Важно: Инструкции по установке**

#### **📋 Предустановочные заметки**

**🚨 Рекомендуемое разделение папок**
- FramePack-oichi2 настоятельно рекомендует построение в **независимой папке**
- Даже если у вас есть существующий FramePack-eichi, постройте в **отдельной папке**
- Хотя перезапись существующих сред технически возможна из-за модульного разделения, это **не рекомендуется** во избежание неожиданного поведения или конфликтов конфигурации

**📁 Пример рекомендуемой структуры папок**
```
C:/
├── framepack(-eichi)/      # FramePack(-eichi) (существующий)
└── framepack-oichi2/       # FramePack-oichi2 (новое создание, рекомендуется)
```

**🎯 О совместном использовании моделей**
- **Файлы моделей** (приблизительно 30ГБ) могут быть совместно использованы между несколькими средами
- Вы можете указать существующую папку модели во время первого запуска "Указание места хранения модели"
- Пример: `C:\Models\hf_download` или `C:\framepack\webui\hf_download`

**Шаг 1: Настройка среды официального FramePack**

1. **Загрузка**
   Загрузите с [Официального FramePack](https://github.com/lllyasviel/FramePack), нажав **"Click Here to Download One-Click Package (CUDA 12.6 + Pytorch 2.6)"**.

2. **Извлечение и настройка**
   ```batch
   # Рекомендуется: Извлечь в выделенную папку (например, C:\framepack-oichi2)
   # Выполните следующее в извлеченной папке:
   
   update.bat  # Обязательно: Применить последние обновления

   ※Если Windows отображает "Windows защитила ваш ПК", нажмите "Подробнее" и выберите "Выполнить в любом случае".
   После выполнения нажмите любую клавишу, чтобы закрыть окно.
   ```

3. **Подтверждение загрузки модели**
   - **Пока пропустите этот процесс (объясняется позже)**
   ```batch
   run.bat     # Первый запуск начинает загрузку модели (приблизительно 30ГБ)
   ```
   - Модели будут автоматически загружены приблизительно 30ГБ при первом запуске
   - Модели сохраняются в папке `framepack\webui\hf_download`
   - Если у вас есть существующие модели и вы хотите их переместить, поместите их в эту папку

4. **Установка высокоскоростных библиотек (Рекомендуется)**
   ```batch
   Загрузите package_installer.zip из [Issue #138](https://github.com/lllyasviel/FramePack/issues/138)
   Извлеките и запустите в корневом каталоге FramePack:
   package_installer.bat

    ※Если Windows отображает "Windows защитила ваш ПК", аналогично нажмите "Подробнее" и выберите "Выполнить в любом случае".
    После выполнения нажмите любую клавишу, чтобы закрыть окно.
   ```

**Шаг 2: Наложенная установка FramePack-oichi2**

#### **🎯 Важно: Метод наложенной установки**

FramePack-oichi2 разработан как **пакет наложения**. Он добавляет или перезаписывает все или минимально необходимые файлы поверх официального FramePack.

1. **Размещение исполняемого файла (минимально необходимо)**
   Поместите следующий файл в корневой каталог FramePack:
   ```
   run_oneframe_ichi2.bat             # Скрипт запуска FramePack-oichi2
   ```

2. **Размещение файлов в папке webui**
   Поместите следующие файлы и папки в папку `webui`:
   ```
   webui/
   ├── oneframe_ichi2.py              # Основное приложение
   ├── version.py                     # Управление версиями
   ├── oichi2_utils/                  # Утилиты, специфичные для oichi2 (18 модулей)
   ├── diffusers_helper/              # Важно: Только 2 файла
   │   ├── bucket_tools.py            # Функциональность бакетов разрешения
   │   └── memory.py                  # Функциональность управления памятью
   ├── common_utils/                  # Общие утилиты
   ├── lora2_utils/                   # Утилиты обработки LoRA
   └── locales/                       # Поддержка нескольких языков
   ```

**Шаг 3: Первый запуск и конфигурация модели**

**Первоначальная настройка и будущий метод запуска**
1. Откройте папку FramePack-oichi2
2. Запустите с помощью `run_oneframe_ichi2.bat`

#### **🎯 Настройка места хранения модели (Важно)**
При первом запуске будет отображен экран настройки места хранения модели HuggingFace:

```batch
============================================================
Настройка места хранения модели HuggingFace
============================================================
Модели не найдены. Пожалуйста, установите место хранения.
Если у вас есть существующая общая папка, введите этот путь.
(Пример: C:\Models\webui\hf_download или /home/user/models/webui/hf_download)
Нажмите Enter без ввода, чтобы использовать локальную папку.
------------------------------------------------------------
Путь к хранилищу модели:
```

**📁 Для совместного использования существующих моделей (Рекомендуется)**
```batch
# Используйте существующие модели FramePack или FramePack-eichi
Пример 1: C:\framepack\webui\hf_download
Пример 2: C:\framepack-eichi\webui\hf_download  
Пример 3: C:\Models\hf_download (общая папка)
```

После настройки это будет записано в `FramePack-oichi2\webui\settings`'s `app_settings.json` как:
```batch
  "hf_settings": {
    "shared_model_paths": [
      "～\webui\hf_download"
    ],
    "local_model_path": "hf_download"
```
Вы можете изменить по необходимости или удалить и перезапустить для повторного указания.

**🆕 Для новой загрузки**
- Нажмите Enter на пустом месте, чтобы использовать локальную папку (`webui\hf_download`)
- Начнется новая загрузка приблизительно 40ГБ

**⚡ Аутентификация HuggingFace**
- Экран аутентификации автоматически пропускается (аутентификация не требуется)

**Шаг 4: Автоматическая загрузка модели**
- Прогресс будет отображаться в консоли
- WebUI запустится автоматически после завершения

#### **🔄 Если вышеописанные шаги не работают**

**Альтернативная процедура: Настройка через официальный маршрут**
1. Запустите официальный FramePack нормально в **отдельной папке** и завершите загрузку модели
2. Наложите файлы FramePack-oichi2 (или скопируйте в новую папку)
3. Запустите с помощью `run_oneframe_ichi2.bat`
4. Укажите существующую папку модели при первом запуске (например, `C:\framepack\webui\hf_download`), или нажмите Enter на пустом месте

**⚠️ Примечание: О прямой перезаписи существующих сред**
- Прямая перезапись существующего FramePack-eichi технически (вероятно) возможна
- Однако это **не рекомендуется** из-за рисков конфликтов конфигурации и интерференции модулей
- Устранение неполадок становится затруднительным в случае проблем

## 📁 Структура файлов

### **🎯 Структура корневого каталога**
```
FramePack-oichi2/
├── .gitignore                      # Настройки исключения Git
├── .gitmodules                     # Настройки Git Submodule
├── LICENSE                         # Информация о лицензии (MIT2.0)
├── README.md                       # Документация проекта (этот файл)
├── run_oneframe_ichi2.bat          # Основной скрипт запуска (японская версия)
├── run_oneframe_ichi2_en.bat       # Скрипт запуска английской версии
├── run_oneframe_ichi2_zh-tw.bat    # Скрипт запуска версии на традиционном китайском
├── run_oneframe_ichi2_ru.bat       # Скрипт запуска русской версии
├── Language_FramePack-oichi2.bat   # Инструмент переключения языков
├── README/                         # 📁 Многоязычная документация
│   ├── README_en.md               # README английской версии
│   ├── README_zh.md               # README версии на традиционном китайском
│   └── README_ru.md               # README русской версии
└── webui/                          # Основное тело приложения WebUI
```

### **📂 Детали каталога webui/**
```
webui/
├── oneframe_ichi2.py              # Основное приложение (Gradio UI)
├── version.py                     # Унифицированное управление версиями (v0.1.0)
├── submodules/                    # Git Submodule
│   └── FramePack/                 # Ссылка на официальный FramePack
├── diffusers_helper/              # 🔥Важно: Только 2 файла (метод наложения)
│   ├── bucket_tools.py            # Функциональность бакетов разрешения (динамическая генерация разрешения)
│   └── memory.py                  # Функциональность управления памятью (динамический своп GPU/CPU)
├── oichi2_utils/                  # 🎯 Модули функциональности, специфичные для oichi2 (18 файлов)
│   ├── __init__.py
│   ├── oichi2_history_manager.py  # Система управления историей (автоматическое управление 20 записями)
│   ├── oichi2_ui_components.py    # Управление компонентами UI
│   ├── oichi2_generation_core.py  # Основное тело движка генерации
│   ├── oichi2_mode_controller.py  # Система управления 4 режимами
│   ├── oichi2_image_processor.py  # Пре/пост-обработка изображений
│   ├── oichi2_prompt_manager.py   # Управление промптами
│   ├── oichi2_settings_manager.py # Сохранение/восстановление настроек
│   ├── oichi2_file_manager.py     # Управление файловыми операциями
│   ├── oichi2_thumbnail_generator.py # Генерация миниатюр
│   ├── oichi2_kisekae_handler.py  # Функциональность kisekaeichi
│   ├── oichi2_1fmc_handler.py     # Функциональность 1f-mc
│   ├── oichi2_custom_handler.py   # Пользовательский режим
│   ├── oichi2_mask_processor.py   # Обработка изображений масок
│   ├── oichi2_controlnet_manager.py # Управление ControlNet
│   ├── oichi2_validation.py       # Валидация входных значений
│   ├── oichi2_error_handler.py    # Обработка ошибок
│   └── oichi2_constants.py        # Определения констант
├── lora2_utils/                   # 🔄 Утилиты обработки LoRA
│   ├── __init__.py
│   ├── lora2_loader.py            # Загрузка LoRA (поддержка до 20)
│   ├── lora2_preset_manager.py    # Управление пресетами (10 сохраненных)
│   ├── lora2_weight_controller.py # Управление весами
│   └── lora2_validation.py        # Валидация настроек LoRA
├── common_utils/                  # 🛠️ Общие утилиты
│   ├── __init__.py
│   ├── path_utils.py              # Кроссплатформенная обработка путей
│   ├── image_utils.py             # Общие функции обработки изображений
│   ├── file_utils.py              # Общие функции файловых операций
│   └── validation_utils.py        # Общие функции валидации
└── locales/                       # 🌍 Поддержка нескольких языков
    ├── i18n.py                    # Ядро функциональности интернационализации
    ├── ja.json                    # Японский перевод
    ├── en.json                    # Английский перевод
    ├── ru.json                    # Русский перевод
    └── zh-tw.json                 # Перевод на традиционный китайский
```

### **🔥 Важно: О diffusers_helper**
- **Метод наложения**: Перезаписывает официальный каталог FramePack с тем же именем только **2 файлами**
- **bucket_tools.py**: Функциональность бакетов разрешения версии oichi2-dev (поддержка 64-шагового)
- **memory.py**: Функциональность управления памятью версии oichi2-dev (кроссплатформенная поддержка)
- **⚠️ Примечание**: Другие файлы используют те, что есть в официальном FramePack как есть

### 🪟 **Инструкции по установке Linux(Mac)**

- В основном то же, что и oichi, поэтому обратитесь к этому и вышеизложенному для установки (будет расширено позже)

## ⚠️ Примечания

- Как **Beta-версия**, может происходить неожиданное поведение
- Настройка официального FramePack требуется как предварительное условие
- Пожалуйста, заранее создайте резервную копию важных данных

## 🐛 Отчеты о проблемах

Пожалуйста, сообщайте об ошибках и запросах в [Issues](https://github.com/git-ai-code/FramePack-oichi2/issues).

## 🤝 Благодарности

FramePack-oichi2 реализован благодаря вкладу следующих замечательных проектов и технологов:

### Фундаментальная технология и источники форков
- **[lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)** - Революционная фундаментальная технология от оригинального автора lllyasviel
- **[nirvash/FramePack](https://github.com/nirvash/FramePack)** - Пионерское расширение функциональности и улучшения от nirvash
- Некая мудрая сущность

### Основная технология и предложения функций
- **[Kohya Tech (kohya-ss)](https://github.com/kohya-ss)** - Основная реализация технологии односеансного вывода, установление [спецификаций musubi-tuner](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack_1f.md)
- **[kohya-ss/FramePack-LoRAReady](https://github.com/kohya-ss/FramePack-LoRAReady)** - Код улучшения производительности и стабильности функциональности LoRA
- **[furusu](https://note.com/gcem156)** - Концепция технологии эталонного изображения функциональности kisekaeichi
- **[mattyamonaca](https://github.com/mattyamonaca)** - Предложение технологии 1f-mc (односеансный мультиконтроль)

Продвинутая функциональность FramePack-oichi2 реализована благодаря преданным исследованиям и разработкам и обмену технологиями этих пионеров. Мы особенно глубоко благодарны каждому технологу за дух открытого исходного кода и приверженность непрерывному улучшению.

## 📄 Лицензия

Этот проект выпущен под [Лицензией Apache 2.0](LICENSE). Это соответствует лицензии оригинального проекта FramePack.

---

**FramePack-oichi2 v0.1.0 (Beta)**