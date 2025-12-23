# Lab 3: LLM Compression (Stage 2) - Fine-Tuning

Цель: Дообучить сжатую модель (полученную на 1 этапе).

Базовая модель: `NOVORDSEC/qwen3-8b-awq-int4` (AWQ 4-bit)
Метод дообучения: **QLoRA** (LoRA адаптеры поверх квантованной модели).
Датасет: `guanaco-llama2-1k`

## Результаты Этапа 2

| Метрика | Значение | Комментарий |
| :--- | :--- | :--- |
| **Original Size** | 15.26 GB | FP16 |
| **Base Model Size** | 5.68 GB | INT4 (AWQ) |
| **Adapter Size** | 0.014 GB | LoRA weights |
| **Total Final Size** | **5.69 GB** | Base + Adapter |
| **Compression Ratio** | **2.68x** | |
| **Baseline Accuracy (MMLU)** | 0.7292 | Stage 1 (Original) |
| **Final Accuracy (MMLU)** | 0.7210 | After QLoRA |
| **Performance Drop** | 1.13% | |
| **Final Score** | **2.65** | `Ratio / (1 + Drop)` |

Детальный отчет по каждой теме бенчмарка доступен в файле: [mmlu_finetuned_detailed.csv](./mmlu_finetuned_detailed.csv)

## Ссылки на модели

1.  **Базовая сжатая модель:** [NOVORDSEC/qwen3-8b-awq-int4](https://huggingface.co/NOVORDSEC/qwen3-8b-awq-int4)
2.  **Адаптеры (LoRA):** [NOVORDSEC/qwen3-8b-awq-finetuned-adapters](https://huggingface.co/NOVORDSEC/qwen3-8b-awq-finetuned-adapters)

---

## Установка и Запуск

Проект можно запустить двумя способами: через готовые Python-скрипты или через Jupyter Notebook.

### Предварительные требования
*   **GPU:** Требуется видеокарта NVIDIA (код тестировался на T4 16GB).
*   **Python:** 3.12

### Установка зависимостей

```bash
pip install -r requirements.txt
```

---

### Способ 1: Python скрипты

Этот способ воспроизводит логику шаг за шагом в терминале.

#### Шаг 1. Обучение (Fine-Tuning)
Скрипт скачивает базовую квантованную модель, инициализирует QLoRA конфигурацию и запускает обучение на датасете Guanaco. Результат (адаптеры) сохраняется локально в папку `./qwen3-8b-awq-finetuned`.

```bash
python train.py
```

#### Шаг 2. Проверка и Бенчмарк
Скрипт скачивает дообученную модель, запускает MMLU бенчмарк (20% выборки для ускорения) и считает Score.

```bash
python inference.py
```

---

### Способ 2: Jupyter Notebook

Весь пайплайн (установка библиотек, обучение, сохранение, бенчмарк) собран в одном ноутбуке. Это тот же код, который запускался на Kaggle.

1. Откройте файл `Stage2_FineTuning_And_Inference.ipynb`.
2. Запустите ячейки последовательно.

---

## Дополнительная информация

*   **Оборудование:** Код запускался на Kaggle (2x Tesla T4).
*   **Методология:** Использовался `SFTTrainer` из библиотеки `trl` + `peft` для QLoRA.
*   **Параметры обучения:** 1 эпоха, learning_rate 2e-4, r=8, alpha=16.
*   **Бенчмарк:** MMLU (Massive Multitask Language Understanding). Использовалась выборка `fraction=0.2` для оптимизации времени выполнения.
