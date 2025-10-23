# Домашнее задание 1 - Прогнозирование оттока клиентов

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![ML](https://img.shields.io/badge/machine-learning-orange)
![Status](https://img.shields.io/badge/status-completed-green)
![PyCharm](https://img.shields.io/badge/IDE-PyCharm-professional)

## Постановка задачи

### Бизнес-задача
Прогнозирование оттока клиентов для телеком-компании с целью снижения затрат на привлечение новых клиентов и увеличения Lifetime Value (LTV).

### ML-задача
Бинарная классификация - предсказание, уйдет ли клиент в течение следующего месяца.

### Набор данных
Синтетический датасет Telco Customer Churn, содержащий 5000 записей с 20+ признаками.

### Метрика качества
**F1-Score** - оптимально балансирует precision и recall при дисбалансе классов.

##  Быстрый старт в PyCharm

### Начальная настройка:
1. **Откройте проект в PyCharm**: File → Open → выберите папку проекта
2. **Настройте интерпретатор**: File → Settings → Project → Python Interpreter
   - Выберите существующий venv или создайте новый
   - Установите зависимости: `pip install -r requirements.txt`

### Запуск проекта:
```bash
# Активация окружения (в терминале PyCharm)
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 1. Генерация данных
python data_generation.py

# 2. EDA анализ
python eda_analysis.py