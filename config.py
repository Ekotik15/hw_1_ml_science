import os
from pathlib import Path

# Пути проекта
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = PROJECT_ROOT / "images"

# Создание директорий если не существуют
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Пути к файлам
DATA_PATH = DATA_DIR / "telco_customer_churn_synthetic.csv"

# Основные параметры
N_SAMPLES = 5000
RANDOM_STATE = 42
TARGET_COLUMN = "Churn"


class DataGenerationConfig:
    """Конфигурация для генерации данных"""
    n_samples = N_SAMPLES
    random_state = RANDOM_STATE

    # Распределения для генерации
    age_params = {"loc": 45, "scale": 15, "min": 18, "max": 80}
    tenure_params = {"scale": 30, "min": 0, "max": 72}
    monthly_charges_params = {"loc": 70, "scale": 30, "min": 20, "max": 120}

    # Вероятности для категориальных признаков
    probabilities = {
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["Yes", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    }


class VisualizationConfig:
    """Конфигурация для визуализаций"""
    style = "seaborn-v0_8"
    colors = ["#2E8B57", "#DC143C"]  # Зеленый для лояльных, Красный для ушедших
    fig_size = (12, 8)
    dpi = 300
    title_fontsize = 16
    label_fontsize = 12
    tick_fontsize = 10


class AnalysisConfig:
    """Конфигурация для анализа"""
    categorical_features = [
        "gender", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]

    numerical_features = ["Age", "tenure", "MonthlyCharges", "TotalCharges"]
    correlation_threshold = 0.5
    outlier_threshold = 1.5


# Глобальные экземпляры для удобного использования
data_config = DataGenerationConfig()
viz_config = VisualizationConfig()
analysis_config = AnalysisConfig()