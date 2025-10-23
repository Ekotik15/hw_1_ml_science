
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random
import logging
from typing import Dict, Any, List
from tqdm import tqdm

from config import (
    DataGenerationConfig,
    DATA_PATH,
    RANDOM_STATE,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataGenerator:
    """Класс для генерации реалистичного датасета оттока клиентов"""

    def __init__(self, config: DataGenerationConfig):
        self.config = config
        self._set_random_seeds()

    def _set_random_seeds(self):
        """Установка случайных seed для воспроизводимости"""
        np.random.seed(self.config.random_state)
        random.seed(self.config.random_state)
        logger.info(f"Random seeds установлены: {self.config.random_state}")

    def generate_demographic_features(self, n_samples: int) -> Dict[str, Any]:
        """Генерация демографических признаков"""
        logger.info("Генерация демографических признаков...")

        ages = np.random.normal(
            self.config.age_params["loc"],
            self.config.age_params["scale"],
            n_samples
        ).astype(int)
        ages = np.clip(ages, self.config.age_params["min"], self.config.age_params["max"])

        return {
            "customerID": [f"CUST{str(i).zfill(5)}" for i in range(n_samples)],
            "gender": np.random.choice(
                self.config.probabilities["gender"],
                n_samples
            ),
            "Age": ages,
            "SeniorCitizen": (ages >= 65).astype(int),
            "Partner": np.random.choice(
                self.config.probabilities["Partner"],
                n_samples,
                p=[0.5, 0.5]
            ),
            "Dependents": np.random.choice(
                self.config.probabilities["Dependents"],
                n_samples,
                p=[0.3, 0.7]
            )
        }

    def generate_service_features(self, n_samples: int) -> Dict[str, Any]:
        """Генерация признаков услуг"""
        logger.info("Генерация признаков услуг...")

        tenure = np.random.exponential(
            self.config.tenure_params["scale"],
            n_samples
        ).astype(int)
        tenure = np.clip(tenure, self.config.tenure_params["min"], self.config.tenure_params["max"])

        monthly_charges = np.random.normal(
            self.config.monthly_charges_params["loc"],
            self.config.monthly_charges_params["scale"],
            n_samples
        )
        monthly_charges = np.clip(
            monthly_charges,
            self.config.monthly_charges_params["min"],
            self.config.monthly_charges_params["max"]
        )

        return {
            "tenure": tenure,
            "PhoneService": np.random.choice(
                self.config.probabilities["PhoneService"],
                n_samples,
                p=[0.9, 0.1]
            ),
            "MultipleLines": np.random.choice(
                self.config.probabilities["MultipleLines"],
                n_samples,
                p=[0.4, 0.5, 0.1]
            ),
            "InternetService": np.random.choice(
                self.config.probabilities["InternetService"],
                n_samples,
                p=[0.4, 0.4, 0.2]
            ),
            "MonthlyCharges": monthly_charges.round(2)
        }

    def generate_additional_services(self, n_samples: int, has_internet: np.ndarray) -> Dict[str, Any]:
        """Генерация дополнительных услуг"""
        logger.info("Генерация дополнительных услуг...")

        services = {}
        service_features = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]

        for service in service_features:
            choices = []
            for i in range(n_samples):
                if not has_internet[i]:
                    choices.append("No internet service")
                else:
                    choices.append(np.random.choice(["Yes", "No"], p=[0.3, 0.7]))
            services[service] = choices

        return services

    def generate_contract_and_payment(self, n_samples: int) -> Dict[str, Any]:
        """Генерация признаков контракта и платежей"""
        logger.info("Генерация признаков контракта и платежей...")

        return {
            "Contract": np.random.choice(
                self.config.probabilities["Contract"],
                n_samples,
                p=[0.5, 0.3, 0.2]
            ),
            "PaperlessBilling": np.random.choice(
                self.config.probabilities["PaperlessBilling"],
                n_samples,
                p=[0.6, 0.4]
            ),
            "PaymentMethod": np.random.choice(
                self.config.probabilities["PaymentMethod"],
                n_samples,
                p=[0.3, 0.2, 0.25, 0.25]
            )
        }

    def calculate_total_charges(self, df: pd.DataFrame) -> pd.Series:
        """Расчет общей суммы платежей"""
        logger.info("Расчет общей суммы платежей...")

        total_charges = (df["MonthlyCharges"] * df["tenure"] +
                         np.random.normal(0, 10, len(df))).round(2)
        return total_charges.clip(lower=0)

    def add_realistic_churn_patterns(self, df: pd.DataFrame, base_target: np.ndarray) -> pd.Series:
        """Добавление реалистичных паттернов оттока"""
        logger.info("Добавление реалистичных паттернов оттока...")

        churn = base_target.copy()

        # Month-to-month клиенты чаще уходят
        mtm_mask = df["Contract"] == "Month-to-month"
        if mtm_mask.any():
            churn[mtm_mask] = np.random.choice([0, 1], mtm_mask.sum(), p=[0.6, 0.4])

        # Клиенты с Fiber optic и высокими платежами чаще уходят
        fiber_high_mask = (df["InternetService"] == "Fiber optic") & (df["MonthlyCharges"] > 80)
        if fiber_high_mask.any():
            churn[fiber_high_mask] = np.random.choice([0, 1], fiber_high_mask.sum(), p=[0.5, 0.5])

        # Долгосрочные клиенты реже уходят
        loyal_mask = df["tenure"] > 24
        if loyal_mask.any():
            churn[loyal_mask] = np.random.choice([0, 1], loyal_mask.sum(), p=[0.9, 0.1])

        return churn

    def generate_dataset(self) -> pd.DataFrame:
        """Основной метод генерации датасета"""
        logger.info(f"Начало генерации датасета с {self.config.n_samples} samples...")

        n_samples = self.config.n_samples

        # Генерация базовых признаков с помощью make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=1,
            weights=[0.75, 0.25],
            random_state=self.config.random_state
        )

        # Создание DataFrame
        df = pd.DataFrame()

        # Добавление различных групп признаков
        demographic_data = self.generate_demographic_features(n_samples)
        service_data = self.generate_service_features(n_samples)

        has_internet = np.array([1 if x != "No" else 0 for x in service_data["InternetService"]])
        additional_services = self.generate_additional_services(n_samples, has_internet)

        contract_payment_data = self.generate_contract_and_payment(n_samples)

        # Объединение всех данных
        all_data = {**demographic_data, **service_data, **additional_services, **contract_payment_data}

        for key, value in all_data.items():
            df[key] = value

        # Расчет TotalCharges
        df["TotalCharges"] = self.calculate_total_charges(df)

        # Добавление целевой переменной с реалистичными паттернами
        df[TARGET_COLUMN] = self.add_realistic_churn_patterns(df, y)
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({0: "No", 1: "Yes"})

        logger.info("Генерация датасета завершена!")
        return df

    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """Валидация сгенерированного датасета"""
        logger.info("Валидация датасета...")

        checks = []

        # Проверка размера
        checks.append(len(df) == self.config.n_samples)

        # Проверка отсутствия пропусков
        checks.append(df.isnull().sum().sum() == 0)

        # Проверка диапазонов числовых признаков
        checks.append((df["Age"] >= 18).all() and (df["Age"] <= 80).all())
        checks.append((df["tenure"] >= 0).all() and (df["tenure"] <= 72).all())
        checks.append((df["MonthlyCharges"] >= 20).all() and (df["MonthlyCharges"] <= 120).all())
        checks.append((df["TotalCharges"] >= 0).all())

        # Проверка распределения оттока
        churn_rate = (df[TARGET_COLUMN] == "Yes").mean()
        checks.append(0.2 <= churn_rate <= 0.3)  # Ожидаемый дисбаланс 20-30%

        is_valid = all(checks)

        if is_valid:
            logger.info(" Валидация пройдена успешно!")
        else:
            logger.error(" Валидация не пройдена!")

        return is_valid


def main():
    """Основная функция выполнения"""
    try:
        # Инициализация генератора
        config = DataGenerationConfig()
        generator = DataGenerator(config)

        # Генерация датасета
        logger.info(" Запуск генерации датасета...")
        df = generator.generate_dataset()

        # Валидация
        if generator.validate_dataset(df):
            # Сохранение
            df.to_csv(DATA_PATH, index=False)
            logger.info(f" Датасет сохранен: {DATA_PATH}")

            # Статистика
            logger.info("\n" + "=" * 50)
            logger.info("СТАТИСТИКА ДАТАСЕТА:")
            logger.info(f"Размер: {df.shape}")
            logger.info(f"Колонки: {list(df.columns)}")
            logger.info(f"Распределение оттока:\n{df[TARGET_COLUMN].value_counts()}")
            logger.info(f"Процент оттока: {(df[TARGET_COLUMN] == 'Yes').mean():.1%}")
            logger.info("=" * 50)

        else:
            logger.error("Генерация прервана из-за ошибок валидации")

    except Exception as e:
        logger.error(f" Ошибка при генерации датасета: {str(e)}")
        raise


if __name__ == "__main__":
    main()