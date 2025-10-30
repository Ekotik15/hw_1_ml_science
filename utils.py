"""
Утилиты для проекта
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import logging

from config import VisualizationConfig, IMAGES_DIR

logger = logging.getLogger(__name__)


class VisualizationUtils:
    """Утилиты для визуализации"""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self._setup_plot_style()

    def _setup_plot_style(self):
        """Настройка стиля графиков"""
        plt.style.use(self.config.style)
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = self.config.fig_size
        plt.rcParams['figure.dpi'] = self.config.dpi

    def save_plot(self, fig: plt.Figure, filename: str,
                  bbox_inches: str = 'tight',
                  dpi: Optional[int] = None) -> None:
        """Сохранение графика"""
        if dpi is None:
            dpi = self.config.dpi

        filepath = IMAGES_DIR / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"📈 График сохранен: {filepath}")

    def create_subplots(self, nrows: int, ncols: int, **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """Создание subplots с настройками"""
        fig, axes = plt.subplots(nrows, ncols, **kwargs)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(-1, 1) if ncols == 1 else axes.reshape(1, -1)
        return fig, axes


class DataAnalysisUtils:
    """Утилиты для анализа данных"""

    @staticmethod
    def analyze_missing_data(df: pd.DataFrame) -> pd.DataFrame:
        """Анализ пропущенных данных"""
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100

        result = pd.DataFrame({
            'missing_count': missing,
            'missing_percent': missing_percent
        }).sort_values('missing_count', ascending=False)

        return result[result['missing_count'] > 0]

    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Обнаружение выбросов с помощью IQR"""
        outlier_info = {}

        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            outlier_info[col] = {
                'count': len(outliers),
                'percent': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }

        return pd.DataFrame(outlier_info).T

    @staticmethod
    def calculate_feature_importance(df: pd.DataFrame, target_col: str,
                                     categorical_features: List[str]) -> pd.DataFrame:
        """Расчет важности признаков через корреляции и ANOVA"""
        from scipy.stats import f_oneway

        importance_data = []

        # Для числовых признаков - корреляция
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = numerical_cols.drop(target_col, errors='ignore')

        for col in numerical_cols:
            if col != target_col:
                correlation = df[col].corr(df[target_col] == "Yes")
                importance_data.append({
                    'feature': col,
                    'importance': abs(correlation),
                    'type': 'numerical',
                    'method': 'correlation'
                })

        # Для категориальных признаков - ANOVA
        for col in categorical_features:
            if col in df.columns:
                groups = [group[target_col].values for name, group in df.groupby(col)]
                if len(groups) > 1:
                    f_stat, p_value = f_oneway(*groups)
                    importance_data.append({
                        'feature': col,
                        'importance': f_stat,
                        'type': 'categorical',
                        'method': 'anova'
                    })

        importance_df = pd.DataFrame(importance_data)
        return importance_df.sort_values('importance', ascending=False)


def setup_logging(level=logging.INFO):
    """Настройка логирования"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('project.log'),
            logging.StreamHandler()
        ]
    )