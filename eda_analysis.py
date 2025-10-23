
import sys
import os

sys.path.append(os.getcwd())

from config import *
from utils import VisualizationUtils, DataAnalysisUtils, setup_logging

# Настройка логирования
setup_logging()

# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class EDAAnalysis:
    """Класс для проведения полного EDA анализа"""

    def __init__(self):
        self.viz_utils = VisualizationUtils(VisualizationConfig())
        self.data_utils = DataAnalysisUtils()
        self.df = None

    def load_data(self):
        """Загрузка и первичный анализ данных"""
        print("=" * 60)
        print(" ЗАГРУЗКА ДАННЫХ И ПЕРВИЧНЫЙ АНАЛИЗ")
        print("=" * 60)

        try:
            self.df = pd.read_csv(DATA_PATH)
            print(f" Датасет загружен успешно!")
            print(f" Размер датасета: {self.df.shape}")

        except FileNotFoundError:
            print(" Файл датасета не найден. Запустите data_generation.py сначала!")
            raise

        print("\n ПЕРВИЧНЫЙ ОБЗОР ДАННЫХ:")
        print("Первые 5 строк:")
        print(self.df.head())

        print("\n ИНФОРМАЦИЯ О ДАННЫХ:")
        print(self.df.info())

        return self.df

    def basic_statistics(self):
        """Базовые статистики"""
        print("\n" + "=" * 60)
        print(" БАЗОВЫЕ СТАТИСТИКИ")
        print("=" * 60)

        print("Статистики числовых признаков:")
        print(self.df[NUMERICAL_FEATURES].describe())

        print("\nСтатистики категориальных признаков:")
        for col in CATEGORICAL_FEATURES:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(self.df[col].value_counts())

    def analyze_target_variable(self):
        """Анализ целевой переменной"""
        print("\n" + "=" * 60)
        print(" АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
        print("=" * 60)

        churn_counts = self.df[TARGET_COLUMN].value_counts()
        churn_rate = (self.df[TARGET_COLUMN] == "Yes").mean() * 100
        loyal_rate = 100 - churn_rate

        # Визуализация
        fig, axes = self.viz_utils.create_subplots(2, 2, figsize=(15, 12))

        # Круговая диаграмма
        axes[0, 0].pie(churn_counts.values, labels=churn_counts.index,
                       autopct='%1.1f%%', colors=COLORS, startangle=90)
        axes[0, 0].set_title('Распределение оттока клиентов', fontsize=14)

        # Столбчатая диаграмма
        sns.countplot(data=self.df, x=TARGET_COLUMN, ax=axes[0, 1], palette=COLORS)
        axes[0, 1].set_title('Абсолютные значения оттока', fontsize=14)
        axes[0, 1].set_xlabel('Отток')
        axes[0, 1].set_ylabel('Количество клиентов')

        # Процентное распределение
        rates = [loyal_rate, churn_rate]
        bars = axes[1, 0].bar(['Лояльные', 'Ушедшие'], rates, color=COLORS)
        axes[1, 0].set_title('Процентное распределение', fontsize=14)
        axes[1, 0].set_ylabel('Процент')

        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height + 1,
                            f'{rate:.1f}%', ha='center', va='bottom', fontsize=12)

        # Отток по полу
        gender_churn = pd.crosstab(self.df['gender'], self.df[TARGET_COLUMN], normalize='index')
        gender_churn.plot(kind='bar', color=COLORS, ax=axes[1, 1], alpha=0.8)
        axes[1, 1].set_title('Отток по полу', fontsize=14)
        axes[1, 1].set_xlabel('Пол')
        axes[1, 1].legend(['Не ушли', 'Ушли'])

        plt.tight_layout()
        self.viz_utils.save_plot(fig, 'churn_distribution.png')
        plt.show()

        print(f" Детальная статистика оттока:")
        print(f"Лояльные клиенты: {churn_counts['No']} ({loyal_rate:.1f}%)")
        print(f"Ушедшие клиенты: {churn_counts['Yes']} ({churn_rate:.1f}%)")
        print(f"Дисбаланс классов: {churn_rate / loyal_rate:.2%}")

    def analyze_numerical_features(self):
        """Анализ числовых признаков"""
        print("\n" + "=" * 60)
        print(" АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ")
        print("=" * 60)

        # Гистограммы
        fig, axes = self.viz_utils.create_subplots(2, 2, figsize=(16, 12))

        for i, col in enumerate(NUMERICAL_FEATURES):
            row, col_idx = i // 2, i % 2
            sns.histplot(data=self.df, x=col, hue=TARGET_COLUMN, bins=30,
                         ax=axes[row, col_idx], alpha=0.7, palette=COLORS, kde=True)
            axes[row, col_idx].set_title(f'Распределение {col} по оттоку', fontsize=12)
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('Плотность')

        plt.tight_layout()
        self.viz_utils.save_plot(fig, 'numerical_features_distribution.png')
        plt.show()

        # Boxplot
        fig, axes = self.viz_utils.create_subplots(2, 2, figsize=(16, 12))

        for i, col in enumerate(NUMERICAL_FEATURES):
            row, col_idx = i // 2, i % 2
            sns.boxplot(data=self.df, y=col, x=TARGET_COLUMN, ax=axes[row, col_idx], palette=COLORS)
            axes[row, col_idx].set_title(f'Распределение {col} по оттоку', fontsize=12)
            axes[row, col_idx].set_xlabel('Отток')
            axes[row, col_idx].set_ylabel(col)

        plt.tight_layout()
        self.viz_utils.save_plot(fig, 'numerical_features_boxplot.png')
        plt.show()

    def analyze_categorical_features(self):
        """Анализ категориальных признаков"""
        print("\n" + "=" * 60)
        print(" АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
        print("=" * 60)

        key_categorical = ['Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling']

        fig, axes = self.viz_utils.create_subplots(2, 2, figsize=(20, 12))

        for i, col in enumerate(key_categorical):
            row, col_idx = i // 2, i % 2

            churn_by_cat = self.df.groupby(col)[TARGET_COLUMN].value_counts(normalize=True).unstack()
            churn_by_cat.plot(kind='bar', ax=axes[row, col_idx], color=COLORS, alpha=0.8)
            axes[row, col_idx].set_title(f'Отток по {col}', fontsize=12)
            axes[row, col_idx].set_xlabel('')
            axes[row, col_idx].tick_params(axis='x', rotation=45)
            axes[row, col_idx].legend(['Не ушли', 'Ушли'])

            for p in axes[row, col_idx].patches:
                height = p.get_height()
                if height > 0:
                    axes[row, col_idx].annotate(f'{height:.1%}',
                                                (p.get_x() + p.get_width() / 2., height),
                                                ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        self.viz_utils.save_plot(fig, 'categorical_features_analysis.png')
        plt.show()

    def correlation_analysis(self):
        """Корреляционный анализ"""
        print("\n" + "=" * 60)
        print(" КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
        print("=" * 60)

        # Подготовка данных
        df_numeric = self.df.copy()
        df_numeric['Churn_numeric'] = df_numeric[TARGET_COLUMN].map({'No': 0, 'Yes': 1})

        categorical_for_corr = ['Contract', 'InternetService', 'PaymentMethod']
        df_encoded = pd.get_dummies(df_numeric, columns=categorical_for_corr, drop_first=True)

        corr_columns = NUMERICAL_FEATURES + ['Churn_numeric'] + \
                       [col for col in df_encoded.columns if any(x in col for x in categorical_for_corr)]
        corr_matrix = df_encoded[corr_columns].corr()

        # Визуализация
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu', center=0,
                    square=True, fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Матрица корреляций признаков', fontsize=16)
        plt.tight_layout()
        self.viz_utils.save_plot(plt.gcf(), 'correlation_matrix.png')
        plt.show()

        print(" Корреляции с целевой переменной (Churn):")
        churn_correlations = corr_matrix['Churn_numeric'].sort_values(key=abs, ascending=False)
        print(churn_correlations.head(10))

    def detailed_analysis(self):
        """Детальный анализ ключевых факторов"""
        print("\n" + "=" * 60)
        print(" ДЕТАЛЬНЫЙ АНАЛИЗ КЛЮЧЕВЫХ ФАКТОРОВ")
        print("=" * 60)

        print("1.  ВЛИЯНИЕ ТИПА КОНТРАКТА:")
        contract_analysis = self.df.groupby('Contract').agg({
            TARGET_COLUMN: lambda x: (x == 'Yes').mean(),
            'tenure': 'mean',
            'MonthlyCharges': 'mean'
        }).round(3)
        print(contract_analysis)

        print("\n2. ВЛИЯНИЕ ТИПА ИНТЕРНЕТ-УСЛУГ:")
        internet_analysis = self.df.groupby('InternetService').agg({
            TARGET_COLUMN: lambda x: (x == 'Yes').mean(),
            'MonthlyCharges': 'mean'
        }).round(3)
        print(internet_analysis)

        print("\n3. ВЛИЯНИЕ МЕТОДА ОПЛАТЫ:")
        payment_analysis = self.df.groupby('PaymentMethod').agg({
            TARGET_COLUMN: lambda x: (x == 'Yes').mean()
        }).round(3).sort_values(TARGET_COLUMN, ascending=False)
        print(payment_analysis)

    def generate_conclusions(self):
        """Генерация выводов и рекомендаций"""
        print("\n" + "=" * 60)
        print(" ВЫВОДЫ И РЕКОМЕНДАЦИИ")
        print("=" * 60)

        churn_rate = (self.df[TARGET_COLUMN] == "Yes").mean() * 100
        loyal_rate = 100 - churn_rate

        contract_analysis = self.df.groupby('Contract').agg({
            TARGET_COLUMN: lambda x: (x == 'Yes').mean()
        })

        internet_analysis = self.df.groupby('InternetService').agg({
            TARGET_COLUMN: lambda x: (x == 'Yes').mean()
        })

        payment_analysis = self.df.groupby('PaymentMethod').agg({
            TARGET_COLUMN: lambda x: (x == 'Yes').mean()
        }).sort_values(TARGET_COLUMN, ascending=False)

        conclusions = f"""
 ОСНОВНЫЕ ВЫВОДЫ:

1. ДИСБАЛАНС КЛАССОВ:
   • Наблюдается значительный дисбаланс ({loyal_rate:.1f}% лояльных vs {churn_rate:.1f}% ушедших)
   • Необходимо использовать стратегии борьбы с дисбалансом

2. КЛЮЧЕВЫЕ ФАКТОРЫ ОТТОКА:
   • Тип контракта: Month-to-month клиенты уходят в {contract_analysis.loc['Month-to-month', 'Churn']:.1%} случаев
   • Интернет-услуги: Fiber optic показывает отток {internet_analysis.loc['Fiber optic', 'Churn']:.1%}
   • Метод оплаты: {payment_analysis.index[0]} ассоциируется с {payment_analysis.iloc[0, 0]:.1%} оттоком
БИЗНЕС-МЕТРИКИ:
• Потерянный доход: ${self.df[self.df[TARGET_COLUMN] == 'Yes']['TotalCharges'].sum():,.2f}
• Средний LTV ушедших: ${self.df[self.df[TARGET_COLUMN] == 'Yes']['TotalCharges'].mean():.2f}
"""
        print(conclusions)

    def run_full_analysis(self):
        """Запуск полного анализа"""
        print("🚀 ЗАПУСК ПОЛНОГО EDA АНАЛИЗА")
        print("=" * 60)

        self.load_data()
        self.basic_statistics()
        self.analyze_target_variable()
        self.analyze_numerical_features()
        self.analyze_categorical_features()
        self.correlation_analysis()
        self.detailed_analysis()
        self.generate_conclusions()

        print("\n" + "=" * 60)
        print(" EDA АНАЛИЗ ЗАВЕРШЕН!")
        print("=" * 60)


def main():
    """Основная функция"""
    analyzer = EDAAnalysis()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()