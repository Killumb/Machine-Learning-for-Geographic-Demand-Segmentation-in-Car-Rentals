
import streamlit as st
import pandas as pd
import lightgbm as lgb
import os
import json

# --- Настройка страницы ---
st.set_page_config(
    page_title="Советник по расширению автопарка",
    page_icon="💡",
    layout="wide"
)

# --- Функция для загрузки артефактов (с кэшированием) ---
@st.cache_resource
def load_artifacts():
    """Загружает модель, статистики, пороги и список колонок."""
    paths = {
        "model": 'models/lgbm_model.txt',
        "city_stats": 'processed_data/city_stats.csv',
        "train_cols": 'processed_data/X_train.csv',
        "thresholds": 'processed_data/demand_thresholds.json'
    }

    if not all(os.path.exists(p) for p in paths.values()):
        st.error("Ошибка: отсутствуют необходимые файлы. Пожалуйста, запустите Jupyter ноутбуки (Этапы 2 и 3).")
        return None, None, None, None

    model = lgb.Booster(model_file=paths["model"])
    city_stats = pd.read_csv(paths["city_stats"])
    train_cols = pd.read_csv(paths["train_cols"]).columns
    
    with open(paths["thresholds"], 'r') as f:
        thresholds = json.load(f)

    # Исключаем 'leaky' признаки
    features_to_drop = ['rating', 'reviewCount']
    train_cols = [col for col in train_cols if col not in features_to_drop]
    
    return model, city_stats, train_cols, thresholds

# Загружаем данные
model, city_stats, train_cols, thresholds = load_artifacts()

# --- Основной интерфейс приложения ---
st.title('💡 Советник по расширению автопарка')
st.markdown("Этот инструмент анализирует потенциал рынка и дает рекомендации по увеличению или сокращению автопарка.")

if model is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("⚙️ Введите сценарий")
        city = st.selectbox('1. Город для анализа:', city_stats['location.city'].unique())
        v_type = st.selectbox('2. Тип автомобиля:', ['car', 'suv', 'truck', 'van', 'minivan'])
        v_age = st.slider('3. Возраст автомобиля (лет):', 1, 15, 1)
        fuel_type = st.radio("4. Тип топлива:", ('GASOLINE', 'HYBRID', 'ELECTRIC'), horizontal=True)
        predict_button = st.button('📈 Получить рекомендацию', use_container_width=True, type="primary")

    if predict_button:
        # --- Подготовка данных для предсказания (логика осталась той же) ---
        current_city_stats = city_stats[city_stats['location.city'] == city].iloc[0]
        features = pd.DataFrame(columns=train_cols)
        features.loc[0] = 0
        
        features['vehicle.age'] = v_age
        features['rate.daily'] = current_city_stats['city_avg_rate']
        features['city_avg_trips'] = current_city_stats['city_avg_trips']
        features['city_car_count'] = current_city_stats['city_car_count'] + 1
        features['city_avg_rate'] = current_city_stats['city_avg_rate']
        
        if f'vehicle.type_{v_type}' in features.columns:
            features[f'vehicle.type_{v_type}'] = True
        if f'fuelType_{fuel_type}' in features.columns:
            features[f'fuelType_{fuel_type}'] = True
        
        features = features.astype(float)
        prediction = model.predict(features)[0]

        # --- НОВАЯ ЛОГИКА: Генерация рекомендаций ---
        with col2:
            st.header("📋 Рекомендация и анализ")
            
            # Блок с основным прогнозом
            st.metric(label="Прогнозируемое количество поездок", value=f"{prediction:.1f}")
            st.caption(f"Для нового автомобиля типа '{v_type}' в городе {city}.")
            
            # Блок с анализом и рекомендацией
            avg_demand = thresholds['average_demand']
            high_demand = thresholds['high_demand']
            
            st.write("---")
            
            if prediction > high_demand:
                st.success(f"**Рекомендация: Увеличить автопарк.**", icon="🚀")
                st.write(f"Прогноз ({prediction:.1f} поездок) **значительно выше** высокого порога спроса по рынку ({high_demand:.1f} поездок). "
                         f"Это сильный сигнал о том, что в городе **{city}** существует неудовлетворенный спрос на автомобили типа **{v_type}**. "
                         "Расширение парка в этой конфигурации имеет высокий потенциал успеха.")
            elif prediction > avg_demand:
                st.info(f"**Рекомендация: Рассмотреть увеличение автопарка.**", icon="👍")
                st.write(f"Прогноз ({prediction:.1f} поездок) **выше среднего** уровня спроса по рынку ({avg_demand:.1f} поездок). "
                         f"Это говорит о стабильном спросе. Расширение является оправданным, но не является первоочередным приоритетом по сравнению с рынками с более высоким потенциалом.")
            else:
                st.warning(f"**Рекомендация: Не увеличивать автопарк / Сократить.**", icon="⚠️")
                st.write(f"Прогноз ({prediction:.1f} поездок) **ниже среднего** уровня спроса по рынку ({avg_demand:.1f} поездок). "
                         f"Это может указывать на насыщение рынка или низкую популярность данной конфигурации (тип/возраст авто) в городе **{city}**. "
                         "Рекомендуется пересмотреть стратегию для этого сегмента или рассмотреть другие города/типы авто.")

            # Дополнительный контекст
            st.write("---")
            st.subheader("Рыночный контекст")
            sub_col1, sub_col2 = st.columns(2)
            sub_col1.metric("Средний спрос в этом городе", f"{current_city_stats['city_avg_trips']:.1f} поездок")
            sub_col2.metric("Уровень конкуренции (авто)", f"{int(current_city_stats['city_car_count'])}")