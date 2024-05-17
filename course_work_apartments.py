import ast

import numpy as np
from selenium.webdriver import Keys
from selenium.webdriver.common.actions.interaction import KEY
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from sklearn.cluster import KMeans
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import selenium.common.exceptions
import psycopg2.errors
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium_stealth import stealth
import json
from bs4 import BeautifulSoup
import re
import json
import folium
import math
import pandas as pd
import openai
import base64
import requests
from folium.plugins import MarkerCluster
from selenium.webdriver import Keys
from selenium.webdriver.common.actions.interaction import KEY
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import selenium.common.exceptions
import psycopg2.errors
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium_stealth import stealth
import json
from bs4 import BeautifulSoup
import re
import json
import folium
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder

'''авито парсер
Не особо оптимизированный (такой задачи не стояло)'''

options = Options()
options.add_argument("--disable-blink-features")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--headless")
options.add_experimental_option("detach", True)
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
        )
all_data = []
for i in range(1, 23):
    driver.get(
        f'https://www.avito.ru/js/1/map/items?page={i}&categoryId=24&locationId=621630&pmin=100&pmax=100000&params%5B201%5D=1060&params%5B504%5D=5257')
    json_data = json.loads(driver.find_element(By.TAG_NAME, 'body').text)
    all_data.extend(json_data['items'])
data_frame = pd.DataFrame(all_data)
data_frame.to_csv('parsed_data.csv', index=False)


'''Предобработка датасета'''
inf = pd.read_csv('parsed_data.csv')
old_data = pd.DataFrame(inf)

columns_needed = ['urlPath', 'title', 'description', 'priceDetailed', 'coords', 'images', 'rating']
data = old_data[columns_needed].copy()
data['price'] = data['priceDetailed'].apply(
    lambda x: int(''.join(re.findall(r'\d+', x)[:2])) if x is not None and re.findall(r'\d+', x) else None
)
data['lat'] = data['coords'].apply(
    lambda x: float(re.findall(r"'lat':\s*'([^']+)'", x)[0]) if x is not None else None
)
data['lng'] = data['coords'].apply(
    lambda x: float(re.findall(r"'lng':\s*'([^']+)'", x)[0]) if x is not None else None
)
data['rooms'] = data['title'].apply(lambda x: x.split(',')[0].strip())
data['rooms'] = data['rooms'].apply(lambda x: 'Студия' if 'студия' in x else int(x.split('-')[0]))
data['area'] = data['title'].apply(lambda x: float(x.split()[-4].strip().replace(',', '.')))
data['room_floor'] = data['title'].apply(lambda x: int(x.split()[-2].split('/')[0].strip()))
data['house_floors'] = data['title'].apply(lambda x: int(x.split()[-2].split('/')[1].strip().split()[0]))


data['images'] = data['images'].apply(lambda x: json.loads(x.replace("'", "\"")))
data['optimized_images'] = data['images'].apply(lambda x: [url for image_set in x for url in image_set.values()])
data['unique_images'] = data['optimized_images'].apply(lambda urls: list({url.split('/')[2].split('.')[0]: url for url in urls}.values()))
data['unique_images'] = data['unique_images'].apply(ast.literal_eval)

data['score'] = data['rating'].apply(lambda x: float(ast.literal_eval(x)['score']) if pd.notna(x) else float('nan'))
data['score'].fillna(data['score'].mean(), inplace=True)
data['unique_images'] = data['unique_images'].apply(ast.literal_eval)
data['reviews_count'] = old_data['rating'].apply(lambda x: float(ast.literal_eval(x)['summary'].split()[0]) if pd.notna(x) else float('nan'))

data['district'] = old_data['geo'].apply(
    lambda x: ast.literal_eval(x)['geoReferences'][0]['content'].split(' ')[1]
    if 'geoReferences' in ast.literal_eval(x) and len(ast.literal_eval(x)['geoReferences']) > 0
       and 'content' in ast.literal_eval(x)['geoReferences'][0] else None
)
data['sellers_advertisement_count'] = old_data['closedItemsText'].apply(
    lambda x: int(x.split()[0]) if x.split()[0].isdigit() else 1
)

data.drop('priceDetailed', axis=1, inplace=True)
data.drop('coords', axis=1, inplace=True)
data.drop('title', axis=1, inplace=True)
data.drop('optimized_images', axis=1, inplace=True)
data.drop('images', axis=1, inplace=True)
data.drop('rating', axis=1, inplace=True)


'''Гипотеза: на цену влиет географическое положение квартиры, для этого нужно провести географическую кластеризацию'''
"""Определяем оптимальное количество географических кластеров методом Локтя"""
X = data[['lat', 'lng']]
inertia = []
for n in range(1, 31): # Проверяем от 1 до 30 кластеров
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

elbow_results = pd.DataFrame({'Количество кластеров': range(1, 31), 'Inertia': inertia})

elbow_results.to_csv('elbow_results.csv', index=False)

plt.figure(figsize=(12, 6))
plt.plot(elbow_results['Количество кластеров'], elbow_results['Inertia'], marker='o')
plt.title('Метод локтя для определения оптимального количества кластеров')
plt.xlabel('Количество кластеров')
plt.ylabel('Inertia')
plt.xticks(range(1, 31))
plt.show() # оптимальное количество между 3 и 10, выбираю 10


"""Визуализация кластеризации по географическому критерию"""
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['geo_cluster'] = kmeans.fit_predict(data[['lat', 'lng']])
geo_cluster_centers = kmeans.cluster_centers_

brn_map = folium.Map(location=[data['lat'].mean(), data['lng'].mean()], zoom_start=10)
colors_list = ['blue', 'lightgreen', 'black', 'red', 'darkgreen', 'orange', 'pink', 'gray', 'purple', 'white']

for index, row in data.iterrows():
    popup_text = f"Цена: {row['price']}<br>Описание: {row['description'][:100]}..."  # Shortened description
    popup = folium.Popup(popup_text, parse_html=True)
    color = colors_list[row['geo_cluster']]
    folium.CircleMarker(
        location=[row['lat'], row['lng']],
        radius=3,
        color=color,
        fill=True,
        popup=popup
    ).add_to(brn_map)

# Добавление маркеров для центров кластеров
for i, center in enumerate(geo_cluster_centers):
    folium.Marker(
        location=[center[0], center[1]],
        icon=folium.Icon(color=colors_list[i], icon='info-sign'),
        popup=f'Геокластер {i}'
    ).add_to(brn_map)

brn_map.save('barnaul_distance_map.html')


'''Гипотеза: квартиры, находящиеся более чем в 2-ух километрах от центров кластеров можно считать выбросами, удаляем их'''
def haversine_distance(lat1, lon1, lat2, lon2):
    # Константы
    R = 6371.0  # Радиус Земли в километрах

    # Перевод градусов в радианы
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Разница координат
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Формула гаверсинуса
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Расстояние
    distance = R * c
    return distance


n_clusters = 10
# Применяем KMeans для определения географических кластеров (повторяем, если еще не сделано)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['geo_cluster'] = kmeans.fit_predict(data[['lat', 'lng']])

cluster_centers = kmeans.cluster_centers_

# Расчет расстояния от каждой квартиры до центра её геокластера
data['distance_to_center'] = data.apply(lambda row: haversine_distance(row['lat'], row['lng'],
                                                                       cluster_centers[row['geo_cluster'], 0],
                                                                       cluster_centers[row['geo_cluster'], 1]), axis=1)

# Фильтрация квартир, удаляем те, что находятся дальше 2 км от центра кластера
data = data[data['distance_to_center'] <= 2]
data.reset_index(drop=True, inplace=True)


'''Гипотеза: ценовая кластеризация влияет на цену квартир. Из-за слишком однородных данных обычная кластеризация не
информативна, решение - создание гибридной кластеризации - внутри каждого географического кластера провести ценовую
кластеризацию'''
"""определение оптимального количества ценовых кластеров внутри географических кластеров"""
# Создание фигуры и массива осей для отображения графиков
fig, axes = plt.subplots(5, 2, figsize=(15, 20))
axes = axes.ravel()

# Проходим по всем геокластерам
for i in range(n_clusters):
    cluster_prices = data[data['geo_cluster'] == i]['price'].values.reshape(-1, 1)
    inertias = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(cluster_prices)
        inertias.append(kmeans.inertia_)

    axes[i].plot(range(1, 10), inertias, marker='o')
    axes[i].set_title(f'Геокластер {i}')
    axes[i].set_xlabel('Количество кластеров')
    axes[i].set_ylabel('Инерция')
    axes[i].grid(True)

plt.tight_layout()
plt.show() # оптимальным количеством выбрано число 3

"""Добавление в датафрейм стобца гибридных кластеров"""
optimal_hybrid_clusters = 3

# Итерация по каждому географическому кластеру для выполнения ценовой кластеризации
for geo_cluster_id in range(data['geo_cluster'].nunique()):
    # Выборка данных для текущего геокластера
    cluster_data = data[data['geo_cluster'] == geo_cluster_id]

    # Проверка наличия данных перед кластеризацией
    if not cluster_data.empty:
        # Инициализация и обучение KMeans
        kmeans = KMeans(n_clusters=optimal_hybrid_clusters, random_state=42)
        # Применение модели и создание столбца hybrid_cluster для данного геокластера
        data.loc[data['geo_cluster'] == geo_cluster_id, 'hybrid_cluster'] = kmeans.fit_predict(cluster_data[['price']])
    else:
        # Если в геокластере нет данных, просто пропускаем его
        continue

"""Визуализация гибридной кластеризации"""
map_with_clusters = folium.Map(location=[data['lat'].mean(), data['lng'].mean()], zoom_start=10)

kmeans_geo = KMeans(n_clusters=10, random_state=42)
data['geo_cluster'] = kmeans_geo.fit_predict(data[['lat', 'lng']])

colors = ['black', 'cadetblue', 'darkblue', 'lightred', 'lightgray', 'blue', 'lightblue', 'gray', 'red', 'orange', 'beige', 'green', 'pink', 'purple', 'lightgreen', 'darkpurple', 'darkred', 'white', 'darkgreen']
geo_colors = colors[:10]
hybrid_colors = colors[-3:]
optimal_hybrid_clusters = 3
for geo_cluster_id in range(10):
    cluster_data = data[data['geo_cluster'] == geo_cluster_id]
    geo_color = geo_colors[geo_cluster_id]
    # Гибридная кластеризация внутри геокластера
    kmeans_hybrid = KMeans(n_clusters=3, random_state=42)
    cluster_data['hybrid_cluster'] = kmeans_hybrid.fit_predict(cluster_data[['price']])
    # Добавление маркеров квартир
    for index, row in cluster_data.iterrows():
        hybrid_color = hybrid_colors[row['hybrid_cluster']]
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=5,
            color=hybrid_color,
            fill=True,
            fill_color=geo_color,
            fill_opacity=1.0,
            popup=f"Цена: {row['price']}<br>Геокластер: {geo_cluster_id}<br>Гибридный кластер: {row['hybrid_cluster']}"
        ).add_to(map_with_clusters)

    # Добавление маркера центра геокластера
    folium.Marker(
        location=kmeans_geo.cluster_centers_[geo_cluster_id],
        icon=folium.Icon(color=geo_color, icon='info-sign'),
        popup=f'Географический кластер {geo_cluster_id}'
    ).add_to(map_with_clusters)

    # Добавление маркеров для центров гибридных кластеров
    for center_num in range(optimal_hybrid_clusters):
        center_data = cluster_data[cluster_data['hybrid_cluster'] == center_num]
        if not center_data.empty:
            center_lat = center_data['lat'].mean()  # Средняя широта квартир в кластере
            center_lng = center_data['lng'].mean()  # Средняя долгота квартир в кластере
            center_color = hybrid_colors[center_num]
            folium.Marker(
                location=[center_lat, center_lng],
                icon=folium.Icon(color=center_color, icon='star'),
                popup=f'Ценовой кластер {center_num}, Средняя цена: {int(center_data["price"].mean())}').add_to(map_with_clusters)

map_with_clusters.save('enhanced_clusters_map.html')


'''Гипотеза: на цену квартир влияет визуальная оценка фотографий квартир. Для оценки использовался API ChatGPT'''
API_OPENAI_TOKEN = ''

client = openai.OpenAI(api_key=API_OPENAI_TOKEN)

data['unique_images'] = data['unique_images'].apply(ast.literal_eval)
for i in data['unique_images']:
    templates = []
    for j in i:

        template = f'''{{
                  "type": "image_url",
                  "image_url": {{
                    "url": "{j}",
                    "detail": "low"
                  }},
                }}'''
        templates.append(template)
    print(templates)
    joined_elements = ",\n".join(templates)
    response = client.chat.completions.create(
        model="gpt-4-1106-vision-preview",
        messages=[
    {
      "role": "user",
      "content": [
          {
          "type": "text",
          "text": "Rate this apartment from 0 to 10 based on these photos. In the answer, write only the number"
          },
    joined_elements,
    ],
    }
  ],
  max_tokens=300,
)
    data['image_rating'] = response.choices[0].message.content


"""Сохраняем все изменения в датафрейме
Это финальный вид файла, с которым можно переходить к следующему этапу - регрессионныму анализу"""
data.to_excel('avito_dataframe_test.xlsx', index=False)

data['images_rating'] = data['images_rating'].replace('3/', 3).astype(float)

# Заполнение пропущенных значений средним значением
data['images_rating'].fillna(data['images_rating'].mean(), inplace=True)
encoder = OneHotEncoder(sparse=False, drop='first')
categorical_features = ['rooms', 'district', 'geo_cluster', 'hybrid_cluster']

# Преобразование всех значений в столбце rooms к строковому типу
data['rooms'] = data['rooms'].astype(str)
"""создание фиктивных переменных"""
# Повторное кодирование категориальных переменных
encoded_data = encoder.fit_transform(data[categorical_features])

# Создание DataFrame из закодированных данных
encoded_columns = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

# Добавление закодированных столбцов к исходным данным
data_encoded = data.join(encoded_df)

# Преобразование sellers_advertisement_count в фиктивную переменную
data_encoded['sellers_advertisement_count'] = (data_encoded['sellers_advertisement_count'] == 1).astype(int)

# Проверка исходного DataFrame с новыми закодированными данными
"""проверка на мультиколлениарность"""
# Выбор количественных и закодированных категориальных переменных для анализа
features = ['lat', 'lng', 'area', 'room_floor', 'house_floors', 'score', 'images_rating',
            'reviews_count', 'distance_to_center', 'sellers_advertisement_count'] + list(encoded_columns)

# Создание матрицы корреляций
correlation_matrix = data_encoded[features].corr()

# Визуализация матрицы корреляций
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Матрица корреляций')
plt.show()



X = data_encoded[features]  # независимые переменные
y = data_encoded['price']

# Заполнение пропущенных значений в reviews_count на 1
X['reviews_count'].fillna(1, inplace=True)

# Проверка, что пропущенные значения обработаны
X['reviews_count'].isnull().sum()
"""Метод обратной пошаговой регрессии"""
def backward_elimination(data, dependent_var, significance_level = 0.05):
    features = data.columns.tolist()
    while len(features) > 0:
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(dependent_var, features_with_constant).fit().pvalues[1:]  # исключаем константу
        max_p_value = p_values.max()  # наибольшее p-значение
        if max_p_value >= significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features

# Применение метода обратной пошаговой регрессии
significant_features = backward_elimination(X[features], y)

# Построение регрессионной модели с учётом только значимых факторов
X_significant = X[significant_features]
model_reduced = sm.OLS(y, sm.add_constant(X_significant)).fit()

# Вывод результатов упрощённой модели
print('метод обр пош рег', model_reduced.summary())


features_adjusted = significant_features + ['hybrid_cluster_2']
features_adjusted.remove('lng')
final_x = X[features_adjusted]
"""вновь смотрим матрицу кореляций"""
# Построение матрицы корреляций для значимых переменных
correlation_matrix_significant = final_x.corr()

# Визуализация матрицы корреляций для значимых переменных
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_significant, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Матрица корреляций для значимых переменных')
plt.show()
'финальная модель'
model = sm.OLS(y, sm.add_constant(final_x)).fit()
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import het_goldfeldquandt

# Вывод результатов регрессии для скорректированной модели
print('final model' ,model.summary())

'проверка на гетероскедастичность'
# Тест Бройша-Пейгана
bp_test = sms.het_breuschpagan(model.resid, model.model.exog)
print(bp_test[0])
print("p-value:", bp_test[1])
print("f-value:", bp_test[2])
print("f p-value:", bp_test[3]) #гетероскедастичности нет
'тест чоу с географическими кластерами'
# График остатков против предсказанных значений графический метод
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('Остатки vs Предсказанные значения')
plt.show() # нету

# Проведение теста Гольдфелда-Квандта
n = 1038
m = 311
'm берем как 30%'
drop_ratio = (n - 2 * m) / n
gq_test = het_goldfeldquandt(y, final_x, split=m/n, drop=drop_ratio)
print(f"F-statistic: {gq_test[0]}, p-value: {gq_test[1]}") # нету

'оценка модель в логарифмах'
y_log = np.log(y)  # Используем y + 1 для обработки нулевых значений

# Если X содержит переменные, которые также нужно преобразовать, примените к ним логарифм
X_transformed = np.log(final_x)

# Построение модели с логарифмированной зависимой переменной
X_with_const = sm.add_constant(final_x)  # Убедитесь, что X_adjusted — это DataFrame с вашими независимыми переменными
model_log = sm.OLS(y_log, X_with_const).fit()

# Вывод результатов
print('log_log model', model_log.summary())

# График остатков против предсказанных значений графический метод смотрим в логарифмах
plt.scatter(model_log.fittedvalues, model.resid)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('Остатки vs Предсказанные значения')
plt.show()
