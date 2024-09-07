#Importacion de bibliotecas
import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

#Ruta de los datos crudos
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Función para descargar los datos
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Función para cargar los datos
@st.cache_data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# Cargar los datos de vivienda
if not os.path.exists(os.path.join(HOUSING_PATH, "housing.csv")):
    st.write("Descargando datos...")
    fetch_housing_data()

housing = load_housing_data()

# Preprocesamiento de los datos
@st.cache_data
def preprocess_data(housing):
    # Excluir median_house_value del pipeline
    housing_num = housing.drop(columns=["ocean_proximity", "median_house_value"])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    
    housing_prepared = full_pipeline.fit_transform(housing)
    return housing_prepared, full_pipeline

# Titulo e introducción
st.markdown("# Proyecto de Ciencia de Datos")
st.markdown("### **Implementacion de Machine Learning para la prediccion de precios de vivienda en California**")
st.markdown("## **Introducción**")
st.markdown("Este proyecto consta de utilizar un dataset con informacion sobre las viviendas en el estado de california, tiene datos sobre el valor medio, la localización, proximidad al oceano, cantidad de viviendas, asi como la cantidad de personas, cuartos y dormitorios. \n Los datos se exploraron y visualizaron previamente para sus analisis, se entreno el modelo con los datos para definir cual seria el mejor algoritmo, y como resultado obtuvimos que el mejor algoritmo para este grupo de datos fue el Random Forest.")

# Visualización de datos

st.title("Precios de Vivienda en California")
st.write("Muestra de Datos originales:")
st.dataframe(housing.head())
with st.expander("Mostrar datos completos"):
    st.dataframe(housing)

with st.expander("Mapa de las Viviendas"):
    map_data = housing[['latitude', 'longitude']]
    map_data.columns = ['lat', 'lon']  # Renombrar columnas para st.map
    st.map(map_data)

with st.expander("Precios por Ubicación"):
    fig, ax = plt.subplots()
    sns.boxplot(x="ocean_proximity", y="median_house_value", data=housing, ax=ax)
    st.pyplot(fig)

with st.expander("Precio vs. Ingreso Mediano"):
    fig, ax = plt.subplots()
    sns.scatterplot(x="median_income", y="median_house_value", data=housing, ax=ax)
    ax.set_xlabel("Ingreso Medio")
    ax.set_ylabel("Precio de la Vivienda")
    st.pyplot(fig)

with st.expander("Matriz de Correlación"):
    corr_matrix = housing.select_dtypes(include=['number']).corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with st.expander("Distribución de Precios de Viviendas"):
    fig, ax = plt.subplots()
    housing["median_house_value"].hist(ax=ax, bins=50)
    ax.set_xlabel("Precio")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

with st.expander("Densidad de Precios de Viviendas"):
    fig, ax = plt.subplots()
    sns.kdeplot(housing["median_house_value"], ax=ax)
    ax.set_xlabel("Precio")
    ax.set_ylabel("Densidad")
    st.pyplot(fig)

with st.expander("Número de Viviendas por Ubicación"):
    fig, ax = plt.subplots()
    housing['ocean_proximity'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel("Ubicación")
    ax.set_ylabel("Número de Viviendas")
    st.pyplot(fig)

# Preprocesar los datos
housing_prepared, full_pipeline = preprocess_data(housing)
housing_labels = housing["median_house_value"].copy()

# Prediccion con datos nuevos

st.header("Predicción de Precios de Viviendas")
st.markdown("Para realizar la predicción del precio de vivienda, porfavor entre los datos en la barra lateral.")

## Crear campos de entrada para las características en un menú desplegable
with st.sidebar:
    st.subheader("Ingrese las características de la vivienda")
    rooms = st.slider( "Total Rooms", min_value=1, max_value=1000, value=300)
    bedrooms = st.slider("Total Bedrooms", min_value=1, max_value=1000, value=300)
    population = st.slider("Population", min_value=1, max_value=1000, value=1000)
    households = st.slider("Households", min_value=1, max_value=1000, value=400)
    latitude = st.slider("Latitude", min_value=30.0, max_value=45.0, value=34.0)
    longitude = st.slider("Longitude", min_value=-125.0, max_value=-114.0, value=-118.0)
    housing_median_age = st.slider("Housing Median Age", min_value=1, max_value=100, value=30)
    median_income = st.number_input("Median Income", value=5)
    proximity = st.selectbox("Ocean Proximity", ['NEAR BAY', 'INLAND', 'NEAR OCEAN', 'ISLAND', '1H OCEAN'])


if st.button("Predecir"):
    # Crear un diccionario con todos los atributos necesarios, excepto median_house_value
    new_data = {
        'total_rooms': [rooms],
        'total_bedrooms': [bedrooms],
        'population': [population],
        'households': [households],
        'housing_median_age': [housing_median_age],
        'median_income': [median_income],
        'latitude': [latitude],
        'longitude': [longitude],
        'ocean_proximity': [proximity],
    }
    new_data_df = pd.DataFrame(new_data)

    # Mostrar los datos introducidos
    st.subheader("Datos de Entrada")
    st.dataframe(new_data_df)

    # Preprocesar los datos nuevos usando el full_pipeline
    new_data_prepared = full_pipeline.transform(new_data_df)

    # Realizar la predicción usando el modelo final (forest_reg )
    forest_reg = RandomForestRegressor(max_features=8, n_estimators=30)
    forest_reg.fit(housing_prepared, housing_labels)
    final_model = forest_reg
    pred = final_model.predict(new_data_prepared)

    # Mostrar el valor predicho
    st.subheader("Predicción")
    st.success(f"Valor Predicho: ${pred[0]:,.2f}")

#Creditos
st.markdown("### Elaborado por: **José Eduardo Conde Hernández 299506** ")
st.markdown("#### Para la materia de Data Science impartida por el docente: **Jesús Roberto López Santillan**")