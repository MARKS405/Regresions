# Econometric Regression Lab · V1

App Streamlit para cargar archivos Excel, transformar variables, correr regresiones OLS/Panel y exportar resultados reproducibles.

## Alcance V1

- Excel como input.
- Frecuencia mensual.
- Serie agregada o panel estilo notebook: `Variable`, `Banco` y fechas como columnas.
- Transformaciones automáticas: `ln_`, `d_`, `_lag1`, `ln_lag1`, `d_ln_`, `d_lag1`, `d_ln_lag1`, variaciones 12 meses y medias móviles opcionales.
- Dummies definidas por usuario y dummies de eventos por rango de fechas.
- OLS, OLS robusto HC1, OLS HAC.
- Panel: Pooled OLS, Fixed Effects entidad, Fixed Effects tiempo, Two-way FE y Random Effects.
- Diagnósticos: missing, correlación, ADF, VIF, autocorrelación, heterocedasticidad.
- Variable objetivo vs dependiente transformada, con reconstrucción del estimado.
- Contrafactuales excluyendo variables.
- Exportación a Excel, JSON e `.ipynb` reproducible.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Contraseña demo local: `bcrp_demo`. Cambiar en `.streamlit/secrets.toml` antes de desplegar.

## Seguridad V1

La app no guarda archivos en disco. Los datos cargados se mantienen en memoria de sesión y se eliminan al cerrar sesión o limpiar sesión.
