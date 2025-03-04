
Gradient Boosting es un método de ensamble que construye un modelo fuerte combinando múltiples modelos débiles, generalmente árboles de decisión. Cada modelo adicional se entrena para corregir los errores del conjunto anterior, minimizando la función de pérdida mediante el descenso de gradiente.

#### Implementación  **  
1. **Generación de datos de prueba**: Se crean datos ruidosos a partir de una función seno.  
2. **Entrenamiento del modelo**:  
   - Se usa la clase `GradientBoosting` del módulo en C++.  
   - Se especifican parámetros como la tasa de aprendizaje (`learning_rate`) y el número de estimadores (`n_estimators`).  
   - El modelo ajusta cada estimador para reducir el error residual.  
3. **Predicción**: Se generan valores predichos `y_pred` para los datos de entrada `X`.  
4. **Visualización**:  
   - **Gráfica de regresión**: Muestra los datos originales y la predicción del modelo.  
   - **Gráfica de flujo de datos (residuos)**: Representa los errores (diferencia entre valores reales y predichos), permitiendo evaluar el ajuste del modelo.  

🔹 **Conclusión**: El programa muestra cómo Gradient Boosting reduce el error progresivamente y visualiza el ajuste del modelo. 
