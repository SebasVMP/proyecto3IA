# Laboratorio de búsqueda y optimización

Implementación reproducible para N-Reinas, Mochila 0/1, coloreado de grafos y Eggholder. Incluye backtracking con filtrado/ordenamiento, recocido simulado (SA), algoritmo genético (GA), semillas, registro de resultados, Excel y gráficas.

El cuaderno listo para Colab está en [`run_experiments.ipynb`](./run_experiments.ipynb).

## Ejecutar localmente

```powershell
py -m pip install -r requirements.txt
py optimizer_lab.py --runs 5 --iterations 2000 --generations 200 --population 50
```

Los resultados se crean en `results/`: `experimental_results.xlsx` contiene las hojas `results`, `convergence` y `summary`; también se generan curvas de convergencia y una gráfica de tiempos. Para una prueba rápida: `py optimizer_lab.py --runs 1 --iterations 100 --generations 20 --population 20`.

Para generar el reporte PDF a partir del Excel: `py generate_report.py`. Se crea `reporte_tecnico.pdf` en la carpeta del proyecto.

## Instancias y alcance experimental

- N-Reinas: N = 8, 50 y 100.
- Mochila: los archivos de la consigna no incluyen la instancia de la tarea anterior; el programa usa por ahora una instancia sintética fija de 30 objetos y capacidad del 35% del peso total. Reemplaza `synthetic_knapsack()` con los pesos, valores y capacidad oficiales antes de reportar resultados.
- Coloreado: grafos aleatorios reproducibles de 50 y 1000 nodos, densidad objetivo de grado medio 4 y 4 colores. Cambia `make_graph()` para cargar los ficheros de instancias oficiales cuando estén disponibles. Backtracking tiene un presupuesto de 200 000 nodos. El Excel marca `termination=node_budget_exceeded` y deja vacío el objetivo si no termina la búsqueda; esto no debe reportarse como prueba de que el grafo es imposible de colorear.
- Eggholder: dominio `[-512, 512] x [-512, 512]`; minimización.

Los parámetros de los experimentos quedan guardados por ejecución. El objetivo de N-Reinas y coloreado es minimizar conflictos; Mochila maximiza el valor factible; Eggholder minimiza la función. Los métodos estocásticos se ejecutan con varias semillas. Aumenta `--runs` para el análisis final. Los tiempos y resultados de esta instancia sintética no sustituyen los datos oficiales de la materia.

## Subir a GitHub y abrir en Google Colab

1. En GitHub crea un repositorio vacío (público si necesitas compartir el código con el docente); no inicialices README si ya usarás este directorio.
2. En PowerShell, desde esta carpeta, configura el remoto y sube el proyecto:

   ```powershell
   git init
   git add .gitignore optimizer_lab.py generate_report.py requirements.txt README.md run_experiments.ipynb test_optimizer_lab.py results/ reporte_tecnico.pdf
   git commit -m "Implementar experimentos de optimizacion"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   git push -u origin main
   ```

   Sustituye `TU_USUARIO` y `TU_REPOSITORIO`. GitHub pedirá autenticación en el navegador o mediante Git Credential Manager; no pongas contraseñas ni tokens en el código.
3. En Colab selecciona **Archivo > Abrir cuaderno > GitHub**, pega la URL del repositorio y abre o crea un notebook. También puedes usar **Archivo > Subir cuaderno** para cargar un `.ipynb`.
4. En una celda de Colab, clona el repositorio e instala dependencias:

   ```python
   !git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   %cd TU_REPOSITORIO
   %pip install -r requirements.txt
   ```

5. Ejecuta los experimentos desde una celda:

   ```python
   !python optimizer_lab.py --runs 5 --iterations 2000 --generations 200 --population 50
   ```

   Descarga el Excel y las gráficas desde el panel **Archivos** de Colab o empaqueta los resultados. Comparte el enlace del notebook con permisos de lector y el enlace del repositorio GitHub en la entrega. Para subir cambios posteriores: `git add .`, `git commit -m "Actualizar experimentos"` y `git push`.

## Reporte

Interpreta las hojas y gráficas; informa media y desviación estándar de resultados estocásticos, factibilidad, semillas, evaluaciones y tiempo. Distingue explícitamente fallos por límite computacional de una demostración de inviabilidad. La instancia de Mochila y grafos debe sustituirse por los datos oficiales antes de entregar.
