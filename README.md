# Sistema de Optimización de Paradas Suburbanas

**Trabajo final de Maestría en Ciencia de Datos – Universidad de Montevideo**  
Autores: Pablo Maurente y Ana Araujo  
Tutor: César Reyes

---

## Descripción

Este proyecto implementa un sistema reproducible para detectar, evaluar y optimizar la localización de paradas de ómnibus suburbanos e interdepartamentales en rutas nacionales de Uruguay.

Fue aplicado como prueba de concepto sobre las rutas **5**, **8** y **9**, combinando visión por computadora, análisis geoespacial y priorización multicriterio (AHP).

---

## Objetivos

- Identificar las paradas existentes usando imágenes satelitales.
- Evaluar su desempeño según criterios de cobertura, accesibilidad, espaciado, seguridad y demanda potencial.
- Generar una propuesta optimizada de paradas para cada corredor.
- Proporcionar una herramienta adaptable a nuevas rutas o escenarios.

---

## Estructura del repositorio

- `Datos/` – Capas geoespaciales e insumos del modelo.
- `Notebooks/` – Jupyter notebooks con análisis y modelos.
- `tesis_tex/` – Proyecto LaTeX de la tesis.
- `Sistema-de-Optimizacion-de-Paradas-Suburbanas.pdf` – Informe final.
- `Bus_Stop_functions.py` – Funciones auxiliares del sistema.
- `mapa_interactivo_.html` – Visualización con mapas comparativos.
  > **⚠️ Para visualizar:** descargar el archivo y abrirlo localmente en un navegador.

---

## Resultados principales

- +450 paradas detectadas y evaluadas.
- Propuestas de rediseño que mejoran cobertura y reducen redundancias.
- Mejora del puntaje global AHP en todos los escenarios optimizados.
- Herramienta replicable en nuevos corredores o territorios.

---

## Aplicaciones

- Planificación de transporte público suburbano/interdepartamental.
- Apoyo a políticas públicas en conectividad territorial.
- Optimización de operaciones para empresas transportistas.

---
