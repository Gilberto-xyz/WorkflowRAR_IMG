# rar_folder_image_info.py

## Descripción

Este script avanzado automatiza el análisis, documentación y compresión de archivos de video organizados en carpetas. Está diseñado para flujos de trabajo exigentes, como la gestión de grandes colecciones de videos, respaldos o distribución de contenido multimedia.

## Características principales

- **Procesamiento por carpetas**: Analiza automáticamente todas las subcarpetas de un directorio base, procesando cada video encontrado.
- **Extracción de metadatos**: Obtiene información detallada de cada video (resolución, duración, peso, pistas de audio y subtítulos, idiomas, canales, formato, etc.) usando `pymediainfo`.
- **Capturas automáticas**: Genera capturas de pantalla (JPG/PNG) en posiciones estratégicas del video usando `ffmpeg`, con soporte para múltiples configuraciones de cantidad y porcentaje.
- **Compresión RAR avanzada**: Comprime cada video en su propio archivo RAR (opcionalmente sin compresión para máxima velocidad), con soporte para división automática de archivos grandes y nombres limpios.
- **Paralelización**: Procesa videos en paralelo usando múltiples hilos, acelerando el análisis y la generación de capturas.
- **Interfaz visual moderna**: Utiliza la biblioteca `rich` para mostrar el progreso, paneles informativos, tablas y logs en consola de forma atractiva y clara.
- **Logs detallados**: Permite guardar logs completos de todo el proceso, incluyendo errores y advertencias, en un archivo.
- **Configuración flexible**: Todos los parámetros clave (extensiones, rutas, workers, compresión, generación de capturas, etc.) son configurables por línea de comandos.
- **Robustez y tolerancia a errores**: Manejo avanzado de excepciones, logs enriquecidos y decoradores de seguridad para evitar que errores individuales detengan el proceso global.
- **Soporte para nombres y rutas complejas**: Limpieza y normalización de nombres de archivos para máxima compatibilidad.

## Ventajas destacadas

- **Automatización total**: Ideal para procesar grandes lotes de videos sin intervención manual.
- **Ahorro de tiempo**: La paralelización y la omisión de pasos innecesarios aceleran el flujo de trabajo.
- **Resultados profesionales**: Documentación visual y técnica de cada video, lista para compartir o archivar.
- **Compatibilidad**: Funciona en Windows y puede adaptarse fácilmente a otros sistemas.
- **Personalización**: Se adapta a distintos escenarios (solo análisis, solo compresión, solo capturas, etc.).
- **Feedback visual**: El usuario siempre sabe el estado y resultado de cada paso.

## Requisitos

- Python 3.8+
- [pymediainfo](https://pypi.org/project/pymediainfo/)
- [rich](https://pypi.org/project/rich/)
- ffmpeg (en el PATH)
- WinRAR (rar.exe, para compresión)

Instalación de dependencias:
```bash
pip install pymediainfo rich
```

## Uso básico

```bash
python rar_folder_image_info.py [directorio_base] [opciones]
```

### Ejemplo:
```bash
python rar_folder_image_info.py "C:\MisVideos" --workers 4 --rar-path "C:\Program Files\WinRAR\rar.exe" --logfile
```

## Opciones principales
- `--workers N`           : Número de hilos para procesar videos en paralelo.
- `--exts .mkv .mp4 ...`  : Extensiones de video a buscar.
- `--skip-img`            : Omitir la generación de capturas.
- `--no-compress`         : Omitir la compresión RAR.
- `--rar-store-only`      : Crear RAR sin compresión (por defecto).
- `--rar-compress`        : Crear RAR con compresión normal.
- `--logfile [ruta]`      : Guardar log detallado en archivo.
- `-v`/`--verbose`        : Modo depuración (más detalles en consola).

## Ejemplo de flujo de trabajo
1. Analiza todas las subcarpetas de un directorio base.
2. Para cada video:
   - Extrae metadatos y muestra información detallada.
   - Genera capturas de pantalla (si no se omite).
   - Comprime el video en un archivo RAR (si no se omite).
3. Muestra un resumen final con métricas globales.

## Créditos
Desarrollado por Gilberto Nava Marcos.

---

¡Este script es ideal para archivistas, uploaders, coleccionistas y cualquier usuario que requiera un flujo de trabajo profesional y automatizado para videos!
