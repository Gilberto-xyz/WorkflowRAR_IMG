# 🎬 rar_folder_image_info.py

## ✨ Descripción general

`rar_folder_image_info.py` es un script **avanzado y automatizado** para analizar, documentar y comprimir archivos de video organizados en carpetas. Está pensado para flujos de trabajo exigentes: gestión de grandes colecciones, respaldos, archivado o distribución profesional de contenido multimedia.

---

## 🚀 Características principales

- 📂 **Procesamiento masivo por carpetas**  
  Analiza todas las subcarpetas de un directorio base y procesa cada video encontrado.

- 🏷️ **Extracción de metadatos avanzada**  
  Obtiene información detallada (resolución, duración, peso, pistas, idiomas, canales, formato, etc.) usando `pymediainfo`.

- 🖼️ **Capturas automáticas**  
  Genera capturas de pantalla (JPG/PNG) en posiciones estratégicas usando `ffmpeg`, configurable en cantidad y porcentaje.

- 📦 **Compresión RAR profesional**  
  Comprime cada video en su propio archivo RAR (con o sin compresión), soporta división automática y nombres limpios.

- ⚡ **Procesamiento paralelo**  
  Usa múltiples hilos para acelerar el análisis y la generación de capturas.

- 🎨 **Interfaz visual moderna**  
  Utiliza `rich` para mostrar progreso, paneles, tablas y logs en consola de forma atractiva y clara.

- 📝 **Logs detallados**  
  Guarda logs completos del proceso, incluyendo errores y advertencias.

- 🛠️ **Configuración flexible**  
  Todos los parámetros clave son configurables por línea de comandos.

- 🦾 **Robustez y tolerancia a errores**  
  Manejo avanzado de excepciones y logs enriquecidos para máxima estabilidad.

- 🧹 **Soporte para nombres y rutas complejas**  
  Limpieza y normalización de nombres para máxima compatibilidad.

---

## 🌟 Ventajas destacadas

- **Automatización total**: Procesa grandes lotes de videos sin intervención manual.
- **Ahorro de tiempo**: Paralelización y omisión de pasos innecesarios.
- **Resultados profesionales**: Documentación visual y técnica lista para compartir o archivar.
- **Compatibilidad**: Funciona en Windows y puede adaptarse a otros sistemas.
- **Personalización**: Se adapta a distintos escenarios (solo análisis, solo compresión, solo capturas, etc.).
- **Feedback visual**: Siempre sabrás el estado y resultado de cada paso.

---

## 🧩 Requisitos

- Python 3.8+
- [pymediainfo](https://pypi.org/project/pymediainfo/)
- [rich](https://pypi.org/project/rich/)
- ffmpeg (en el PATH)
- WinRAR (`rar.exe`, para compresión)

**Instalación de dependencias:**
```bash
pip install pymediainfo rich
```

---

## 🏁 Uso rápido

```bash
python rar_folder_image_info.py [directorio_base] [opciones]
```

> 💡 **TIP:** Usa `--help` para ver todos los parámetros y opciones disponibles:
> ```bash
> python rar_folder_image_info.py --help
> ```

### Ejemplo práctico

```bash
python rar_folder_image_info.py "C:\MisVideos" --workers 4 --rar-path "C:\Program Files\WinRAR\rar.exe" --logfile
```

---

## ⚙️ Opciones principales

| Opción                | Descripción                                               |
|-----------------------|----------------------------------------------------------|
| `--workers N`         | Número de hilos para procesar videos en paralelo         |
| `--exts .mkv .mp4 ...`| Extensiones de video a buscar                            |
| `--skip-img`          | Omitir la generación de capturas                         |
| `--no-compress`       | Omitir la compresión RAR                                 |
| `--rar-store-only`    | Crear RAR sin compresión (por defecto)                   |
| `--rar-compress`      | Crear RAR con compresión normal                          |
| `--logfile [ruta]`    | Guardar log detallado en archivo                         |
| `-v`/`--verbose`      | Modo depuración (más detalles en consola)                |

---

## 🔄 Ejemplo de flujo de trabajo

1. Analiza todas las subcarpetas de un directorio base.
2. Para cada video:
   - Extrae metadatos y muestra información detallada.
   - Genera capturas de pantalla (si no se omite).
   - Comprime el video en un archivo RAR (si no se omite).
3. Muestra un resumen final con métricas globales.

---

## 👨‍💻 Créditos

Desarrollado por **Gilberto Nava Marcos**.

---

> 🎥 **Ideal para archivistas, uploaders, coleccionistas y cualquier usuario que requiera un flujo de trabajo profesional y automatizado para videos**
