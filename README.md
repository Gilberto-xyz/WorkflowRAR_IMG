# 🎬 rar_folder_image_info.py

## ✨ Descripción general
`rar_folder_image_info.py` es un script **avanzado y automatizado** para analizar, documentar y comprimir videos organizados en carpetas. Está pensado para flujos de trabajo exigentes: gestión de grandes colecciones, respaldos, archivado o distribución profesional de contenido multimedia.
Procesa **subcarpetas directas** del directorio base (no procesa videos que estén directamente en el directorio base).

---

## 🚀 Características principales
- 📂 **Procesamiento masivo por carpetas**: analiza subcarpetas y procesa cada video encontrado.
- 🏷️ **Metadatos avanzados**: resolución, duración, peso, pistas, idiomas, canales, formato, etc.
- 🖼️ **Capturas automáticas**: JPG o PNG en posiciones estratégicas con `ffmpeg`.
- 📦 **Compresión RAR profesional**: RAR por video, con o sin compresión y división automática.
- ⚡ **Procesamiento paralelo**: hilos configurables para acelerar el flujo.
- 📝 **Logs detallados**: consola con `rich` y log opcional a archivo.

---

## 🧩 Requisitos
- Python 3.10+
- `ffmpeg` en el PATH (solo para capturas)
- WinRAR (`rar.exe`) (solo si quieres compresión)
- Dependencias Python: `pymediainfo`, `rich`

Si `ffmpeg` no está disponible, las capturas se omiten automáticamente.  
Si `rar.exe` no está disponible, la compresión se desactiva automáticamente.

**Instalación de dependencias:**
```bash
pip install pymediainfo rich
```

---

## 🏁 Uso básico
```bash
python rar_folder_image_info.py [directorio_base] [opciones]
```

**Ejemplo rápido:**
```bash
python rar_folder_image_info.py "C:\MisVideos" --workers 4 --rar-path "C:\Program Files\WinRAR\rar.exe"
```

---

## ⚙️ Parámetros (principales y extra)
| Opción                  | Descripción |
|-------------------------|-------------|
| `directorio_base`       | Carpeta base con subcarpetas directas a procesar (posicional). |
| `--workers N`           | Hilos de procesamiento por carpeta. |
| `--exts .mkv .mp4 ...`  | Extensiones de video a buscar. |
| `--skip-img`            | Omitir capturas. |
| `--num-capturas N`      | Número de capturas por video (reemplaza el valor por defecto). |
| `--img-format`          | Formato de capturas: `jpg` o `png` (alta calidad). |
| `--no-compress`         | Omitir compresión RAR. |
| `--rar-path`            | Ruta completa a `rar.exe`. |
| `--rar-store-only`      | RAR sin compresión (por defecto). |
| `--rar-compress`        | RAR con compresión normal. |
| `--rar-password`        | Contraseña para cifrar RAR (o `RAR_PASSWORD` en entorno). |
| `--logfile [ruta]`      | Guardar log detallado en archivo. |
| `-v` / `--verbose`      | Modo depuración. |

> Tip: usa `--help` para ver todas las opciones y valores por defecto.

---

## 🧭 Flujo real del script (resumen)
- El directorio base solo se usa para **listar subcarpetas directas**.
- Dentro de cada subcarpeta, busca videos **recursivamente** (excluye `capturas`).
- Genera metadatos y, si aplica, crea `capturas\` y `RARs\` dentro de cada subcarpeta.
- Cada video genera su **RAR individual** con nombre ofuscado + sufijo `[GDriveLatinoHD]`.
- La contraseña RAR es `--rar-password`, o `RAR_PASSWORD`, o el valor por defecto `GDriveLatinoHD`.
- Si un video supera 20 GB, el RAR se **divide en partes de 15 GB**.

---

## 🧠 Notas importantes sobre capturas
- Si hay **1 solo video** en la carpeta: **100 capturas** entre **2% y 98%**.   
- Si hay **varios videos**: **50 capturas** entre **8% y 96%**.
- `--num-capturas` reemplaza esos valores y mantiene el rango según el caso.
- `--img-format png` genera PNG sin perdida (archivos mas pesados).
- `--num-capturas 0` permite desactivar capturas sin usar `--skip-img`.

---

## ✅ Ejemplos utiles
**1) Solo analisis y capturas (sin RAR):**
```bash
python rar_folder_image_info.py "C:\MisVideos" --no-compress
```

**2) Capturas PNG (alta calidad) con 20 imagenes:**
```bash
python rar_folder_image_info.py "C:\MisVideos" --num-capturas 20 --img-format png
```

**3) Solo compresion (sin capturas):**
```bash
python rar_folder_image_info.py "C:\MisVideos" --skip-img
```

**4) Filtrar extensiones y guardar log:**
```bash
python rar_folder_image_info.py "C:\MisVideos" --exts .mkv .mp4 --logfile
```

---

## 👨‍💻 Créditos
Desarrollado por **Gilberto Nava Marcos**.

---

> 🎥 Ideal para archivistas, uploaders, coleccionistas y cualquier usuario que requiera un flujo de trabajo profesional y automatizado para videos.
