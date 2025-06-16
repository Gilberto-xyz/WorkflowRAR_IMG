#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow avanzado v3: capturas JPG, info detallada y compresión RAR por carpeta.
Combina mejoras de robustez y configuración.
"""

import os
import re
import shutil
import sys
import argparse # <<< argparse
import subprocess
import logging
import unicodedata # <<< unicodedata
from pathlib import Path
from datetime import datetime
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# --- Third-party Libraries ---
try:
    from pymediainfo import MediaInfo
except ImportError:
    print("Error: Necesitas instalar 'pymediainfo'. Ejecuta: pip install pymediainfo")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (Progress, BarColumn, TextColumn, TimeRemainingColumn,
                               SpinnerColumn, TaskProgressColumn)
    from rich.logging import RichHandler
    from rich.theme import Theme # <<< rich.theme
    from rich.traceback import install as install_rich_traceback
except ImportError:
    print("Error: Necesitas instalar la biblioteca 'rich'. Ejecuta: pip install rich")
    sys.exit(1)

# --- CONFIGURACIÓN GLOBAL ---
# (Se pueden sobrescribir con argparse)
DEFAULT_DIRECTORIO  = r"C:\Users\Default\Videos"
DEFAULT_EXTS_VIDEO  = ('.mkv', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mpeg', '.mpg')
DEFAULT_PCTS_UNICOS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96]  # 24 capturas + 1 extra = 25
DEFAULT_PCTS_MULTI  = [20, 40, 60, 80]  # Para múltiples videos, mantener pocas capturas
DEFAULT_RAR_EXE     = r"C:\Program Files\WinRAR\rar.exe" # Ruta por defecto a rar.exe
DEFAULT_LOG_LEVEL   = logging.WARNING # Nivel de log por defecto
DEFAULT_WORKERS     = max(1, os.cpu_count() // 2) # Al menos 1 worker

NOMBRE_CARPETA_CAPTURAS = "capturas" # Nombre de la carpeta donde se guardarán las capturas
RAR_FILENAME_SUFFIX     = "[GDriveLatinoHD]" # Sufijo para nombres de RARs
RAR_SPLIT_THRESHOLD_GB  = 20 # Umbral para dividir RARs
RAR_SPLIT_SIZE_GB       = 15 # Tamaño de cada parte RAR. # Si el archivo es mayor a RAR_SPLIT_THRESHOLD_GB, se divide en partes de este tamaño (RAR_SPLIT_SIZE_GB)
LOG_FILENAME            = "process_log.log" # Nombre de log por defecto

# --- Inicialización Global ---
# Instalar traceback mejorado de Rich
install_rich_traceback(show_locals=False)

# Configurar tema de Rich
console = Console(theme=Theme({
    "info" : "dim cyan",
    "good" : "bold green",
    "warn" : "bold yellow",
    "fail" : "bold red",
    "head" : "bold magenta",
    "path" : "dim blue",
    "detail": "italic"
}), record=True)

# Configuración de Logging (se ajustará en main con argparse)
log_handler = RichHandler(console=console, rich_tracebacks=True, show_path=False, level=DEFAULT_LOG_LEVEL)
logging.basicConfig(
    level=DEFAULT_LOG_LEVEL,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[log_handler] # Inicialmente solo a consola, se añade FileHandler en main si se pide
)
log = logging.getLogger(__name__) # Usar logger específico

# --- Decorador de Seguridad ---
def safe_run(fn):
    """Decorador: captura excepciones, las loggea y devuelve un valor por defecto (usualmente None o False)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            # Usar el logger configurado
            log.error(f"Error en '{fn.__name__}': {e}", exc_info=True) # Log con traceback
            # No imprimir directamente a consola aquí, el logger ya lo hace si el nivel es apropiado
            # Devolver un valor que indique fallo
            if "comprimir" in fn.__name__: return False
            if "capturas" in fn.__name__: return []
            if "info_media" in fn.__name__: return {"error": f"Excepción en {fn.__name__}: {e}"}
            return None # Valor por defecto genérico
    return wrapper

# --- Utilidades Varias ---
def clean_name(text: str) -> str:
    """Normaliza a ASCII, quita etiquetas [], (), {}, emojis, múltiples espacios."""
    if not isinstance(text, str):
        return ""
    # Quitar etiquetas comunes primero
    text = re.sub(r'\[[^\]]*?\]|\([^\)]*?\)|\{[^\}]*?\}', ' ', text)
    try:
        # Normalizar a ASCII decomposed form (NFKD), ignorar lo no convertible
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    except Exception:
        log.warning(f"Fallo en normalización unicodedata para: '{text[:50]}...'")
        # Fallback: quitar no-ascii básico
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Quitar caracteres no alfanuméricos (permitiendo ., -, _)
    text = re.sub(r'[^A-Za-z0-9._-]+', ' ', text)
    # Reemplazar puntos/underscores rodeados de espacios o al final (simplificado)
    text = text.replace('.', ' ').replace('_', ' ')
    # Consolidar espacios y limpiar bordes
    return re.sub(r'\s{2,}', ' ', text).strip()

def formatear_peso(peso_bytes: int | float) -> str:
    """Formatea bytes a KB, MB, GB o TB."""
    try:
        peso_bytes = float(peso_bytes)
        if peso_bytes < 1024: return f"{peso_bytes:.0f} B"
        elif peso_bytes < 1024**2: return f"{peso_bytes / 1024:.2f} KB"
        elif peso_bytes < 1024**3: return f"{peso_bytes / (1024**2):.2f} MB"
        elif peso_bytes < 1024**4: return f"{peso_bytes / (1024**3):.2f} GB"
        else: return f"{peso_bytes / (1024**4):.2f} TB"
    except (ValueError, TypeError):
        return "Inválido"

# --- MediaInfo Helpers ---
LANGS = {
    "es": "Español", "en": "Inglés", "ja": "Japonés", "fr": "Francés",
    "de": "Alemán", "it": "Italiano", "ru": "Ruso", "pt": "Portugués",
    "es-419": "Español Lat.", "pt-br": "Portugués BR", "zh": "Chino",
    "ko": "Coreano", "und": "Indet.", "zxx": "N/A", "mul": "Multi"
}

def normalize_lang(code: str | None) -> str:
    if not code: return "Indet."
    code_low = code.lower()
    if code_low in LANGS: return LANGS[code_low]
    base_code = code_low.split('-')[0]
    if base_code in LANGS: return LANGS[base_code] # Fallback
    return code.capitalize() # Devolver original capitalizado si no se conoce

def normalize_channels(ch_count: int | None) -> str:
    if not isinstance(ch_count, int): return "?"
    return {1: "1.0", 2: "2.0", 6: "5.1", 7: "6.1", 8: "7.1"}.get(ch_count, str(ch_count))

# Aplicar decorador a la función principal de parseo
@safe_run
def obtener_info_media(ruta_archivo: Path) -> dict:
    """Extrae información de audio, subtítulos, duración y resolución."""
    resultado = {
        "audio": [], "subtitulos": [], "duracion_s": None,
        "resolucion": "Desconocida", "formato_video": "Desconocido",
        "error": None
    }
    media_info = MediaInfo.parse(str(ruta_archivo)) # pymediainfo necesita str

    for track in media_info.tracks:
        track_type = getattr(track, 'track_type', None)

        if track_type == "General":
            if getattr(track, "duration", None):
                resultado["duracion_s"] = float(track.duration) / 1000.0
            fmt = getattr(track, "format", None)
            # Evitar formatos contenedores genéricos si hay pista de video específica
            if fmt and fmt not in ("BDAV", "Matroska", "MPEG-4") and resultado["formato_video"] == "Desconocido":
                resultado["formato_video"] = fmt

        elif track_type == "Video":
            w = getattr(track, 'width', None)
            h = getattr(track, 'height', None)
            if w and h: resultado["resolucion"] = f"{w}x{h}"
            fmt = getattr(track, "format", None)
            if fmt: resultado["formato_video"] = fmt

        elif track_type == "Audio":
            fmt = getattr(track, "format", "Desc.").upper()
            lang_code = getattr(track, 'language', None)
            ch_str = getattr(track, 'channel_s', None) or getattr(track, 'channels', None)
            try: ch_count = int(ch_str) if ch_str else None
            except ValueError: ch_count = None

            default = "[D]" if getattr(track, 'default', 'No') == 'Yes' else ""
            forced = "[F]" if getattr(track, 'forced', 'No') == 'Yes' else ""
            flags = f"{default}{forced}".strip()

            pista_info = {
                "lang": normalize_lang(lang_code), "format": fmt,
                "channels": normalize_channels(ch_count), "flags": flags
            }
            # Evitar duplicados exactos
            if pista_info not in resultado["audio"]:
                resultado["audio"].append(pista_info)

        elif track_type == "Text":
            fmt = getattr(track, "format", "Desc.").upper()
            # Cambiar UTF-8 a SRT para subtítulos
            if fmt == "UTF-8":
                fmt = "SRT"
            lang_code = getattr(track, 'language', None)
            default = "[D]" if getattr(track, 'default', 'No') == 'Yes' else ""
            forced = "[F]" if getattr(track, 'forced', 'No') == 'Yes' else ""
            flags = f"{default}{forced}".strip()
        
            sub_info = {
                "lang": normalize_lang(lang_code), "format": fmt, "flags": flags
            }
            if sub_info not in resultado["subtitulos"]:
                 resultado["subtitulos"].append(sub_info)

    # Ordenar para consistencia
    resultado["audio"].sort(key=lambda x: (x['lang'], x['format'], x['channels']))
    resultado["subtitulos"].sort(key=lambda x: (x['lang'], x['format']))
    return resultado

def armar_cadena_agrupada(pistas: list[dict], tipo='audio') -> str:
    """Agrupa pistas por idioma y formato/canales, usando conteo simple.
    Para subtítulos, muestra 'Forzado' si corresponde.
    """
    if not pistas:
        return "[detail]Ninguna[/detail]"

    key_func = lambda p: (
        p['lang'],
        p['format'],
        p.get('channels', '') if tipo == 'audio' else '',
        '[F]' in p.get('flags', '') if tipo == 'subs' else False  # True si es forzado
    )
    contador = defaultdict(int)
    for p in pistas:
        contador[key_func(p)] += 1

    if not contador:
        return "[detail]Ninguna[/detail]"

    partes = []
    sorted_keys = sorted(contador.keys(), key=lambda k: (
        0 if k[0] == 'Español' else (1 if k[0] == 'Inglés' else 2),
        k[0], k[1], k[2], k[3]
    ))

    for key in sorted_keys:
        lang, fmt, channels, forzado = key
        count = contador[key]
        count_str = f" [detail]x{count}[/detail]" if count > 1 else ""
        channel_str = f" {channels}" if channels else ""
        forzado_str = " [bold yellow]Forzado[/bold yellow]" if tipo == 'subs' and forzado else ""
        partes.append(f"[info]{lang}[/info] [bold]{fmt}[/bold]{channel_str}{forzado_str}{count_str}")

    return " - ".join(partes)

# --- ffmpeg Capturas (JPEG) ---
@safe_run
def generar_capturas(
    ruta_video: Path,
    nombre_base_capturas: str,
    porcentajes: list[int],
    carpeta_destino: Path,
    indice_archivo: int,
    duracion_s: float,
    progress: Progress,
    task_id,
) -> list[Path]:
    """Genera capturas de pantalla en formato JPG usando ffmpeg."""
    capturas_generadas = []
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        log.error("'ffmpeg' no encontrado en el PATH. No se pueden generar capturas.")
        return capturas_generadas  # Devuelve lista vacía en error

    # Duración ya obtenida previamente

    if not duracion_s or duracion_s <= 0:
        log.warning(
            f"No se generarán capturas para '{ruta_video.name}': Duración inválida o no encontrada."
        )
        progress.update(task_id, description=f"[cyan]{ruta_video.name}[/] [warn]- Sin duración[/warn]")
        return capturas_generadas

    os.makedirs(carpeta_destino, exist_ok=True)
    log.info(f"Generando {len(porcentajes)} capturas JPG para '{ruta_video.name}'")
    progress.update(task_id, total=len(porcentajes), description=f"[cyan]{ruta_video.name}[/] [yellow]- Capturas JPG...[/yellow]")

    for idx, pct in enumerate(porcentajes, 1):
        tiempo = max(0.1, duracion_s * (pct / 100.0))
        nombre_captura = f"{nombre_base_capturas}[{indice_archivo}]_Captura[{idx:02d}].jpg"  # Cambia extensión a .jpg
        ruta_captura = carpeta_destino / nombre_captura
    
        cmd = [
            ffmpeg_path, "-hide_banner", "-loglevel", "error",
            "-ss", str(tiempo), "-i", str(ruta_video),
            "-vframes", "1", "-q:v", "2", "-y", str(ruta_captura),  # -q:v 2 ≈ calidad 85-90
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=45, encoding='utf-8', errors='ignore')
            if ruta_captura.exists():
                capturas_generadas.append(ruta_captura)
                log.debug(f"Captura JPG generada: {nombre_captura}")
            else:
                log.warning(f"ffmpeg OK, pero falta archivo captura JPG: {nombre_captura}")
            progress.update(task_id, advance=1, description=f"[cyan]{ruta_video.name}[/] [yellow]- Captura {idx}/{len(porcentajes)}[/yellow]")
        except subprocess.CalledProcessError as e:
            log.error(f"Error ffmpeg (captura JPG {idx}@{pct}%): {e.stderr.strip()}")
        except subprocess.TimeoutExpired:
            log.warning(f"Timeout ffmpeg (captura JPG {idx}@{pct}%)")
        except Exception as e_inner:
            log.error(f"Error inesperado interno en ffmpeg (captura JPG {idx}): {e_inner}", exc_info=True)


    log.info(
        f"Generadas {len(capturas_generadas)}/{len(porcentajes)} capturas JPG para '{ruta_video.name}'."
    )
    progress.update(task_id, description=f"[cyan]{ruta_video.name}[/] [green]- Capturas OK[/green]")
    return capturas_generadas # @safe_run devolverá [] si hubo excepción grave

# --- Compresión RAR (Individual por archivo) ---
@safe_run
def comprimir_carpeta_rar(carpeta_path: Path, video_files_to_compress: list[Path],
                           rar_exe_path: str, rar_store_only: bool, progress: Progress, task_id) -> bool:
    """Comprime cada video de la carpeta en su propio RAR, usando el nombre limpio del archivo y guardando en 'RARs'."""
    if not video_files_to_compress:
        log.info(f"No hay videos para comprimir en '{carpeta_path.name}'")
        progress.update(task_id, completed=1, total=1, description=f"[cyan]{carpeta_path.name}[/] [info]- Sin compresión (no videos)[/info]")
        return True

    if not rar_exe_path or not Path(rar_exe_path).exists():
        log.error(f"Compresión fallida: 'rar.exe' no encontrado en '{rar_exe_path}'.")
        progress.update(task_id, description=f"[cyan]{carpeta_path.name}[/] [fail]- RAR no encontrado[/fail]")
        return False

    # Crear subcarpeta RARs si no existe
    rars_dir = carpeta_path / "RARs"
    rars_dir.mkdir(exist_ok=True)

    total_files = len(video_files_to_compress)
    total_size_bytes = sum(f.stat().st_size for f in video_files_to_compress if f.exists())
    
    # Determinar el modo de compresión
    compression_mode = "sin compresión (solo almacenar)" if rar_store_only else "con compresión"
    log.info(f"Iniciando creación RAR {compression_mode} para carpeta '{carpeta_path.name}'. Videos: {total_files}, Tamaño total: {formatear_peso(total_size_bytes)}")
    progress.update(task_id, total=total_files, description=f"[cyan]{carpeta_path.name}[/] [magenta]- Iniciando RAR {compression_mode}...[/magenta]")

    ok = True
    for idx, video_file in enumerate(video_files_to_compress, 1):
        nombre_limpio = clean_name(video_file.stem)
        rar_filename = f"{nombre_limpio}{RAR_FILENAME_SUFFIX}.rar"
        rar_filepath = rars_dir / rar_filename

        # Actualiza la descripción para mostrar qué archivo se está procesando AHORA
        progress.update(task_id, description=f"[magenta]RAR {idx}/{total_files}:[/] [cyan]{video_file.name}[/]")

        size_gb = video_file.stat().st_size / (1024**3)
        cmd = [rar_exe_path, "a"]
        
        # Configurar método de compresión
        if rar_store_only:
            cmd.append("-m0")  # Sin compresión (solo almacenar)
        else:
            cmd.append("-m3")  # Compresión normal
        
        split_msg = ""
        if size_gb > RAR_SPLIT_THRESHOLD_GB:
            split_size_param = f"-v{RAR_SPLIT_SIZE_GB}g"
            cmd.append(split_size_param)
            split_msg = f" (Dividiendo en {RAR_SPLIT_SIZE_GB} GB)"

        cmd.extend([
            "-ep1", "-o+", "-idq",
            str(rar_filepath.resolve()),
            str(video_file.resolve())
        ])

        try:
            action_verb = "Almacenando" if rar_store_only else "Comprimiendo"
            log.info(f"{action_verb} [{idx}/{total_files}] '{video_file.name}' -> '{rar_filename}'{split_msg}")
            log.debug(f"Ejecutando RAR: {' '.join(cmd)}")
            timeout_compresion = 6 * 60 * 60  # 6 horas
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=timeout_compresion)
            log.info(f"Creación RAR [{idx}/{total_files}] exitosa: '{rar_filename}'")
            progress.update(task_id, advance=1)
        except subprocess.CalledProcessError as e:
            error_message = e.stderr or e.stdout or "Sin salida de error específica."
            log.error(f"Error durante la creación RAR para '{video_file.name}' (código {e.returncode}):\n{error_message.strip()}")
            progress.update(task_id, advance=1, description=f"[fail]Error RAR {idx}/{total_files}:[/] [cyan]{video_file.name}[/]")
            ok = False
        except subprocess.TimeoutExpired:
            log.error(f"Timeout durante creación RAR para '{video_file.name}' (límite {timeout_compresion / 3600:.1f}h).")
            progress.update(task_id, advance=1, description=f"[fail]Timeout RAR {idx}/{total_files}:[/] [cyan]{video_file.name}[/]")
            ok = False
        except Exception as e_inner:
            log.error(f"Error inesperado interno en creación RAR para '{video_file.name}': {e_inner}", exc_info=True)
            progress.update(task_id, advance=1, description=f"[fail]Error interno RAR {idx}/{total_files}:[/] [cyan]{video_file.name}[/]")
            ok = False

    # Actualización final de la tarea
    action_name = "almacenado" if rar_store_only else "compresión"
    final_description = f"[cyan]{carpeta_path.name}[/] [green]- {action_name.capitalize()} RAR finalizado[/]" if ok else f"[cyan]{carpeta_path.name}[/] [fail]- {action_name.capitalize()} RAR con errores[/]"
    progress.update(task_id, description=final_description)

    return ok

# --- Procesamiento Individual de Video (Dentro de Carpeta) ---
# No necesita @safe_run porque lo llamaremos desde un futuro que captura errores
def procesar_un_video(ruta_video: Path, indice_archivo: int, total_archivos: int,
                       porcentajes_capturas: list[int], args: argparse.Namespace,
                       progress: Progress, task_id_capturas) -> dict:
    """Procesa un único video: extrae info y opcionalmente genera capturas."""
    resultado_video = {
        "ruta": ruta_video,
        "nombre_original": ruta_video.name,
        "nombre_limpio": None,
        "info_media": {"error": "No procesado"}, # Default error
        "peso_str": "Desconocido",
        "peso_bytes": 0,
        "capturas_generadas": [], # Lista de Paths de capturas
        "error": None # Error específico de esta función
    }
    try:
        log.info(f"Procesando video [{indice_archivo}/{total_archivos}]: {ruta_video.name}")

        # 1. Limpiar nombre base (para capturas y referencia)
        # Usamos la nueva función clean_name
        resultado_video["nombre_limpio"] = clean_name(ruta_video.stem) or ruta_video.stem # Fallback

        # 2. Obtener información de medios (ya está decorada con @safe_run)
        resultado_video["info_media"] = obtener_info_media(ruta_video)
        # Verificar si la info falló
        if not isinstance(resultado_video["info_media"], dict) or resultado_video["info_media"].get("error"):
            log.warning(f"Fallo al obtener info media para {ruta_video.name}. Error: {resultado_video['info_media'].get('error', 'Desconocido')}")
            # No marcamos error fatal aquí, aún podemos intentar obtener peso
            # El error ya está en info_media["error"]

        # 3. Obtener peso
        try:
            resultado_video["peso_bytes"] = ruta_video.stat().st_size
            resultado_video["peso_str"] = formatear_peso(resultado_video["peso_bytes"])
        except OSError as e:
            log.warning(f"No se pudo obtener el peso de '{ruta_video.name}': {e}")
            resultado_video["error"] = resultado_video["error"] or f"Error obteniendo peso: {e}" # Añadir error si no había uno

        # 4. Generar Capturas (si no se omite y no hubo error fatal antes)
        if not args.skip_img:
            if resultado_video["info_media"] and not resultado_video["info_media"].get("error"):
                duracion_s = resultado_video["info_media"].get("duracion_s")
                # generar_capturas está decorado con @safe_run
                resultado_video["capturas_generadas"] = generar_capturas(
                    ruta_video,
                    resultado_video["nombre_limpio"],
                    porcentajes_capturas,
                    ruta_video.parent / NOMBRE_CARPETA_CAPTURAS,
                    indice_archivo,
                    duracion_s,
                    progress,
                    task_id_capturas,
                )
                # Si generar_capturas falla, devolverá [] y loggeará el error.
            else:
                 log.warning(f"Omitiendo capturas para '{ruta_video.name}' debido a error previo o falta de info.")
                 progress.update(task_id_capturas, description=f"[cyan]{ruta_video.name}[/] [warn]- Sin capturas (error previo)[/warn]")
        else:
             # Si se omiten las capturas, actualizar la tarea para que no quede "pendiente"
             progress.update(task_id_capturas, completed=1, total=1, description=f"[cyan]{ruta_video.name}[/] [info]- Capturas omitidas[/info]")


        log.info(f"Finalizado procesamiento básico para: {ruta_video.name}")
        return resultado_video

    except Exception as e:
        # Captura genérica por si algo más falla aquí
        log.error(f"Fallo crítico procesando '{ruta_video.name}': {e}", exc_info=True)
        resultado_video["error"] = f"Error crítico en procesar_un_video: {e}"
        # Asegurar que la tarea de progreso se marque como fallida
        try:
            progress.update(task_id_capturas, description=f"[cyan]{ruta_video.name}[/] [fail]- Error Crítico[/fail]")
        except Exception:
            pass # Ignorar si falla la actualización de progreso
        return resultado_video


# --- Procesamiento por Carpeta ---
def procesar_carpeta(carpeta_path: Path, args: argparse.Namespace, progress: Progress):
    """Procesa una carpeta: busca videos, los procesa en paralelo, muestra info y comprime."""
    nombre_carpeta = carpeta_path.name
    log.info(f"Iniciando procesamiento de carpeta: {nombre_carpeta}")
    console.print(Panel(f"[head]Carpeta:[/head] [path]{nombre_carpeta}[/path]", expand=False, border_style="blue"))

    # Usar Path.rglob para buscar videos
    try:
        archivos_video = sorted([
            p for p in carpeta_path.rglob('*')
            if p.is_file() and p.suffix.lower() in args.exts
               and NOMBRE_CARPETA_CAPTURAS not in p.parts # Evitar buscar in carpeta de capturas
        ])
    except Exception as e:
         log.error(f"Error buscando videos en '{carpeta_path}': {e}", exc_info=True)
         console.print(f"  [fail]Error buscando videos en esta carpeta.[/fail]")
         return 0, 0

    if not archivos_video:
        log.warning(f"No se encontraron videos en la carpeta '{nombre_carpeta}'.")
        console.print("  [warn]No se encontraron archivos de video compatibles.[/warn]")
        return 0, 0

    num_videos = len(archivos_video)
    log.info(f"Encontrados {num_videos} archivo(s) de video en '{nombre_carpeta}'.")
    console.print(f"  Encontrados {num_videos} video(s) a procesar.")

    porcentajes = DEFAULT_PCTS_UNICOS if num_videos == 1 else DEFAULT_PCTS_MULTI

    resultados_videos = []
    videos_para_comprimir = []
    peso_total_videos_carpeta = 0
    archivos_procesados_ok = 0

    # Tarea principal para la carpeta: N videos + 1 paso para compresión (si aplica)
    num_pasos_carpeta = num_videos + (1 if args.compress else 0)
    task_folder = progress.add_task(f"[blue]Carpeta {nombre_carpeta}[/]", total=num_pasos_carpeta)

    # --- Procesamiento de videos en paralelo ---
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures_to_tasks = {}
        for idx, ruta_video in enumerate(archivos_video, 1):
            # Tarea hija para capturas (si no se omite)
            task_capturas = progress.add_task(f"[cyan]{ruta_video.name}[/] [dim]- Pendiente[/]", total=1, visible=not args.skip_img, parent=task_folder)
            future = executor.submit(procesar_un_video, ruta_video, idx, num_videos,
                                     porcentajes, args, progress, task_capturas)
            futures_to_tasks[future] = task_capturas

        for future in as_completed(futures_to_tasks):
            task_id_capturas = futures_to_tasks[future]
            try:
                resultado = future.result()
                resultados_videos.append(resultado)

                if resultado and not resultado.get("error"):
                    archivos_procesados_ok += 1
                    peso_total_videos_carpeta += resultado.get("peso_bytes", 0)
                    videos_para_comprimir.append(resultado["ruta"])
                    if not args.skip_img:
                        # Asegura que la tarea de captura esté completa
                         progress.update(task_id_capturas, completed=progress.tasks[task_id_capturas].total, description=f"[cyan]{resultado.get('nombre_original', '??')}[/] [green]- Capturas OK[/green]") # Actualiza descripción final OK
                else:
                    error_msg = resultado.get('error', 'Desconocido') if isinstance(resultado, dict) else 'Error crítico futuro'
                    log.warning(f"Video con error: {resultado.get('nombre_original', 'N/A')} - {error_msg}")
                    # Marca la tarea de captura como fallida si existe
                    if not args.skip_img:
                        progress.update(task_id_capturas, description=f"[cyan]{resultado.get('nombre_original', '??')}[/] [fail]- Error Proc.[/fail]")

            except Exception as exc:
                log.critical(f"Error crítico obteniendo resultado del futuro: {exc}", exc_info=True)
                if not args.skip_img:
                    try: progress.update(task_id_capturas, description="[fail]- Error Futuro[/fail]")
                    except Exception: pass
            finally:
                 # Avanza la tarea principal de la carpeta por cada video procesado (fase 1)
                 progress.update(task_folder, advance=1)
    # --- Fin procesamiento paralelo ---

    # --- Mostrar resultados individuales ---
    console.print(f"\n  [head]Resultados del procesamiento ({archivos_procesados_ok}/{num_videos} OK):[/head]")
    resultados_videos.sort(key=lambda x: x.get('ruta', Path('.'))) # Ordenar por ruta
    for res in resultados_videos:
        if not isinstance(res, dict): # Si el futuro falló gravemente
             console.print(Panel(f"[fail]Fallo crítico al procesar un archivo no identificado.[/fail]", border_style="red"))
             continue

        if res.get("error"):
             console.print(Panel(f"[fail]Archivo:[/fail] {res.get('nombre_original', 'N/A')}\n[fail]Error:[/fail] {res['error']}",
                               title="Error Procesando", border_style="red", expand=False))
             continue

        # Info del video procesado OK
        info = res.get('info_media', {})
        if not isinstance(info, dict): info = {} # Seguridad extra

        duracion_s = info.get("duracion_s")
        duracion_str = f"{int(duracion_s // 3600):02d}:{int((duracion_s % 3600) // 60):02d}:{int(duracion_s % 60):02d}" if duracion_s else "N/A"
        audio_str = armar_cadena_agrupada(info.get('audio', []), 'audio')
        subtitulos_str = armar_cadena_agrupada(info.get('subtitulos', []), 'subs')
        capturas_paths = res.get('capturas_generadas', [])
        capturas_nombres = [c.name for c in capturas_paths]
        if capturas_nombres:
            if len(capturas_nombres) == 1:
                capturas_str = f"[good]JPG[/good] (1): {capturas_nombres[0]}"
            else:
                capturas_str = f"[good]JPG[/good] ({len(capturas_nombres)})"
        else:
            capturas_str = "[detail]Ninguna[/detail]"
        if args.skip_img:
            capturas_str = "[info]Omitidas[/info]"

        info_panel = (
            f"[info]Limpio:[/info]  [detail]{res.get('nombre_limpio', 'N/A')}[/detail]\n"
            f"[info]Resolucion.:[/info]  {info.get('resolucion', 'N/A')} ({info.get('formato_video', 'N/A')})\n"
            f"[info]Duracion.:[/info]  {duracion_str}\n"
            f"[info]Peso:[/info]    {res.get('peso_str', 'N/A')}\n"
            f"[info]Audio:[/info]   {audio_str}\n"
            f"[info]Subtitulos:[/info]   {subtitulos_str}\n"
            f"[info]Capturas:[/info]    {capturas_str}"
        )
        console.print(Panel(info_panel, title=f"[cyan]{res.get('nombre_original', 'N/A')}[/cyan]", border_style="cyan", expand=False))
    # --- Fin mostrar resultados ---

    # --- Compresión de la carpeta ---
    if args.compress and videos_para_comprimir: # Solo comprime si hay videos OK
        # Añade una tarea hija específica para la compresión de esta carpeta
        task_compress = progress.add_task(f"[magenta]Procesando RAR {nombre_carpeta}...[/magenta]", total=1, parent=task_folder)
        success = comprimir_carpeta_rar(carpeta_path, videos_para_comprimir, args.rar_path, args.rar_store_only, progress, task_compress)
        if not success:
            log.error(f"Fallo en la creación RAR de la carpeta {nombre_carpeta}")
        # Avanza la tarea principal de la carpeta por la etapa de compresión (fase 2)
        progress.update(task_folder, advance=1)
    elif args.compress and not videos_para_comprimir:
        log.warning(f"Omitiendo creación RAR para '{nombre_carpeta}' porque no hay videos procesados correctamente.")
        progress.update(task_folder, advance=1)
        task_compress_skipped = progress.add_task(f"[magenta]RAR {nombre_carpeta}[/]", total=1, parent=task_folder, completed=1, description="[info]- Omitida (sin videos OK)[/info]")


    # --- Fin compresión ---

    log.info(f"Finalizado procesamiento de carpeta: {nombre_carpeta}. Videos OK: {archivos_procesados_ok}/{num_videos}")
    console.print(f"  [info]Fin procesamiento carpeta: {nombre_carpeta}[/info]\n")
    # Asegurarse de que la tarea principal de la carpeta esté completa
    progress.update(task_folder, completed=progress.tasks[task_folder].total)
    return archivos_procesados_ok, peso_total_videos_carpeta


# ========= Main =========

def main():
    # --- Configuración de Argparse ---
    ap = argparse.ArgumentParser(description="Workflow de análisis y compresión de videos con Rich Progress.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("basedir", nargs='?', default=DEFAULT_DIRECTORIO,
                    help="Directorio base que contiene las subcarpetas a procesar.")
    ap.add_argument("--rar-path", default=DEFAULT_RAR_EXE,
                    help="Ruta completa al ejecutable 'rar.exe'.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="Número de hilos para procesar videos en paralelo por carpeta.")
    ap.add_argument("--exts", nargs='+', default=list(DEFAULT_EXTS_VIDEO),
                    help="Extensiones de video a buscar.")
    ap.add_argument("--skip-img", action="store_true",
                    help="Omitir la generación de capturas de pantalla.")
    ap.add_argument("--no-compress", dest='compress', action="store_false",
                    help="Omitir la compresión RAR.")
    ap.add_argument("--rar-store-only", action="store_true", default=True,
                help="Crear RAR sin compresión (solo almacenar archivos). ACTIVADO POR DEFECTO.")
    ap.add_argument("--rar-compress", action="store_false", dest="rar_store_only",
                help="Crear RAR con compresión normal (anula --rar-store-only).")
    ap.add_argument("--logfile", nargs='?', const=LOG_FILENAME, default=None,
                    help=f"Guardar log detallado en un archivo (por defecto: {LOG_FILENAME} si se especifica la opción sin ruta).")
    ap.add_argument("-v", "--verbose", action="store_const", const=logging.DEBUG, default=DEFAULT_LOG_LEVEL,
                   help="Mostrar mensajes de depuración (DEBUG).")
    args = ap.parse_args()

    # --- Configurar Logging basado en Args ---
    log_level = args.verbose
    log_handler.setLevel(log_level) # Actualizar nivel del handler de consola
    log.setLevel(log_level)       # Actualizar nivel del logger raíz

    if args.logfile:
        log_file_path = Path(args.logfile).resolve()
        try:
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"))
            file_handler.setLevel(logging.DEBUG) # Guardar todo en el archivo
            logging.getLogger().addHandler(file_handler) # Añadir handler al logger raíz
            log.info(f"Logging configurado a nivel {logging.getLevelName(log_level)}, guardando log DEBUG en: {log_file_path}")
        except Exception as e:
            log.error(f"No se pudo configurar el log a archivo '{log_file_path}': {e}")
            console.print(f"[fail]Error configurando log a archivo: {e}[/fail]")
    else:
         log.info(f"Logging configurado a nivel {logging.getLevelName(log_level)} (solo consola).")

    # --- Validaciones Iniciales ---
    base_path = Path(args.basedir).resolve()
    if not base_path.is_dir():
        log.critical(f"El directorio base especificado no existe o no es un directorio: {base_path}")
        console.print(f"[fail]Error:[/fail] Directorio base no válido: [path]{base_path}[/path]")
        sys.exit(1)

    if not shutil.which('ffmpeg') and not args.skip_img:
        log.warning("'ffmpeg' no encontrado en el PATH. Se omitirán las capturas.")
        console.print("[warn]Advertencia:[/warn] 'ffmpeg' no encontrado. Se omitirán las capturas (--skip-img implícito).")
        args.skip_img = True # Forzar omisión

    rar_path_obj = Path(args.rar_path)
    if args.compress and not rar_path_obj.exists():
        log.warning(f"'rar.exe' no encontrado en '{args.rar_path}'. Se desactivará la compresión.")
        console.print(f"[warn]Advertencia:[/warn] 'rar.exe' no encontrado en [path]{args.rar_path}[/path]. Se desactivará la compresión (--no-compress implícito).")
        args.compress = False # Forzar desactivación

    console.print(Panel(f"[head]== INICIO DEL PROCESO ==[/head]\n"
                      f"Directorio Base: [path]{base_path}[/path]\n"
                      f"Workers: {args.workers} | Capturas: {'Sí' if not args.skip_img else 'No'} | Compresión: {'Sí' if args.compress else 'No'}",
                      expand=False, border_style="magenta"))
    start_time = datetime.now()

    # --- Búsqueda de Subcarpetas ---
    try:
        # Listar solo directorios directamente dentro de base_path
        subcarpetas = sorted([d for d in base_path.iterdir() if d.is_dir()])
    except Exception as e:
        log.critical(f"Error listando subcarpetas en '{base_path}': {e}", exc_info=True)
        console.print(f"[fail]Error fatal:[/fail] No se pudo listar el contenido de '{base_path}'.")
        sys.exit(1)

    if not subcarpetas:
        log.warning(f"No se encontraron subcarpetas directas en '{base_path}'.")
        console.print(f"[warn]Advertencia:[/warn] No se encontraron subcarpetas para procesar en [path]{base_path}[/path].")
        sys.exit(0)

    log.info(f"Encontradas {len(subcarpetas)} subcarpetas para analizar.")

    # --- Procesamiento Principal con Rich Progress ---
    stats = {"total_carpetas": 0, "carpetas_con_videos": 0, "videos_procesados_ok": 0, "peso_total_bytes": 0}
    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(), # Muestra completado/total
        TimeRemainingColumn(),
    )

    # Crear la instancia de Progress
    with Progress(*progress_columns, console=console, transient=False) as progress: # transient=False para que quede visible al final
        # Tarea global para el progreso general de carpetas
        task_overall = progress.add_task(f"[magenta]Progreso General Carpetas", total=len(subcarpetas))

        for carpeta in subcarpetas:
            try:
                # Pasar 'progress' a la función para que añada tareas hijas
                videos_ok, peso_carpeta = procesar_carpeta(carpeta, args, progress)
                stats["total_carpetas"] += 1
                if videos_ok > 0:
                    stats["carpetas_con_videos"] += 1
                    stats["videos_procesados_ok"] += videos_ok
                    stats["peso_total_bytes"] += peso_carpeta
            except Exception as e:
                # Captura de errores inesperados a nivel de carpeta
                log.critical(f"Error fatal no controlado procesando carpeta '{carpeta.name}': {e}", exc_info=True)
                console.print(f"[fail]Error Inesperado:[/fail] Fallo procesando la carpeta '{carpeta.name}'. Ver logs.")
            finally:
                # Avanzar la tarea general sin importar el resultado de la carpeta
                progress.update(task_overall, advance=1)
    # --- Fin Progress ---

    # --- Resumen Final ---
    end_time = datetime.now()
    console.print("\n" + "="*60)
    console.print(Panel(f"[head]== RESUMEN GENERAL DEL PROCESO ==[/head]\n"
                      f"Tiempo Total: {end_time - start_time}",
                      expand=False, border_style="magenta"))

    summary_table = Table(show_header=True, header_style="bold blue", show_lines=True)
    summary_table.add_column("Métrica", style="dim", width=35)
    summary_table.add_column("Valor", style="bold")

    summary_table.add_row("Total de carpetas analizadas", str(stats["total_carpetas"]))
    summary_table.add_row("Carpetas con videos procesados", str(stats["carpetas_con_videos"]))
    summary_table.add_row("Total archivos de video OK", str(stats["videos_procesados_ok"]))

    if stats["videos_procesados_ok"] > 0:
        peso_total_str = formatear_peso(stats["peso_total_bytes"])
        peso_promedio_bytes = stats["peso_total_bytes"] / stats["videos_procesados_ok"]
        peso_promedio_str = formatear_peso(peso_promedio_bytes)
        summary_table.add_row("Peso total de videos procesados", peso_total_str)
        summary_table.add_row("Peso promedio por video", peso_promedio_str)
    else:
        summary_table.add_row("Peso total de videos procesados", "[detail]N/A[/detail]")
        summary_table.add_row("Peso promedio por video", "[detail]N/A[/detail]")

    if args.logfile:
        summary_table.add_row("Log detallado guardado en", f"[path]{Path(args.logfile).resolve()}[/path]")
    else:
        summary_table.add_row("Log detallado", "[info]No guardado (solo consola)[/info]")

    console.print(summary_table)
    log.info("--- Script Finalizado ---")

    # Opcional: Guardar salida de consola si se grabó
    # console.save_html("processing_summary.html")


if __name__ == "__main__":
    
    main()
