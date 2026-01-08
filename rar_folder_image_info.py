#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow avanzado v3: capturas JPG/PNG, info detallada y compresión RAR por carpeta.
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
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional

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
# Más capturas por defecto en ambos casos
def generar_porcentajes_equiespaciados(total_capturas: int, inicio_pct: float, fin_pct: float) -> list[float]:
    """Genera una lista de porcentajes equiespaciados entre 'inicio_pct' y 'fin_pct'."""
    if total_capturas <= 0:
        return []
    if total_capturas == 1:
        return [round((inicio_pct + fin_pct) / 2, 4)]
    paso = (fin_pct - inicio_pct) / (total_capturas - 1)
    return [round(inicio_pct + paso * i, 4) for i in range(total_capturas)]

# Un solo archivo: capturas del 2% al 98% (~0.97% entre cada una) => 100 capturas
DEFAULT_PCTS_UNICOS = generar_porcentajes_equiespaciados(100, 2.0, 98.0)
# Varios archivos: del 8% al 96% (~1.8% entre cada una) => 50 capturas
DEFAULT_PCTS_MULTI  = generar_porcentajes_equiespaciados(50, 8.0, 96.0)
DEFAULT_RAR_EXE     = r"C:\Program Files\WinRAR\rar.exe" # Ruta por defecto a rar.exe
DEFAULT_RAR_PASSWORD = "GDriveLatinoHD"  # Contraseña por defecto para cifrar RARs
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
def calcular_workers_capturas(total_capturas: int, workers_videos: int) -> int:
    """Calcula workers para capturas sin saturar CPU."""
    cpu = os.cpu_count() or 1
    if total_capturas <= 1:
        return 1
    max_por_video = max(1, cpu // max(1, workers_videos))
    return max(1, min(4, max_por_video, total_capturas))

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

def display_name_until_parenthesis(filename: str) -> str:
    """Devuelve el nombre recortado hasta el primer ')' (incluyéndolo) para mostrar en consola."""
    if not isinstance(filename, str):
        return ""
    pos = filename.find(")")
    if pos != -1:
        truncated = filename[:pos + 1].strip()
        return truncated or filename
    return filename

def obfuscate_rar_basename(original: str) -> str:
    """Genera un nombre compacto que usa sustituciones numéricas ligeras para disfrazar el título."""
    if not isinstance(original, str):
        original = ""

    text = re.sub(r'[\s_]+', ' ', original).strip()
    if not text:
        text = "archivo"

    parts = [p.strip() for p in re.split(r'\s*-\s*', text) if p.strip()]
    selected_parts = []
    if len(parts) >= 2:
        selected_parts = parts[:2]
    else:
        match = re.search(r'(S\d{1,2}E\d{1,2})', text, re.IGNORECASE)
        if match:
            title = text[:match.start()].strip(" -_.")
            episode = match.group(1).upper()
            if title:
                selected_parts.append(title)
            selected_parts.append(episode)
        else:
            selected_parts = [text]

    cleaned_parts = []
    for part in selected_parts:
        normalized = unicodedata.normalize("NFKD", part).encode("ascii", "ignore").decode("ascii")
        normalized = re.sub(r'[^A-Za-z0-9\s-]+', ' ', normalized)
        normalized = re.sub(r'\s{2,}', ' ', normalized).strip()
        if normalized:
            cleaned_parts.append(normalized.replace(' ', '_'))

    if not cleaned_parts:
        cleaned_parts = ["archivo"]

    base_name = "-".join(cleaned_parts)
    base_upper = base_name.upper()

    letter_to_digit = {
        'A': '4',
        'E': '3',
        'I': '1',
        'O': '0',
        'S': '5',
        'T': '7',
        'B': '8',
        'G': '6'
    }

    def transform_token(token: str) -> str:
        if re.fullmatch(r'S\d{1,2}E\d{1,2}', token, re.IGNORECASE):
            return token.upper()
        return ''.join(letter_to_digit.get(ch, ch) for ch in token.upper())

    segments = re.split(r'([_-])', base_upper)
    obfuscated_segments = []
    for segment in segments:
        if not segment:
            continue
        if segment in {"-", "_"}:
            obfuscated_segments.append(segment)
            continue
        obfuscated_segments.append(transform_token(segment))

    obfuscated_core = ''.join(obfuscated_segments).strip('_-') or "ARCHIVO"

    digest = hashlib.sha1(base_upper.encode("utf-8")).hexdigest()[:4].upper()
    suffix = f"_X{digest}"

    max_length = 80
    core_limit = max_length - len(suffix)
    if core_limit < 1:
        core_limit = max_length
    if len(obfuscated_core) > core_limit:
        trimmed = obfuscated_core[:core_limit].rstrip('_-')
        obfuscated_core = trimmed or obfuscated_core[:core_limit]

    return f"{obfuscated_core}{suffix}"

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
    # Nota: para Español aplicamos lógica especial más abajo (Latino vs Castellano).
    "en": "Inglés", "ja": "Japonés", "fr": "Francés",
    "de": "Alemán", "it": "Italiano", "ru": "Ruso", "pt": "Portugués",
    "pt-br": "Portugués BR", "zh": "Chino", "ko": "Coreano",
    "und": "Indet.", "zxx": "N/A", "mul": "Multi"
}

# Patrones para distinguir variantes de Español
SPANISH_SPAIN_PAT = re.compile(r"\b(castellano|espa(?:ñ|n)a|europa|europeo|spain|ib[eé]rico)\b", re.IGNORECASE)
SPANISH_LATAM_PAT = re.compile(r"\b(latino|latam|am[eé]rica\s*latina|mexic|argentin|chile|colombi|per[uú]|venez|lat)\b", re.IGNORECASE)

def _normalize_spanish_variant(code_low: str, hint: str) -> str:
    """Devuelve etiqueta de Español priorizando 'Español Latino' automáticamente,
    a menos que se detecte explícitamente Castellano (España).
    """
    # Códigos que implican Latino explícito
    if any(x in code_low for x in ["es-419", "es_mx", "es-mx", "es_ar", "es-ar", "es_cl", "es-co", "es-pe", "es-ve", "es-uy", "es-bo", "es-ec", "es-pa", "es-pr", "es-cr", "es-do", "es-sv", "es-gt", "es-hn", "es-ni", "es-py"]):
        return "Español Latino"
    # Códigos que implican España
    if any(x in code_low for x in ["es-es", "es_es", "españa"]):
        return "Castellano"

    # Inspección por texto auxiliar (título, other_language, etc.)
    hint_low = hint.lower() if hint else ""
    if SPANISH_SPAIN_PAT.search(hint_low):
        return "Castellano"
    if SPANISH_LATAM_PAT.search(hint_low):
        return "Español Latino"

    # Por defecto, Español no calificado -> Latino
    return "Español Latino"

def normalize_lang(code: str | None, hint: str | None = None) -> str:
    if not code and not hint:
        return "Indet."
    code_low = (code or "").lower()

    # Tratar Español con lógica propia
    base_code = code_low.split('-')[0].replace("_", "-")
    if code_low.startswith("es") or base_code == "es" or code_low == "spa":
        return _normalize_spanish_variant(code_low, hint or "")

    # Resto de idiomas por mapa conocido
    if code_low in LANGS:
        return LANGS[code_low]
    if base_code in LANGS:
        return LANGS[base_code]  # Fallback
    return (code or "Indet.").capitalize()  # Devolver original capitalizado si no se conoce

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
            # Recolectar pistas de pistas/hints para distinguir Español
            other_lang = getattr(track, 'other_language', None)
            if isinstance(other_lang, (list, tuple)):
                other_lang = " ".join([str(x) for x in other_lang if x])
            title_hint = getattr(track, 'title', None)
            other_title = getattr(track, 'other_title', None)
            if isinstance(other_title, (list, tuple)):
                other_title = " ".join([str(x) for x in other_title if x])
            hint_text = " ".join([x for x in [str(lang_code or ""), other_lang or "", title_hint or "", other_title or ""] if x])
            ch_str = getattr(track, 'channel_s', None) or getattr(track, 'channels', None)
            try: ch_count = int(ch_str) if ch_str else None
            except ValueError: ch_count = None

            default = "[D]" if getattr(track, 'default', 'No') == 'Yes' else ""
            forced = "[F]" if getattr(track, 'forced', 'No') == 'Yes' else ""
            flags = f"{default}{forced}".strip()

            pista_info = {
                "lang": normalize_lang(lang_code, hint_text), "format": fmt,
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
            other_lang = getattr(track, 'other_language', None)
            if isinstance(other_lang, (list, tuple)):
                other_lang = " ".join([str(x) for x in other_lang if x])
            title_hint = getattr(track, 'title', None)
            other_title = getattr(track, 'other_title', None)
            if isinstance(other_title, (list, tuple)):
                other_title = " ".join([str(x) for x in other_title if x])
            hint_text = " ".join([x for x in [str(lang_code or ""), other_lang or "", title_hint or "", other_title or ""] if x])
            default = "[D]" if getattr(track, 'default', 'No') == 'Yes' else ""
            forced = "[F]" if getattr(track, 'forced', 'No') == 'Yes' else ""
            flags = f"{default}{forced}".strip()
        
            sub_info = {
                "lang": normalize_lang(lang_code, hint_text), "format": fmt, "flags": flags
            }
            if sub_info not in resultado["subtitulos"]:
                 resultado["subtitulos"].append(sub_info)

    # Ordenar para consistencia
    resultado["audio"].sort(key=lambda x: (x['lang'], x['format'], x['channels']))
    resultado["subtitulos"].sort(key=lambda x: (x['lang'], x['format']))
    return resultado

def armar_cadena_agrupada(pistas: list[dict], tipo='audio') -> str:
    """Agrupa pistas por idioma y formato/canales.
    Cambios:
    - Español por defecto se muestra como 'Español Latino' salvo detectar Castellano.
    - Para subtítulos, formatea como 'Idioma (FORMATO)'. Si es forzado, agrega
      la etiqueta 'Forzados' sin perder el idioma.
    - Separador: ' – ' para audio, ', ' para subtítulos.
    """
    if not pistas:
        if tipo == 'subs':
            return "[detail]Ninguno[/detail]"
        return "[detail]Ninguna[/detail]"

    def is_forced(p):
        return '[F]' in p.get('flags', '')

    key_func = lambda p: (
        p['lang'],
        p['format'],
        p.get('channels', '') if tipo == 'audio' else '',
        is_forced(p) if tipo == 'subs' else False
    )
    contador = defaultdict(int)
    for p in pistas:
        contador[key_func(p)] += 1

    if not contador:
        if tipo == 'subs':
            return "[detail]Ninguno[/detail]"
        return "[detail]Ninguna[/detail]"

    partes = []
    def lang_priority(name: str) -> int:
        if name in ("Español", "Español Latino", "Castellano"):
            return 0
        if name == "Inglés":
            return 1
        return 3

    sorted_keys = sorted(contador.keys(), key=lambda k: (
        lang_priority(k[0]), k[0], k[1], k[2], k[3]
    ))

    for key in sorted_keys:
        lang, fmt, channels, forzado = key
        count = contador[key]
        count_str = f" [detail]x{count}[/detail]" if count > 1 else ""
        if tipo == 'audio':
            channel_str = f" {channels}" if channels else ""
            partes.append(f"[info]{lang}[/info] [bold]{fmt}[/bold]{channel_str}{count_str}")
        else:
            # Subtítulos: 'Idioma (FORMATO)'; si es forzado, conservar idioma.
            forced_tag = ", [detail]Forzados[/detail]" if forzado else ""
            partes.append(
                f"[info]{lang}[/info] ([bold]{fmt}[/bold]{forced_tag}){count_str}"
            )

    return (" – " if tipo == 'audio' else ", ").join(partes)

# --- ffmpeg Capturas (Imagen) ---
@safe_run
def generar_capturas(
    ruta_video: Path,
    nombre_base_capturas: str,
    porcentajes: list[float],
    image_format: str,
    carpeta_destino: Path,
    indice_archivo: int,
    duracion_s: float,
    progress: Progress,
    task_id,
    workers_capturas: int | None = None,
) -> list[Path]:
    """Genera capturas de pantalla en formato JPG/PNG usando ffmpeg."""
    capturas_generadas = []
    nombre_display = display_name_until_parenthesis(ruta_video.name)
    image_format = (image_format or "jpg").lower()
    if image_format not in {"jpg", "png"}:
        log.warning(f"Formato de imagen no valido '{image_format}', usando 'jpg'.")
        image_format = "jpg"
    formato_ext = "png" if image_format == "png" else "jpg"
    formato_label = "PNG" if formato_ext == "png" else "JPG"
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        log.error("'ffmpeg' no encontrado en el PATH. No se pueden generar capturas.")
        return capturas_generadas  # Devuelve lista vacía en error

    # Duración ya obtenida previamente

    if not duracion_s or duracion_s <= 0:
        log.warning(
            f"No se generarán capturas para '{nombre_display}': Duración inválida o no encontrada."
        )
        progress.update(task_id, description=f"[cyan]{nombre_display}[/] [warn]- Sin duración[/warn]")
        return capturas_generadas

    os.makedirs(carpeta_destino, exist_ok=True)
    total_capturas = len(porcentajes)
    log.info(f"Generando {total_capturas} capturas {formato_label} para '{nombre_display}'")
    progress.update(task_id, total=total_capturas, description=f"[cyan]{nombre_display}[/] [yellow]- Capturas {formato_label}...[/yellow]")

    if total_capturas <= 0:
        progress.update(task_id, completed=1, total=1, description=f"[cyan]{nombre_display}[/] [warn]- Sin capturas[/warn]")
        return capturas_generadas

    workers_capturas = max(1, min(workers_capturas or 1, total_capturas))
    if workers_capturas > 1:
        log.debug(f"Capturas en paralelo ({workers_capturas} workers) para '{nombre_display}'")

    def _ejecutar_captura(idx: int, pct: float):
        tiempo = max(0.1, duracion_s * (pct / 100.0))
        nombre_captura = f"{nombre_base_capturas}[{indice_archivo}]_Captura[{idx:02d}].{formato_ext}"
        ruta_captura = carpeta_destino / nombre_captura

        if formato_ext == "png":
            cmd = [
                ffmpeg_path, "-hide_banner", "-loglevel", "error",
                "-ss", str(tiempo), "-i", str(ruta_video),
                "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vframes", "1", "-pix_fmt", "rgb24", "-compression_level", "0",
                "-y", str(ruta_captura),
            ]
        else:
            cmd = [
                ffmpeg_path, "-hide_banner", "-loglevel", "error",
                "-ss", str(tiempo), "-i", str(ruta_video),
                "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vframes", "1", "-q:v", "0", "-pix_fmt", "yuvj444p",
                "-y", str(ruta_captura),
            ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=45, encoding='utf-8', errors='ignore')
            if ruta_captura.exists():
                return idx, pct, ruta_captura, None, None
            return idx, pct, ruta_captura, "warn", f"ffmpeg OK, pero falta archivo captura {formato_label}: {nombre_captura}"
        except subprocess.CalledProcessError as e:
            error_text = (e.stderr or e.stdout or "Sin salida de error especifica.").strip()
            return idx, pct, None, "error", f"Error ffmpeg (captura {formato_label} {idx}@{pct}%): {error_text}"
        except subprocess.TimeoutExpired:
            return idx, pct, None, "warn", f"Timeout ffmpeg (captura {formato_label} {idx}@{pct}%)"
        except Exception as e_inner:
            return idx, pct, None, "error", f"Error inesperado interno en ffmpeg (captura {formato_label} {idx}): {e_inner}"

    completadas = 0
    resultados = [None] * total_capturas
    if workers_capturas == 1:
        for idx, pct in enumerate(porcentajes, 1):
            idx, pct, ruta_captura, nivel, mensaje = _ejecutar_captura(idx, pct)
            completadas += 1
            if nivel == "warn":
                log.warning(mensaje)
            elif nivel == "error":
                log.error(mensaje)
            else:
                resultados[idx - 1] = ruta_captura
                log.debug(f"Captura {formato_label} generada: {ruta_captura.name}")
            progress.update(task_id, advance=1, description=f"[cyan]{nombre_display}[/] [yellow]- Captura {completadas}/{total_capturas}[/yellow]")
    else:
        with ThreadPoolExecutor(max_workers=workers_capturas) as executor:
            futures = {
                executor.submit(_ejecutar_captura, idx, pct): (idx, pct)
                for idx, pct in enumerate(porcentajes, 1)
            }
            for future in as_completed(futures):
                idx_ref, pct_ref = futures[future]
                try:
                    idx, pct, ruta_captura, nivel, mensaje = future.result()
                except Exception as e_inner:
                    idx, pct = idx_ref, pct_ref
                    ruta_captura = None
                    nivel = "error"
                    mensaje = f"Error inesperado interno en ffmpeg (captura {formato_label} {idx}): {e_inner}"

                completadas += 1
                if nivel == "warn":
                    log.warning(mensaje)
                elif nivel == "error":
                    log.error(mensaje)
                else:
                    resultados[idx - 1] = ruta_captura
                    if ruta_captura:
                        log.debug(f"Captura {formato_label} generada: {ruta_captura.name}")
                progress.update(task_id, advance=1, description=f"[cyan]{nombre_display}[/] [yellow]- Captura {completadas}/{total_capturas}[/yellow]")

    capturas_generadas = [c for c in resultados if c]
    log.info(
        f"Generadas {len(capturas_generadas)}/{len(porcentajes)} capturas {formato_label} para '{nombre_display}'."
    )
    progress.update(task_id, description=f"[cyan]{nombre_display}[/] [green]- Capturas OK[/green]")
    return capturas_generadas # @safe_run devolverá [] si hubo excepción grave

# --- Compresión RAR (Individual por archivo) ---
@safe_run
def comprimir_carpeta_rar(carpeta_path: Path, video_files_to_compress: list[Path],
                           rar_exe_path: str, rar_store_only: bool, rar_password: Optional[str],
                           progress: Progress, task_id) -> bool:
    """Empaqueta cada video en su propio RAR cifrado (sin compresión por defecto), usando el nombre limpio y guardando en 'RARs'."""
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
        obfuscated_name = obfuscate_rar_basename(nombre_limpio)
        rar_filename = f"{obfuscated_name}{RAR_FILENAME_SUFFIX}.rar"
        rar_filepath = rars_dir / rar_filename
        nombre_display = display_name_until_parenthesis(video_file.name)

        # Actualiza la descripción para mostrar qué archivo se está procesando AHORA
        progress.update(task_id, description=f"[magenta]RAR {idx}/{total_files}:[/] [cyan]{nombre_display}[/]")

        size_gb = video_file.stat().st_size / (1024**3)
        cmd = [rar_exe_path, "a", "-ma5"]
        
        # Configurar método de compresión
        if rar_store_only:
            cmd.append("-m0")  # Sin compresión (solo almacenar)
        else:
            cmd.append("-m3")  # Compresión normal

        if rar_password:
            cmd.append(f"-hp{rar_password}")  # Cifra datos y nombres
        
        split_msg = ""
        if size_gb > RAR_SPLIT_THRESHOLD_GB:
            split_size_param = f"-v{RAR_SPLIT_SIZE_GB}g"
            cmd.append(split_size_param)
            split_msg = f" (Dividiendo en {RAR_SPLIT_SIZE_GB} GB)"

        cmd.extend([
            "-ep1", "-o+",
            str(rar_filepath.resolve()),
            str(video_file.resolve())
        ])

        try:
            action_verb = "Almacenando" if rar_store_only else "Comprimiendo"
            log.info(f"{action_verb} [{idx}/{total_files}] '{nombre_display}' -> '{rar_filename}'{split_msg}")
            cmd_log_safe = ["-hp********" if part.startswith("-hp") else part for part in cmd]
            log.debug(f"Ejecutando RAR: {' '.join(cmd_log_safe)}")
            timeout_compresion = 6 * 60 * 60  # 6 horas
            progress.update(task_id, completed=idx - 1)
            output_chunks = []
            digit_buf = ""
            last_pct = -1

            def read_rar_output(proc: subprocess.Popen):
                nonlocal digit_buf, last_pct
                while True:
                    ch = proc.stdout.read(1)
                    if ch == "":
                        break
                    output_chunks.append(ch)
                    if ch.isdigit():
                        digit_buf += ch
                        if len(digit_buf) > 3:
                            digit_buf = digit_buf[-3:]
                        continue
                    if ch == "%":
                        if digit_buf:
                            try:
                                pct = int(digit_buf)
                            except ValueError:
                                pct = None
                            if pct is not None:
                                pct = max(0, min(100, pct))
                                if pct != last_pct:
                                    progress.update(task_id, completed=(idx - 1) + (pct / 100.0))
                                    last_pct = pct
                            digit_buf = ""
                        continue
                    digit_buf = ""

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            reader = threading.Thread(target=read_rar_output, args=(proc,), daemon=True)
            reader.start()
            try:
                proc.wait(timeout=timeout_compresion)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise
            finally:
                reader.join(timeout=2)

            output_text = "".join(output_chunks)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd, output=output_text)

            progress.update(task_id, completed=idx)
            log.info(f"Creación RAR [{idx}/{total_files}] exitosa: '{rar_filename}'")
        except subprocess.CalledProcessError as e:
            error_message = e.stderr or e.stdout or "Sin salida de error específica."
            log.error(f"Error durante la creación RAR para '{nombre_display}' (código {e.returncode}):\n{error_message.strip()}")
            progress.update(task_id, completed=idx, description=f"[fail]Error RAR {idx}/{total_files}:[/] [cyan]{nombre_display}[/]")
            ok = False
        except subprocess.TimeoutExpired:
            log.error(f"Timeout durante creación RAR para '{nombre_display}' (límite {timeout_compresion / 3600:.1f}h).")
            progress.update(task_id, completed=idx, description=f"[fail]Timeout RAR {idx}/{total_files}:[/] [cyan]{nombre_display}[/]")
            ok = False
        except Exception as e_inner:
            log.error(f"Error inesperado interno en creación RAR para '{nombre_display}': {e_inner}", exc_info=True)
            progress.update(task_id, completed=idx, description=f"[fail]Error interno RAR {idx}/{total_files}:[/] [cyan]{nombre_display}[/]")
            ok = False

    # Actualización final de la tarea
    action_name = "almacenado" if rar_store_only else "compresión"
    final_description = f"[cyan]{carpeta_path.name}[/] [green]- {action_name.capitalize()} RAR finalizado[/]" if ok else f"[cyan]{carpeta_path.name}[/] [fail]- {action_name.capitalize()} RAR con errores[/]"
    progress.update(task_id, description=final_description)

    return ok

# --- Procesamiento Individual de Video (Dentro de Carpeta) ---
# No necesita @safe_run porque lo llamaremos desde un futuro que captura errores
def procesar_un_video(ruta_video: Path, indice_archivo: int, total_archivos: int,
                       porcentajes_capturas: list[float], args: argparse.Namespace,
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
    nombre_display = display_name_until_parenthesis(ruta_video.name)
    try:
        log.info(f"Procesando video [{indice_archivo}/{total_archivos}]: {nombre_display}")

        # 1. Limpiar nombre base (para capturas y referencia)
        # Usamos la nueva función clean_name
        resultado_video["nombre_limpio"] = clean_name(ruta_video.stem) or ruta_video.stem # Fallback

        # 2. Obtener información de medios (ya está decorada con @safe_run)
        resultado_video["info_media"] = obtener_info_media(ruta_video)
        # Verificar si la info falló
        if not isinstance(resultado_video["info_media"], dict) or resultado_video["info_media"].get("error"):
            log.warning(f"Fallo al obtener info media para {nombre_display}. Error: {resultado_video['info_media'].get('error', 'Desconocido')}")
            # No marcamos error fatal aquí, aún podemos intentar obtener peso
            # El error ya está en info_media["error"]

        # 3. Obtener peso
        try:
            resultado_video["peso_bytes"] = ruta_video.stat().st_size
            resultado_video["peso_str"] = formatear_peso(resultado_video["peso_bytes"])
        except OSError as e:
            log.warning(f"No se pudo obtener el peso de '{nombre_display}': {e}")
            resultado_video["error"] = resultado_video["error"] or f"Error obteniendo peso: {e}" # Añadir error si no había uno

        # 4. Generar Capturas (si no se omite y no hubo error fatal antes)
        if not args.skip_img:
            if resultado_video["info_media"] and not resultado_video["info_media"].get("error"):
                duracion_s = resultado_video["info_media"].get("duracion_s")
                workers_capturas = calcular_workers_capturas(len(porcentajes_capturas), args.workers)
                # generar_capturas está decorado con @safe_run
                resultado_video["capturas_generadas"] = generar_capturas(
                    ruta_video,
                    resultado_video["nombre_limpio"],
                    porcentajes_capturas,
                    args.img_format,
                    ruta_video.parent / NOMBRE_CARPETA_CAPTURAS,
                    indice_archivo,
                    duracion_s,
                    progress,
                    task_id_capturas,
                    workers_capturas,
                )
                # Si generar_capturas falla, devolverá [] y loggeará el error.
            else:
                 log.warning(f"Omitiendo capturas para '{nombre_display}' debido a error previo o falta de info.")
                 progress.update(task_id_capturas, description=f"[cyan]{nombre_display}[/] [warn]- Sin capturas (error previo)[/warn]")
        else:
             # Si se omiten las capturas, actualizar la tarea para que no quede "pendiente"
             progress.update(task_id_capturas, completed=1, total=1, description=f"[cyan]{nombre_display}[/] [info]- Capturas omitidas[/info]")


        log.info(f"Finalizado procesamiento básico para: {nombre_display}")
        return resultado_video

    except Exception as e:
        # Captura genérica por si algo más falla aquí
        log.error(f"Fallo crítico procesando '{nombre_display}': {e}", exc_info=True)
        resultado_video["error"] = f"Error crítico en procesar_un_video: {e}"
        # Asegurar que la tarea de progreso se marque como fallida
        try:
            progress.update(task_id_capturas, description=f"[cyan]{nombre_display}[/] [fail]- Error Crítico[/fail]")
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

    if num_videos == 1:
        pct_inicio, pct_fin = 2.0, 98.0
        porcentajes_default = DEFAULT_PCTS_UNICOS
    else:
        pct_inicio, pct_fin = 8.0, 96.0
        porcentajes_default = DEFAULT_PCTS_MULTI
    if args.num_capturas is not None:
        porcentajes = generar_porcentajes_equiespaciados(args.num_capturas, pct_inicio, pct_fin)
    else:
        porcentajes = porcentajes_default
    capturas_label = "PNG" if args.img_format == "png" else "JPG"

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
            nombre_display = display_name_until_parenthesis(ruta_video.name)
            # Tarea hija para capturas (si no se omite)
            task_capturas = progress.add_task(f"[cyan]{nombre_display}[/] [dim]- Pendiente[/]", total=1, visible=not args.skip_img, parent=task_folder)
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
                        nombre_result_display = display_name_until_parenthesis(resultado.get('nombre_original', '??'))
                        # Asegura que la tarea de captura esté completa
                        progress.update(task_id_capturas, completed=progress.tasks[task_id_capturas].total, description=f"[cyan]{nombre_result_display}[/] [green]- Capturas OK[/green]") # Actualiza descripción final OK
                else:
                    error_msg = resultado.get('error', 'Desconocido') if isinstance(resultado, dict) else 'Error crítico futuro'
                    nombre_result_display = display_name_until_parenthesis(resultado.get('nombre_original', 'N/A')) if isinstance(resultado, dict) else 'N/A'
                    log.warning(f"Video con error: {nombre_result_display} - {error_msg}")
                    # Marca la tarea de captura como fallida si existe
                    if not args.skip_img:
                        progress.update(task_id_capturas, description=f"[cyan]{nombre_result_display}[/] [fail]- Error Proc.[/fail]")

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
                capturas_str = f"[good]{capturas_label}[/good] (1): {capturas_nombres[0]}"
            else:
                capturas_str = f"[good]{capturas_label}[/good] ({len(capturas_nombres)})"
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
        success = comprimir_carpeta_rar(carpeta_path, videos_para_comprimir, args.rar_path, args.rar_store_only, args.rar_password, progress, task_compress)
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
    ap.add_argument("-i", "--skip-img", action="store_true",
                    help="No generar imágenes (capturas) de los videos.")
    ap.add_argument("-r", "--no-compress", dest='compress', action="store_false",
                    help="No generar archivos RAR (omitir compresión).")
    ap.add_argument("--num-capturas", type=int, default=None,
                    help="Numero de capturas a generar por video. Si no se indica, se usan los valores por defecto.")
    ap.add_argument("--img-format", choices=["jpg", "png"], default="jpg",
                    help="Formato de las capturas: jpg o png (alta calidad).")
    ap.add_argument("--rar-store-only", action="store_true", default=True,
                help="Crear RAR sin compresión (solo almacenar archivos). ACTIVADO POR DEFECTO.")
    ap.add_argument("--rar-compress", action="store_false", dest="rar_store_only",
                help="Crear RAR con compresión normal (anula --rar-store-only).")
    ap.add_argument("--rar-password", default=None,
                    help=f"Contraseña para cifrar los RAR. Si no se especifica, se usa el valor por defecto ({DEFAULT_RAR_PASSWORD}). También puede definirse con la variable de entorno RAR_PASSWORD.")
    ap.add_argument("--logfile", nargs='?', const=LOG_FILENAME, default=None,
                    help=f"Guardar log detallado en un archivo (por defecto: {LOG_FILENAME} si se especifica la opción sin ruta).")
    ap.add_argument("-v", "--verbose", action="store_const", const=logging.DEBUG, default=DEFAULT_LOG_LEVEL,
                   help="Mostrar mensajes de depuración (DEBUG).")
    args = ap.parse_args()

    if args.rar_password is None:
        env_password = os.getenv("RAR_PASSWORD")
        args.rar_password = env_password if env_password else DEFAULT_RAR_PASSWORD

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

    if args.num_capturas is not None and args.num_capturas < 0:
        log.critical("El numero de capturas no puede ser negativo.")
        console.print("[fail]Error:[/fail] --num-capturas no puede ser negativo.")
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
