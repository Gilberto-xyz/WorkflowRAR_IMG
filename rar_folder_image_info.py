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
import json
import secrets
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

# Un solo archivo: capturas del 2% al 98% => 300 capturas
DEFAULT_PCTS_UNICOS = generar_porcentajes_equiespaciados(300, 2.0, 98.0)
# Varios archivos: del 8% al 96% => 100 capturas
DEFAULT_PCTS_MULTI  = generar_porcentajes_equiespaciados(100, 8.0, 96.0)
DEFAULT_RAR_EXE     = r"C:\Program Files\WinRAR\rar.exe" # Ruta por defecto a rar.exe
DEFAULT_RAR_PASSWORD = "GDRIVELatinoHD.NET"  # Contraseña por defecto para cifrar RARs
DEFAULT_LOG_LEVEL   = logging.WARNING # Nivel de log por defecto
DEFAULT_WORKERS     = max(1, os.cpu_count() // 2) # Al menos 1 worker
CAPTURAS_FIXED_WORKERS_100 = 4 # Perfil fijo para tandas de 100 capturas.
CAPTURAS_FIXED_WORKERS_300 = 8 # Perfil fijo para tandas de 300 capturas.
CAPTURAS_ONE_PASS_MAX_INTERVAL_S = 2.5 # One-pass solo compensa en muestreos extremadamente densos.

NOMBRE_CARPETA_CAPTURAS = "capturas" # Nombre de la carpeta donde se guardarán las capturas
RAR_FILENAME_SUFFIX     = "[GDRIVELatinoHD.NET]" # Sufijo para nombres de RARs
RAR_MAX_VOLUME_SIZE_GB  = 15 # Tamaño máximo permitido por volumen RAR.
RAR_MAX_VOLUME_SIZE_BYTES = int(RAR_MAX_VOLUME_SIZE_GB * (1024**3))
RAR_SPLIT_OVERHEAD_PADDING_BYTES = 4 * 1024 * 1024 # Holgura para cabeceras/metadata del RAR.
RAR_TINY_LAST_VOLUME_THRESHOLD_BYTES = 16 * 1024 * 1024 # Si la última parte es minúscula, se reajusta.
RAR_SPLIT_RETRY_LIMIT = 3 # Intentos máximos de reajuste cuando aparece una cola mínima.
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
    """Devuelve un número fijo de workers según la tanda de capturas."""
    _ = workers_videos  # Se mantiene por compatibilidad, pero la política ya es fija.
    if total_capturas <= 1:
        return 1
    if total_capturas >= 300:
        return min(CAPTURAS_FIXED_WORKERS_300, total_capturas)
    if total_capturas >= 100:
        return min(CAPTURAS_FIXED_WORKERS_100, total_capturas)
    return min(CAPTURAS_FIXED_WORKERS_100, total_capturas)

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

def normalize_rar_source_text(text: str) -> str:
    """Normaliza un stem preservando el año y descartando tags decorativas."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\[[^\]]*?\]|\{[^\}]*?\}', ' ', text)
    try:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    except Exception:
        log.warning(f"Fallo en normalización unicodedata para RAR: '{text[:50]}...'")
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    text = re.sub(r'[._]+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9()\s-]+', ' ', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def extract_rar_release_tag(text: str) -> str | None:
    """Extrae una etiqueta corta de fuente/release para evitar colisiones entre ediciones."""
    normalized = normalize_rar_source_text(text)
    if not normalized:
        return None

    release_patterns = (
        (r'\b(?:BD|BLURAY)\s*REMUX\b|\bBDREMUX\b', "BDREMUX"),
        (r'\bWEB[- ]?DL\b|\bWEBDL\b', "WEBDL"),
        (r'\bWEBRIP\b', "WEBRIP"),
        (r'\bWEBCAP\b', "WEBCAP"),
        (r'\bBDRIP\b', "BDRIP"),
        (r'\bBRRIP\b', "BRRIP"),
        (r'\bHDCAM\b|\bHD[- ]?CAM\b', "HDCAM"),
        (r'\bHDTS\b|\bHD[- ]?TS\b', "HDTS"),
        (r'\bCAM\b', "CAM"),
        (r'\b(?:TELE|TELE)?SYNC\b|\bTS\b', "TS"),
        (r'\bTELECINE\b|\bTC\b', "TC"),
        (r'\bDVDRIP\b', "DVDRIP"),
        (r'\bDVD(?:-| )?SCR\b|\bDVDSCR\b', "DVDSCR"),
        (r'\bSCREENER\b|\bSCR\b', "SCR"),
        (r'\bHDRIP\b', "HDRIP"),
        (r'\bR5\b', "R5"),
        (r'\bHDTV\b', "HDTV"),
        (r'\bPDTV\b', "PDTV"),
        (r'\bTVRIP\b', "TVRIP"),
        (r'\bSATRIP\b', "SATRIP"),
        (r'\bDSRIP\b', "DSRIP"),
        (r'\bDVBRIP\b', "DVBRIP"),
        (r'\bVHSRIP\b', "VHSRIP"),
        (r'\bBLURAY\b', "BLURAY"),
        (r'\bBDMV\b', "BDMV"),
        (r'\bREMUX\b', "REMUX"),
    )

    for pattern, label in release_patterns:
        if re.search(pattern, normalized, re.IGNORECASE):
            return label
    return None

def select_movie_rar_parts(text: str) -> list[str]:
    """Extrae un set compacto para películas: título + año + resolución + release tag."""
    normalized = normalize_rar_source_text(text)
    if not normalized:
        return []

    year_match = re.search(r'\((19|20)\d{2}\)|\b(19|20)\d{2}\b', normalized)
    resolution_match = re.search(r'\b(?:4K|UHD|2160P|1080P|720P|480P)\b', normalized, re.IGNORECASE)
    release_tag = extract_rar_release_tag(normalized)
    metadata_match = re.search(
        r'\b(?:WEB[- ]?DL|WEBRIP|WEBCAP|BLURAY|BDMV|BDRIP|BRRIP|BDREMUX|REMUX|HDCAM|HD[- ]?CAM|HDTS|HD[- ]?TS|CAM|(?:TELE|TELE)?SYNC|TS|TELECINE|TC|DVDRIP|DVD(?:-| )?SCR|DVDSCR|SCREENER|SCR|HDRIP|R5|HDTV|PDTV|TVRIP|SATRIP|DSRIP|DVBRIP|VHSRIP|HDR10\+?|DOLBY|VISION|X26[45]|H\.?26[45]|HEVC|AV1|BITS?|AAC|DDP?|TRUEHD|ATMOS|DUAL|AUDIO|SUB(?:TIT(?:LES?)?)?|CAST?|LAT(?:INO)?|ESP(?:ANOL)?|ES-419|ENG|KOR|JAP)\b',
        normalized,
        re.IGNORECASE)

    cut_positions = [match.start() for match in (year_match, resolution_match, metadata_match) if match]
    title_part = normalized[:min(cut_positions)].strip(" -_.") if cut_positions else normalized
    title_words = [word for word in re.split(r'\s+', title_part) if word]

    selected_parts = title_words[:4]
    if year_match:
        year = re.sub(r'[^0-9]', '', year_match.group(0))
        if year:
            selected_parts.append(year)

    if resolution_match:
        resolution = resolution_match.group(0).upper()
        selected_parts.append("4K" if resolution in {"UHD", "2160P"} else resolution)

    if release_tag:
        selected_parts.append(release_tag)

    return selected_parts or (title_words[:4] if title_words else [normalized])

def obfuscate_rar_basename(original: str) -> str:
    """Genera un nombre compacto que usa sustituciones numéricas ligeras para disfrazar el título."""
    if not isinstance(original, str):
        original = ""

    text = normalize_rar_source_text(original)
    if not text:
        text = "archivo"

    match = re.search(r'\bS\d{1,2}\s*E\d{1,2}\b', text, re.IGNORECASE)
    if not match:
        match = re.search(r'\bE\s*\d{1,3}\b', text, re.IGNORECASE)
    if match:
        episode = re.sub(r'\s+', '', match.group(0)).upper()
        title_part = text[:match.start()].strip(" -_.")
        release_tag = extract_rar_release_tag(text)
        if title_part:
            title_short = re.split(r'\s+', title_part, maxsplit=1)[0]
            selected_parts = [title_short, episode]
        else:
            selected_parts = [episode]
        if release_tag:
            selected_parts.append(release_tag)
    else:
        selected_parts = select_movie_rar_parts(original)

    cleaned_parts = []
    for part in selected_parts:
        normalized = unicodedata.normalize("NFKD", part).encode("ascii", "ignore").decode("ascii")
        normalized = re.sub(r'[^A-Za-z0-9\s-]+', ' ', normalized)
        normalized = re.sub(r'\s{2,}', ' ', normalized).strip()
        if normalized:
            cleaned_parts.append(normalized.replace(' ', '_'))

    if not cleaned_parts:
        cleaned_parts = ["archivo"]

    joiner = "-" if match else "_"
    base_name = joiner.join(cleaned_parts)
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
        if re.fullmatch(r'(S\d{1,2}E\d{1,2}|E\d{1,3})', token, re.IGNORECASE):
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

def calcular_plan_volumenes_rar(total_size_bytes: int, total_parts: int | None = None,
                                extra_padding_bytes: int | None = None) -> dict | None:
    """Calcula un reparto donde N-1 partes son equivalentes y la última absorbe el sobrante."""
    total_size_bytes = max(0, int(total_size_bytes))
    if extra_padding_bytes is None:
        extra_padding_bytes = 0
    if total_size_bytes <= RAR_MAX_VOLUME_SIZE_BYTES and total_parts is None:
        return None

    if total_parts is None:
        total_parts = max(2, (total_size_bytes + RAR_MAX_VOLUME_SIZE_BYTES - 1) // RAR_MAX_VOLUME_SIZE_BYTES)

    base_volume_bytes = total_size_bytes // total_parts
    base_volume_bytes = max(1, min(base_volume_bytes, RAR_MAX_VOLUME_SIZE_BYTES))
    last_volume_bytes = total_size_bytes - (base_volume_bytes * (total_parts - 1))

    if extra_padding_bytes > 0 and last_volume_bytes + extra_padding_bytes <= RAR_MAX_VOLUME_SIZE_BYTES:
        last_volume_bytes += extra_padding_bytes

    if last_volume_bytes > RAR_MAX_VOLUME_SIZE_BYTES:
        overflow_bytes = last_volume_bytes - RAR_MAX_VOLUME_SIZE_BYTES
        if total_parts <= 1:
            return None
        adjust_bytes = (overflow_bytes + (total_parts - 2)) // (total_parts - 1)
        base_volume_bytes = min(RAR_MAX_VOLUME_SIZE_BYTES, base_volume_bytes + adjust_bytes)
        last_volume_bytes = total_size_bytes - (base_volume_bytes * (total_parts - 1))

    if last_volume_bytes > RAR_MAX_VOLUME_SIZE_BYTES or last_volume_bytes <= 0:
        return None

    volume_sizes = [base_volume_bytes] * (total_parts - 1)
    volume_sizes.append(last_volume_bytes)
    volume_switches = [f"-v{size}b" for size in volume_sizes]

    split_msg = (
        f" (Dividiendo en {total_parts} partes: "
        f"{total_parts - 1} x {formatear_peso(base_volume_bytes)}"
        f" + última de hasta {formatear_peso(last_volume_bytes)})"
    )

    return {
        "total_parts": total_parts,
        "base_volume_bytes": base_volume_bytes,
        "last_volume_bytes": last_volume_bytes,
        "volume_sizes": volume_sizes,
        "volume_switches": volume_switches,
        "split_msg": split_msg,
    }

def estimar_plan_volumenes_rar(source_size_bytes: int,
                               extra_padding_bytes: int | None = None) -> dict | None:
    """Estima el plan de volúmenes a partir del tamaño origen más una pequeña holgura."""
    if extra_padding_bytes is None:
        extra_padding_bytes = RAR_SPLIT_OVERHEAD_PADDING_BYTES
    estimated_archive_bytes = int(source_size_bytes) + max(0, int(extra_padding_bytes))
    if estimated_archive_bytes <= RAR_MAX_VOLUME_SIZE_BYTES:
        return None

    total_parts = (estimated_archive_bytes + RAR_MAX_VOLUME_SIZE_BYTES - 1) // RAR_MAX_VOLUME_SIZE_BYTES
    return calcular_plan_volumenes_rar(
        source_size_bytes,
        total_parts=total_parts,
        extra_padding_bytes=extra_padding_bytes)

def obtener_volumenes_rar_generados(rar_filepath: Path) -> list[Path]:
    """Devuelve los volúmenes generados para un RAR, soportando nombres RAR5 y legacy."""
    base_name = rar_filepath.stem
    parent = rar_filepath.parent

    rar5_volumes = sorted(
        (
            p for p in parent.iterdir()
            if p.is_file() and re.search(rf"^{re.escape(base_name)}\.part(\d+)\.rar$", p.name, re.IGNORECASE)
        ),
        key=lambda path: int(re.search(r"\.part(\d+)\.rar$", path.name, re.IGNORECASE).group(1))
        if re.search(r"\.part(\d+)\.rar$", path.name, re.IGNORECASE) else 0)
    if rar5_volumes:
        return rar5_volumes

    legacy_volumes = []
    if rar_filepath.exists():
        legacy_volumes.append(rar_filepath)
    legacy_volumes.extend(
        sorted(
            (p for p in parent.iterdir()
             if p.is_file() and re.search(rf"^{re.escape(base_name)}\.r\d+$", p.name, re.IGNORECASE)),
            key=lambda path: int(re.search(r"\.r(\d+)$", path.name, re.IGNORECASE).group(1))
        )
    )
    return legacy_volumes

def eliminar_volumenes_rar(paths: list[Path]) -> None:
    """Elimina los volúmenes generados para permitir un reintento limpio."""
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            log.warning(f"No se pudo eliminar el volumen temporal '{path.name}': {exc}")

def requiere_reajuste_split(volumes: list[Path], split_config: dict | None) -> bool:
    """Detecta el caso de una última parte diminuta causada por overhead del contenedor."""
    if not split_config or len(volumes) <= split_config["total_parts"]:
        return False
    try:
        last_size = volumes[-1].stat().st_size
    except OSError:
        return False
    return last_size <= RAR_TINY_LAST_VOLUME_THRESHOLD_BYTES

def ejecutar_comando_rar(cmd: list[str], idx: int, total_files: int, nombre_display: str,
                         progress: Progress, task_id, timeout_compresion: int) -> str:
    """Ejecuta rar.exe y propaga errores conservando el progreso incremental."""
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
    return output_text

def strip_rich_markup(text: str) -> str:
    """Elimina etiquetas simples de Rich para reutilizar el texto en salidas máquina-app."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\[/?[^\]]+\]", "", text).strip()

# --- MediaInfo Helpers ---
FALLBACK_LANG_NAMES = {
    "und": "Indefinido",
    "zxx": "Sin contenido lingüístico",
    "mul": "Múltiples idiomas",
    "es": "Español",
    "spa": "Español",
    "es-419": "Español (Latinoamérica)",
    "es-es": "Español (España)",
    "en": "Inglés",
    "eng": "Inglés",
    "ja": "Japonés",
    "jpn": "Japonés",
    "fr": "Francés",
    "fra": "Francés",
    "fre": "Francés",
    "de": "Alemán",
    "deu": "Alemán",
    "ger": "Alemán",
    "it": "Italiano",
    "ita": "Italiano",
    "ru": "Ruso",
    "rus": "Ruso",
    "pt": "Portugués",
    "por": "Portugués",
    "pt-br": "Portugués (Brasil)",
    "zh": "Chino",
    "zho": "Chino",
    "chi": "Chino",
    "ko": "Coreano",
    "kor": "Coreano",
    "fa": "Persa",
    "fas": "Persa",
    "per": "Persa",
    "th": "Tailandés",
    "tha": "Tailandés",
}

FALLBACK_CANONICAL_CODES = {
    "und": "und",
    "zxx": "zxx",
    "mul": "mul",
    "es": "es",
    "spa": "es",
    "es-419": "es-419",
    "es-la": "es-419",
    "es-es": "es-es",
    "en": "en",
    "eng": "en",
    "ja": "ja",
    "jpn": "ja",
    "fr": "fr",
    "fra": "fr",
    "fre": "fr",
    "de": "de",
    "deu": "de",
    "ger": "de",
    "it": "it",
    "ita": "it",
    "ru": "ru",
    "rus": "ru",
    "pt": "pt",
    "por": "pt",
    "pt-br": "pt-br",
    "zh": "zh",
    "zho": "zh",
    "chi": "zh",
    "ko": "ko",
    "kor": "ko",
    "fa": "fa",
    "fas": "fa",
    "per": "fa",
    "th": "th",
    "tha": "th",
}


def normalize_lang_code(code: str | None) -> str:
    if not code:
        return "und"
    normalized = str(code).strip().lower().replace("_", "-")
    return normalized or "und"


def load_language_catalog() -> tuple[dict[str, str], dict[str, str]]:
    language_names = dict(FALLBACK_LANG_NAMES)
    canonical_codes = dict(FALLBACK_CANONICAL_CODES)
    catalog_path = Path(__file__).with_name("track_languages.json")

    if not catalog_path.exists():
        return language_names, canonical_codes

    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        display_names = raw.get("displayNames", {})
        if isinstance(display_names, dict):
            for code, label in display_names.items():
                if not isinstance(code, str) or not isinstance(label, str):
                    continue
                normalized = normalize_lang_code(code)
                clean_label = label.strip()
                if normalized and clean_label:
                    language_names[normalized] = clean_label

        raw_canonical_codes = raw.get("canonicalBaseCodes", {})
        if isinstance(raw_canonical_codes, dict):
            for code, canonical in raw_canonical_codes.items():
                if not isinstance(code, str) or not isinstance(canonical, str):
                    continue
                normalized = normalize_lang_code(code)
                normalized_canonical = normalize_lang_code(canonical)
                if normalized and normalized_canonical:
                    canonical_codes[normalized] = normalized_canonical
    except Exception:
        return language_names, canonical_codes

    return language_names, canonical_codes


LANG_NAMES, LANG_CANONICAL_CODES = load_language_catalog()

# Patrones para distinguir variantes de Español
SPANISH_SPAIN_PAT = re.compile(r"\b(castellano|espa(?:ñ|n)a|europa|europeo|spain|ib[eé]rico)\b", re.IGNORECASE)
SPANISH_LATAM_PAT = re.compile(r"\b(latino|latam|am[eé]rica\s*latina|mexic|argentin|chile|colombi|per[uú]|venez|lat)\b", re.IGNORECASE)

def _resolve_spanish_variant_code(code_low: str, hint: str) -> str:
    """Resuelve la variante de español sin inventar una región si el metadato es ambiguo."""
    # Códigos que implican Latino explícito
    if any(x in code_low for x in ["es-419", "es_mx", "es-mx", "es_ar", "es-ar", "es_cl", "es-co", "es-pe", "es-ve", "es-uy", "es-bo", "es-ec", "es-pa", "es-pr", "es-cr", "es-do", "es-sv", "es-gt", "es-hn", "es-ni", "es-py"]):
        return "es-419"
    # Códigos que implican España
    if any(x in code_low for x in ["es-es", "es_es", "españa"]):
        return "es-es"

    # Inspección por texto auxiliar (título, other_language, etc.)
    hint_low = hint.lower() if hint else ""
    if SPANISH_SPAIN_PAT.search(hint_low):
        return "es-es"
    if SPANISH_LATAM_PAT.search(hint_low):
        return "es-419"

    return "es"


def resolve_display_name(code: str | None) -> str | None:
    normalized = normalize_lang_code(code)
    candidates = [
        normalized,
        LANG_CANONICAL_CODES.get(normalized),
    ]

    base_code = normalized.split("-")[0]
    if base_code:
        candidates.extend(
            [
                base_code,
                LANG_CANONICAL_CODES.get(base_code),
            ])

    for candidate in candidates:
        if candidate and candidate in LANG_NAMES:
            return LANG_NAMES[candidate]

    return None

def normalize_lang(code: str | None, hint: str | None = None) -> str:
    if not code and not hint:
        return resolve_display_name("und") or "Indefinido"

    code_low = normalize_lang_code(code)

    # Tratar Español con lógica propia
    base_code = code_low.split('-')[0]
    if code_low.startswith("es") or base_code == "es" or code_low == "spa":
        resolved_spanish = resolve_display_name(_resolve_spanish_variant_code(code_low, hint or ""))
        if resolved_spanish:
            return resolved_spanish

    resolved = resolve_display_name(code_low)
    if resolved:
        return resolved

    return (code or "Indefinido").capitalize()

def normalize_channels(ch_count: int | None) -> str:
    if not isinstance(ch_count, int): return "?"
    return {1: "1.0", 2: "2.0", 6: "5.1", 7: "6.1", 8: "7.1"}.get(ch_count, str(ch_count))

def find_ffprobe() -> str | None:
    """Localiza ffprobe para fallback de metadatos cuando MediaInfo no lee pistas."""
    path_cmd = shutil.which("ffprobe")
    if path_cmd:
        return path_cmd
    common = Path(r"C:\ffmpeg\bin\ffprobe.exe")
    return str(common) if common.exists() else None

def normalize_ffprobe_codec(codec_name: str | None, profile: str | None = None, tipo: str = "") -> str:
    codec = (codec_name or "").strip().lower()
    prof = (profile or "").strip()
    audio_map = {
        "aac": "AAC",
        "ac3": "AC-3",
        "eac3": "E-AC-3",
        "dts": "DTS",
        "truehd": "TRUEHD",
        "flac": "FLAC",
        "mp3": "MP3",
        "pcm_s16le": "PCM",
        "pcm_s24le": "PCM",
    }
    sub_map = {
        "subrip": "SRT",
        "ass": "ASS",
        "ssa": "SSA",
        "webvtt": "WEBVTT",
        "hdmv_pgs_subtitle": "PGS",
        "dvd_subtitle": "VOBSUB",
    }
    video_map = {
        "hevc": "HEVC",
        "h264": "AVC",
        "av1": "AV1",
        "mpeg4": "MPEG-4",
        "vc1": "VC-1",
    }
    if tipo == "audio":
        if codec == "dts" and prof:
            return "DTS-HD MA" if "MA" in prof.upper() else "DTS"
        return audio_map.get(codec, codec.upper() if codec else "Desc.")
    if tipo == "subs":
        return sub_map.get(codec, codec.upper() if codec else "Desc.")
    return video_map.get(codec, codec.upper() if codec else "Desconocido")

def obtener_info_media_ffprobe(ruta_archivo: Path) -> dict:
    ffprobe = find_ffprobe()
    if not ffprobe:
        return {}

    cmd = [
        ffprobe, "-hide_banner", "-v", "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,profile,width,height,channels:stream_tags=language,title:stream_disposition=default,forced",
        "-of", "json", str(ruta_archivo)
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=30,
        )
        data = json.loads(proc.stdout or "{}")
    except Exception as exc:
        log.debug(f"ffprobe no pudo leer metadatos de '{ruta_archivo.name}': {exc}")
        return {}

    resultado = {
        "audio": [],
        "subtitulos": [],
        "duracion_s": None,
        "resolucion": "Desconocida",
        "formato_video": "Desconocido",
    }

    duration = (data.get("format") or {}).get("duration")
    try:
        if duration:
            resultado["duracion_s"] = float(duration)
    except (TypeError, ValueError):
        pass

    for stream in data.get("streams") or []:
        stream_type = stream.get("codec_type")
        tags = stream.get("tags") or {}
        disposition = stream.get("disposition") or {}
        lang_code = tags.get("language")
        title_hint = tags.get("title")
        hint_text = " ".join(str(x) for x in [lang_code or "", title_hint or ""] if x)
        default = "[D]" if int(disposition.get("default") or 0) else ""
        forced = "[F]" if int(disposition.get("forced") or 0) else ""
        flags = f"{default}{forced}".strip()

        if stream_type == "video" and resultado["resolucion"] == "Desconocida":
            w = stream.get("width")
            h = stream.get("height")
            if w and h:
                resultado["resolucion"] = f"{w}x{h}"
            resultado["formato_video"] = normalize_ffprobe_codec(
                stream.get("codec_name"), stream.get("profile"), "video"
            )
        elif stream_type == "audio":
            pista_info = {
                "lang": normalize_lang(lang_code, hint_text),
                "format": normalize_ffprobe_codec(stream.get("codec_name"), stream.get("profile"), "audio"),
                "channels": normalize_channels(stream.get("channels")),
                "flags": flags,
            }
            if pista_info not in resultado["audio"]:
                resultado["audio"].append(pista_info)
        elif stream_type == "subtitle":
            sub_info = {
                "lang": normalize_lang(lang_code, hint_text),
                "format": normalize_ffprobe_codec(stream.get("codec_name"), stream.get("profile"), "subs"),
                "flags": flags,
            }
            if sub_info not in resultado["subtitulos"]:
                resultado["subtitulos"].append(sub_info)

    return resultado

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

    if (
        resultado["resolucion"] == "Desconocida"
        or resultado["formato_video"] == "Desconocido"
        or resultado["duracion_s"] is None
        or not resultado["audio"]
        or not resultado["subtitulos"]
    ):
        fallback = obtener_info_media_ffprobe(ruta_archivo)
        if fallback:
            if resultado["resolucion"] == "Desconocida" and fallback.get("resolucion") != "Desconocida":
                resultado["resolucion"] = fallback["resolucion"]
            if resultado["formato_video"] == "Desconocido" and fallback.get("formato_video") != "Desconocido":
                resultado["formato_video"] = fallback["formato_video"]
            if resultado["duracion_s"] is None and fallback.get("duracion_s") is not None:
                resultado["duracion_s"] = fallback["duracion_s"]
            if not resultado["audio"]:
                resultado["audio"] = fallback.get("audio", [])
            if not resultado["subtitulos"]:
                resultado["subtitulos"] = fallback.get("subtitulos", [])

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
_FFMPEG_CUDA_CACHE: dict[str, bool] = {}
_FFMPEG_CUDA_CACHE_LOCK = threading.Lock()

def _ffmpeg_tiene_cuda(ffmpeg_path: str) -> bool:
    """Devuelve True si la build actual de ffmpeg anuncia soporte CUDA/NVDEC."""
    with _FFMPEG_CUDA_CACHE_LOCK:
        if ffmpeg_path in _FFMPEG_CUDA_CACHE:
            return _FFMPEG_CUDA_CACHE[ffmpeg_path]

    disponible = False
    try:
        proc = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-hwaccels"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="ignore",
        )
        salida = f"{proc.stdout}\n{proc.stderr}"
        disponible = re.search(r"(?im)^\s*cuda\s*$", salida) is not None
    except Exception:
        disponible = False

    with _FFMPEG_CUDA_CACHE_LOCK:
        _FFMPEG_CUDA_CACHE[ffmpeg_path] = disponible
    return disponible

def _calcular_plan_muestreo_capturas(porcentajes: list[float], duracion_s: float) -> Optional[dict]:
    """
    Calcula una ventana útil y la tasa de muestreo exacta para obtener N capturas
    en una sola pasada sobre el video.
    """
    if not porcentajes or duracion_s <= 0:
        return None

    porcentajes_ordenados = sorted(max(0.0, min(100.0, float(p))) for p in porcentajes)
    total_capturas = len(porcentajes_ordenados)
    inicio_s = max(0.0, duracion_s * (porcentajes_ordenados[0] / 100.0))
    fin_s = min(duracion_s, duracion_s * (porcentajes_ordenados[-1] / 100.0))
    ventana_s = max(0.0, fin_s - inicio_s)
    if total_capturas <= 1 or ventana_s <= 0.001:
        return None

    # Una captura por subventana temporal: se adapta automáticamente a 20 min, 1 h o una película larga.
    fps = total_capturas / ventana_s
    timeout_s = int(max(180, min(1800, (ventana_s * 0.08) + 90)))
    return {
        "inicio_s": inicio_s,
        "fin_s": fin_s,
        "ventana_s": ventana_s,
        "fps": fps,
        "total_capturas": total_capturas,
        "intervalo_s": ventana_s / total_capturas,
        "timeout_s": timeout_s,
    }

def _listar_capturas_generadas(
    carpeta_destino: Path,
    nombre_base_capturas: str,
    indice_archivo: int,
    formato_ext: str,
) -> list[Path]:
    prefijo = f"{nombre_base_capturas}[{indice_archivo}]_Captura["
    sufijo = f"].{formato_ext}"
    patron = re.compile(
        rf"^{re.escape(prefijo)}(\d+){re.escape(sufijo)}$",
        re.IGNORECASE,
    )
    encontradas: list[tuple[int, Path]] = []
    for path in carpeta_destino.iterdir():
        if not path.is_file():
            continue
        match = patron.match(path.name)
        if match:
            encontradas.append((int(match.group(1)), path))
    encontradas.sort(key=lambda item: item[0])
    return [path for _, path in encontradas]

def _limpiar_capturas_generadas(
    carpeta_destino: Path,
    nombre_base_capturas: str,
    indice_archivo: int,
    formato_ext: str,
) -> None:
    for path in _listar_capturas_generadas(carpeta_destino, nombre_base_capturas, indice_archivo, formato_ext):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except Exception as exc:
            log.debug(f"No se pudo borrar captura parcial '{path.name}': {exc}")

def _construir_comando_capturas_one_pass(
    ffmpeg_path: str,
    ruta_video: Path,
    ruta_salida_patron: Path,
    plan_muestreo: dict,
    formato_ext: str,
    usar_cuda: bool,
) -> list[str]:
    filtros = [
        f"trim=start={plan_muestreo['inicio_s']:.6f}:end={plan_muestreo['fin_s']:.6f}",
        f"fps={plan_muestreo['fps']:.12f}",
    ]
    if usar_cuda:
        filtros.extend(["hwdownload", "format=nv12"])

    cmd = [ffmpeg_path, "-hide_banner", "-loglevel", "error"]
    if usar_cuda:
        cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])

    cmd.extend([
        "-i", str(ruta_video),
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        "-vf", ",".join(filtros),
        "-frames:v", str(plan_muestreo["total_capturas"]),
        "-start_number", "1",
    ])

    if formato_ext == "png":
        cmd.extend(["-pix_fmt", "rgb24", "-compression_level", "0"])
    else:
        cmd.extend(["-q:v", "0", "-pix_fmt", "yuvj444p"])

    cmd.extend(["-y", str(ruta_salida_patron)])
    return cmd

def _ejecutar_capturas_one_pass(
    ffmpeg_path: str,
    ruta_video: Path,
    nombre_base_capturas: str,
    carpeta_destino: Path,
    indice_archivo: int,
    formato_ext: str,
    plan_muestreo: dict,
) -> tuple[list[Path], Optional[str], Optional[str]]:
    ruta_salida_patron = carpeta_destino / (
        f"{nombre_base_capturas.replace('%', '%%')}[{indice_archivo}]_Captura[%02d].{formato_ext}"
    )
    modos = ["cpu"]
    if _ffmpeg_tiene_cuda(ffmpeg_path):
        modos.insert(0, "cuda")

    errores: list[str] = []
    for modo in modos:
        usar_cuda = modo == "cuda"
        _limpiar_capturas_generadas(carpeta_destino, nombre_base_capturas, indice_archivo, formato_ext)
        cmd = _construir_comando_capturas_one_pass(
            ffmpeg_path,
            ruta_video,
            ruta_salida_patron,
            plan_muestreo,
            formato_ext,
            usar_cuda,
        )
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=plan_muestreo["timeout_s"],
                encoding="utf-8",
                errors="ignore",
            )
            capturas = _listar_capturas_generadas(
                carpeta_destino,
                nombre_base_capturas,
                indice_archivo,
                formato_ext,
            )
            if len(capturas) == plan_muestreo["total_capturas"]:
                return capturas, modo.upper(), None
            errores.append(
                f"{modo.upper()}: generó {len(capturas)}/{plan_muestreo['total_capturas']} capturas."
            )
        except subprocess.CalledProcessError as exc:
            error_text = (exc.stderr or exc.stdout or "Sin salida de error específica.").strip()
            errores.append(f"{modo.upper()}: {error_text}")
        except subprocess.TimeoutExpired:
            errores.append(f"{modo.upper()}: timeout tras {plan_muestreo['timeout_s']} s")
        except Exception as exc:
            errores.append(f"{modo.upper()}: error interno {exc}")

    _limpiar_capturas_generadas(carpeta_destino, nombre_base_capturas, indice_archivo, formato_ext)
    return [], None, " | ".join(errores) if errores else "Fallo desconocido en extracción one-pass."

def _construir_comando_captura_seek(
    ffmpeg_path: str,
    ruta_video: Path,
    ruta_captura: Path,
    tiempo: float,
    formato_ext: str,
    usar_cuda: bool,
) -> list[str]:
    cmd = [ffmpeg_path, "-hide_banner", "-loglevel", "error"]
    if usar_cuda:
        # Mantener solo la aceleración de decodificación evita problemas de compatibilidad
        # con ciertos flujos HEVC Main 10 al descargar frames desde GPU.
        cmd.extend(["-hwaccel", "cuda"])

    cmd.extend([
        "-ss", str(tiempo), "-i", str(ruta_video),
        "-map", "0:v:0", "-an", "-sn", "-dn",
    ])

    if formato_ext == "png":
        cmd.extend(["-vframes", "1", "-pix_fmt", "rgb24", "-compression_level", "0"])
    else:
        cmd.extend(["-vframes", "1", "-q:v", "0", "-pix_fmt", "yuvj444p"])

    cmd.extend(["-y", str(ruta_captura)])
    return cmd

def _generar_capturas_por_seek_individual(
    ffmpeg_path: str,
    ruta_video: Path,
    nombre_base_capturas: str,
    porcentajes: list[float],
    formato_ext: str,
    formato_label: str,
    carpeta_destino: Path,
    indice_archivo: int,
    duracion_s: float,
    progress: Progress,
    task_id,
    nombre_display: str,
    workers_capturas: int,
) -> list[Path]:
    """Ruta de compatibilidad: una ejecución de ffmpeg por captura."""
    workers_capturas = max(1, min(workers_capturas or 1, len(porcentajes)))
    cuda_disponible = _ffmpeg_tiene_cuda(ffmpeg_path)
    cuda_deshabilitada = threading.Event()
    if workers_capturas > 1:
        log.debug(f"Capturas fallback en paralelo ({workers_capturas} workers) para '{nombre_display}'")
    if cuda_disponible:
        log.debug(f"Capturas por seek con CUDA habilitado para '{nombre_display}'")

    def _run_cmd(cmd: list[str], timeout_s: int):
        return subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            errors="ignore",
        )

    def _ejecutar_captura(idx: int, pct: float, forzar_cpu: bool = False):
        tiempo = max(0.1, duracion_s * (pct / 100.0))
        nombre_captura = f"{nombre_base_capturas}[{indice_archivo}]_Captura[{idx:02d}].{formato_ext}"
        ruta_captura = carpeta_destino / nombre_captura
        usar_cuda = cuda_disponible and not forzar_cpu and not cuda_deshabilitada.is_set()
        cmd = _construir_comando_captura_seek(
            ffmpeg_path,
            ruta_video,
            ruta_captura,
            tiempo,
            formato_ext,
            usar_cuda,
        )
        try:
            _run_cmd(cmd, 45)
            if ruta_captura.exists():
                return idx, pct, ruta_captura, None, None
            return idx, pct, ruta_captura, "warn", f"ffmpeg OK, pero falta archivo captura {formato_label}: {nombre_captura}"
        except subprocess.CalledProcessError as exc:
            if usar_cuda:
                cuda_deshabilitada.set()
                try:
                    if ruta_captura.exists():
                        ruta_captura.unlink()
                except Exception:
                    pass
                log.debug(
                    f"Fallo CUDA en captura {formato_label} {idx}@{pct}% para '{nombre_display}'. "
                    "Reintentando en CPU."
                )
                return _ejecutar_captura(idx, pct, forzar_cpu=True)
            error_text = (exc.stderr or exc.stdout or "Sin salida de error especifica.").strip()
            return idx, pct, None, "error", f"Error ffmpeg (captura {formato_label} {idx}@{pct}%): {error_text}"
        except subprocess.TimeoutExpired:
            if usar_cuda:
                cuda_deshabilitada.set()
                try:
                    if ruta_captura.exists():
                        ruta_captura.unlink()
                except Exception:
                    pass
                return _ejecutar_captura(idx, pct, forzar_cpu=True)
            return idx, pct, None, "warn", f"Timeout ffmpeg (captura {formato_label} {idx}@{pct}%)"
        except Exception as exc:
            if usar_cuda:
                cuda_deshabilitada.set()
                try:
                    if ruta_captura.exists():
                        ruta_captura.unlink()
                except Exception:
                    pass
                return _ejecutar_captura(idx, pct, forzar_cpu=True)
            return idx, pct, None, "error", f"Error inesperado interno en ffmpeg (captura {formato_label} {idx}): {exc}"

    completadas = 0
    resultados = [None] * len(porcentajes)
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
            progress.update(task_id, advance=1, description=f"[cyan]{nombre_display}[/] [yellow]- Captura {completadas}/{len(porcentajes)}[/yellow]")
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
                except Exception as exc:
                    idx, pct = idx_ref, pct_ref
                    ruta_captura = None
                    nivel = "error"
                    mensaje = f"Error inesperado interno en ffmpeg (captura {formato_label} {idx}): {exc}"

                completadas += 1
                if nivel == "warn":
                    log.warning(mensaje)
                elif nivel == "error":
                    log.error(mensaje)
                else:
                    resultados[idx - 1] = ruta_captura
                    if ruta_captura:
                        log.debug(f"Captura {formato_label} generada: {ruta_captura.name}")
                progress.update(task_id, advance=1, description=f"[cyan]{nombre_display}[/] [yellow]- Captura {completadas}/{len(porcentajes)}[/yellow]")

    return [captura for captura in resultados if captura]

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
    """Genera capturas de pantalla priorizando una sola pasada de ffmpeg por video."""
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

    plan_muestreo = _calcular_plan_muestreo_capturas(porcentajes, duracion_s)
    usar_one_pass = bool(
        plan_muestreo and plan_muestreo["intervalo_s"] <= CAPTURAS_ONE_PASS_MAX_INTERVAL_S
    )
    if usar_one_pass:
        log.info(
            f"Capturas {formato_label} one-pass para '{nombre_display}': "
            f"{total_capturas} imágenes, intervalo medio ~{plan_muestreo['intervalo_s']:.2f}s."
        )
        progress.update(
            task_id,
            completed=0,
            total=total_capturas,
            description=f"[cyan]{nombre_display}[/] [yellow]- Capturas {formato_label} one-pass...[/yellow]",
        )
        capturas_generadas, modo_usado, error_one_pass = _ejecutar_capturas_one_pass(
            ffmpeg_path,
            ruta_video,
            nombre_base_capturas,
            carpeta_destino,
            indice_archivo,
            formato_ext,
            plan_muestreo,
        )
        if capturas_generadas:
            log.info(
                f"Generadas {len(capturas_generadas)}/{total_capturas} capturas {formato_label} "
                f"para '{nombre_display}' usando {modo_usado}."
            )
            progress.update(
                task_id,
                completed=total_capturas,
                description=f"[cyan]{nombre_display}[/] [green]- Capturas OK ({modo_usado})[/green]",
            )
            return capturas_generadas

        log.warning(
            f"Extracción one-pass falló para '{nombre_display}'. "
            f"Se usará fallback por seek individual. Detalle: {error_one_pass}"
        )
    elif plan_muestreo:
        log.info(
            f"Capturas {formato_label} por seek para '{nombre_display}': "
            f"intervalo medio ~{plan_muestreo['intervalo_s']:.2f}s, por encima del umbral one-pass."
        )

    workers_capturas = max(1, min(workers_capturas or 1, total_capturas))
    progress.update(
        task_id,
        completed=0,
        total=total_capturas,
        description=f"[cyan]{nombre_display}[/] [yellow]- Capturas {formato_label} fallback...[/yellow]",
    )
    capturas_generadas = _generar_capturas_por_seek_individual(
        ffmpeg_path,
        ruta_video,
        nombre_base_capturas,
        porcentajes,
        formato_ext,
        formato_label,
        carpeta_destino,
        indice_archivo,
        duracion_s,
        progress,
        task_id,
        nombre_display,
        workers_capturas,
    )
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
        obfuscated_name = obfuscate_rar_basename(video_file.stem)
        rar_filename = f"{obfuscated_name}{RAR_FILENAME_SUFFIX}.rar"
        rar_filepath = rars_dir / rar_filename
        rar_internal_folder = f"{obfuscated_name}_{secrets.token_hex(3).upper()}"
        nombre_display = display_name_until_parenthesis(video_file.name)

        # Actualiza la descripción para mostrar qué archivo se está procesando AHORA
        progress.update(task_id, description=f"[magenta]RAR {idx}/{total_files}:[/] [cyan]{nombre_display}[/]")

        size_bytes = video_file.stat().st_size
        split_config = estimar_plan_volumenes_rar(size_bytes)
        timeout_compresion = 6 * 60 * 60  # 6 horas

        def build_rar_command(active_split_config: dict | None) -> list[str]:
            cmd = [rar_exe_path, "a", "-ma5"]
            if rar_store_only:
                cmd.append("-m0")  # Sin compresión (solo almacenar)
            else:
                cmd.append("-m3")  # Compresión normal
            if rar_password:
                cmd.append(f"-hp{rar_password}")  # Cifra datos y nombres
            if active_split_config:
                cmd.extend(active_split_config["volume_switches"])
            cmd.extend([
                "-ep1", f"-ap{rar_internal_folder}", "-o+",
                str(rar_filepath.resolve()),
                str(video_file.resolve())
            ])
            return cmd

        try:
            action_verb = "Almacenando" if rar_store_only else "Comprimiendo"
            split_msg = split_config["split_msg"] if split_config else ""
            existing_volumes = obtener_volumenes_rar_generados(rar_filepath)
            if existing_volumes:
                log.debug(
                    f"Limpiando volúmenes RAR previos para '{nombre_display}': "
                    f"{', '.join(path.name for path in existing_volumes)}"
                )
                eliminar_volumenes_rar(existing_volumes)
            log.info(f"{action_verb} [{idx}/{total_files}] '{nombre_display}' -> '{rar_filename}'{split_msg}")

            generated_volumes: list[Path] = []
            for retry_index in range(RAR_SPLIT_RETRY_LIMIT + 1):
                cmd = build_rar_command(split_config)
                cmd_log_safe = ["-hp********" if part.startswith("-hp") else part for part in cmd]
                if retry_index == 0:
                    log.debug(f"Ejecutando RAR: {' '.join(cmd_log_safe)}")
                else:
                    log.debug(f"Reintentando RAR ajustado (intento {retry_index + 1}): {' '.join(cmd_log_safe)}")

                ejecutar_comando_rar(cmd, idx, total_files, nombre_display, progress, task_id, timeout_compresion)
                generated_volumes = obtener_volumenes_rar_generados(rar_filepath)
                if not requiere_reajuste_split(generated_volumes, split_config):
                    break

                if retry_index >= RAR_SPLIT_RETRY_LIMIT:
                    log.warning(
                        f"Se alcanzó el límite de reajustes para '{nombre_display}'. "
                        f"Se conservarán los volúmenes generados en el último intento.")
                    break

                total_archive_bytes = sum(path.stat().st_size for path in generated_volumes if path.exists())
                last_volume_size = generated_volumes[-1].stat().st_size
                retry_padding_bytes = max(
                    RAR_SPLIT_OVERHEAD_PADDING_BYTES,
                    last_volume_size + 1024)
                retry_split_config = estimar_plan_volumenes_rar(
                    total_archive_bytes,
                    extra_padding_bytes=retry_padding_bytes)
                if not retry_split_config:
                    log.warning(
                        f"No se pudo recalcular un reparto mejor para '{nombre_display}'. "
                        f"Se conservarán los volúmenes actuales.")
                    break

                log.warning(
                    f"RAR generó un volumen final mínimo ({formatear_peso(last_volume_size)}) para '{nombre_display}'. "
                    f"Se reintentará con un reparto coherente para absorber ese peso residual.")
                eliminar_volumenes_rar(generated_volumes)
                split_config = retry_split_config
                split_msg = f"{split_config['split_msg']} (reajustado)"

            progress.update(task_id, completed=idx)
            log.info(
                f"Creación RAR [{idx}/{total_files}] exitosa: '{rar_filename}' "
                f"(carpeta interna: '{rar_internal_folder}')")
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
        "nombre_rar": None,
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
        resultado_video["nombre_rar"] = f"{obfuscate_rar_basename(ruta_video.stem)}{RAR_FILENAME_SUFFIX}"

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
def procesar_carpeta(carpeta_path: Path, args: argparse.Namespace, progress: Progress,
                     archivos_video_override: list[Path] | None = None,
                     nombre_carpeta_override: str | None = None):
    """Procesa una carpeta: busca videos, los procesa en paralelo, muestra info y comprime."""
    nombre_carpeta = nombre_carpeta_override or carpeta_path.name
    log.info(f"Iniciando procesamiento de carpeta: {nombre_carpeta}")
    console.print(Panel(f"[head]Carpeta:[/head] [path]{nombre_carpeta}[/path]", expand=False, border_style="blue"))

    if archivos_video_override is None:
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
    else:
        archivos_video = sorted([
            p.resolve() for p in archivos_video_override
            if p.is_file() and p.suffix.lower() in args.exts
               and NOMBRE_CARPETA_CAPTURAS not in p.parts
        ])

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
        audio_plain = strip_rich_markup(audio_str)
        subtitulos_plain = strip_rich_markup(subtitulos_str)
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
            f"[info]RAR:[/info]     [detail]{res.get('nombre_rar', 'N/A')}[/detail]\n"
            f"[info]Limpio:[/info]  [detail]{res.get('nombre_limpio', 'N/A')}[/detail]\n"
            f"[info]Resolucion.:[/info]  {info.get('resolucion', 'N/A')} ({info.get('formato_video', 'N/A')})\n"
            f"[info]Duracion.:[/info]  {duracion_str}\n"
            f"[info]Peso:[/info]    {res.get('peso_str', 'N/A')}\n"
            f"[info]Audio:[/info]   {audio_str}\n"
            f"[info]Subtitulos:[/info]   {subtitulos_str}\n"
            f"[info]Capturas:[/info]    {capturas_str}"
        )
        console.print(Panel(info_panel, title=f"[cyan]{res.get('nombre_original', 'N/A')}[/cyan]", border_style="cyan", expand=False))
        print(
            "MWO_RAW_DATA\t"
            f"{res.get('nombre_original', 'N/A')}\t"
            f"{info.get('resolucion', 'N/A')}\t"
            f"{res.get('peso_str', 'N/A')}\t"
            f"{audio_plain}\t"
            f"{subtitulos_plain}\t"
            f"{res.get('nombre_limpio', 'N/A')}\t"
            f"{res.get('nombre_rar', 'N/A')}"
        )
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
                    help="Directorio base que contiene las subcarpetas a procesar, o un archivo de video directo.")
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
    input_video_path: Path | None = None
    if base_path.is_file():
        if base_path.suffix.lower() not in args.exts:
            log.critical(f"El archivo especificado no es un video compatible: {base_path}")
            console.print(f"[fail]Error:[/fail] Archivo de video no válido: [path]{base_path}[/path]")
            sys.exit(1)
        input_video_path = base_path
    elif not base_path.is_dir():
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

    input_label = "Archivo Base" if input_video_path else "Directorio Base"
    console.print(Panel(f"[head]== INICIO DEL PROCESO ==[/head]\n"
                      f"{input_label}: [path]{base_path}[/path]\n"
                      f"Workers: {args.workers} | Capturas: {'Sí' if not args.skip_img else 'No'} | Compresión: {'Sí' if args.compress else 'No'}",
                      expand=False, border_style="magenta"))
    start_time = datetime.now()

    subcarpetas: list[Path] = []
    videos_directos_base: list[Path] = []
    if input_video_path is None:
        # --- Búsqueda de contenido directo ---
        try:
            elementos_directos = list(base_path.iterdir())
            subcarpetas = sorted([d for d in elementos_directos if d.is_dir()])
            videos_directos_base = sorted([
                p.resolve() for p in elementos_directos
                if p.is_file()
                   and p.suffix.lower() in args.exts
                   and NOMBRE_CARPETA_CAPTURAS not in p.parts
            ])
        except Exception as e:
            log.critical(f"Error listando contenido en '{base_path}': {e}", exc_info=True)
            console.print(f"[fail]Error fatal:[/fail] No se pudo listar el contenido de '{base_path}'.")
            sys.exit(1)

        if not subcarpetas and not videos_directos_base:
            log.warning(f"No se encontraron subcarpetas ni videos directos en '{base_path}'.")
            console.print(f"[warn]Advertencia:[/warn] No se encontraron subcarpetas ni videos directos para procesar en [path]{base_path}[/path].")
            sys.exit(0)

        log.info(
            f"Encontradas {len(subcarpetas)} subcarpetas y {len(videos_directos_base)} video(s) directos en '{base_path}'."
        )
    else:
        log.info(f"Procesando archivo individual como entrada directa: {input_video_path}")

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
        total_contextos = 1 if input_video_path else len(subcarpetas) + (1 if videos_directos_base else 0)
        task_overall = progress.add_task(f"[magenta]Progreso General Carpetas", total=total_contextos)

        if input_video_path:
            try:
                videos_ok, peso_carpeta = procesar_carpeta(
                    input_video_path.parent,
                    args,
                    progress,
                    archivos_video_override=[input_video_path],
                    nombre_carpeta_override=input_video_path.stem)
                stats["total_carpetas"] += 1
                if videos_ok > 0:
                    stats["carpetas_con_videos"] += 1
                    stats["videos_procesados_ok"] += videos_ok
                    stats["peso_total_bytes"] += peso_carpeta
            except Exception as e:
                log.critical(f"Error fatal no controlado procesando archivo '{input_video_path.name}': {e}", exc_info=True)
                console.print(f"[fail]Error Inesperado:[/fail] Fallo procesando el archivo '{input_video_path.name}'. Ver logs.")
            finally:
                progress.update(task_overall, advance=1)
        else:
            if videos_directos_base:
                try:
                    nombre_directos = videos_directos_base[0].stem if len(videos_directos_base) == 1 else f"{base_path.name} (directos)"
                    videos_ok, peso_carpeta = procesar_carpeta(
                        base_path,
                        args,
                        progress,
                        archivos_video_override=videos_directos_base,
                        nombre_carpeta_override=nombre_directos)
                    stats["total_carpetas"] += 1
                    if videos_ok > 0:
                        stats["carpetas_con_videos"] += 1
                        stats["videos_procesados_ok"] += videos_ok
                        stats["peso_total_bytes"] += peso_carpeta
                except Exception as e:
                    log.critical(f"Error fatal no controlado procesando videos directos en '{base_path}': {e}", exc_info=True)
                    console.print(f"[fail]Error Inesperado:[/fail] Fallo procesando los videos directos en '{base_path.name}'. Ver logs.")
                finally:
                    progress.update(task_overall, advance=1)

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
        peso_total_str = "N/A"
        peso_promedio_str = "N/A"
        summary_table.add_row("Peso total de videos procesados", "[detail]N/A[/detail]")
        summary_table.add_row("Peso promedio por video", "[detail]N/A[/detail]")

    if args.logfile:
        summary_table.add_row("Log detallado guardado en", f"[path]{Path(args.logfile).resolve()}[/path]")
    else:
        summary_table.add_row("Log detallado", "[info]No guardado (solo consola)[/info]")

    console.print(summary_table)
    print(f"MWO_WEIGHT_SUMMARY\t{peso_total_str}\t{peso_promedio_str}")
    log.info("--- Script Finalizado ---")

    # Opcional: Guardar salida de consola si se grabó
    # console.save_html("processing_summary.html")


if __name__ == "__main__":
    
    main()
