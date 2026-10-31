# Archivo: python/gcapi_wrapper.py
"""Puente FFI (Foreign Function Interface) hacia el motor nativo Capi-Carpi (C++17 / AVX2).

Gestiona la carga de la libreria compartida (.dll / .so), la validacion de la
cabecera binaria del modelo y la exposicion de los tensores de salida hacia
NumPy sin copias intermedias de memoria.
"""

import ctypes
import os
import struct
import sys
from typing import Optional
import numpy as np


class CapiModel:
  """Envoltura de alto nivel sobre la API C de Capi-Carpi."""

  HEADER_FORMAT = "<4i5Q"
  HEADER_SIZE_BYTES = 56

  def __init__(
      self, model_path: str = "capi_test.bin", dll_path: Optional[str] = None
  ) -> None:
    if not os.path.exists(model_path):
      raise FileNotFoundError(
          f"[CapiModel] No se encontró el archivo del modelo: {model_path}"
      )

    self.model_path = os.path.abspath(model_path)
    self._parse_model_header()

    resolved_dll = self._resolve_library_path(dll_path)
    self.lib = self._load_shared_library(resolved_dll)
    self._bind_c_functions()

    if not self.lib.init_model(self.model_path.encode("utf-8")):
      raise RuntimeError(
          f"[CapiModel] El núcleo C++ falló al mapear '{self.model_path}' vía"
          " mmap."
      )

  def _parse_model_header(self) -> None:
    """Lee y valida la estructura ModelHeader (56 bytes) del archivo binario."""
    with open(self.model_path, "rb") as f:
      header_bytes = f.read(self.HEADER_SIZE_BYTES)

    if len(header_bytes) < self.HEADER_SIZE_BYTES:
      raise ValueError(
          f"[CapiModel] Archivo binario truncado ({len(header_bytes)} bytes)."
      )

    unpacked = struct.unpack(self.HEADER_FORMAT, header_bytes)
    self.d_model: int = unpacked[0]
    self.num_experts: int = unpacked[1]
    self.top_k: int = unpacked[2]
    self.vocab_size: int = unpacked[3]
    self.embedding_offset: int = unpacked[4]

    if self.d_model <= 0 or self.vocab_size <= 0 or self.num_experts <= 0:
      raise ValueError(
          "[CapiModel] Dimensiones inválidas detectadas en la cabecera del"
          " modelo."
      )

  @staticmethod
  def _resolve_library_path(custom_path: Optional[str]) -> str:
    """Resuelve la ruta de la librería compilada según el sistema operativo."""
    if custom_path:
      lib_path = os.path.abspath(custom_path)
    else:
      lib_name = "gcapi_core.dll" if os.name == "nt" else "libgcapi_core.so"
      lib_path = os.path.abspath(os.path.join("build", lib_name))

    if not os.path.exists(lib_path):
      raise FileNotFoundError(
          f"[CapiModel] Librería nativa no encontrada en: {lib_path}"
      )
    return lib_path

  @staticmethod
  def _load_shared_library(lib_path: str) -> ctypes.CDLL:
    """Carga la librería dinámica manejando las restricciones de rutas en Windows."""
    if os.name == "nt":
      try:
        return ctypes.CDLL(lib_path, winmode=0)
      except TypeError:
        if hasattr(os, "add_dll_directory"):
          os.add_dll_directory(os.path.dirname(lib_path))
        return ctypes.CDLL(lib_path)
    return ctypes.CDLL(lib_path)

  def _bind_c_functions(self) -> None:
    """Declara los contratos de tipos (ABI) para las funciones exportadas en C++."""
    self.lib.init_model.argtypes = [ctypes.c_char_p]
    self.lib.init_model.restype = ctypes.c_bool

    self.lib.forward_step.argtypes = [ctypes.c_int]
    self.lib.forward_step.restype = ctypes.POINTER(ctypes.c_float)

    self.lib.limpiar_memoria_secuencia.argtypes = []
    self.lib.limpiar_memoria_secuencia.restype = None

    # Enlaces opcionales para rutinas experimentales de perturbación estocástica (MeZO)
    if hasattr(self.lib, "forward_step_mezo"):
      self.lib.forward_step_mezo.argtypes = [
          ctypes.c_int,
          ctypes.c_uint32,
          ctypes.c_float,
      ]
      self.lib.forward_step_mezo.restype = ctypes.POINTER(ctypes.c_float)

  def forward_step(self, token_id: int) -> np.ndarray:
    """Ejecuta un paso autorregresivo en C++ y retorna una vista NumPy de los logits."""
    if token_id < 0 or token_id >= self.vocab_size:
      raise ValueError(
          f"[CapiModel] token_id ({token_id}) fuera del rango del vocabulario"
          f" [0, {self.vocab_size})."
      )

    ptr = self.lib.forward_step(token_id)
    if not ptr:
      raise RuntimeError(
          f"[CapiModel] El motor C++ devolvió un puntero nulo para token_id="
          f"{token_id}."
      )

    return np.ctypeslib.as_array(ptr, shape=(self.vocab_size,))

  def reset_state(self) -> None:
    """Reinicia el estado oculto recurrente (SSM) entre secuencias independientes."""
    self.lib.limpiar_memoria_secuencia()

  def limpiar_memoria_secuencia(self) -> None:
    """Alias mantenido por retrocompatibilidad con scripts previos."""
    self.reset_state()


if __name__ == "__main__":
  test_bin = sys.argv[1] if len(sys.argv) > 1 else "capi_test.bin"
  model = CapiModel(model_path=test_bin)
  print(
      f"[OK] Modelo conectado | d_model={model.d_model} |"
      f" vocab={model.vocab_size} | experts={model.num_experts}"
      f" (top-{model.top_k})"
  )
  logits = model.forward_step(1)
  print(
      f"[OK] Inferencia FFI verificada | Shape={logits.shape} |"
      f" Argmax={int(np.argmax(logits))}"
  )