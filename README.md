# ⚡ Capi-Carpi: motor de inferencia SSM + MoE ternario en C++17 puro

**🇪🇸 Español** · [🇬🇧 English](README-English.md)

![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=c%2B%2B)
![AVX2 SIMD](https://img.shields.io/badge/SIMD-AVX2%20%7C%20FMA-FF6F00?style=for-the-badge)
![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-2E7D32?style=for-the-badge)
![Python ctypes](https://img.shields.io/badge/FFI-Python%20ctypes-3776AB?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**Capi-Carpi** es un motor de inferencia de bajo nivel escrito desde cero en **C++17**, sin PyTorch, LibTorch, ONNX ni BLAS. Ejecuta una arquitectura híbrida de **State Space Models (SSM)** con **Sparse Mixture of Experts (MoE)** y pesos ternarios de **1.58 bits** `{-1, 0, +1}`, usando intrínsecos **AVX2/FMA**, paralelismo **OpenMP** y carga **zero-copy con `mmap`**, todo sobre una CPU de consumo.

| | |
|---|---|
| **Rendimiento** | 236.29 tokens/s (4.23 ms/token) en CPU, sobre un modelo de 23.07M de parámetros |
| **Memoria** | Pesos mapeados con mmap en 26.79 ms; 4 pesos ternarios por byte |
| **Dependencias** | Ninguna. Solo compilador C++17 con soporte AVX2 |
| **Hardware de desarrollo** | Ryzen 7 5700G (16 hilos), 32 GB DDR4, sin GPU, SSD SATA III |

---

## Por qué existe este proyecto

Empecé con una pregunta ambiciosa: **¿se puede entrenar y ejecutar un modelo de escala 1T en una PC doméstica sin GPU?** El hardware: 32 GB de RAM DDR4, 0 GB de VRAM y un SSD ADATA SU650 SATA III de 960 GB (~600 MB/s).

Para intentarlo combiné las técnicas más agresivas que encontré en la literatura:

1. **State Space Models:** recurrencia lineal con estado de tamaño constante por token, en lugar de atención cuadrática $O(N^2)$.
2. **Cuantización ternaria (1.58 bits):** 4 pesos por byte (uint8_t), un 93.75 % menos de memoria que FP32 y sin multiplicaciones en punto flotante en los expertos.
3. **Sparse MoE sobre disco:** ruteo Top-K que activa solo una fracción de los expertos, residentes en el SSD vía mmap.
4. **Entrenamiento zeroth-order (MeZO y una variante propia):** entrenar sin backpropagation para no guardar activaciones.

El resultado fue doble. **El motor de inferencia funciona, es estable y es rápido cuando el modelo cabe en RAM.** Y **las cuentas del I/O demostraron que el objetivo original (1T en este hardware) es físicamente inviable**. Este repositorio publica lo primero y documenta con números lo segundo.

---

## Arquitectura

```text
[Token ID] ──► [Embedding lookup sobre mmap (zero-copy)]
                          │
                          ▼
        [Router Top-K MoE (AVX2)] ──► Softmax vectorizado
                          │          (reducción de rango IEEE-754 + minimax grado 6)
              ┌───────────┴───────────┐
              ▼                       ▼
     [Experto activo 1]      [Experto activo K]
      SSM ternario 1.58-bit   SSM ternario 1.58-bit
      4 pesos/byte, sin       estado recurrente O(1)
      multiplicaciones        por token
              └───────────┬───────────┘
                          ▼
     [Fusión ponderada + clamping] ──► [LM Head Fast-Path AVX2] ──► [Logits]
```

**Topología del benchmark:** `d_model = 512`, `vocab_size = 16,384`, 8 expertos SSM con Top-2 activos por token. Total: 23.07M de parámetros, 9.96M activos por token, 73.02 MB en disco.

---

## Resultados de rendimiento

Inferencia autorregresiva continua de 200 tokens (`benchmark_capi.cpp`):

| Versión | Carga `mmap` | Latencia / token | Throughput | Tiempo (200 tok) | Speedup |
|---|---|---|---|---|---|
| v1.0 (baseline AVX2 + transposición por llamada) | 33.15 ms | 84.00 ms | 11.90 tok/s | 16.7999 s | 1.00x |
| v1.1 (Fast-Path `M=1`, zero-allocation) | 26.79 ms | **4.23 ms** | **236.29 tok/s** | 0.8464 s | **19.86x** |

```text
=================================================================
  CAPI-CARPI AVX2 INFERENCE ENGINE - BENCHMARK DE RENDIMIENTO
  Arquitectura: State Space Model (SSM) + Sparse MoE (1.58-bit)
=================================================================

[ENGINE] Header verificado | d_model: 512 | vocab: 16384 | experts: 8 (top-2)
[OK] Mapeo Zero-Copy (mmap) completado en: 26.79 ms
[>>] Ejecutando inferencia continua de 200 tokens...

-----------------------------------------------------------------
  RESULTADOS DEL BENCHMARK (CPU AVX2 + FMA + OpenMP)
-----------------------------------------------------------------
  Tokens procesados : 200
  Tiempo total      : 0.8464 s
  Latencia / Token  : 4.23 ms/tok
  Throughput        : 236.29 tokens/seg
=================================================================
```

> **Cómo leer estos números.** El benchmark corre sobre un modelo pequeño con pesos generados por `model_builder` (no un modelo entrenado). En inferencia determinista el throughput depende de las dimensiones tensoriales y de las instrucciones SIMD ejecutadas, no del valor de los pesos, por eso sirve para medir el motor, pero **no es una comparación contra modelos de lenguaje reales**. El speedup de 19.86x viene de eliminar una asignación y transposición de 33.55 MB en cada llamada al LM Head (ver bug #4), es decir, de corregir un cuello de botella evitable de la v1.0.

---

## El muro de I/O: por qué 1T no entra en esta máquina

Cuando el modelo activo entra en los 32 GB de RAM (~35 000 MB/s de ancho de banda) el motor es rápido. Cuando no entra, `mmap` pagina desde el SSD y todo cambia:

| Escenario | Cálculo | Resultado |
|---|---|---|
| Inferencia con expertos fuera de RAM | 3 GB de pesos activos por token a 600 MB/s | ≈ **5 s/token** de piso físico, sin importar cuán optimizado esté el código AVX2 |
| Entrenamiento MeZO sobre un binario de 17 GB | 2 pasadas forward (θ+εz, θ−εz) + mutación de pesos por lote | ≈ **56 s por lote** solo de I/O, unos **327 días** para 500 000 iteraciones |
| Desgaste del SSD | Escrituras completas repetidas sobre las celdas NAND | Agotamiento prematuro del TBW |

La brecha de ancho de banda es de **~58x** entre RAM DDR4 y SSD SATA III, y de **~3 000x** frente a la memoria HBM de una GPU. Concluí que el cuello de botella está en el bus de almacenamiento y no en el software.

---

## ️ Cuatro bugs de bajo nivel resueltos

Al no usar frameworks, tuve que diagnosticar fallos que normalmente quedan ocultos detrás de ellos.

### 1. `alignas(32)` inflaba 4x el salto de punteros (`src/tensor/ternary_tensor.h`)
`TernaryWeightBlock` empaqueta 32 pesos de 2 bits en **8 bytes**, pero tenía `alignas(32)`. Como `sizeof(T)` debe ser múltiplo de `alignof(T)`, el compilador lo infló silenciosamente a **32 bytes** (24 de padding). Toda la aritmética de punteros avanzaba 32 bytes donde el generador había escrito 8: el SSM leía 16 MB donde había 4 MB e invadía la memoria del experto siguiente. **Solución:** eliminar el `alignas` y blindar la estructura con `static_assert(sizeof(TernaryWeightBlock) == 8)`.

### 2. Divergencia del polinomio de Taylor en el softmax del router (`src/attention/softmax.cpp`)
La primera versión vectorizada de `exp` usaba un polinomio de Taylor de grado 4 alrededor de 0, sin reducción de rango. Para $x = -10$ el polinomio da **+291.0** en lugar de **0.0000454**, así que el router MoE asignaba la mayor probabilidad a los peores expertos. **Solución:** reducción de rango IEEE-754 ($x = n \ln 2 + g$), minimax de grado 6 con FMA (`_mm256_fmadd_ps`) y reconstrucción del exponente desplazando 23 bits (`_mm256_slli_epi32`). Error relativo **< 3 × 10⁻⁷**.

### 3. Inanición de hilos en OpenMP con `M = 1` (`src/avx/avx_ops.cpp`)
Paralelizar solo el bucle externo `for (i < M)` de la multiplicación matricial deja **un único hilo trabajando** durante la inferencia token a token (`M = 1`), mientras el resto de los núcleos queda inactivo. Además, OpenMP 2.0 (el que ofrece, por ejemplo, MSVC) no soporta `collapse(2)`. **Solución:** aplanar manualmente el espacio `M × N` en un solo bucle y crear un Fast-Path paralelo sobre las `N` columnas.

### 4. Cache thrashing por transposición dinámica (**19.86x**) (`src/avx/avx_ops.cpp`)
En la v1.0, cada llamada a `matrix_multiply` del LM Head reservaba **33.55 MB** con `_mm_malloc` y transponía la matriz completa en cada token, con millones de cache misses. **Solución:** Fast-Path zero-allocation para `M = 1`. El vector de entrada (2 KB) queda residente en caché L1 mientras los hilos leen filas contiguas directamente desde el `mmap`. Latencia por token: **84.00 ms → 4.23 ms**.

---

## Estructura del repositorio

```text
capi-carpi-inference-engine/
├── src/
│   ├── attention/   # Router Top-K MoE, softmax AVX2 y recurrencia SSM O(1)
│   ├── avx/         # Kernels SIMD AVX2/FMA (FP32 Fast-Path y ternario 1.58-bit)
│   ├── model/       # Orquestador CapiModel, buffers alineados (_mm_malloc) y LM Head
│   ├── tensor/      # Cargador zero-copy (mmap / MapViewOfFile) y empaquetado de 2 bits
│   ├── tools/       # Generador CLI de modelos binarios (model_builder) y verificación
│   └── utils/       # Transposición matricial multihilo (OpenMP)
├── python/
│   └── gcapi_wrapper.py   # Puente FFI hacia Python mediante ctypes
├── benchmark_capi.cpp     # Benchmark de latencia (ms/token) y throughput (tokens/s)
├── LICENSE
└── README.md
```

---

##  Compilación y reproducción

**Requisitos:** compilador con C++17 y AVX2: `g++` / MinGW-w64 en Windows, o GCC/Clang en Linux.

### Windows (PowerShell)

```powershell
# 1. Carpeta de compilación
New-Item -ItemType Directory -Force -Path build | Out-Null

# 2. Generador de modelos
g++ -O3 -std=c++17 src/tools/model_builder.cpp -o build/model_builder.exe

# 3. Generar un modelo binario de prueba (73 MB, 23M parámetros)
./build/model_builder.exe --d-model 512 --vocab 16384 --experts 8 --top-k 2 --out capi_test.bin

# 4. Compilar el motor y el benchmark (AVX2 + FMA + OpenMP)
g++ -O3 -std=c++17 -mavx2 -mfma -fopenmp `
    benchmark_capi.cpp `
    src/tensor/mmap_loader.cpp `
    src/utils/transpose.cpp `
    src/avx/avx_ops.cpp `
    src/avx/ternary_ops.cpp `
    src/attention/softmax.cpp `
    src/attention/moe_router.cpp `
    src/attention/ssm_memory.cpp `
    src/attention/ssm_forward.cpp `
    src/model/capi_model.cpp `
    -o build/benchmark_capi.exe

# 5. Benchmark de 200 pasos autorregresivos
./build/benchmark_capi.exe capi_test.bin 200
```

### Linux / macOS con GCC (bash)

> Los comandos de Linux siguen el mismo esquema; el cargador `mmap` usa la API POSIX en lugar de `MapViewOfFile`.

```bash
mkdir -p build
g++ -O3 -std=c++17 src/tools/model_builder.cpp -o build/model_builder
./build/model_builder --d-model 512 --vocab 16384 --experts 8 --top-k 2 --out capi_test.bin

g++ -O3 -std=c++17 -mavx2 -mfma -fopenmp \
    benchmark_capi.cpp \
    src/tensor/mmap_loader.cpp src/utils/transpose.cpp \
    src/avx/avx_ops.cpp src/avx/ternary_ops.cpp \
    src/attention/softmax.cpp src/attention/moe_router.cpp \
    src/attention/ssm_memory.cpp src/attention/ssm_forward.cpp \
    src/model/capi_model.cpp \
    -o build/benchmark_capi

./build/benchmark_capi capi_test.bin 200
```

### Como librería dinámica para Python

```powershell
g++ -O3 -std=c++17 -mavx2 -mfma -fopenmp -shared `
    src/tensor/mmap_loader.cpp src/utils/transpose.cpp `
    src/avx/avx_ops.cpp src/avx/ternary_ops.cpp `
    src/attention/softmax.cpp src/attention/moe_router.cpp `
    src/attention/ssm_memory.cpp src/attention/ssm_forward.cpp `
    src/model/capi_model.cpp `
    -o build/gcapi_core.dll
```

En Linux usá `-fPIC -shared` y `-o build/libgcapi_core.so`. El puente `python/gcapi_wrapper.py` carga la librería con `ctypes`.

---

## Alcance y limitaciones

- Los benchmarks usan un modelo de 23M de parámetros con pesos generados, no un modelo de lenguaje entrenado.
- El motor está pensado para modelos que **caben en RAM**. Fuera de RAM, el límite es el ancho de banda del almacenamiento (ver sección del muro de I/O).
- Solo CPU x86-64 con AVX2/FMA. No hay backend GPU ni AVX-512.

## Próximos pasos posibles

- Profiling por componente (router, SSM, LM Head) para saber dónde se va cada milisegundo.
- Probar el motor con pesos de un SSM/MoE entrenado de verdad y validar la salida contra una implementación de referencia.
- Mover el LM Head a cuantización de menor precisión.
- Entrenar un modelo pequeño que sí quepa en RAM usando la variante de entrenamiento zeroth-order ya desarrollada.

## Qué demuestra este proyecto

Diseño de un motor de inferencia de punta a punta; vectorización SIMD con intrínsecos; paralelismo con OpenMP; I/O con memoria virtual y `mmap`; análisis de ancho de banda y de límites físicos del hardware; y depuración de bugs de layout de memoria, estabilidad numérica y rendimiento de caché.

---

## Autor

**Thiago Hugo Daniel Norambuena**
Licenciatura en Ciencia de Datos, Universidad Nacional Guillermo Brown (UNaB)
Buenos Aires, Argentina

[LinkedIn](https://www.linkedin.com/in/thiago-norambuena/) · [Email](mailto:thiago.norambuena727@gmail.com)

## Licencia

Distribuido bajo licencia MIT. Ver [`LICENSE`](LICENSE).
