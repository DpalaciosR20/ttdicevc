# Decisiones de preprocesamiento por técnica (F1.4)

> Tarea F1.4 del plan v2 · Roles: SV + CE · Estado: **criterio fijado; resultados pendientes**
> Las secciones 1 a 5 se versionan con commit propio **antes** de ejecutar cualquier prueba. Las secciones 6 y 7 se completan después.

## 1. Propósito

Cada imagen analítica es una función determinista de los cinco componentes UMAP, $X = G(\mathbf{u})$. Antes de llegar al clasificador, la imagen pasa por un preprocesamiento (redimensionamiento, replicación a tres canales, normalización) y, durante el entrenamiento, por aumentos de datos. Si alguna de esas transformaciones destruye información sobre $\mathbf{u}$, la clasificación visual subestimaría la información conservada por razones ajenas a la técnica generativa; es la amenaza a la validez interna «preprocesamiento que destruye información de **u**» (sección 5 del plan). Esta tarea determina, por técnica, qué preprocesamiento lo conserva.

## 2. Criterio

La decisión se toma **sin etiquetas de clase** y **solo con el conjunto de entrenamiento** (908 muestras, validación cruzada de cinco particiones). El objetivo de las pruebas es $\mathbf{u}$, no la clase. Por construcción, F1.4 no lee imágenes de validación ni de prueba y no consume la información de validación reservada para el *early stopping* y la selección del finalista (F3.3).

La primera pasada del grid (F3.2, preprocesamiento `px224-flip`) ya se ejecutó cuando se fijó este criterio. Sus F1-Val no intervienen en la decisión, y sus resultados se consideran superados por la repetición posterior a esta tarea.

## 3. Transformaciones evaluadas

| Código | Transformación | Determinista |
|---|---|---|
| `V0` | Referencia: 256 × 256 nativo | Sí |
| `V1` | Redimensionamiento a 224 × 224 con los parámetros exactos de `run_grid.py` (interpolación y *antialias*: §6.1) | Sí |
| `V2` | Recorte central 224 × 224 | Sí |
| `F` | Volteo horizontal (aumento de entrenamiento) | No |
| Otras | Cualquier otro aumento presente en `run_grid.py` (§6.1) se evalúa con la prueba P2 | — |

Todas las variantes incluyen la replicación a tres canales y la normalización con la media y desviación de ImageNet, que no alteran la información.

## 4. Pruebas

| Prueba | Método | Estadístico |
|---|---|---|
| **P1. Recuperabilidad de u** | Características de la penúltima capa (512) de ResNet-18 preentrenada en ImageNet y congelada, en modo de evaluación. Regresión *ridge* por componente, con estandarización y α elegido por validación cruzada interna (`RidgeCV`, α ∈ 10^[−3, 3], 13 valores), ajustadas dentro de cada partición. KFold de 5 particiones con `shuffle=True`, `random_state=2026`. R² calculado sobre las predicciones fuera de partición | $\Delta R^2_j(V) = R^2_j(V0) - R^2_j(V)$, para j = 0…4 |
| **P2. Sensibilidad al aumento** | Mismo regresor, entrenado con la variante determinista sin aumentar y evaluado sobre la partición retenida aumentada (volteada) | $\Delta R^2_{j}(F) = R^2_j(V) - R^2_j(F(V))$ |
| **P3. Espectro** | Fracción de energía espectral por encima de 112 ciclos por imagen (nueva frecuencia de Nyquist a 224 px), por imagen, correlacionada con cada $u_j$ (Spearman) | Explicativo |
| **P4. Orientación** | Estadísticos con signo, que cambian de signo con el volteo: desplazamiento horizontal del centroide de intensidad y asimetría (*skewness*) del perfil de intensidad por columnas. Correlación de Spearman con cada $u_j$; Holm sobre 2 estadísticos × 5 componentes × 3 técnicas (30 contrastes), α = 0.05 | ρ y p ajustado |

Para todos los ΔR² se reporta un IC del 95 % por *bootstrap* pareado de las predicciones fuera de partición (B = 2,000, semilla 2026), con fines informativos.

## 5. Regla de decisión

Se aplica por técnica, igual para todas las arquitecturas y para las estrategias `frozen` y `finetune`.

1. **Tamaño.** Se admite una variante determinista V si $\Delta R^2_j(V) \le 0.02$ para los cinco componentes (P1). Orden de preferencia: `V1`, `V2`, `V0`.
2. **ViT.** Si se admite `V1` o `V2`, ViT usa la misma variante. Si solo se admite `V0`, ViT usa 256 px con *positional embeddings* interpolados, siempre que CE confirme que es viable. Si no lo es, ViT usa `V1` y sus resultados para esa técnica se reportan como cota inferior, declarándolo en la matriz de experimentos (F1.3).
3. **Volteo.** Se permite si y solo si $\Delta R^2_j(F) \le 0.02$ para los cinco componentes (P2) **y** ningún contraste de P4 resulta significativo tras Holm. La misma regla se aplica a cualquier otro aumento identificado en §6.1, salvo P4, que solo aplica al volteo.
4. **P3** es explicativa y no decide.
5. Ninguna regla se modifica después de ver los resultados. Toda desviación se reporta como tal.

**Limitación declarada.** P1 mide la información disponible para un lector lineal de características de ImageNet, no toda la información de la imagen; un modelo afinado podría recuperar más. El criterio es, por tanto, conservador respecto de la pérdida.

## 6. Resultados

### 6.1 Preprocesamiento vigente en `run_grid.py`

`run_grid.py` no define transformaciones: importa `make_loaders` de `src/training/data.py` (`run_grid.py:54`, llamada en `run_grid.py:370-373`), que construye la cadena con `get_transforms` (`data.py:43-52`). Versiones: torch 2.11.0, torchvision 0.26.0 (`requirements.txt`).

| Paso | Entrenamiento | Evaluación (val, test) | Archivo y línea |
|---|---|---|---|
| Lectura | `Image.open(...).convert('L')` | Ídem | `data.py:97` |
| Tres canales | `transforms.Grayscale(num_output_channels=3)`: replica el canal (R = G = B) | Ídem | `data.py:46` |
| Tamaño | `transforms.Resize((img_size, img_size))` si `img_size != 256`; con 256 no se redimensiona. Por defecto `img_size = 224` (`data.py:35`, `run_grid.py:309`) | Ídem, mismo `img_size` (`data.py:149, 153`) | `data.py:47-48` |
| Interpolación | `InterpolationMode.BILINEAR` (valor por defecto de `Resize`, no se pasa explícitamente) | Ídem | torchvision `transforms.py:338` |
| *Antialias* | `antialias=True` (por defecto); sobre imágenes PIL el *antialias* se aplica siempre | Ídem | torchvision `transforms.py:318-325, 338` |
| Volteo | `transforms.RandomHorizontalFlip()` con p = 0.5 (por defecto), solo si `hflip=True` (desactivable con `--no-hflip`) | No se aplica | `data.py:49-50`; torchvision `transforms.py:707`; `run_grid.py:311, 327` |
| Otros aumentos | Ninguno | Ninguno | `data.py:43-52`; `train_model.py` no transforma entradas |
| Normalización | `ToTensor()` + `Normalize(IMAGENET_MEAN, IMAGENET_STD)`, con media (0.485, 0.456, 0.406) y desviación (0.229, 0.224, 0.225) | Ídem | `data.py:36-37, 51` |

**Por arquitectura.** La cadena es la misma para ResNet-18, EfficientNet-B0 y ViT-B-16: `build_model.py` solo construye el modelo y no añade transformaciones, y `make_loaders` se llama una vez por técnica y se reutiliza entre arquitecturas (`run_grid.py:367-374`). La única diferencia es una restricción: `preflight` rechaza `vit_b_16` con `img_size != 224` (`run_grid.py:141-143`).

**No hay aumentos distintos del redimensionamiento y el volteo horizontal**, así que P2 se aplica solo al volteo.

**ViT a 256 px.** El `vit_b_16` preentrenado de torchvision exige que la entrada coincida con `image_size` (`torchvision/models/vision_transformer.py:271`, `"Wrong image height! Expected 224"`). Admite 256 px con este procedimiento: se construye `vit_b_16(weights=None, image_size=256)` y se cargan los pesos `IMAGENET1K_V1` tras pasar el `state_dict` por `torchvision.models.vision_transformer.interpolate_embeddings(256, 16, state_dict)` (`vision_transformer.py:789`). La función interpola bicúbicamente los *positional embeddings* de una malla de 14 × 14 parches a una de 16 × 16 (256 / 16 = 16) y conserva el *token* de clase. Hoy el código no lo implementa (habría que cambiar `build_model.py` y la restricción de `run_grid.py:141-143`); su viabilidad la confirma CE según la regla 2.

**Precisiones operativas fijadas antes de ejecutar.** Concretan §3–§5 sin modificar el criterio:

1. **Variantes.** `V0 = get_transforms(False, 256)` y `V1 = get_transforms(False, 224)`, importadas de `data.py`. `V2` (que no existe en el código) es `Grayscale(3) → CenterCrop(224) → ToTensor → Normalize` con las mismas constantes de `data.py`.
2. **P1.** Características de 512 = salida del *average pooling* de `resnet18(weights=IMAGENET1K_V1)` con `fc = Identity`, en `eval()` y sin gradientes, en CPU. Un regresor por componente: `Pipeline(StandardScaler, RidgeCV(alphas=logspace(-3, 3, 13)))` (validación cruzada interna eficiente *leave-one-out* de `RidgeCV`). R² sobre las 908 predicciones fuera de partición concatenadas.
3. **P2.** La partición retenida se voltea **completa** (p = 1), no con la p = 0.5 del entrenamiento. Se reporta para V0, V1 y V2; la regla 3 se aplica con la variante V elegida por la regla 1.
4. **IC.** *Bootstrap* pareado sobre los índices de las 908 muestras (las mismas muestras remuestreadas para ambos términos de ΔR²), B = 2,000, `numpy.random.default_rng(2026)`, percentiles 2.5 y 97.5.
5. **P3.** Sobre la imagen nativa (256 × 256, valores en [0, 1]): fracción de |FFT2|² con max(|f_x|, |f_y|) > 112 ciclos por imagen respecto de la energía total, ambas sin la componente DC. Se usa el corte por eje porque es el que impone redimensionar a 224 × 224.
6. **P4.** Sobre la imagen nativa, con x = 0…255 e I la intensidad: (a) desplazamiento del centroide, Σ x·I / Σ I − 127.5; (b) asimetría de la posición x ponderada por el perfil de columnas p(x) = Σ_y I(x, y), es decir, el tercer momento central estandarizado de x con pesos p(x). Ambas cambian de signo con el volteo (la *skewness* de los valores del perfil no lo haría). Spearman con cada u_j y Holm sobre los 30 contrastes, α = 0.05.
7. **Insumos.** Solo las 908 filas `split = train` de `meta/sample_index.csv` (columnas `sample_id`, `idx`, `img_*`; `y_true` no se lee) y `u = load_umap_arrays(...)['train']['params']`. El SHA-256 de `umap_params.pkl` se verifica contra `meta/data_manifest.csv`.
8. **Fuera del alcance de F1.4.** Que `run_grid.py` lea `configs/preprocesamiento.json` es una tarea aparte de CE.

### 6.2 P1 y P2

[completar: tabla técnica × variante × componente con R², ΔR² e IC]

### 6.3 P3 y P4

[completar]

## 7. Decisión

| Técnica | Tamaño | Volteo | ViT | Etiqueta de variante |
|---|---|---|---|---|
| `perlin` | [completar] | [completar] | [completar] | [completar] |
| `gray_scott` | [completar] | [completar] | [completar] | [completar] |
| `lorenz` | [completar] | [completar] | [completar] | [completar] |

La decisión se materializa en `configs/preprocesamiento.json`, que es la única fuente que lee `run_grid.py`.
