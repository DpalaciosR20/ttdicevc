# Decisiones de preprocesamiento por técnica (F1.4)

> Tarea F1.4 del plan v2 · Roles: SV + CE · Estado: **criterio preregistrado (commit `5b9831c`); resultados y decisión completados**
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

Ejecución: `results/f14_preprocesamiento/f14_recuperabilidad.py` (CPU, 11.2 min; entorno, insumos y commits en el `README.md` de esa carpeta). Salidas: `r2_por_componente.csv` (P1) y `sensibilidad_volteo.csv` (P2). En negrita, los ΔR² mayores que el umbral de 0.02.

| Técnica | V | Comp. | R²(V) | ΔR²(V) | IC 95 % ΔR²(V) | R²(F(V)) | ΔR²(F) | IC 95 % ΔR²(F) |
|---|---|---|---|---|---|---|---|---|
| `perlin` | V0 | u0 | 0.9433 | 0 | — | 0.9289 | +0.0144 | [+0.0075, +0.0212] |
| `perlin` | V0 | u1 | 0.7249 | 0 | — | 0.7016 | **+0.0233** | [+0.0016, +0.0460] |
| `perlin` | V0 | u2 | 0.9689 | 0 | — | 0.9628 | +0.0061 | [+0.0026, +0.0098] |
| `perlin` | V0 | u3 | 0.8625 | 0 | — | 0.8625 | -0.0001 | [-0.0088, +0.0087] |
| `perlin` | V0 | u4 | 0.9392 | 0 | — | 0.9265 | +0.0127 | [+0.0066, +0.0189] |
| `perlin` | V1 | u0 | 0.9409 | +0.0024 | [-0.0039, +0.0088] | 0.9356 | +0.0053 | [-0.0011, +0.0116] |
| `perlin` | V1 | u1 | 0.7151 | +0.0098 | [-0.0145, +0.0358] | 0.6920 | **+0.0230** | [+0.0002, +0.0460] |
| `perlin` | V1 | u2 | 0.9615 | +0.0074 | [+0.0032, +0.0119] | 0.9613 | +0.0001 | [-0.0040, +0.0042] |
| `perlin` | V1 | u3 | 0.8633 | -0.0009 | [-0.0124, +0.0103] | 0.8536 | +0.0097 | [-0.0002, +0.0202] |
| `perlin` | V1 | u4 | 0.9343 | +0.0049 | [-0.0008, +0.0109] | 0.9319 | +0.0024 | [-0.0030, +0.0077] |
| `perlin` | V2 | u0 | 0.9345 | +0.0089 | [+0.0021, +0.0169] | 0.9356 | -0.0011 | [-0.0078, +0.0048] |
| `perlin` | V2 | u1 | 0.7054 | +0.0195 | [-0.0046, +0.0443] | 0.6943 | +0.0111 | [-0.0104, +0.0323] |
| `perlin` | V2 | u2 | 0.9612 | +0.0077 | [+0.0043, +0.0114] | 0.9651 | -0.0038 | [-0.0077, +0.0000] |
| `perlin` | V2 | u3 | 0.8599 | +0.0025 | [-0.0060, +0.0114] | 0.8584 | +0.0015 | [-0.0083, +0.0104] |
| `perlin` | V2 | u4 | 0.9323 | +0.0069 | [+0.0018, +0.0120] | 0.9310 | +0.0013 | [-0.0039, +0.0062] |
| `gray_scott` | V0 | u0 | 0.9851 | 0 | — | 0.9839 | +0.0012 | [-0.0004, +0.0027] |
| `gray_scott` | V0 | u1 | 0.9069 | 0 | — | 0.9082 | -0.0013 | [-0.0098, +0.0072] |
| `gray_scott` | V0 | u2 | 0.9628 | 0 | — | 0.9600 | +0.0027 | [-0.0005, +0.0059] |
| `gray_scott` | V0 | u3 | 0.9173 | 0 | — | 0.9019 | +0.0154 | [+0.0083, +0.0229] |
| `gray_scott` | V0 | u4 | 0.9854 | 0 | — | 0.9842 | +0.0012 | [-0.0003, +0.0027] |
| `gray_scott` | V1 | u0 | 0.9826 | +0.0025 | [-0.0006, +0.0066] | 0.9825 | +0.0002 | [-0.0024, +0.0023] |
| `gray_scott` | V1 | u1 | 0.9179 | -0.0110 | [-0.0207, -0.0019] | 0.9144 | +0.0035 | [-0.0039, +0.0111] |
| `gray_scott` | V1 | u2 | 0.9638 | -0.0011 | [-0.0047, +0.0027] | 0.9611 | +0.0028 | [-0.0005, +0.0059] |
| `gray_scott` | V1 | u3 | 0.9042 | +0.0131 | [+0.0005, +0.0306] | 0.8862 | +0.0180 | [+0.0100, +0.0265] |
| `gray_scott` | V1 | u4 | 0.9845 | +0.0010 | [-0.0007, +0.0026] | 0.9835 | +0.0009 | [-0.0005, +0.0023] |
| `gray_scott` | V2 | u0 | 0.9858 | -0.0007 | [-0.0023, +0.0010] | 0.9846 | +0.0012 | [-0.0002, +0.0026] |
| `gray_scott` | V2 | u1 | 0.9027 | +0.0042 | [-0.0047, +0.0141] | 0.9022 | +0.0005 | [-0.0078, +0.0090] |
| `gray_scott` | V2 | u2 | 0.9637 | -0.0009 | [-0.0041, +0.0022] | 0.9624 | +0.0013 | [-0.0018, +0.0043] |
| `gray_scott` | V2 | u3 | 0.9099 | +0.0075 | [-0.0012, +0.0176] | 0.8970 | +0.0129 | [+0.0039, +0.0234] |
| `gray_scott` | V2 | u4 | 0.9845 | +0.0009 | [-0.0005, +0.0024] | 0.9838 | +0.0007 | [-0.0009, +0.0021] |
| `lorenz` | V0 | u0 | 0.9517 | 0 | — | 0.8822 | **+0.0696** | [+0.0584, +0.0817] |
| `lorenz` | V0 | u1 | 0.6765 | 0 | — | 0.0639 | **+0.6126** | [+0.5147, +0.7230] |
| `lorenz` | V0 | u2 | 0.9193 | 0 | — | 0.7910 | **+0.1284** | [+0.1079, +0.1500] |
| `lorenz` | V0 | u3 | 0.8558 | 0 | — | 0.6715 | **+0.1843** | [+0.1558, +0.2173] |
| `lorenz` | V0 | u4 | 0.9915 | 0 | — | 0.9412 | **+0.0503** | [+0.0436, +0.0575] |
| `lorenz` | V1 | u0 | 0.9623 | -0.0106 | [-0.0174, -0.0047] | 0.7995 | **+0.1628** | [+0.1445, +0.1822] |
| `lorenz` | V1 | u1 | 0.6707 | +0.0058 | [-0.0264, +0.0387] | 0.0333 | **+0.6374** | [+0.5333, +0.7641] |
| `lorenz` | V1 | u2 | 0.9272 | -0.0079 | [-0.0162, +0.0003] | 0.7295 | **+0.1977** | [+0.1760, +0.2205] |
| `lorenz` | V1 | u3 | 0.8521 | +0.0037 | [-0.0109, +0.0191] | 0.4793 | **+0.3729** | [+0.3339, +0.4150] |
| `lorenz` | V1 | u4 | 0.9921 | -0.0006 | [-0.0020, +0.0009] | 0.9104 | **+0.0817** | [+0.0735, +0.0904] |
| `lorenz` | V2 | u0 | 0.9506 | +0.0011 | [-0.0054, +0.0078] | 0.8716 | **+0.0791** | [+0.0664, +0.0914] |
| `lorenz` | V2 | u1 | 0.6792 | -0.0026 | [-0.0357, +0.0309] | 0.2501 | **+0.4291** | [+0.3526, +0.5167] |
| `lorenz` | V2 | u2 | 0.9266 | -0.0072 | [-0.0148, -0.0002] | 0.8174 | **+0.1091** | [+0.0915, +0.1291] |
| `lorenz` | V2 | u3 | 0.8549 | +0.0009 | [-0.0107, +0.0127] | 0.6671 | **+0.1879** | [+0.1603, +0.2162] |
| `lorenz` | V2 | u4 | 0.9912 | +0.0003 | [-0.0010, +0.0015] | 0.9273 | **+0.0640** | [+0.0550, +0.0733] |

**P1 (tamaño).** Con las tres técnicas, V1 y V2 cumplen ΔR²_j ≤ 0.02 en los cinco componentes, así que las tres variantes se admiten. El mayor ΔR² de V1 es 0.0131 (Gray-Scott, u3) y el de V2, 0.0195 (Perlin, u1). Algunos ΔR² negativos indican que la variante reducida predice algo mejor que la nativa; quedan dentro del ruido de la validación cruzada.

**P2 (volteo, evaluado sobre V1, la variante elegida por la regla 1).**
- `gray_scott`: el máximo es ΔR²(F) = 0.0180 (u3). Cumple.
- `perlin`: ΔR²(F) = 0.0230 en u1 (IC [+0.0002, +0.0460]). Supera el umbral.
- `lorenz`: los cinco componentes lo superan por mucho (0.08 a 0.64). En u1, R² cae de 0.67 a 0.03: casi toda la información de u1 depende de la orientación horizontal.

### 6.3 P3 y P4

**P3 (espectro, explicativa; `espectro.csv`).** Fracción media de la energía (sin DC) con max(|f_x|, |f_y|) > 112 ciclos por imagen:

| Técnica | Media | Máximo | ρ de Spearman con u0…u4 |
|---|---|---|---|
| `perlin` | 0.138 | 0.234 | −0.55, 0.13, 0.56, 0.63, 0.59 |
| `gray_scott` | 6.4 × 10⁻⁵ | 4.2 × 10⁻⁴ | 0.72, −0.32, −0.95, −0.79, −0.81 |
| `lorenz` | 6.2 × 10⁻³ | 2.5 × 10⁻² | 0.23, −0.09, −0.40, −0.31, −0.32 |

Perlin concentra el 14 % de su energía en las frecuencias que el redimensionamiento a 224 px elimina, y esa fracción depende de u. Aun así, P1 no detecta pérdida de información para un lector lineal de ImageNet (ΔR²(V1) ≤ 0.0098). En Gray-Scott la fracción correlaciona fuertemente con u (ρ = −0.95 con u2), pero su energía por encima del corte es despreciable.

**P4 (orientación; `orientacion.csv`).** Spearman entre cada estadístico con signo y cada u_j; Holm sobre los 30 contrastes, α = 0.05.

| Técnica | Contrastes significativos tras Holm | ρ máximo en valor absoluto |
|---|---|---|
| `perlin` | 9 de 10 (todos salvo el centroide con u1, p_holm = 0.10) | 0.575 (centroide, u2) |
| `gray_scott` | 0 de 10 (p_holm mínimo = 0.065) | 0.093 (skewness, u0) |
| `lorenz` | 8 de 10 (salvo el centroide con u1 y la skewness con u2) | 0.452 (centroide, u2) |

En Perlin y Lorenz, la asimetría horizontal de la imagen codifica u: el volteo invierte esa relación y crea ejemplos de entrenamiento con un u efectivo distinto del verdadero. En Gray-Scott no hay asociación detectable.

## 7. Decisión

| Técnica | Tamaño | Volteo | ViT | Etiqueta de variante |
|---|---|---|---|---|
| `perlin` | `V1` (resize 224 × 224, bilineal, *antialias*) | No | `V1` | `px224-noflip` |
| `gray_scott` | `V1` (resize 224 × 224, bilineal, *antialias*) | Sí | `V1` | `px224-flip` |
| `lorenz` | `V1` (resize 224 × 224, bilineal, *antialias*) | No | `V1` | `px224-noflip` |

Aplicación de la regla de §5:
1. **Tamaño.** V1, V2 y V0 se admiten en las tres técnicas; por orden de preferencia se elige V1, que coincide con el preprocesamiento vigente de `data.py`.
2. **ViT.** Como se admite V1, ViT usa V1 (224 px) en las tres técnicas. No hace falta interpolar *positional embeddings*, ni confirmación de CE, ni declarar una cota inferior.
3. **Volteo.**
   - `gray_scott`: se permite (ΔR²(F) ≤ 0.0180 y ningún contraste de P4 significativo).
   - `perlin`: se excluye por los dos criterios (ΔR²(F) = 0.0230 en u1 y 9 contrastes significativos).
   - `lorenz`: se excluye por los dos criterios.
4. No hay otros aumentos que evaluar (§6.1).
5. No hubo desviaciones: la regla se aplicó tal como estaba preregistrada, con las precisiones operativas de §6.1.

**Consecuencia para el grid.** La primera pasada (F3.2) usó `px224-flip` en las tres técnicas. Para Perlin y Lorenz, la repetición debe correr con `px224-noflip` (`run_grid.py --no-hflip`). Para Gray-Scott, la variante vigente ya es la decidida.

La decisión se materializa en `configs/preprocesamiento.json`, que es la única fuente que lee `run_grid.py`. Esa lectura todavía no está implementada: es una tarea de CE fuera de F1.4 (§6.1, precisión 8). Mientras tanto, la variante se fija con `--img-size` y `--no-hflip`. `tests/test_preprocesamiento_config.py` verifica que el JSON coincida con esta tabla.
