# Reporte D2 — Ejecución confirmatoria de la Fase 5 y F4.6

Reporte sin interpretación. Cada cifra se copió a precisión completa (repr de Python) de `results/f5_contraste/resultados.json` o `results/f5_contraste/f46_resultados.json`. Junto a cada cifra va su ruta como `archivo:clave.clave[índice]`. Las claves con tuplas (p. ej. `('img', 'final', 'gray_scott')`) son literales de `observados`. Orden de semillas en las listas por semilla: [42, 7, 123, 2024, 99] (`configs/protocolo.json:modelos.seeds`).

## a) Comprobaciones previas y del log

### Punto 0 (rama `sv/f5-resultados`, creada desde `main` en `dd2e462`)

| Comprobación | Resultado |
|---|---|
| `git checkout main && git pull origin main` | Already up to date; `main` = `dd2e462` |
| `git merge-base --is-ancestor enmienda-5 HEAD` | código 0 |
| `git show enmienda-5:docs/protocolo_preregistrado.md` contiene «## Enmienda 5 (2026-11-07)» | sí, línea 329 |
| `git diff --stat enmienda-5 HEAD -- docs/protocolo_preregistrado.md configs/protocolo.json docs/plan_fase4.md` | vacío |
| `git status --porcelain` | vacío |
| `results/f5_contraste/` | no existía |
| `python -m src.protocol.run_fase5 --verificar-insumos` | 56 OK (45 corridas visuales + 11 referencias); prueba n = 114, soportes 0/1 = 70/44; u de train (908, 5) alineado |
| Variables | `CANON_UMAP_PARAMS=G:\Mi unidad\tt2-insumos\umap_params.pkl`, `CANON_STRICT=1`; `TT_DATA_ROOT=G:\Mi unidad\tt2-insumos` (ya definida) |

### Commits

| Paso | Commit |
|---|---|
| HEAD al ejecutar `--ejecutar-prueba` | dd2e462ca7d2f40326392da6997916a34c7c3606 (`resultados.json:meta.git_commit`) |
| Punto 2: «[F5] Resultados de la ejecución confirmatoria (run_fase5 --ejecutar-prueba)» | `e02104fcb15affed677a64b972305bc72f97c325` (2026-11-07 12:05:15 −06:00) |
| HEAD al ejecutar `--analisis-f46` | e02104fcb15affed677a64b972305bc72f97c325 (`f46_resultados.json:meta.git_commit`) |
| Punto 3: «[F4.6] Kappa y contención de errores (después del veredicto)» | `43d933cde335b2f323360c7667189396b68c3b79` (2026-11-07 12:10:30 −06:00) |

### Metadatos de `resultados.json`

- `created_at`: 2026-11-07T18:05:09+00:00 (`resultados.json:meta.created_at`)
- `gate_tag`: enmienda-5 (`resultados.json:meta.gate_tag`)
- `gate_tag_commit`: dd2e462ca7d2f40326392da6997916a34c7c3606 (`resultados.json:meta.gate_tag_commit`)
- `config_sha256_lf`: 068f33521968cecd35c461a21f6fc456407d1e5cc44bc9ae8a5d4af4a6351544 (`resultados.json:meta.config_sha256_lf`)
- `canonical_sha256_lf`: 8d1d9ea215b582e50be772a032fc16a7487fc428080fddf48e06b659a5c53bfa (`resultados.json:meta.canonical_sha256_lf`)
- `bootstrap.call`: rng.integers(0, 114, size=(10000, 114)) (`resultados.json:meta.bootstrap.call`)
- `bootstrap.seed`: 2026 (`resultados.json:meta.bootstrap.seed`)
- `bootstrap.idx_sha256`: 322edaaa5f1175bb4d4a16409d7f852fa7e67fa5daa8a504bb4976e67d0a7e37 (`resultados.json:meta.bootstrap.idx_sha256`)

Metadatos de `f46_resultados.json`:

- `created_at`: 2026-11-07T18:10:11+00:00 (`f46_resultados.json:meta.created_at`)
- `gate_tag`: enmienda-5 (`f46_resultados.json:meta.gate_tag`)
- `gate_tag_commit`: dd2e462ca7d2f40326392da6997916a34c7c3606 (`f46_resultados.json:meta.gate_tag_commit`)
- `config_sha256_lf`: 068f33521968cecd35c461a21f6fc456407d1e5cc44bc9ae8a5d4af4a6351544 (`f46_resultados.json:meta.config_sha256_lf`)
- `canonical_sha256_lf`: 8d1d9ea215b582e50be772a032fc16a7487fc428080fddf48e06b659a5c53bfa (`f46_resultados.json:meta.canonical_sha256_lf`)
- `bootstrap.idx_sha256`: 322edaaa5f1175bb4d4a16409d7f852fa7e67fa5daa8a504bb4976e67d0a7e37 (`f46_resultados.json:meta.bootstrap.idx_sha256`)
- `meta.inputs` idéntico entre ambos archivos: sí

### Hashes de los insumos (`resultados.json:meta.inputs`, también en `run_fase5.log`)

| Archivo | sha256 |
|---|---|
| `results/predictions/final/gray_scott__efficientnet_b0__finetune__seed42.csv` | `60e78e5c2f3d4cbde2584a7622ad5ff148d9e83257858e50142889a8556fc6bb` |
| `results/predictions/final/gray_scott__efficientnet_b0__finetune__seed7.csv` | `3ead6bb773fbf2e586282a76f761c529ca24198f3d88005de0740835016acaf1` |
| `results/predictions/final/gray_scott__efficientnet_b0__finetune__seed123.csv` | `b51a2d0227a2bb9110b076fa9cb6ede73497fde3a67a26754707ca795f80bf41` |
| `results/predictions/final/gray_scott__efficientnet_b0__finetune__seed2024.csv` | `8c5c9b633612e41facc8cd8e46d1af804da9b46faaec191dc966ae15edcc374a` |
| `results/predictions/final/gray_scott__efficientnet_b0__finetune__seed99.csv` | `e72dc2ae668f54ed1581ad172abc802cb73b99a405656a0f430b11e3490799d7` |
| `results/predictions/final/perlin__vit_b16__frozen__seed42.csv` | `2807eec87fca0391d8ce06538921571854c8b77ec6744a3145685981f5688f0d` |
| `results/predictions/final/perlin__vit_b16__frozen__seed7.csv` | `f2ec421680245dadfaad2eea27b0f26a5416eba09ac291aae207abe70ececec2` |
| `results/predictions/final/perlin__vit_b16__frozen__seed123.csv` | `de5d1a7c55bb629174cccacaa6da27e7acece57930262147b19f0248ca93108d` |
| `results/predictions/final/perlin__vit_b16__frozen__seed2024.csv` | `52588352631a8a3e7e8108e071a50a07e4e5c5f0a423b9781f880934d2788920` |
| `results/predictions/final/perlin__vit_b16__frozen__seed99.csv` | `27f74169fca562b1711d10ccb26411c74c5d295b2743f5ebaa284a2d3f39f199` |
| `results/predictions/final/lorenz__vit_b16__frozen__seed42.csv` | `fcac7c6b35a380637b0462ee4d541ec69d911278eed02d5f634cd38b67f91b76` |
| `results/predictions/final/lorenz__vit_b16__frozen__seed7.csv` | `67b61ae0124831f7f7011be3d30d68a01fd8d89079783f5a320614f8a6387e38` |
| `results/predictions/final/lorenz__vit_b16__frozen__seed123.csv` | `0b66f60777549bcf2181f931bc10e85d2dd77ffc02d71d321c74359b9484b065` |
| `results/predictions/final/lorenz__vit_b16__frozen__seed2024.csv` | `5e2e09af39e19ec568d24ab765ee340eb8d697314c4a72138f450e89a76c640e` |
| `results/predictions/final/lorenz__vit_b16__frozen__seed99.csv` | `2ef17e032c1d861cc3b161a1ec6a47dc41b75f7cf8ecdddac6346bde30ce6d30` |
| `results/predictions/control_permutacion/gray_scott__efficientnet_b0__finetune__seed42.csv` | `802f2d3143b3369027ecf6c5d86fa72c464c581f8311045dc549cb29bbbf6dbc` |
| `results/predictions/control_permutacion/gray_scott__efficientnet_b0__finetune__seed7.csv` | `7abad718cd67923a8bbe88e6d68c282204b700f19fbff0e879741f932f233cc7` |
| `results/predictions/control_permutacion/gray_scott__efficientnet_b0__finetune__seed123.csv` | `a78c6a0c488508cfb9b7ede881c109a5847bfbedad05646c421a7295bee17767` |
| `results/predictions/control_permutacion/gray_scott__efficientnet_b0__finetune__seed2024.csv` | `d0bfa556abd14ab29374ed7bd2e17c0f3a9391f1586b1f3e55a03b93c3717ad0` |
| `results/predictions/control_permutacion/gray_scott__efficientnet_b0__finetune__seed99.csv` | `3b69706f50dd061ad6611d23b7f628133edaccbe9f9324b7596b9aed24489125` |
| `results/predictions/control_permutacion/perlin__vit_b16__frozen__seed42.csv` | `4670a24a37d61e1e8c5adb6bd645b1f8147911a051cfa85d8f1dd097bee348e6` |
| `results/predictions/control_permutacion/perlin__vit_b16__frozen__seed7.csv` | `58d101eea5df718ae07a17e54a253dc26b797e57340bb2f280607233e226e602` |
| `results/predictions/control_permutacion/perlin__vit_b16__frozen__seed123.csv` | `92da843b7d92e4dcabaeaa0947564923232964f06751480adbc6915d9e04ec04` |
| `results/predictions/control_permutacion/perlin__vit_b16__frozen__seed2024.csv` | `bca015ec3a0179fad98d1c00b5b5932cba927cc4529eb703d3912acfef33a01e` |
| `results/predictions/control_permutacion/perlin__vit_b16__frozen__seed99.csv` | `f16ba0fc86c345d3e2d50de09c26c8d3c2730228c385928aa24fbe20e7162a5c` |
| `results/predictions/control_permutacion/lorenz__vit_b16__frozen__seed42.csv` | `f272063680c0d81558d8b8447fe7323c49ff376f6512d2230281715a0fe72ef8` |
| `results/predictions/control_permutacion/lorenz__vit_b16__frozen__seed7.csv` | `23dc97cded39c49d5304dc291e5c4130b7504e4c65318d2de955f2abcd97bd18` |
| `results/predictions/control_permutacion/lorenz__vit_b16__frozen__seed123.csv` | `63f90471574326b1dcf2136fe4dd8ffbb9130471ea8415d32b072d9e31c6e50b` |
| `results/predictions/control_permutacion/lorenz__vit_b16__frozen__seed2024.csv` | `4b45ffd6d7844d5f9169f7b3126628114e3fdefc0a50320dd3a42da7baa9fc7d` |
| `results/predictions/control_permutacion/lorenz__vit_b16__frozen__seed99.csv` | `dce8e1428b257a1cf3aed55cb92cabb1bf8c429e9d83cab789341fa7cc0960f2` |
| `results/predictions/sensibilidad/gray_scott__resnet18__finetune__seed42.csv` | `8db328356c77a886edae1ee7eb132f56776734710b156b2b9af35394a708f3cf` |
| `results/predictions/sensibilidad/gray_scott__resnet18__finetune__seed7.csv` | `8b30a94d339f1a9e4230addfc1058dacafc402e3586e59db3d170f9b5bf5a9f5` |
| `results/predictions/sensibilidad/gray_scott__resnet18__finetune__seed123.csv` | `010b75c6c546582e2e91beb3f51578934a1e1ca1aa9195ba0d9b10ef3bd75fd4` |
| `results/predictions/sensibilidad/gray_scott__resnet18__finetune__seed2024.csv` | `ed7db9b98fa6427f88da79b7de0fa115183a2523b52e7f3e6568f85b440ca325` |
| `results/predictions/sensibilidad/gray_scott__resnet18__finetune__seed99.csv` | `513acfa971f22b00661be2e685033d7c21d63f8f448229a9b57d0a2792c3be4d` |
| `results/predictions/sensibilidad/perlin__efficientnet_b0__frozen__seed42.csv` | `53a32af82b18a73e9eb7cb44da43698007da69232b52713cea58f63be3e0e447` |
| `results/predictions/sensibilidad/perlin__efficientnet_b0__frozen__seed7.csv` | `aa10c07988cddf8feb2cc77ed99b798971cb9ec45a13ca67bb88869028241424` |
| `results/predictions/sensibilidad/perlin__efficientnet_b0__frozen__seed123.csv` | `adae665f11ac829d1e7765c062fd7672a1effa1678bd42d81d0422e0ef149d37` |
| `results/predictions/sensibilidad/perlin__efficientnet_b0__frozen__seed2024.csv` | `cdf46053eb806708ce9db7811d2462b5a4ff3fd27ad23d6790ec89880da32b2a` |
| `results/predictions/sensibilidad/perlin__efficientnet_b0__frozen__seed99.csv` | `045a8c533f5291c385a6c6fa2d38059ee9ec89672f146ce99e80548194bcc6c9` |
| `results/predictions/sensibilidad/lorenz__efficientnet_b0__finetune__seed42.csv` | `f19b54953dcafff28663af8b2f88fc120914ba94cb255ffd88f5d711935aef65` |
| `results/predictions/sensibilidad/lorenz__efficientnet_b0__finetune__seed7.csv` | `c6c7f9d49a3f2a68a82fd13cd790e0c4be7ed0e078b360fb4d7a658c1c4646c2` |
| `results/predictions/sensibilidad/lorenz__efficientnet_b0__finetune__seed123.csv` | `2da90da74434010236f50febc410bb51b37c7bab16fd87cfc4e8e59d482f5262` |
| `results/predictions/sensibilidad/lorenz__efficientnet_b0__finetune__seed2024.csv` | `842591a0d2404514e92166331ea661a9934383939dce3030c635f2dfb9797649` |
| `results/predictions/sensibilidad/lorenz__efficientnet_b0__finetune__seed99.csv` | `99cd0a5067583ae5b7a7616afe3cbc3452ee0ae4b3f21bad93ebbfd061b5f452` |
| `results/predictions/referencia/text__beto__finetune__seed42.csv` | `3d3587469545e592e39fa62d3b00b0acd52889bfe21dceb0a5c2816a28f7916e` |
| `results/predictions/referencia/emb768__logreg__none__seed42.csv` | `b8c3d471a860b943df962fa8e4652d2ff400e3fcbcf6b9af2e84871091cb778a` |
| `results/predictions/referencia/umap5d__svm_rbf__none__seed42.csv` | `c061b61580e3132b4b5ad0093cb8a5b032d7065b8ef9fb18f2cafa2e03cf426f` |
| `results/predictions/referencia/none__majority__none__seed42.csv` | `14c4b6efbcfcaaedb6112eab13b9b06edf49150101d5f38b68e9602beb6ef819` |
| `results/predictions/referencia/text_length__logreg__none__seed42.csv` | `9aabed280470c07c5641a6858e8a401ed5f9f8144b1e26c5b71c245ca7c24f29` |
| `results/predictions/referencia/gray_scott__brightness__none__seed42.csv` | `f5183e70a6327deaa7d6c1ca0f7e28957900f4244621808aaa81d7e47bee37fb` |
| `results/predictions/referencia/gray_scott__descriptors__none__seed42.csv` | `a464b321e40d9601067352e349f40a8d2f9baee608a5b2f31276a185bedbb51b` |
| `results/predictions/referencia/perlin__brightness__none__seed42.csv` | `93e6a60e61b98bad816c46b9916110b15b6e7726e3ccc204571158021c8d3369` |
| `results/predictions/referencia/perlin__descriptors__none__seed42.csv` | `92f45fd2252a2545843d9adab1398f6b0a4ba8f233a8867ab0af510588342538` |
| `results/predictions/referencia/lorenz__brightness__none__seed42.csv` | `de67b6e5a7531b5cd854a016c6e2c520d7759bb9b8c74fbc09d929a246f1a43b` |
| `results/predictions/referencia/lorenz__descriptors__none__seed42.csv` | `98017425fe294d932d5082353e1bcfb369e8db988f3dba674bb60333cae8e4f0` |

Total: 56 archivos.

## b) Contraste principal por técnica

Parámetros: `fase` = final, `r` = 0.5, `formulation` = B, `delta` = null, `beto_shift` = 0.0, `floor_mode` = identidad, `ties` = min, `floors` = [majority, stratified, brightness, visual_permutado], `seed_index` = null (`resultados.json:resultados.principal.params`); `alpha` = 0.05 (`resultados.json:resultados.principal.alpha`); `B` = 10000 (`resultados.json:resultados.principal.B`).

### gray_scott

- F1 del finalista por semilla: semilla 42 = 0.7448849104859334, semilla 7 = 0.7266624040920716, semilla 123 = 0.6982433624769391, semilla 2024 = 0.736678614097969, semilla 99 = 0.7185185185185186 (`resultados.json:resultados.observados.('img', 'final', 'gray_scott')[0..4]`)
- F1 del finalista (media de 5 semillas): 0.7249975619342864 (`resultados.json:resultados.principal.techniques.gray_scott.f1_img_obs`)
- F1 de BETO: 0.7038961038961039 (`resultados.json:resultados.principal.techniques.gray_scott.f1_beto_obs`)
- F1 del piso `majority`: 0.3804347826086957 (`resultados.json:resultados.principal.techniques.gray_scott.piso_f1_obs.majority`)
- F1 del piso `stratified`: 0.5 (`resultados.json:resultados.principal.techniques.gray_scott.piso_f1_obs.stratified`)
- F1 del piso `brightness`: 0.44503042596348885 (`resultados.json:resultados.principal.techniques.gray_scott.piso_f1_obs.brightness`)
- F1 del piso `visual_permutado`: 0.5269755062041923 (`resultados.json:resultados.principal.techniques.gray_scott.piso_f1_obs.visual_permutado`)
- F1 del control permutado por semilla: semilla 42 = 0.566468253968254, semilla 7 = 0.46710526315789475, semilla 123 = 0.42731554160125584, semilla 2024 = 0.7185185185185186, semilla 99 = 0.4554699537750385 (`resultados.json:resultados.observados.('img', 'control_permutacion', 'gray_scott')[0..4]`)
- Piso aplicado: [visual_permutado] (`resultados.json:resultados.principal.techniques.gray_scott.piso`); empate: false (`resultados.json:resultados.principal.techniques.gray_scott.empate`); `mode`: identidad (`resultados.json:resultados.principal.techniques.gray_scott.mode`); `ties`: min (`resultados.json:resultados.principal.techniques.gray_scott.ties`)
- θ observado: 0.12304950998623448 (`resultados.json:resultados.principal.techniques.gray_scott.theta_obs`)
- No inferioridad: p = 0.0 (`resultados.json:resultados.principal.techniques.gray_scott.no_inferioridad.p`); `below_resolution` = true (`resultados.json:resultados.principal.techniques.gray_scott.no_inferioridad.below_resolution`); rango de Holm = 1 (`resultados.json:resultados.principal.techniques.gray_scott.no_inferioridad.rank`); α_h = 0.016666666666666666 (`resultados.json:resultados.principal.techniques.gray_scott.no_inferioridad.alpha_h`); rechazo = true (`resultados.json:resultados.principal.techniques.gray_scott.no_inferioridad.reject`)
- IC de θ: [0.06880818233277372, 0.17379796798270544] (`resultados.json:resultados.principal.techniques.gray_scott.theta_interval`); nivel = 0.9666666666666667 (`resultados.json:resultados.principal.techniques.gray_scott.interval_level`)
- D observado: 0.19802205573009413 (`resultados.json:resultados.principal.techniques.gray_scott.D_obs`)
- Superioridad sobre el piso: p = 0.0 (`resultados.json:resultados.principal.techniques.gray_scott.superioridad_piso.p`); `below_resolution` = true (`resultados.json:resultados.principal.techniques.gray_scott.superioridad_piso.below_resolution`); rango de Holm = 1 (`resultados.json:resultados.principal.techniques.gray_scott.superioridad_piso.rank`); α_h = 0.016666666666666666 (`resultados.json:resultados.principal.techniques.gray_scott.superioridad_piso.alpha_h`); rechazo = true (`resultados.json:resultados.principal.techniques.gray_scott.superioridad_piso.reject`)
- IC de D: [0.09782748805479571, 0.2908685790617239] (`resultados.json:resultados.principal.techniques.gray_scott.D_interval`); nivel = 0.9666666666666667 (`resultados.json:resultados.principal.techniques.gray_scott.D_interval_level`)
- **Veredicto: V1 (`resultados.json:resultados.principal.techniques.gray_scott.verdict`)**

### perlin

- F1 del finalista por semilla: semilla 42 = 0.6581439393939394, semilla 7 = 0.6902173913043479, semilla 123 = 0.6859998426064373, semilla 2024 = 0.6599686028257457, semilla 99 = 0.6650170120630993 (`resultados.json:resultados.observados.('img', 'final', 'perlin')[0..4]`)
- F1 del finalista (media de 5 semillas): 0.671869357638714 (`resultados.json:resultados.principal.techniques.perlin.f1_img_obs`)
- F1 de BETO: 0.7038961038961039 (`resultados.json:resultados.principal.techniques.perlin.f1_beto_obs`)
- F1 del piso `majority`: 0.3804347826086957 (`resultados.json:resultados.principal.techniques.perlin.piso_f1_obs.majority`)
- F1 del piso `stratified`: 0.5 (`resultados.json:resultados.principal.techniques.perlin.piso_f1_obs.stratified`)
- F1 del piso `brightness`: 0.45344880915558305 (`resultados.json:resultados.principal.techniques.perlin.piso_f1_obs.brightness`)
- F1 del piso `visual_permutado`: 0.5483271786205862 (`resultados.json:resultados.principal.techniques.perlin.piso_f1_obs.visual_permutado`)
- F1 del control permutado por semilla: semilla 42 = 0.550249169435216, semilla 7 = 0.6420561586076972, semilla 123 = 0.5587855852756516, semilla 2024 = 0.47148247978436664, semilla 99 = 0.5190625 (`resultados.json:resultados.observados.('img', 'control_permutacion', 'perlin')[0..4]`)
- Piso aplicado: [visual_permutado] (`resultados.json:resultados.principal.techniques.perlin.piso`); empate: false (`resultados.json:resultados.principal.techniques.perlin.empate`); `mode`: identidad (`resultados.json:resultados.principal.techniques.perlin.mode`); `ties`: min (`resultados.json:resultados.principal.techniques.perlin.ties`)
- θ observado: 0.06992130569066207 (`resultados.json:resultados.principal.techniques.perlin.theta_obs`)
- No inferioridad: p = 0.0046 (`resultados.json:resultados.principal.techniques.perlin.no_inferioridad.p`); `below_resolution` = false (`resultados.json:resultados.principal.techniques.perlin.no_inferioridad.below_resolution`); rango de Holm = 3 (`resultados.json:resultados.principal.techniques.perlin.no_inferioridad.rank`); α_h = 0.05 (`resultados.json:resultados.principal.techniques.perlin.no_inferioridad.alpha_h`); rechazo = true (`resultados.json:resultados.principal.techniques.perlin.no_inferioridad.reject`)
- IC de θ: [0.02456167142914315, 0.11282917237536824] (`resultados.json:resultados.principal.techniques.perlin.theta_interval`); nivel = 0.9 (`resultados.json:resultados.principal.techniques.perlin.interval_level`)
- D observado: 0.1235421790181278 (`resultados.json:resultados.principal.techniques.perlin.D_obs`)
- Superioridad sobre el piso: p = 0.002 (`resultados.json:resultados.principal.techniques.perlin.superioridad_piso.p`); `below_resolution` = false (`resultados.json:resultados.principal.techniques.perlin.superioridad_piso.below_resolution`); rango de Holm = 3 (`resultados.json:resultados.principal.techniques.perlin.superioridad_piso.rank`); α_h = 0.05 (`resultados.json:resultados.principal.techniques.perlin.superioridad_piso.alpha_h`); rechazo = true (`resultados.json:resultados.principal.techniques.perlin.superioridad_piso.reject`)
- IC de D: [0.05381508112061247, 0.19178784816300895] (`resultados.json:resultados.principal.techniques.perlin.D_interval`); nivel = 0.9 (`resultados.json:resultados.principal.techniques.perlin.D_interval_level`)
- **Veredicto: V1 (`resultados.json:resultados.principal.techniques.perlin.verdict`)**

### lorenz

- F1 del finalista por semilla: semilla 42 = 0.7121212121212122, semilla 7 = 0.7315541601255887, semilla 123 = 0.7448849104859334, semilla 2024 = 0.7448849104859334, semilla 99 = 0.7203007518796992 (`resultados.json:resultados.observados.('img', 'final', 'lorenz')[0..4]`)
- F1 del finalista (media de 5 semillas): 0.7307491890196733 (`resultados.json:resultados.principal.techniques.lorenz.f1_img_obs`)
- F1 de BETO: 0.7038961038961039 (`resultados.json:resultados.principal.techniques.lorenz.f1_beto_obs`)
- F1 del piso `majority`: 0.3804347826086957 (`resultados.json:resultados.principal.techniques.lorenz.piso_f1_obs.majority`)
- F1 del piso `stratified`: 0.5 (`resultados.json:resultados.principal.techniques.lorenz.piso_f1_obs.stratified`)
- F1 del piso `brightness`: 0.4735221674876847 (`resultados.json:resultados.principal.techniques.lorenz.piso_f1_obs.brightness`)
- F1 del piso `visual_permutado`: 0.5329193226291082 (`resultados.json:resultados.principal.techniques.lorenz.piso_f1_obs.visual_permutado`)
- F1 del control permutado por semilla: semilla 42 = 0.5063408598824621, semilla 7 = 0.5612684729064039, semilla 123 = 0.5145180023228804, semilla 2024 = 0.5913978494623655, semilla 99 = 0.4910714285714286 (`resultados.json:resultados.observados.('img', 'control_permutacion', 'lorenz')[0..4]`)
- Piso aplicado: [visual_permutado] (`resultados.json:resultados.principal.techniques.lorenz.piso`); empate: false (`resultados.json:resultados.principal.techniques.lorenz.empate`); `mode`: identidad (`resultados.json:resultados.principal.techniques.lorenz.mode`); `ties`: min (`resultados.json:resultados.principal.techniques.lorenz.ties`)
- θ observado: 0.12880113707162139 (`resultados.json:resultados.principal.techniques.lorenz.theta_obs`)
- No inferioridad: p = 0.0 (`resultados.json:resultados.principal.techniques.lorenz.no_inferioridad.p`); `below_resolution` = true (`resultados.json:resultados.principal.techniques.lorenz.no_inferioridad.below_resolution`); rango de Holm = 1 (`resultados.json:resultados.principal.techniques.lorenz.no_inferioridad.rank`); α_h = 0.016666666666666666 (`resultados.json:resultados.principal.techniques.lorenz.no_inferioridad.alpha_h`); rechazo = true (`resultados.json:resultados.principal.techniques.lorenz.no_inferioridad.reject`)
- IC de θ: [0.07198636459962342, 0.18189215074697387] (`resultados.json:resultados.principal.techniques.lorenz.theta_interval`); nivel = 0.9666666666666667 (`resultados.json:resultados.principal.techniques.lorenz.interval_level`)
- D observado: 0.1978298663905651 (`resultados.json:resultados.principal.techniques.lorenz.D_obs`)
- Superioridad sobre el piso: p = 0.0 (`resultados.json:resultados.principal.techniques.lorenz.superioridad_piso.p`); `below_resolution` = true (`resultados.json:resultados.principal.techniques.lorenz.superioridad_piso.below_resolution`); rango de Holm = 1 (`resultados.json:resultados.principal.techniques.lorenz.superioridad_piso.rank`); α_h = 0.016666666666666666 (`resultados.json:resultados.principal.techniques.lorenz.superioridad_piso.alpha_h`); rechazo = true (`resultados.json:resultados.principal.techniques.lorenz.superioridad_piso.reject`)
- IC de D: [0.09816577974747169, 0.2973528238449148] (`resultados.json:resultados.principal.techniques.lorenz.D_interval`); nivel = 0.9666666666666667 (`resultados.json:resultados.principal.techniques.lorenz.D_interval_level`)
- **Veredicto: V1 (`resultados.json:resultados.principal.techniques.lorenz.verdict`)**

**Veredicto global: G1 (`resultados.json:resultados.principal.global`)**

Otros F1 observados (`resultados.json:resultados.observados`):

- `beto`: 0.7038961038961039 (`resultados.json:resultados.observados.beto`)
- `emb768`: 0.6984126984126984 (`resultados.json:resultados.observados.emb768`)
- `techo5d`: 0.6933235509904623 (`resultados.json:resultados.observados.techo5d`)
- `majority`: 0.3804347826086957 (`resultados.json:resultados.observados.majority`)
- `longitud`: 0.5142045454545454 (`resultados.json:resultados.observados.longitud`)
- `('brightness', 'gray_scott')`: 0.44503042596348885 (`resultados.json:resultados.observados.('brightness', 'gray_scott')`)
- `('descriptors', 'gray_scott')`: 0.6795634920634921 (`resultados.json:resultados.observados.('descriptors', 'gray_scott')`)
- `('brightness', 'perlin')`: 0.45344880915558305 (`resultados.json:resultados.observados.('brightness', 'perlin')`)
- `('descriptors', 'perlin')`: 0.6420722135007849 (`resultados.json:resultados.observados.('descriptors', 'perlin')`)
- `('brightness', 'lorenz')`: 0.4735221674876847 (`resultados.json:resultados.observados.('brightness', 'lorenz')`)
- `('descriptors', 'lorenz')`: 0.5853174603174602 (`resultados.json:resultados.observados.('descriptors', 'lorenz')`)

## c) Retención, descomposición y frente a descriptores

### Retención ρ̂

- gray_scott: ρ̂ = 1.1034912273210227 (`resultados.json:resultados.retencion.gray_scott.rho_obs`); IC = [0.9207770397128984, 1.4549966687532847] (`resultados.json:resultados.retencion.gray_scott.interval`); réplicas con denominador no positivo (−inf) = 0 (`resultados.json:resultados.retencion.gray_scott.n_denominador_no_positivo`)
- perlin: ρ̂ = 0.8429261489287247 (`resultados.json:resultados.retencion.perlin.rho_obs`); IC = [0.5878206489673888, 1.1842410233580951] (`resultados.json:resultados.retencion.perlin.interval`); réplicas con denominador no positivo (−inf) = 0 (`resultados.json:resultados.retencion.perlin.n_denominador_no_positivo`)
- lorenz: ρ̂ = 1.1316998442366144 (`resultados.json:resultados.retencion.lorenz.rho_obs`); IC = [0.8988488279469173, 1.5861895981860032] (`resultados.json:resultados.retencion.lorenz.interval`); réplicas con denominador no positivo (−inf) = 0 (`resultados.json:resultados.retencion.lorenz.n_denominador_no_positivo`)

### Descomposición

**gray_scott**

- cabeza: obs = 0.005483405483405535 (`resultados.json:resultados.descomposicion.gray_scott.cabeza.obs`); IC = [-0.04196906304702251, 0.054088085942356465] (`resultados.json:resultados.descomposicion.gray_scott.cabeza.interval`)
- reduccion: obs = 0.0050891474222360955 (`resultados.json:resultados.descomposicion.gray_scott.reduccion.obs`); IC = [-0.025957779961985793, 0.03808948470496814] (`resultados.json:resultados.descomposicion.gray_scott.reduccion.interval`)
- renderizado: obs = -0.03167401094382416 (`resultados.json:resultados.descomposicion.gray_scott.renderizado.obs`); IC = [-0.07428175598918436, 0.005262856471583586] (`resultados.json:resultados.descomposicion.gray_scott.renderizado.interval`)
- total: -0.02110145803818253 (`resultados.json:resultados.descomposicion.gray_scott.total_obs`)
- BETO − imagen: -0.02110145803818253 (`resultados.json:resultados.descomposicion.gray_scott.beto_menos_img_obs`)

**perlin**

- cabeza: obs = 0.005483405483405535 (`resultados.json:resultados.descomposicion.perlin.cabeza.obs`); IC = [-0.04196906304702251, 0.054088085942356465] (`resultados.json:resultados.descomposicion.perlin.cabeza.interval`)
- reduccion: obs = 0.0050891474222360955 (`resultados.json:resultados.descomposicion.perlin.reduccion.obs`); IC = [-0.025957779961985793, 0.03808948470496814] (`resultados.json:resultados.descomposicion.perlin.reduccion.interval`)
- renderizado: obs = 0.021454193351748252 (`resultados.json:resultados.descomposicion.perlin.renderizado.obs`); IC = [-0.04456346420136281, 0.08473274066015725] (`resultados.json:resultados.descomposicion.perlin.renderizado.interval`)
- total: 0.03202674625738988 (`resultados.json:resultados.descomposicion.perlin.total_obs`)
- BETO − imagen: 0.03202674625738988 (`resultados.json:resultados.descomposicion.perlin.beto_menos_img_obs`)

**lorenz**

- cabeza: obs = 0.005483405483405535 (`resultados.json:resultados.descomposicion.lorenz.cabeza.obs`); IC = [-0.04196906304702251, 0.054088085942356465] (`resultados.json:resultados.descomposicion.lorenz.cabeza.interval`)
- reduccion: obs = 0.0050891474222360955 (`resultados.json:resultados.descomposicion.lorenz.reduccion.obs`); IC = [-0.025957779961985793, 0.03808948470496814] (`resultados.json:resultados.descomposicion.lorenz.reduccion.interval`)
- renderizado: obs = -0.03742563802921106 (`resultados.json:resultados.descomposicion.lorenz.renderizado.obs`); IC = [-0.09420960101726263, 0.014087938808884226] (`resultados.json:resultados.descomposicion.lorenz.renderizado.interval`)
- total: -0.02685308512356943 (`resultados.json:resultados.descomposicion.lorenz.total_obs`)
- BETO − imagen: -0.02685308512356943 (`resultados.json:resultados.descomposicion.lorenz.beto_menos_img_obs`)

### Visual − descriptores

- gray_scott: obs = 0.04543406987079435 (`resultados.json:resultados.frente_a_descriptores.gray_scott.obs`); IC = [-0.016468478070513985, 0.10910761012410698] (`resultados.json:resultados.frente_a_descriptores.gray_scott.interval`)
- perlin: obs = 0.02979714413792911 (`resultados.json:resultados.frente_a_descriptores.perlin.obs`); IC = [-0.048729275463953906, 0.10898332462111919] (`resultados.json:resultados.frente_a_descriptores.perlin.interval`)
- lorenz: obs = 0.1454317287022131 (`resultados.json:resultados.frente_a_descriptores.lorenz.obs`); IC = [0.04853381041037734, 0.24342529140282904] (`resultados.json:resultados.frente_a_descriptores.lorenz.interval`)

## d) Contraste de longitud

- m = 3 (`resultados.json:resultados.longitud.m`)
- Técnicas evaluadas: gray_scott, perlin, lorenz (`resultados.json:resultados.longitud.techniques`)

- gray_scott: L = 0.21079301647974102 (`resultados.json:resultados.longitud.techniques.gray_scott.L_obs`); p = 0.0001 (`resultados.json:resultados.longitud.techniques.gray_scott.p`); `below_resolution` = false (`resultados.json:resultados.longitud.techniques.gray_scott.below_resolution`); rango = 1 (`resultados.json:resultados.longitud.techniques.gray_scott.rank`); α_h = 0.016666666666666666 (`resultados.json:resultados.longitud.techniques.gray_scott.alpha_h`); rechazo = true (`resultados.json:resultados.longitud.techniques.gray_scott.reject`); calificación = null (`resultados.json:resultados.longitud.techniques.gray_scott.calificacion`)
- perlin: L = 0.1576648121841686 (`resultados.json:resultados.longitud.techniques.perlin.L_obs`); p = 0.0029 (`resultados.json:resultados.longitud.techniques.perlin.p`); `below_resolution` = false (`resultados.json:resultados.longitud.techniques.perlin.below_resolution`); rango = 3 (`resultados.json:resultados.longitud.techniques.perlin.rank`); α_h = 0.05 (`resultados.json:resultados.longitud.techniques.perlin.alpha_h`); rechazo = true (`resultados.json:resultados.longitud.techniques.perlin.reject`); calificación = null (`resultados.json:resultados.longitud.techniques.perlin.calificacion`)
- lorenz: L = 0.21654464356512793 (`resultados.json:resultados.longitud.techniques.lorenz.L_obs`); p = 0.0001 (`resultados.json:resultados.longitud.techniques.lorenz.p`); `below_resolution` = false (`resultados.json:resultados.longitud.techniques.lorenz.below_resolution`); rango = 1 (`resultados.json:resultados.longitud.techniques.lorenz.rank`); α_h = 0.016666666666666666 (`resultados.json:resultados.longitud.techniques.lorenz.alpha_h`); rechazo = true (`resultados.json:resultados.longitud.techniques.lorenz.reject`); calificación = null (`resultados.json:resultados.longitud.techniques.lorenz.calificacion`)

## e) Cochran y McNemar

- Cochran `Q`: 3.2941176470588234 (`resultados.json:resultados.mcnemar.cochran.Q`)
- Cochran `p`: 0.19261559277384016 (`resultados.json:resultados.mcnemar.cochran.p`)
- Cochran `gl`: 2 (`resultados.json:resultados.mcnemar.cochran.gl`)
- Cochran `alpha`: 0.05 (`resultados.json:resultados.mcnemar.cochran.alpha`)
- Cochran `todos_concordantes`: false (`resultados.json:resultados.mcnemar.cochran.todos_concordantes`)
- Pares: null (`resultados.json:resultados.mcnemar.pares`)

### Frente a `beto`

- gray_scott: b = 1 (`resultados.json:resultados.mcnemar.frente_a.beto.gray_scott.b`); c = 3 (`resultados.json:resultados.mcnemar.frente_a.beto.gray_scott.c`); p = 0.9375 (`resultados.json:resultados.mcnemar.frente_a.beto.gray_scott.p`); lado = unilateral (`resultados.json:resultados.mcnemar.frente_a.beto.gray_scott.lado`); rango de Holm = 2 (`resultados.json:resultados.mcnemar.frente_a.beto.gray_scott.rank`); α_h = 0.025 (`resultados.json:resultados.mcnemar.frente_a.beto.gray_scott.alpha_h`); rechazo = false (`resultados.json:resultados.mcnemar.frente_a.beto.gray_scott.reject`)
- perlin: b = 7 (`resultados.json:resultados.mcnemar.frente_a.beto.perlin.b`); c = 5 (`resultados.json:resultados.mcnemar.frente_a.beto.perlin.c`); p = 0.38720703125 (`resultados.json:resultados.mcnemar.frente_a.beto.perlin.p`); lado = unilateral (`resultados.json:resultados.mcnemar.frente_a.beto.perlin.lado`); rango de Holm = 1 (`resultados.json:resultados.mcnemar.frente_a.beto.perlin.rank`); α_h = 0.016666666666666666 (`resultados.json:resultados.mcnemar.frente_a.beto.perlin.alpha_h`); rechazo = false (`resultados.json:resultados.mcnemar.frente_a.beto.perlin.reject`)
- lorenz: b = 3 (`resultados.json:resultados.mcnemar.frente_a.beto.lorenz.b`); c = 7 (`resultados.json:resultados.mcnemar.frente_a.beto.lorenz.c`); p = 0.9453125 (`resultados.json:resultados.mcnemar.frente_a.beto.lorenz.p`); lado = unilateral (`resultados.json:resultados.mcnemar.frente_a.beto.lorenz.lado`); rango de Holm = 3 (`resultados.json:resultados.mcnemar.frente_a.beto.lorenz.rank`); α_h = 0.05 (`resultados.json:resultados.mcnemar.frente_a.beto.lorenz.alpha_h`); rechazo = false (`resultados.json:resultados.mcnemar.frente_a.beto.lorenz.reject`)

### Frente a `techo5d`

- gray_scott: b = 1 (`resultados.json:resultados.mcnemar.frente_a.techo5d.gray_scott.b`); c = 4 (`resultados.json:resultados.mcnemar.frente_a.techo5d.gray_scott.c`); p = 0.96875 (`resultados.json:resultados.mcnemar.frente_a.techo5d.gray_scott.p`); lado = unilateral (`resultados.json:resultados.mcnemar.frente_a.techo5d.gray_scott.lado`); rango de Holm = 3 (`resultados.json:resultados.mcnemar.frente_a.techo5d.gray_scott.rank`); α_h = 0.05 (`resultados.json:resultados.mcnemar.frente_a.techo5d.gray_scott.alpha_h`); rechazo = false (`resultados.json:resultados.mcnemar.frente_a.techo5d.gray_scott.reject`)
- perlin: b = 7 (`resultados.json:resultados.mcnemar.frente_a.techo5d.perlin.b`); c = 6 (`resultados.json:resultados.mcnemar.frente_a.techo5d.perlin.c`); p = 0.5 (`resultados.json:resultados.mcnemar.frente_a.techo5d.perlin.p`); lado = unilateral (`resultados.json:resultados.mcnemar.frente_a.techo5d.perlin.lado`); rango de Holm = 1 (`resultados.json:resultados.mcnemar.frente_a.techo5d.perlin.rank`); α_h = 0.016666666666666666 (`resultados.json:resultados.mcnemar.frente_a.techo5d.perlin.alpha_h`); rechazo = false (`resultados.json:resultados.mcnemar.frente_a.techo5d.perlin.reject`)
- lorenz: b = 3 (`resultados.json:resultados.mcnemar.frente_a.techo5d.lorenz.b`); c = 8 (`resultados.json:resultados.mcnemar.frente_a.techo5d.lorenz.c`); p = 0.96728515625 (`resultados.json:resultados.mcnemar.frente_a.techo5d.lorenz.p`); lado = unilateral (`resultados.json:resultados.mcnemar.frente_a.techo5d.lorenz.lado`); rango de Holm = 2 (`resultados.json:resultados.mcnemar.frente_a.techo5d.lorenz.rank`); α_h = 0.025 (`resultados.json:resultados.mcnemar.frente_a.techo5d.lorenz.alpha_h`); rechazo = false (`resultados.json:resultados.mcnemar.frente_a.techo5d.lorenz.reject`)

## f) Sensibilidad (§8)

| Variante | gray_scott | perlin | lorenz | Global | Ruta |
|---|---|---|---|---|---|
| principal | V1 | V1 | V1 | G1 | `resultados.json:resultados.principal` |
| retencion_exigida/r=0.75 | V1 | V4 | V1 | G1 | `resultados.json:resultados.sensibilidad.retencion_exigida.r=0.75` |
| retencion_exigida/r=0.8 | V1 | V4 | V1 | G1 | `resultados.json:resultados.sensibilidad.retencion_exigida.r=0.8` |
| formulacion_A/r=0.5 | V1 | V1 | V1 | G1 | `resultados.json:resultados.sensibilidad.formulacion_A.r=0.5` |
| formulacion_A/r=0.75 | V1 | V4 | V1 | G1 | `resultados.json:resultados.sensibilidad.formulacion_A.r=0.75` |
| formulacion_A/r=0.8 | V1 | V4 | V1 | G1 | `resultados.json:resultados.sensibilidad.formulacion_A.r=0.8` |
| variabilidad_beto/+0.012814512390867394 | V1 | V1 | V1 | G1 | `resultados.json:resultados.sensibilidad.variabilidad_beto.+0.012814512390867394` |
| variabilidad_beto/-0.012814512390867394 | V1 | V1 | V1 | G1 | `resultados.json:resultados.sensibilidad.variabilidad_beto.-0.012814512390867394` |
| por_semilla/42 | V1 | V4 | V1 | G1 | `resultados.json:resultados.sensibilidad.por_semilla.por_semilla.42` |
| por_semilla/7 | V1 | V1 | V1 | G1 | `resultados.json:resultados.sensibilidad.por_semilla.por_semilla.7` |
| por_semilla/123 | V1 | V1 | V1 | G1 | `resultados.json:resultados.sensibilidad.por_semilla.por_semilla.123` |
| por_semilla/2024 | V1 | V4 | V1 | G1 | `resultados.json:resultados.sensibilidad.por_semilla.por_semilla.2024` |
| por_semilla/99 | V1 | V1 | V1 | G1 | `resultados.json:resultados.sensibilidad.por_semilla.por_semilla.99` |
| segundo_mejor | V1 | V1 | V1 | G1 | `resultados.json:resultados.sensibilidad.segundo_mejor` |
| piso_sin_permutado | V1 | V1 | V1 | G1 | `resultados.json:resultados.sensibilidad.piso_sin_permutado` |

- `n_semillas_V1`: {'gray_scott': 5, 'perlin': 3, 'lorenz': 5} (`resultados.json:resultados.sensibilidad.por_semilla.n_semillas_V1`)

Parámetros registrados en cada variante (`params`):

- `retencion_exigida/r=0.75`: r = 0.75, formulation = B, δ = null, beto_shift = 0.0, floors = [majority, stratified, brightness, visual_permutado], fase = final
- `retencion_exigida/r=0.8`: r = 0.8, formulation = B, δ = null, beto_shift = 0.0, floors = [majority, stratified, brightness, visual_permutado], fase = final
- `formulacion_A/r=0.5`: r = 0.5, formulation = A, δ = 0.10194805194805195, beto_shift = 0.0, floors = [majority, stratified, brightness, visual_permutado], fase = final
- `formulacion_A/r=0.75`: r = 0.5, formulation = A, δ = 0.05097402597402598, beto_shift = 0.0, floors = [majority, stratified, brightness, visual_permutado], fase = final
- `formulacion_A/r=0.8`: r = 0.5, formulation = A, δ = 0.04077922077922077, beto_shift = 0.0, floors = [majority, stratified, brightness, visual_permutado], fase = final
- `variabilidad_beto/+0.012814512390867394`: r = 0.5, formulation = B, δ = null, beto_shift = 0.012814512390867394, floors = [majority, stratified, brightness, visual_permutado], fase = final
- `variabilidad_beto/-0.012814512390867394`: r = 0.5, formulation = B, δ = null, beto_shift = -0.012814512390867394, floors = [majority, stratified, brightness, visual_permutado], fase = final
- `segundo_mejor`: r = 0.5, formulation = B, δ = null, beto_shift = 0.0, floors = [majority, stratified, brightness, visual_permutado], fase = sensibilidad
- `piso_sin_permutado`: r = 0.5, formulation = B, δ = null, beto_shift = 0.0, floors = [majority, stratified, brightness], fase = final

Nota de transcripción (sin interpretar): en `formulacion_A/r=0.75` y `formulacion_A/r=0.8`, `params.r` registra 0.5 y `params.delta` difiere entre variantes, con los valores de arriba.

Cifras por técnica de cada variante: `veredictos.csv` y `resultados.json:resultados.sensibilidad.<variante>.techniques.<técnica>`.

## g) Robustez

| # | Alternativa | gray_scott | perlin | lorenz | Global | Cambia técnica | Cambia global | Nota |
|---|---|---|---|---|---|---|---|---|
| 0 | A2: máximo por réplica | V1 | V1 | V1 | G1 | {"gray_scott": false, "perlin": false, "lorenz": false} | false | null |
| 1 | A3: primero de los empatados según pisos.order (Enmienda 5, P2) | V1 | V1 | V1 | G1 | {"gray_scott": false, "perlin": false, "lorenz": false} | false | sin empate en el máximo observado: igual al principal |
| 2 | A7: m = 3 y calificación en cada variante de §8 | V1 | V1 | V1 | G1 | {"gray_scott": false, "perlin": false, "lorenz": false} | false | L solo califica; V y G no cambian |
| 3 | A12: McNemar bilateral frente a BETO y al techo | V1 | V1 | V1 | G1 | {"gray_scott": false, "perlin": false, "lorenz": false} | false | §7: McNemar no modifica veredictos |
| 4 | A14: piso permutado de la semilla emparejada, si es mayor (por réplica; Enmienda 5, P1) | — | — | — | — | — | — | null |

Ruta: `resultados.json:resultados.robustez[i]`. Campos adicionales:

- `robustez[2].calificaciones` (L, p, rechazo y calificación de longitud por variante):

  | Variante | Técnica | L | p | α_h | rechazo | calificación |
  |---|---|---|---|---|---|---|
  | principal | gray_scott | 0.21079301647974102 | 0.0001 | 0.016666666666666666 | true | null |
  | principal | perlin | 0.1576648121841686 | 0.0029 | 0.05 | true | null |
  | principal | lorenz | 0.21654464356512793 | 0.0001 | 0.016666666666666666 | true | null |
  | retencion_exigida/r=0.75 | gray_scott | 0.21079301647974102 | 0.0001 | 0.016666666666666666 | true | null |
  | retencion_exigida/r=0.75 | perlin | 0.1576648121841686 | 0.0029 | 0.05 | true | null |
  | retencion_exigida/r=0.75 | lorenz | 0.21654464356512793 | 0.0001 | 0.016666666666666666 | true | null |
  | retencion_exigida/r=0.8 | gray_scott | 0.21079301647974102 | 0.0001 | 0.016666666666666666 | true | null |
  | retencion_exigida/r=0.8 | perlin | 0.1576648121841686 | 0.0029 | 0.05 | true | null |
  | retencion_exigida/r=0.8 | lorenz | 0.21654464356512793 | 0.0001 | 0.016666666666666666 | true | null |
  | formulacion_A/r=0.5 | gray_scott | 0.21079301647974102 | 0.0001 | 0.016666666666666666 | true | null |
  | formulacion_A/r=0.5 | perlin | 0.1576648121841686 | 0.0029 | 0.05 | true | null |
  | formulacion_A/r=0.5 | lorenz | 0.21654464356512793 | 0.0001 | 0.016666666666666666 | true | null |
  | formulacion_A/r=0.75 | gray_scott | 0.21079301647974102 | 0.0001 | 0.016666666666666666 | true | null |
  | formulacion_A/r=0.75 | perlin | 0.1576648121841686 | 0.0029 | 0.05 | true | null |
  | formulacion_A/r=0.75 | lorenz | 0.21654464356512793 | 0.0001 | 0.016666666666666666 | true | null |
  | formulacion_A/r=0.8 | gray_scott | 0.21079301647974102 | 0.0001 | 0.016666666666666666 | true | null |
  | formulacion_A/r=0.8 | perlin | 0.1576648121841686 | 0.0029 | 0.05 | true | null |
  | formulacion_A/r=0.8 | lorenz | 0.21654464356512793 | 0.0001 | 0.016666666666666666 | true | null |
  | por_semilla/42 | gray_scott | 0.230680365031388 | 0.0 | 0.016666666666666666 | true | null |
  | por_semilla/42 | perlin | 0.14393939393939403 | 0.0147 | 0.05 | true | null |
  | por_semilla/42 | lorenz | 0.19791666666666674 | 0.0004 | 0.025 | true | null |
  | por_semilla/7 | gray_scott | 0.2124578586375262 | 0.0002 | 0.025 | true | null |
  | por_semilla/7 | perlin | 0.17601284584980248 | 0.002 | 0.05 | true | null |
  | por_semilla/7 | lorenz | 0.2173496146710433 | 0.0 | 0.016666666666666666 | true | null |
  | por_semilla/123 | gray_scott | 0.18403881702239366 | 0.0016 | 0.05 | true | null |
  | por_semilla/123 | perlin | 0.1717952971518919 | 0.0013 | 0.025 | true | null |
  | por_semilla/123 | lorenz | 0.230680365031388 | 0.0001 | 0.016666666666666666 | true | null |
  | por_semilla/2024 | gray_scott | 0.22247406864342356 | 0.0001 | 0.016666666666666666 | true | null |
  | por_semilla/2024 | perlin | 0.14576405737120024 | 0.0117 | 0.05 | true | null |
  | por_semilla/2024 | lorenz | 0.230680365031388 | 0.0001 | 0.016666666666666666 | true | null |
  | por_semilla/99 | gray_scott | 0.20431397306397314 | 0.0003 | 0.025 | true | null |
  | por_semilla/99 | perlin | 0.15081246660855385 | 0.0088 | 0.05 | true | null |
  | por_semilla/99 | lorenz | 0.2060962064251538 | 0.0002 | 0.016666666666666666 | true | null |
  | variabilidad_beto/+0.012814512390867394 | gray_scott | 0.21079301647974102 | 0.0001 | 0.016666666666666666 | true | null |
  | variabilidad_beto/+0.012814512390867394 | perlin | 0.1576648121841686 | 0.0029 | 0.05 | true | null |
  | variabilidad_beto/+0.012814512390867394 | lorenz | 0.21654464356512793 | 0.0001 | 0.016666666666666666 | true | null |
  | variabilidad_beto/-0.012814512390867394 | gray_scott | 0.21079301647974102 | 0.0001 | 0.016666666666666666 | true | null |
  | variabilidad_beto/-0.012814512390867394 | perlin | 0.1576648121841686 | 0.0029 | 0.05 | true | null |
  | variabilidad_beto/-0.012814512390867394 | lorenz | 0.21654464356512793 | 0.0001 | 0.016666666666666666 | true | null |
  | segundo_mejor | gray_scott | 0.20081685420338458 | 0.0008 | 0.016666666666666666 | true | null |
  | segundo_mejor | perlin | 0.17209374826837742 | 0.003 | 0.05 | true | null |
  | segundo_mejor | lorenz | 0.1799511210232161 | 0.0011 | 0.025 | true | null |
  | piso_sin_permutado | gray_scott | 0.21079301647974102 | 0.0001 | 0.016666666666666666 | true | null |
  | piso_sin_permutado | perlin | 0.1576648121841686 | 0.0029 | 0.05 | true | null |
  | piso_sin_permutado | lorenz | 0.21654464356512793 | 0.0001 | 0.016666666666666666 | true | null |

- `robustez[3]` (A12: McNemar bilateral frente a BETO y al techo):
  - `cambia_rechazo_mcnemar`: `{"beto": {"gray_scott": false, "perlin": false, "lorenz": false}, "techo5d": {"gray_scott": false, "perlin": false, "lorenz": false}}`
- `robustez[4]` (A14: piso permutado de la semilla emparejada, si es mayor (por réplica; Enmienda 5, P1)):
  - `n_semillas_V1`: `{"gray_scott": 4, "perlin": 2, "lorenz": 5}`
  - `cambia_n_semillas_V1`: `{"gray_scott": true, "perlin": true, "lorenz": false}`
  - `por_semilla`: `{"42": {"veredictos": {"gray_scott": "V1", "perlin": "V4", "lorenz": "V1"}, "global": "G1"}, "7": {"veredictos": {"gray_scott": "V1", "perlin": "V3", "lorenz": "V1"}, "global": "G1"}, "123": {"veredictos": {"gray_scott": "V1", "perlin": "V1", "lorenz": "V1"}, "global": "G1"}, "2024": {"veredictos": {"gray_scott": "V3", "perlin": "V4", "lorenz": "V1"}, "global": "G1"}, "99": {"veredictos": {"gray_scott": "V1", "perlin": "V1", "lorenz": "V1"}, "global": "G1"}}`

  Tabla de `robustez[4].por_semilla` (sin agregado global; ver `n_semillas_V1` y `cambia_n_semillas_V1` arriba):

  | Semilla | gray_scott | perlin | lorenz | Global |
  |---|---|---|---|---|
  | 42 | V1 | V4 | V1 | G1 |
  | 7 | V1 | V3 | V1 | G1 |
  | 123 | V1 | V1 | V1 | G1 |
  | 2024 | V3 | V4 | V1 | G1 |
  | 99 | V1 | V1 | V1 | G1 |


## h) Terciles de longitud y Spearman con u

- Cortes: [86.33333333333331, 381.0] (`resultados.json:resultados.terciles_longitud.cortes`)

**T1**: n = 36 (`resultados.json:resultados.terciles_longitud.estratos.T1.n`); n_dep = 16 (`resultados.json:resultados.terciles_longitud.estratos.T1.n_dep`); BETO = 0.71875 (`resultados.json:resultados.terciles_longitud.estratos.T1.beto`)

- gray_scott: media = 0.7419231237026184 (`resultados.json:resultados.terciles_longitud.estratos.T1.gray_scott.media`); por semilla [42, 7, 123, 2024, 99] = [0.7770897832817337, 0.7482517482517483, 0.6884343036978757, 0.7770897832817337, 0.71875] (`resultados.json:resultados.terciles_longitud.estratos.T1.gray_scott.por_semilla`)
- perlin: media = 0.7741128995216728 (`resultados.json:resultados.terciles_longitud.estratos.T1.perlin.media`); por semilla [42, 7, 123, 2024, 99] = [0.7401764234161989, 0.8017309205350118, 0.7482517482517483, 0.775, 0.8054054054054054] (`resultados.json:resultados.terciles_longitud.estratos.T1.perlin.por_semilla`)
- lorenz: media = 0.7661766498918202 (`resultados.json:resultados.terciles_longitud.estratos.T1.lorenz.media`); por semilla [42, 7, 123, 2024, 99] = [0.7498069498069498, 0.7770897832817337, 0.7770897832817337, 0.7770897832817337, 0.7498069498069498] (`resultados.json:resultados.terciles_longitud.estratos.T1.lorenz.por_semilla`)

**T2**: n = 42 (`resultados.json:resultados.terciles_longitud.estratos.T2.n`); n_dep = 19 (`resultados.json:resultados.terciles_longitud.estratos.T2.n_dep`); BETO = 0.7083333333333333 (`resultados.json:resultados.terciles_longitud.estratos.T2.beto`)

- gray_scott: media = 0.7416744366744367 (`resultados.json:resultados.terciles_longitud.estratos.T2.gray_scott.media`); por semilla [42, 7, 123, 2024, 99] = [0.7367521367521368, 0.7367521367521368, 0.7367521367521368, 0.7367521367521368, 0.7613636363636364] (`resultados.json:resultados.terciles_longitud.estratos.T2.gray_scott.por_semilla`)
- perlin: media = 0.5988032121100494 (`resultados.json:resultados.terciles_longitud.estratos.T2.perlin.media`); por semilla [42, 7, 123, 2024, 99] = [0.5714285714285714, 0.641025641025641, 0.6190476190476191, 0.595008508224617, 0.5675057208237986] (`resultados.json:resultados.terciles_longitud.estratos.T2.perlin.por_semilla`)
- lorenz: media = 0.693968253968254 (`resultados.json:resultados.terciles_longitud.estratos.T2.lorenz.media`); por semilla [42, 7, 123, 2024, 99] = [0.6888888888888889, 0.7142857142857143, 0.6888888888888889, 0.6888888888888889, 0.6888888888888889] (`resultados.json:resultados.terciles_longitud.estratos.T2.lorenz.por_semilla`)

**T3**: n = 36 (`resultados.json:resultados.terciles_longitud.estratos.T3.n`); n_dep = 9 (`resultados.json:resultados.terciles_longitud.estratos.T3.n_dep`); BETO = 0.6538461538461539 (`resultados.json:resultados.terciles_longitud.estratos.T3.beto`)

- gray_scott: media = 0.6492897487464927 (`resultados.json:resultados.terciles_longitud.estratos.T3.gray_scott.media`); por semilla [42, 7, 123, 2024, 99] = [0.6782522343594837, 0.6538461538461539, 0.6302521008403361, 0.6538461538461539, 0.6302521008403361] (`resultados.json:resultados.terciles_longitud.estratos.T3.gray_scott.por_semilla`)
- perlin: media = 0.6258838188374332 (`resultados.json:resultados.terciles_longitud.estratos.T3.perlin.media`); por semilla [42, 7, 123, 2024, 99] = [0.6629787234042553, 0.6072727272727273, 0.6727272727272727, 0.5847382431233363, 0.6017021276595744] (`resultados.json:resultados.terciles_longitud.estratos.T3.perlin.por_semilla`)
- lorenz: media = 0.7084874025442984 (`resultados.json:resultados.terciles_longitud.estratos.T3.lorenz.media`); por semilla [42, 7, 123, 2024, 99] = [0.6727272727272727, 0.6727272727272727, 0.7497517378351539, 0.7497517378351539, 0.6974789915966386] (`resultados.json:resultados.terciles_longitud.estratos.T3.lorenz.por_semilla`)

### Spearman u – longitud (train, n = 908)

- `u0__log(n_tokens_beto)`: ρ = -0.03228805721620575 (`resultados.json:resultados.spearman_u_longitud.u0__log(n_tokens_beto).rho`); p = 0.3311249710030675 (`resultados.json:resultados.spearman_u_longitud.u0__log(n_tokens_beto).p`)
- `u0__truncated_128`: ρ = 0.0016207233914768194 (`resultados.json:resultados.spearman_u_longitud.u0__truncated_128.rho`); p = 0.9611025284696256 (`resultados.json:resultados.spearman_u_longitud.u0__truncated_128.p`)
- `u1__log(n_tokens_beto)`: ρ = 0.03720555720234443 (`resultados.json:resultados.spearman_u_longitud.u1__log(n_tokens_beto).rho`); p = 0.2627308567714919 (`resultados.json:resultados.spearman_u_longitud.u1__log(n_tokens_beto).p`)
- `u1__truncated_128`: ρ = 0.025965694756081254 (`resultados.json:resultados.spearman_u_longitud.u1__truncated_128.rho`); p = 0.43452069067900334 (`resultados.json:resultados.spearman_u_longitud.u1__truncated_128.p`)
- `u2__log(n_tokens_beto)`: ρ = 0.04615882890252544 (`resultados.json:resultados.spearman_u_longitud.u2__log(n_tokens_beto).rho`); p = 0.16461071856798706 (`resultados.json:resultados.spearman_u_longitud.u2__log(n_tokens_beto).p`)
- `u2__truncated_128`: ρ = 0.07137154008677132 (`resultados.json:resultados.spearman_u_longitud.u2__truncated_128.rho`); p = 0.03152187883500595 (`resultados.json:resultados.spearman_u_longitud.u2__truncated_128.p`)
- `u3__log(n_tokens_beto)`: ρ = 0.23432772600619392 (`resultados.json:resultados.spearman_u_longitud.u3__log(n_tokens_beto).rho`); p = 8.610131744063864e-13 (`resultados.json:resultados.spearman_u_longitud.u3__log(n_tokens_beto).p`)
- `u3__truncated_128`: ρ = 0.22678185308317225 (`resultados.json:resultados.spearman_u_longitud.u3__truncated_128.rho`); p = 4.695302008663352e-12 (`resultados.json:resultados.spearman_u_longitud.u3__truncated_128.p`)
- `u4__log(n_tokens_beto)`: ρ = 0.032814716957135774 (`resultados.json:resultados.spearman_u_longitud.u4__log(n_tokens_beto).rho`); p = 0.32329419623266653 (`resultados.json:resultados.spearman_u_longitud.u4__log(n_tokens_beto).p`)
- `u4__truncated_128`: ρ = 0.02232333218681493 (`resultados.json:resultados.spearman_u_longitud.u4__truncated_128.rho`); p = 0.5016938866288099 (`resultados.json:resultados.spearman_u_longitud.u4__truncated_128.p`)

## i) F4.6: kappa y contención

### Kappa (`y_hat`)

- beto__emb768: obs = 0.8870541611624834 (`f46_resultados.json:resultados.kappa.y_hat.beto__emb768.obs`); IC = [0.7907907907907908, 0.9632968448164841] (`f46_resultados.json:resultados.kappa.y_hat.beto__emb768.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.y_hat.beto__emb768.n_replicas_indefinidas`)
- beto__techo5d: obs = 0.9070753179002282 (`f46_resultados.json:resultados.kappa.y_hat.beto__techo5d.obs`); IC = [0.8201325339223728, 0.9810568295114656] (`f46_resultados.json:resultados.kappa.y_hat.beto__techo5d.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.y_hat.beto__techo5d.n_replicas_indefinidas`)
- beto__gray_scott: obs = 0.9272030651340997 (`f46_resultados.json:resultados.kappa.y_hat.beto__gray_scott.obs`); IC = [0.8484546360917249, 0.9823474759987612] (`f46_resultados.json:resultados.kappa.y_hat.beto__gray_scott.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.y_hat.beto__gray_scott.n_replicas_indefinidas`)
- beto__perlin: obs = 0.785175879396985 (`f46_resultados.json:resultados.kappa.y_hat.beto__perlin.obs`); IC = [0.6657440775747373, 0.8923512747875353] (`f46_resultados.json:resultados.kappa.y_hat.beto__perlin.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.y_hat.beto__perlin.n_replicas_indefinidas`)
- beto__lorenz: obs = 0.818007662835249 (`f46_resultados.json:resultados.kappa.y_hat.beto__lorenz.obs`); IC = [0.6998946998947, 0.922998986828774] (`f46_resultados.json:resultados.kappa.y_hat.beto__lorenz.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.y_hat.beto__lorenz.n_replicas_indefinidas`)
- emb768__techo5d: obs = 0.9432082364662904 (`f46_resultados.json:resultados.kappa.y_hat.emb768__techo5d.obs`); IC = [0.8692232055063914, 1.0] (`f46_resultados.json:resultados.kappa.y_hat.emb768__techo5d.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.y_hat.emb768__techo5d.n_replicas_indefinidas`)
- emb768__gray_scott: obs = 0.8527131782945736 (`f46_resultados.json:resultados.kappa.y_hat.emb768__gray_scott.obs`); IC = [0.7487283825025431, 0.9447495961227788] (`f46_resultados.json:resultados.kappa.y_hat.emb768__gray_scott.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.y_hat.emb768__gray_scott.n_replicas_indefinidas`)
- emb768__perlin: obs = 0.7117572692793932 (`f46_resultados.json:resultados.kappa.y_hat.emb768__perlin.obs`); IC = [0.5767315343497541, 0.8360498561840843] (`f46_resultados.json:resultados.kappa.y_hat.emb768__perlin.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.y_hat.emb768__perlin.n_replicas_indefinidas`)
- emb768__lorenz: obs = 0.7422480620155038 (`f46_resultados.json:resultados.kappa.y_hat.emb768__lorenz.obs`); IC = [0.6076696165191741, 0.8599938593797973] (`f46_resultados.json:resultados.kappa.y_hat.emb768__lorenz.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.y_hat.emb768__lorenz.n_replicas_indefinidas`)
- techo5d__gray_scott: obs = 0.9087415946205571 (`f46_resultados.json:resultados.kappa.y_hat.techo5d__gray_scott.obs`); IC = [0.8214844973379266, 0.9814150635800456] (`f46_resultados.json:resultados.kappa.y_hat.techo5d__gray_scott.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.y_hat.techo5d__gray_scott.n_replicas_indefinidas`)
- techo5d__perlin: obs = 0.7669078326517772 (`f46_resultados.json:resultados.kappa.y_hat.techo5d__perlin.obs`); IC = [0.6393293269492553, 0.8761254268860603] (`f46_resultados.json:resultados.kappa.y_hat.techo5d__perlin.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.y_hat.techo5d__perlin.n_replicas_indefinidas`)
- techo5d__lorenz: obs = 0.7992315081652258 (`f46_resultados.json:resultados.kappa.y_hat.techo5d__lorenz.obs`); IC = [0.6778649921507063, 0.906283906480076] (`f46_resultados.json:resultados.kappa.y_hat.techo5d__lorenz.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.y_hat.techo5d__lorenz.n_replicas_indefinidas`)

### Kappa (`vector_acierto`)

- beto__emb768: obs = 0.8696646341463413 (`f46_resultados.json:resultados.kappa.vector_acierto.beto__emb768.obs`); IC = [0.7544065804935371, 0.95877034358047] (`f46_resultados.json:resultados.kappa.vector_acierto.beto__emb768.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.vector_acierto.beto__emb768.n_replicas_indefinidas`)
- beto__techo5d: obs = 0.8924122310305775 (`f46_resultados.json:resultados.kappa.vector_acierto.beto__techo5d.obs`); IC = [0.7905181918412347, 0.9776207302709069] (`f46_resultados.json:resultados.kappa.vector_acierto.beto__techo5d.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.vector_acierto.beto__techo5d.n_replicas_indefinidas`)
- beto__gray_scott: obs = 0.9114219114219114 (`f46_resultados.json:resultados.kappa.vector_acierto.beto__gray_scott.obs`); IC = [0.815083282746776, 0.9798515376458111] (`f46_resultados.json:resultados.kappa.vector_acierto.beto__gray_scott.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.vector_acierto.beto__gray_scott.n_replicas_indefinidas`)
- beto__perlin: obs = 0.7442034405385192 (`f46_resultados.json:resultados.kappa.vector_acierto.beto__perlin.obs`); IC = [0.5941600751399639, 0.8698630136986301] (`f46_resultados.json:resultados.kappa.vector_acierto.beto__perlin.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.vector_acierto.beto__perlin.n_replicas_indefinidas`)
- beto__lorenz: obs = 0.7741679873217114 (`f46_resultados.json:resultados.kappa.vector_acierto.beto__lorenz.obs`); IC = [0.6288095238095239, 0.9003496503496503] (`f46_resultados.json:resultados.kappa.vector_acierto.beto__lorenz.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.vector_acierto.beto__lorenz.n_replicas_indefinidas`)
- emb768__techo5d: obs = 0.9354473386183466 (`f46_resultados.json:resultados.kappa.vector_acierto.emb768__techo5d.obs`); IC = [0.850785340314136, 1.0] (`f46_resultados.json:resultados.kappa.vector_acierto.emb768__techo5d.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.vector_acierto.emb768__techo5d.n_replicas_indefinidas`)
- emb768__gray_scott: obs = 0.8228438228438228 (`f46_resultados.json:resultados.kappa.vector_acierto.emb768__gray_scott.obs`); IC = [0.6931285531783948, 0.9328621908127209] (`f46_resultados.json:resultados.kappa.vector_acierto.emb768__gray_scott.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.vector_acierto.emb768__gray_scott.n_replicas_indefinidas`)
- emb768__perlin: obs = 0.6589379207180255 (`f46_resultados.json:resultados.kappa.vector_acierto.emb768__perlin.obs`); IC = [0.4914798421519988, 0.8051948051948051] (`f46_resultados.json:resultados.kappa.vector_acierto.emb768__perlin.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.vector_acierto.emb768__perlin.n_replicas_indefinidas`)
- emb768__lorenz: obs = 0.6838351822503961 (`f46_resultados.json:resultados.kappa.vector_acierto.emb768__lorenz.obs`); IC = [0.5210084033613444, 0.8294051627384961] (`f46_resultados.json:resultados.kappa.vector_acierto.emb768__lorenz.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.vector_acierto.emb768__lorenz.n_replicas_indefinidas`)
- techo5d__gray_scott: obs = 0.8904267589388696 (`f46_resultados.json:resultados.kappa.vector_acierto.techo5d__gray_scott.obs`); IC = [0.7854454203262232, 0.9766105867870332] (`f46_resultados.json:resultados.kappa.vector_acierto.techo5d__gray_scott.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.vector_acierto.techo5d__gray_scott.n_replicas_indefinidas`)
- techo5d__perlin: obs = 0.725250278086763 (`f46_resultados.json:resultados.kappa.vector_acierto.techo5d__perlin.obs`); IC = [0.5705660131517556, 0.8546448087431695] (`f46_resultados.json:resultados.kappa.vector_acierto.techo5d__perlin.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.vector_acierto.techo5d__perlin.n_replicas_indefinidas`)
- techo5d__lorenz: obs = 0.7544065804935371 (`f46_resultados.json:resultados.kappa.vector_acierto.techo5d__lorenz.obs`); IC = [0.6059204496325116, 0.8817427385892115] (`f46_resultados.json:resultados.kappa.vector_acierto.techo5d__lorenz.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.kappa.vector_acierto.techo5d__lorenz.n_replicas_indefinidas`)

### Contención (errores de la imagen contenidos en los de la referencia)

**gray_scott**

- frente a techo5d: C = 0.9666666666666667 (`f46_resultados.json:resultados.contencion.gray_scott.techo5d.C_obs`); nula = 0.2894736842105263 (`f46_resultados.json:resultados.contencion.gray_scott.techo5d.nula_obs`); IC = [0.8888888888888888, 1.0] (`f46_resultados.json:resultados.contencion.gray_scott.techo5d.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.contencion.gray_scott.techo5d.n_replicas_indefinidas`); errores imagen = 30 (`f46_resultados.json:resultados.contencion.gray_scott.techo5d.n_errores_vis`); errores referencia = 33 (`f46_resultados.json:resultados.contencion.gray_scott.techo5d.n_errores_ref`); intersección = 29 (`f46_resultados.json:resultados.contencion.gray_scott.techo5d.n_interseccion`)
- frente a beto: C = 0.9666666666666667 (`f46_resultados.json:resultados.contencion.gray_scott.beto.C_obs`); nula = 0.2807017543859649 (`f46_resultados.json:resultados.contencion.gray_scott.beto.nula_obs`); IC = [0.8888888888888888, 1.0] (`f46_resultados.json:resultados.contencion.gray_scott.beto.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.contencion.gray_scott.beto.n_replicas_indefinidas`); errores imagen = 30 (`f46_resultados.json:resultados.contencion.gray_scott.beto.n_errores_vis`); errores referencia = 32 (`f46_resultados.json:resultados.contencion.gray_scott.beto.n_errores_ref`); intersección = 29 (`f46_resultados.json:resultados.contencion.gray_scott.beto.n_interseccion`)
- frente a emb768: C = 0.9 (`f46_resultados.json:resultados.contencion.gray_scott.emb768.C_obs`); nula = 0.2807017543859649 (`f46_resultados.json:resultados.contencion.gray_scott.emb768.nula_obs`); IC = [0.7777777777777778, 1.0] (`f46_resultados.json:resultados.contencion.gray_scott.emb768.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.contencion.gray_scott.emb768.n_replicas_indefinidas`); errores imagen = 30 (`f46_resultados.json:resultados.contencion.gray_scott.emb768.n_errores_vis`); errores referencia = 32 (`f46_resultados.json:resultados.contencion.gray_scott.emb768.n_errores_ref`); intersección = 27 (`f46_resultados.json:resultados.contencion.gray_scott.emb768.n_interseccion`)
- **Condición 1**: true (`f46_resultados.json:resultados.contencion.gray_scott.condicion_1`)
- **Condición 2**: PENDIENTE (P6): requiere el IC de ΔF1 de F4.1 (`f46_resultados.json:resultados.contencion.gray_scott.condicion_2_F41`)

**perlin**

- frente a techo5d: C = 0.7941176470588235 (`f46_resultados.json:resultados.contencion.perlin.techo5d.C_obs`); nula = 0.2894736842105263 (`f46_resultados.json:resultados.contencion.perlin.techo5d.nula_obs`); IC = [0.6470588235294118, 0.9230769230769231] (`f46_resultados.json:resultados.contencion.perlin.techo5d.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.contencion.perlin.techo5d.n_replicas_indefinidas`); errores imagen = 34 (`f46_resultados.json:resultados.contencion.perlin.techo5d.n_errores_vis`); errores referencia = 33 (`f46_resultados.json:resultados.contencion.perlin.techo5d.n_errores_ref`); intersección = 27 (`f46_resultados.json:resultados.contencion.perlin.techo5d.n_interseccion`)
- frente a beto: C = 0.7941176470588235 (`f46_resultados.json:resultados.contencion.perlin.beto.C_obs`); nula = 0.2807017543859649 (`f46_resultados.json:resultados.contencion.perlin.beto.nula_obs`); IC = [0.6470588235294118, 0.9210526315789473] (`f46_resultados.json:resultados.contencion.perlin.beto.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.contencion.perlin.beto.n_replicas_indefinidas`); errores imagen = 34 (`f46_resultados.json:resultados.contencion.perlin.beto.n_errores_vis`); errores referencia = 32 (`f46_resultados.json:resultados.contencion.perlin.beto.n_errores_ref`); intersección = 27 (`f46_resultados.json:resultados.contencion.perlin.beto.n_interseccion`)
- frente a emb768: C = 0.7352941176470589 (`f46_resultados.json:resultados.contencion.perlin.emb768.C_obs`); nula = 0.2807017543859649 (`f46_resultados.json:resultados.contencion.perlin.emb768.nula_obs`); IC = [0.5789473684210527, 0.8787878787878788] (`f46_resultados.json:resultados.contencion.perlin.emb768.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.contencion.perlin.emb768.n_replicas_indefinidas`); errores imagen = 34 (`f46_resultados.json:resultados.contencion.perlin.emb768.n_errores_vis`); errores referencia = 32 (`f46_resultados.json:resultados.contencion.perlin.emb768.n_errores_ref`); intersección = 25 (`f46_resultados.json:resultados.contencion.perlin.emb768.n_interseccion`)
- **Condición 1**: true (`f46_resultados.json:resultados.contencion.perlin.condicion_1`)
- **Condición 2**: PENDIENTE (P6): requiere el IC de ΔF1 de F4.1 (`f46_resultados.json:resultados.contencion.perlin.condicion_2_F41`)

**lorenz**

- frente a techo5d: C = 0.8928571428571429 (`f46_resultados.json:resultados.contencion.lorenz.techo5d.C_obs`); nula = 0.2894736842105263 (`f46_resultados.json:resultados.contencion.lorenz.techo5d.nula_obs`); IC = [0.7646671826625386, 1.0] (`f46_resultados.json:resultados.contencion.lorenz.techo5d.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.contencion.lorenz.techo5d.n_replicas_indefinidas`); errores imagen = 28 (`f46_resultados.json:resultados.contencion.lorenz.techo5d.n_errores_vis`); errores referencia = 33 (`f46_resultados.json:resultados.contencion.lorenz.techo5d.n_errores_ref`); intersección = 25 (`f46_resultados.json:resultados.contencion.lorenz.techo5d.n_interseccion`)
- frente a beto: C = 0.8928571428571429 (`f46_resultados.json:resultados.contencion.lorenz.beto.C_obs`); nula = 0.2807017543859649 (`f46_resultados.json:resultados.contencion.lorenz.beto.nula_obs`); IC = [0.7619047619047619, 1.0] (`f46_resultados.json:resultados.contencion.lorenz.beto.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.contencion.lorenz.beto.n_replicas_indefinidas`); errores imagen = 28 (`f46_resultados.json:resultados.contencion.lorenz.beto.n_errores_vis`); errores referencia = 32 (`f46_resultados.json:resultados.contencion.lorenz.beto.n_errores_ref`); intersección = 25 (`f46_resultados.json:resultados.contencion.lorenz.beto.n_interseccion`)
- frente a emb768: C = 0.8214285714285714 (`f46_resultados.json:resultados.contencion.lorenz.emb768.C_obs`); nula = 0.2807017543859649 (`f46_resultados.json:resultados.contencion.lorenz.emb768.nula_obs`); IC = [0.6666666666666666, 0.96] (`f46_resultados.json:resultados.contencion.lorenz.emb768.interval`); réplicas indefinidas = 0 (`f46_resultados.json:resultados.contencion.lorenz.emb768.n_replicas_indefinidas`); errores imagen = 28 (`f46_resultados.json:resultados.contencion.lorenz.emb768.n_errores_vis`); errores referencia = 32 (`f46_resultados.json:resultados.contencion.lorenz.emb768.n_errores_ref`); intersección = 23 (`f46_resultados.json:resultados.contencion.lorenz.emb768.n_interseccion`)
- **Condición 1**: true (`f46_resultados.json:resultados.contencion.lorenz.condicion_1`)
- **Condición 2**: PENDIENTE (P6): requiere el IC de ΔF1 de F4.1 (`f46_resultados.json:resultados.contencion.lorenz.condicion_2_F41`)

### Contención por semilla (frente a techo5d)

| Técnica | Semilla | C | nula | IC | errores imagen | errores techo | intersección | indefinidas |
|---|---|---|---|---|---|---|---|---|
| gray_scott | 42 | 0.9642857142857143 | 0.2894736842105263 | [0.88, 1.0] | 28 | 33 | 27 | 0 |
| gray_scott | 7 | 0.9666666666666667 | 0.2894736842105263 | [0.8888888888888888, 1.0] | 30 | 33 | 29 | 0 |
| gray_scott | 123 | 0.9393939393939394 | 0.2894736842105263 | [0.84375, 1.0] | 33 | 33 | 31 | 0 |
| gray_scott | 2024 | 0.9655172413793104 | 0.2894736842105263 | [0.8846153846153846, 1.0] | 29 | 33 | 28 | 0 |
| gray_scott | 99 | 0.9354838709677419 | 0.2894736842105263 | [0.8333333333333334, 1.0] | 31 | 33 | 29 | 0 |
| perlin | 42 | 0.6842105263157895 | 0.2894736842105263 | [0.5277777777777778, 0.8285714285714286] | 38 | 33 | 26 | 0 |
| perlin | 7 | 0.7647058823529411 | 0.2894736842105263 | [0.6153846153846154, 0.9047619047619048] | 34 | 33 | 26 | 0 |
| perlin | 123 | 0.6857142857142857 | 0.2894736842105263 | [0.52, 0.8333333333333334] | 35 | 33 | 24 | 0 |
| perlin | 2024 | 0.6842105263157895 | 0.2894736842105263 | [0.53125, 0.8276108374384232] | 38 | 33 | 26 | 0 |
| perlin | 99 | 0.6578947368421053 | 0.2894736842105263 | [0.5, 0.8] | 38 | 33 | 25 | 0 |
| lorenz | 42 | 0.84375 | 0.2894736842105263 | [0.7083333333333334, 0.9655172413793104] | 32 | 33 | 27 | 0 |
| lorenz | 7 | 0.8 | 0.2894736842105263 | [0.6521739130434783, 0.9333333333333333] | 30 | 33 | 24 | 0 |
| lorenz | 123 | 0.9285714285714286 | 0.2894736842105263 | [0.8181818181818182, 1.0] | 28 | 33 | 26 | 0 |
| lorenz | 2024 | 0.8928571428571429 | 0.2894736842105263 | [0.7646671826625386, 1.0] | 28 | 33 | 25 | 0 |
| lorenz | 99 | 0.8709677419354839 | 0.2894736842105263 | [0.7419354838709677, 0.9714285714285714] | 31 | 33 | 27 | 0 |

Ruta: `f46_resultados.json:resultados.contencion_por_semilla.<técnica>.<semilla>`.

## j) Tiempos, avisos e incidente

### Tiempos

| Ejecución | Inicio (−06:00) | Duración total | Líneas de tiempo del log | Código |
|---|---|---|---|---|
| `--verificar-insumos` | 2026-11-07, antes de 12:05 | — | — | 0 |
| `--ejecutar-prueba` | `created_at` = 2026-11-07T18:05:09+00:00 (UTC) | 5.08 s (PowerShell) | F1 de 20 clasificadores en 10000 réplicas: 0.1 s; cálculo completo: 0.1 s | 0 |
| `--analisis-f46` (1.ª, interrumpida) | entre 12:05:15 y 12:09:45 | 4.89 s | — | −1 |
| `--analisis-f46` (2.ª, autorizada) | 12:10:06 (`created_at` = 2026-11-07T18:10:11+00:00 UTC) | 5.56 s | — (F4.6 no registra tiempos) | 0 |

### Avisos de los logs

- `run_fase5.log`: «AVISO (no bloquea): P6: condición 2 de F4.6 (IC de ΔF1 de F4.1), hasta el bloque E»
- `run_f46.log`: «AVISO (no bloquea): P6: condición 2 de F4.6 (IC de ΔF1 de F4.1), hasta el bloque E»

Líneas finales de `run_fase5.log`: «escrito C:\Users\DELL\Desktop\TT\TT2\ttdicevc\results\f5_contraste\resultados.json». `run_fase5.log` tiene 61 líneas y `run_f46.log` 58. Ambos listan los mismos 56 hashes que `meta.inputs`.

### Incidente en la primera ejecución de `--analisis-f46`

- **Hora:** 2026-11-07, entre 12:05:15 (commit `e02104f`) y 12:09:45 −06:00. No se registró la hora exacta de inicio. Duración: 4.89 s.
- **Comando:** `python -m src.protocol.run_fase5 --analisis-f46 | Select-Object -First 3` (PowerShell 5.1, con `CANON_UMAP_PARAMS` y `CANON_STRICT=1` definidos).
- **Código de salida:** −1.
- **Causa:** el comando pasó la salida por `Select-Object -First 3` para mostrar solo las primeras líneas. PowerShell detiene la tubería al recibir la tercera línea y terminó el proceso de Python. El script no produjo error ni traceback.
- **Líneas que alcanzó a imprimir:**

  ```
  AVISO (no bloquea): P6: condición 2 de F4.6 (IC de ΔF1 de F4.1), hasta el bloque E
  Compuerta enmienda-5: OK; árbol limpio; HEAD e02104fcb15affed677a64b972305bc72f97c325
    sha256 a78c6a0c488508cfb9b7ede881c109a5847bfbedad05646c421a7295bee17767  results/predictions/control_permutacion/gray_scott__efficientnet_b0__finetune__seed123.csv
  ```

- **Efecto:** el proceso terminó al imprimir el primer hash de insumos, antes de cualquier cálculo de F4.6. No escribió archivos: después, `results/f5_contraste/` contenía solo los 11 archivos del commit `e02104f`, no existían `f46_resultados.json` ni `run_f46.log`, y `git status --porcelain` estaba vacío.
- **Veredicto:** ya estaba commiteado en `e02104fcb15affed677a64b972305bc72f97c325` antes del incidente.
- **Autorización:** SV autorizó repetir `--analisis-f46` una sola vez, con dos condiciones: verificar antes que el punto 1 no estuviera truncado y ejecutar sin filtros ni tuberías, con la salida completa redirigida a un archivo.
- **Verificación previa a la repetición:** el comando del punto 1 no tuvo tubería ni filtro. `run_fase5.log` tiene 61 líneas (aviso, compuerta, 56 hashes, 2 tiempos y la línea final «escrito …resultados.json»). `resultados.json` es JSON válido con `meta` y las 11 secciones de `resultados`.
- **Repetición:** a las 12:10:06 −06:00, `python -m src.protocol.run_fase5 --analisis-f46 *> <scratchpad>\f46_consola.log`. Código 0, 5.56 s. Desviación de la ruta indicada: la salida no se redirigió directamente a `results/f5_contraste/f46_consola.log`. Esa ruta no está en `.gitignore`, PowerShell crea el archivo antes de lanzar Python y `check_clean_tree` habría rechazado la ejecución por el archivo sin seguimiento. Se redirigió a un archivo temporal fuera del repo y se copió sin cambios a `results/f5_contraste/f46_consola.log`. Su contenido, sin el BOM ni los CRLF de PowerShell, es idéntico línea por línea a `run_f46.log` (58 líneas).

