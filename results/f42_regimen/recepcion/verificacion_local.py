import csv, json, hashlib, sys
from pathlib import Path
import numpy as np
REPO = Path(sys.argv[1]); sys.path[:0] = [str(REPO), str(REPO/'src'/'training')]
D = REPO/'results/f42_regimen/reintentos'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
m = json.loads((D/'reintentos_gs_manifest.json').read_text())
rows = list(csv.DictReader(open(D/'reintentos_gs_train.csv', newline='', encoding='utf-8')))
idx = sorted((r for r in csv.DictReader(open(REPO/'meta/sample_index.csv', encoding='utf-8')) if r['split']=='train'), key=lambda r: r['sample_id'])
sums = {l.split()[1].split('/')[-1]: l.split()[0] for l in (D/'f42_sha256sum.txt').read_text().splitlines()}
from umap_io import load_umap_arrays
from src import paths
from src.build_manifest import verify_manifest
arr = load_umap_arrays(paths.umap_params_path())
chk = {}
chk['commit_txt'] = (D/'commit.txt').read_text().strip()
chk['manifest_git_commit'] = m['git_commit']
chk['commit_coinciden'] = chk['commit_txt'] == m['git_commit']
chk['smoke'] = m['smoke']; chk['limit'] = m['limit']; chk['split'] = m['split']; chk['ruta'] = m['ruta']
chk['csv_filas'] = len(rows); chk['manifest_csv_filas'] = m['csv']['filas']
chk['csv_sha256'] = sha(D/'reintentos_gs_train.csv')
chk['csv_sha_eq_manifest'] = chk['csv_sha256'] == m['csv']['sha256']
chk['csv_sha_eq_sha256sum'] = chk['csv_sha256'] == sums['reintentos_gs_train.csv']
chk['manifest_sha256'] = sha(D/'reintentos_gs_manifest.json')
chk['manifest_sha_eq_sha256sum'] = chk['manifest_sha256'] == sums['reintentos_gs_manifest.json']
chk['u_dtype_manifest'] = m['u_dtype']; chk['u_dtype_pkl'] = str(arr['train']['params'].dtype)
chk['verify_manifest_UMAP_PARAMS'] = verify_manifest('UMAP_PARAMS') or 'OK'
chk['umap_sha_manifest_eq_local'] = m['umap_params_sha256'] == sha(paths.umap_params_path())
chk['orden_sample_id'] = [r['sample_id'] for r in rows] == [r['sample_id'] for r in idx]
chk['idx_coincide'] = [r['idx'] for r in rows] == [r['idx'] for r in idx]
chk['y_true_coincide_index'] = [r['y_true'] for r in rows] == [r['y_true'] for r in idx]
order = np.array([int(r['idx']) for r in idx]); U = arr['train']['params'][order]
chk['y_true_coincide_pkl'] = [int(r['y_true']) for r in rows] == arr['train']['labels'][order].astype(int).tolist()
chk['u_vs_pkl_max_abs'] = float(np.abs(np.array([[float(r[f'u{j}']) for j in range(5)] for r in rows]) - U.astype(np.float64)).max())
T = lambda v: v == 'True'
re_ = [int(r['reintentos']) for r in rows]; resp = [T(r['usa_respaldo']) for r in rows]; nl = [int(r['n_llamadas']) for r in rows]
chk['reintentos_en_0_3'] = all(0 <= x <= 3 for x in re_)
chk['respaldo_sii_reintentos_3'] = all(b == (x == 3) for b, x in zip(resp, re_))
chk['n_llamadas_coherente'] = all(n == (4 if b else x + 1) for n, x, b in zip(nl, re_, resp))
vv = [[float(v) for v in r['var_por_llamada'].split(';')] for r in rows]
chk['len_var_eq_n_llamadas'] = all(len(v) == n for v, n in zip(vv, nl))
chk['ultima_var_gt_1e-4_sin_respaldo'] = all(v[-1] > 1e-4 for v, b in zip(vv, resp) if not b)
chk['intentos_previos_var_le_1e-4'] = all(all(x <= 1e-4 for x in v[:-1]) for v in vv)
for c in ('sha256_coincide', 'pixeles_coinciden', 'almacenada_integra'):
    chk['n_' + c] = sum(T(r[c]) for r in rows)
chk['n_col_regen_eq_col_manifiesto'] = sum(r['sha256_regenerada'] == r['sha256_manifiesto'] for r in rows)
G = Path(paths.data_root()) / 'images_grayscale_classblind' / 'gray_scott' if hasattr(paths, 'data_root') else Path(sys.argv[2])
chk['n_png_local_sha_eq_col_manifiesto'] = sum(sha(G / r['archivo']) == r['sha256_manifiesto'] for r in rows)
chk['nombre_archivo_coherente'] = all(r['archivo'] == f"rd_train_{int(r['idx']):04d}_label{r['y_true']}.png" for r in rows)
chk['reintentos_mayor_0_por_clase'] = {c: sum(1 for r, x in zip(rows, re_) if x > 0 and r['y_true'] == c) for c in '01'}
chk['resumen_manifest_reintentos'] = m['resumen']['reintentos_mayor_0_por_clase']
chk['distribucion'] = {k: re_.count(k) for k in range(4)}
chk['n_por_clase'] = {c: sum(r['y_true'] == c for r in rows) for c in '01'}
chk['no_coinciden_sha256'] = m['resumen']['no_coinciden_sha256']; chk['no_coinciden_pixeles'] = m['resumen']['no_coinciden_pixeles']
print(json.dumps(chk, indent=1, default=str))
