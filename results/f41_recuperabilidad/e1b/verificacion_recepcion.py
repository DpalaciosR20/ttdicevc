import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
R = Path('results/f41_recuperabilidad/caracteristicas'); NPZ = Path('f41_caracteristicas')
ok = True
def chk(c, msg):
    global ok
    print(('OK   ' if c else 'FALLA') + ' ' + msg); ok &= bool(c)
meta = json.loads((R/'caracteristicas_meta.json').read_text())
chk(meta['git_commit'] == '9f3b7b016bcb24d41b58d46ffdb8af7f26267295', f"git_commit {meta['git_commit']}")
chk(meta['smoke'] is False and meta['limit'] is None, 'smoke=false, limit=None')
mv = json.loads((R/'modelos_verificacion.json').read_text())
chk(mv['n_ok'] == 15 and mv['n_expected'] == 15 and mv['passed'], f"modelos {mv['n_ok']}/{mv['n_expected']} passed={mv['passed']}")
chk((R/'datos_verificacion.log').read_text().strip() == 'manifiesto OK', 'datos_verificacion: manifiesto OK')
dm = {r['logical_name']: r for r in csv.DictReader(open('meta/data_manifest.csv', encoding='utf-8'))}
for t, m in meta['tecnicas'].items():
    chk(m['dir_digest'] == dm[f'IMAGES[{t}]']['sha256'], f'{t} dir_digest = data_manifest')
    chk(m['n_imagenes'] == 1136, f'{t} n_imagenes 1136')
    if t in ('perlin', 'lorenz'):
        ids = [s for s, x in m['modelos'].items() if x['backbone_vs_imagenet']['identico'] and x['backbone_vs_imagenet']['iguales'] == x['backbone_vs_imagenet']['n_tensores']]
        chk(sorted(ids) == sorted(['42','7','123','2024','99']), f'{t} backbone == ImageNet en semillas {sorted(ids, key=int)}')
# hashes de modelos del meta vs artefactos_manifest
am = {r.get('file_name') or r.get('archivo') or list(r.values())[0]: r for r in csv.DictReader(open('meta/artefactos_manifest.csv', encoding='utf-8'))}
amtxt = Path('meta/artefactos_manifest.csv').read_text(encoding='utf-8')
for t, m in meta['tecnicas'].items():
    for s, x in m['modelos'].items():
        chk(x['sha256'] in amtxt, f'{t} seed{s} sha256 de modelo en artefactos_manifest')
sums = dict(l.split()[::-1] for l in (R/'npz_sha256sum.txt').read_text().splitlines() if l.strip())
man = list(csv.DictReader(open(R/'caracteristicas_manifest.csv', encoding='utf-8')))
chk(len(man) == 8 and len(sums) == 8, f'manifiesto {len(man)} filas, sha256sum {len(sums)}')
idx = list(csv.DictReader(open('meta/sample_index.csv', encoding='utf-8')))
print('sample_index cols:', list(idx[0].keys()))
for r in man:
    p = NPZ/r['archivo']; h = hashlib.sha256(p.read_bytes()).hexdigest()
    chk(h == r['sha256'] == sums[r['archivo']], f"{r['archivo']} sha256 {h[:12]} = manifiesto = sha256sum")
    z = np.load(p, allow_pickle=False)
    F = z['features']; d = int(r['d'])
    chk(F.dtype == np.float32 and F.shape == (1136, d) and np.isfinite(F).all(), f"  features {F.dtype} {F.shape} finitas")
    t = r['tecnica']
    rows = [x for x in idx if x.get('technique', t) == t] if 'technique' in idx[0] else idx
    sid = [x['sample_id'] for x in rows]; spl = [x['split'] for x in rows]
    chk(list(z['sample_id'].astype(str)) == sid and list(z['split'].astype(str)) == spl, f'  sample_id/split en orden de sample_index ({len(sid)})')
    print('   claves npz:', list(z.keys()))
print('TODO OK' if ok else 'HAY FALLAS')
