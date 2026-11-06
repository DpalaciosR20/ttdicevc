"""verify_artefacts: emparejamiento por nombre y comparación de huellas, con archivos sintéticos."""
import csv
import hashlib

from src import verify_artefacts as va

FIELDS = ['logical_name', 'task', 'fase', 'rel_path', 'kind', 'size_bytes', 'sha256', 'run_json', 'git_commit']


def _setup(tmp_path, contents):
    root = tmp_path / 'input' / 'tt2-artefactos' / 'f35_final' / 'perlin' / 'models'
    root.mkdir(parents=True)
    rows = []
    for name, data in contents.items():
        (root / name).write_bytes(data)
        rows.append(dict(logical_name=name, task='F3.5', fase='final', rel_path=f'f35_final/perlin/models/{name}',
                         kind='state_dict', size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                         run_json='x', git_commit='abc1234'))
    man = tmp_path / 'manifest.csv'
    with open(man, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return tmp_path / 'input', root, man


def test_todo_coincide(tmp_path):
    inp, _, man = _setup(tmp_path, {'a.pt': b'uno', 'b.pt': b'dos'})
    out = tmp_path / 'rep.json'
    assert va.main(['--root', str(inp), '--manifest', str(man), '--out', str(out)]) == 0
    rep = va.verify(str(inp), str(man))
    assert rep['passed'] and rep['n_ok'] == 2 and rep['dataset_dirs'] == ['tt2-artefactos']
    assert out.exists()


def test_falta_distinto_y_extra(tmp_path):
    inp, root, man = _setup(tmp_path, {'a.pt': b'uno', 'b.pt': b'dos'})
    (root / 'a.pt').write_bytes(b'otro')
    (root / 'b.pt').unlink()
    (root / 'c.pt').write_bytes(b'tres')
    rep = va.verify(str(inp), str(man))
    assert not rep['passed']
    assert rep['mismatch'] == ['a.pt'] and rep['missing'] == ['b.pt'] and rep['extra'] == ['c.pt']
    assert va.main(['--root', str(inp), '--manifest', str(man)]) == 1


def test_duplicado_falla(tmp_path):
    inp, root, man = _setup(tmp_path, {'a.pt': b'uno'})
    copia = inp / 'otra_version'
    copia.mkdir()
    (copia / 'a.pt').write_bytes(b'uno')
    rep = va.verify(str(inp), str(man))
    assert rep['duplicates'] == ['a.pt'] and not rep['passed']


def test_hash_models_y_verificacion(tmp_path):
    import pytest
    from src import hash_models
    res = tmp_path / 'results' / 'f36_permutacion'
    (res / 'perlin' / 'models').mkdir(parents=True)
    (res / 'perlin' / 'models' / 'm.pt').write_bytes(b'abc')
    out, n = hash_models.write_hashes(str(res), expected=1)
    rows = list(csv.DictReader(open(out)))
    assert n == 1 and rows[0]['path'] == 'results/f36_permutacion/perlin/models/m.pt'
    assert rows[0]['sha256'] == hashlib.sha256(b'abc').hexdigest() and rows[0]['bytes'] == '3'
    with pytest.raises(SystemExit):
        hash_models.write_hashes(str(res), expected=15)
