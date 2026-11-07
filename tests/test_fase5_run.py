"""Fase 5 — run_fase5: negativas, ejecución de punta a punta con insumos SINTÉTICOS y rendimiento.

Ninguna prueba carga predicciones reales: cargar() se sustituye por insumos
sintéticos y git por un doble controlado."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import fase5_sintetico as S  # noqa: E402
from src.protocol import insumos as ins_mod  # noqa: E402
from src.protocol import run_fase5 as rf  # noqa: E402


class FakeGit:
    def __init__(self, ancestor=0, diff=0, status='', tracked=0, diff_head=0):
        self.ancestor, self.diff, self.status, self.tracked, self.diff_head = ancestor, diff, status, tracked, diff_head

    def __call__(self, *args):
        rc, out = 0, ''
        if args[0] == 'merge-base':
            rc = self.ancestor
        elif args[0] == 'diff' and args[2] == rf.GATE_TAG:
            rc = self.diff
        elif args[0] == 'diff' and args[2] == 'HEAD':
            rc = self.diff_head
        elif args[0] == 'status':
            out = self.status
        elif args[0] == 'ls-files':
            rc = self.tracked
        elif args[0] == 'rev-parse':
            out = 'f' * 40
        return subprocess.CompletedProcess(args, rc, out, '')


@pytest.fixture
def entorno(monkeypatch, tmp_path):
    """Compuerta y árbol simulados; cargar() devuelve insumos sintéticos; config con A14 fijado."""
    calls = {'cargar': 0, 'verificar': 0}

    def fake_cargar(cfg, with_u=True):
        calls['cargar'] += 1
        return S.make()

    def fake_verificar(cfg, log=print, check_umap=True):
        calls['verificar'] += 1
        return {}

    monkeypatch.setattr(rf, '_git', FakeGit())
    monkeypatch.setattr(ins_mod, 'cargar', fake_cargar)
    monkeypatch.setattr(ins_mod, 'verificar_insumos', fake_verificar)
    monkeypatch.setattr(ins_mod, 'load_config', lambda path=None: S.cfg())
    return calls, tmp_path / 'f5_contraste'


def run(argv, out):
    lines = []
    rc = rf.main(argv, out_dir=out, log=lines.append)
    return rc, '\n'.join(map(str, lines))


def test_sin_bandera_solo_verifica(entorno):
    calls, out = entorno
    rc, log = run([], out)
    assert rc == 0 and calls == {'cargar': 0, 'verificar': 1} and not out.exists()
    rc, _ = run(['--verificar-insumos'], out)
    assert rc == 0 and calls['cargar'] == 0 and not out.exists()


@pytest.mark.parametrize('git,msg', [(FakeGit(ancestor=1), 'no es ancestro'),
                                     (FakeGit(diff=1), 'cambiaron frente a'),
                                     (FakeGit(status=' M src/x.py\n'), 'no está limpio')])
def test_se_niega_sin_compuerta_o_con_arbol_sucio(entorno, monkeypatch, git, msg):
    calls, out = entorno
    monkeypatch.setattr(rf, '_git', git)
    rc, log = run(['--ejecutar-prueba'], out)
    assert rc == 2 and 'NEGATIVA' in log and msg in log
    assert calls['cargar'] == 0 and not out.exists()


def test_se_niega_con_pendientes(entorno, monkeypatch):
    calls, out = entorno
    monkeypatch.setattr(ins_mod, 'load_config', lambda path=None: S.cfg(sin_enmienda_5=True))
    rc, log = run(['--ejecutar-prueba'], out)
    assert rc == 2 and 'PENDIENTES' in log and 'P1' in log and calls['cargar'] == 0


def test_config_real_sin_pendientes_que_bloqueen():
    """Con la Enmienda 5 no queda ningún punto que bloquee --ejecutar-prueba; solo el aviso de P6."""
    cfg = ins_mod.load_config()
    assert rf.pendientes(cfg) == []
    av = rf.avisos(cfg)
    assert len(av) == 1 and av[0].startswith('P6') and 'F4.1' in av[0]


def test_compuerta_en_una_sola_constante_y_se_niega_sin_la_etiqueta(monkeypatch):
    """La compuerta toma la etiqueta de rf.GATE_TAG (enmienda-5) y se niega si no está en la historia."""
    assert rf.GATE_TAG == 'enmienda-5'
    src = Path(rf.__file__).read_text(encoding='utf-8')
    assert src.count("'enmienda-5'") == 1
    monkeypatch.setattr(rf, 'GATE_TAG', 'etiqueta-que-no-existe-f5')
    with pytest.raises(rf.Negativa, match='no es ancestro'):
        rf.check_gate()


def test_aviso_P6_en_el_log(entorno):
    calls, out = entorno
    rc, log = run(['--ejecutar-prueba'], out)
    assert rc == 0 and 'AVISO (no bloquea): P6' in log


def test_ejecucion_sintetica_y_no_sobrescribe(entorno):
    calls, out = entorno
    rc, log = run(['--ejecutar-prueba'], out)
    assert rc == 0, log
    doc = json.loads((out / 'resultados.json').read_text(encoding='utf-8'),
                     parse_constant=lambda c: pytest.fail(f'constante no estándar {c} en el JSON'))
    res = doc['resultados']
    assert set(res['principal']['techniques']) == set(S.TECHS) and res['principal']['global'] in ('G1', 'G2', 'G3', 'G4')
    assert doc['meta']['bootstrap']['call'] == 'rng.integers(0, 114, size=(10000, 114))'
    for name in ('veredictos', 'robustez', 'retencion', 'descomposicion', 'frente_a_descriptores', 'longitud',
                 'mcnemar', 'terciles_longitud', 'spearman_u_longitud'):
        assert (out / f'{name}.csv').is_file(), name
    assert (out / 'run_fase5.log').is_file()
    before = (out / 'resultados.json').read_bytes()
    rc, log = run(['--ejecutar-prueba'], out)
    assert rc == 2 and 'nunca sobrescribe' in log and (out / 'resultados.json').read_bytes() == before


def test_f46_exige_veredicto_commiteado(entorno, monkeypatch):
    calls, out = entorno
    rc, log = run(['--analisis-f46'], out)
    assert rc == 2 and 'no está versionado' in log
    out.mkdir(parents=True)
    (out / 'resultados.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(rf, '_git', FakeGit(tracked=1))
    rc, log = run(['--analisis-f46'], out)
    assert rc == 2 and 'no está versionado' in log
    monkeypatch.setattr(rf, '_git', FakeGit(diff_head=1))
    rc, log = run(['--analisis-f46'], out)
    assert rc == 2 and 'sin commitear' in log
    monkeypatch.setattr(rf, '_git', FakeGit())
    rc, log = run(['--analisis-f46'], out)
    assert rc == 0 and (out / 'f46_resultados.json').is_file()
    rc, log = run(['--analisis-f46'], out)
    assert rc == 2 and 'nunca sobrescribe' in log


def test_rendimiento_B10000_tamano_real(capsys):
    """B = 10,000 sobre insumos sintéticos del tamaño real: debe caber en minutos en local."""
    ins, cfg = S.make(), S.cfg()
    idx = S.idx(cfg)
    t0 = time.time()
    rf.ejecutar(ins, cfg, S.CANON, idx, log=lambda *_: None)
    t1 = time.time() - t0
    rf.ejecutar_f46(ins, cfg, idx)
    t2 = time.time() - t0 - t1
    with capsys.disabled():
        print(f'\n[rendimiento] Fase 5 sintética: {t1:.1f} s; F4.6: {t2:.1f} s (B = {idx.shape[0]})')
    assert t1 < 300 and t2 < 300
