"""
v3_ckpt_11marzo.py
──────────────────
Diagnóstico de solo lectura: V1 y V3 (especificacion_f22 §3) contra el
checkpoint de la corrida del 11 de marzo (exp_4_1_11_marzo/checkpoint-342),
candidato a ser el modelo que generó EMB_41 (commit 6cfc1ce del repositorio
de TT-I). Mismo procedimiento que f22_beto.check_v3. No escribe
predicciones; registra el SHA-256 de cada archivo del checkpoint.

Uso (desde la raíz del repo):
    python results/f22_referencias_textuales/diagnostico_v3/v3_ckpt_11marzo.py
"""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from common import Log, load_index, paths, sha256, verify_file  # noqa: E402
from f22_beto import EMB_KEYS, V3_COS_MIN, check_v1, cls_embeddings, load_texts  # noqa: E402

REL_CKPT = 'exp_4_1_11_marzo/checkpoint-342'
log = Log(HERE / 'v3_ckpt_11marzo.log')


def main():
    t0 = time.time()
    torch.manual_seed(42)
    ckpt = paths.resolve(REL_CKPT)
    log(f'Checkpoint: {REL_CKPT}')
    files, _ = paths.list_files(ckpt)
    for p in files:
        log(f'  sha256  {p.name:24s} {sha256(p)}')
    verify_file(paths.REL['EMB_41'], log)
    for s in paths.SPLITS:
        verify_file(paths.REL['CSV'][s], log)

    index = load_index()
    texts = load_texts(index, log)
    emb = joblib.load(paths.emb_41_path())

    log('\n— V1')
    _, f1_epoch3, history = check_v1(ckpt, log)
    log(f'  eval_f1_macro por época: {history}')

    log('\n— V3')
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt).eval()
    rows, ok = [], True
    for s in paths.SPLITS:
        new = cls_embeddings(model, tok, texts[s])
        ref = np.asarray(emb[EMB_KEYS[s]], dtype=np.float64)
        a = new / np.linalg.norm(new, axis=1, keepdims=True)
        b = ref / np.linalg.norm(ref, axis=1, keepdims=True)
        sim = a @ b.T
        cos = np.diag(sim)
        is_self = sim.argmax(axis=1) == np.arange(len(cos))
        maxabs = np.abs(new - ref).max()
        n_cos, n_nn = int((cos < V3_COS_MIN).sum()), int((~is_self).sum())
        ok &= n_cos == 0 and n_nn == 0
        log(f'V3 {s}: coseno mín {cos.min():.8f}, medio {cos.mean():.8f}; filas < {V3_COS_MIN}: {n_cos}; '
            f'vecino ≠ propia: {n_nn}; |Δ| máx {maxabs:.3e}')
        rows.append(pd.DataFrame({'sample_id': index[s]['sample_id'], 'split': s, 'cosine': cos,
                                  'nearest_is_self': is_self}))
    pd.concat(rows, ignore_index=True).to_csv(HERE / 'v3_ckpt_11marzo.csv', index=False, lineterminator='\n')
    log(f'V3: {"OK" if ok else "FALLA"}')
    log(f'Duración: {time.time() - t0:.0f} s')


if __name__ == '__main__':
    try:
        main()
    finally:
        log.write()
