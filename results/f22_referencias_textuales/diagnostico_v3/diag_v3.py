"""
diag_v3.py
──────────
Diagnóstico de solo lectura de la falla de V3 (F2.2): ¿qué procedimiento, si
alguno, reproduce EMB_41? No escribe predicciones ni toca TT_DATA_ROOT.

Variantes, sobre las 1,136 filas, en el orden del índice maestro:
  (a) [CLS] de BertForSequenceClassification(checkpoint-342).bert, eval y train
  (b) pooler_output y pooling medio (con máscara) de checkpoint-342, eval
  (c) dccuchile/bert-base-spanish-wwm-cased sin fine-tuning: [CLS], pooler y medio
  (d) cada modelo con padding dinámico y con padding='max_length' (128)
Para cada una: coseno por fila contra EMB_41 y si el vecino más cercano
(dentro del split) es la propia fila.

Además: LogReg sobre EMB_41 con el protocolo de especificacion_f22 §2, solo
F1-Val (sin test).

Uso (desde la raíz del repo):
    python results/f22_referencias_textuales/diagnostico_v3/diag_v3.py
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from common import Log, load_index, paths, verify_file  # noqa: E402
from f22_beto import EMB_KEYS, load_texts  # noqa: E402

BASE = 'dccuchile/bert-base-spanish-wwm-cased'
BATCH, MAX_LENGTH, SEED = 16, 128, 42
log = Log(HERE / 'diag_v3.log')


@torch.no_grad()
def encode(bert, tok, texts, padding):
    """Devuelve {'cls', 'pooler', 'mean'} (n, 768) float64."""
    out = {'cls': [], 'pooler': [], 'mean': []}
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=padding, truncation=True, max_length=MAX_LENGTH,
                  return_tensors='pt')
        o = bert(**enc)
        h = o.last_hidden_state
        m = enc['attention_mask'].unsqueeze(-1).to(h.dtype)
        out['cls'].append(h[:, 0, :].numpy())
        out['pooler'].append(o.pooler_output.numpy() if o.pooler_output is not None
                             else np.full((h.shape[0], h.shape[2]), np.nan))
        out['mean'].append(((h * m).sum(1) / m.sum(1)).numpy())
    return {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}


def compare(new, ref):
    a = new / np.linalg.norm(new, axis=1, keepdims=True)
    b = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    sim = a @ b.T
    cos = np.diag(sim)
    return cos, sim.argmax(axis=1) == np.arange(len(cos))


def main():
    t0 = time.time()
    verify_file(paths.REL['EMB_41'], log)
    for s in paths.SPLITS:
        verify_file(paths.REL['CSV'][s], log)
    index = load_index()
    texts = load_texts(index, log)
    emb = joblib.load(paths.emb_41_path())
    log(f'EMB_41: model_name={emb.get("model_name")!r}, checkpoint={emb.get("checkpoint")!r}')
    ref = {s: np.asarray(emb[EMB_KEYS[s]], dtype=np.float64) for s in paths.SPLITS}
    for s in paths.SPLITS:
        n = np.linalg.norm(ref[s], axis=1)
        log(f'  norma de EMB_41[{s}]: media {n.mean():.3f}, mín {n.min():.3f}, máx {n.max():.3f}')

    ckpt = paths.ckpt_41_path()
    tok_ckpt = AutoTokenizer.from_pretrained(ckpt)
    clf = AutoModelForSequenceClassification.from_pretrained(ckpt)
    base_tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModel.from_pretrained(BASE)
    if tok_ckpt.get_vocab() != base_tok.get_vocab():
        log('AVISO: el vocabulario del tokenizador de checkpoint-342 difiere del de ' + BASE)
    else:
        log('Tokenizador de checkpoint-342 = vocabulario de ' + BASE)

    models = [('ckpt342.bert', clf.bert, tok_ckpt, 'eval'),
              ('ckpt342.bert', clf.bert, tok_ckpt, 'train'),
              ('base_cased', base, base_tok, 'eval')]
    rows, per_row = [], []
    for name, bert, tok, mode in models:
        for padding in ('dynamic', 'max_length'):
            bert.train(mode == 'train')
            torch.manual_seed(SEED)
            t1 = time.time()
            reps = {s: encode(bert, tok, texts[s], True if padding == 'dynamic' else 'max_length')
                    for s in paths.SPLITS}
            for pooling in ('cls', 'pooler', 'mean'):
                cos_all, self_all = [], []
                for s in paths.SPLITS:
                    cos, is_self = compare(reps[s][pooling], ref[s])
                    cos_all.append(cos)
                    self_all.append(is_self)
                    per_row.append(pd.DataFrame({'variant': f'{name}|{mode}|{padding}|{pooling}',
                                                 'sample_id': index[s]['sample_id'], 'split': s,
                                                 'cosine': cos, 'nearest_is_self': is_self}))
                cos_all, self_all = np.concatenate(cos_all), np.concatenate(self_all)
                row = {'model': name, 'mode': mode, 'padding': padding, 'pooling': pooling,
                       'cos_mean': cos_all.mean(), 'cos_min': cos_all.min(), 'cos_max': cos_all.max(),
                       'n_cos_ge_0_9999': int((cos_all >= 0.9999).sum()),
                       'n_nearest_is_self': int(self_all.sum()), 'n': len(cos_all)}
                rows.append(row)
                log(f'{name:13s} {mode:5s} {padding:10s} {pooling:6s}  coseno medio {row["cos_mean"]:.6f} '
                    f'[{row["cos_min"]:.4f}, {row["cos_max"]:.4f}]  ≥0.9999: {row["n_cos_ge_0_9999"]:4d}/1136  '
                    f'vecino = propia: {row["n_nearest_is_self"]:4d}/1136')
            log(f'  ({time.time() - t1:.0f} s)')
    table = pd.DataFrame(rows)
    table.to_csv(HERE / 'variantes.csv', index=False, lineterminator='\n')
    pd.concat(per_row, ignore_index=True).to_csv(HERE / 'variantes_por_fila.csv', index=False,
                                                 lineterminator='\n')
    best = table.sort_values('cos_mean', ascending=False).iloc[0]
    reproducible = bool((table['n_cos_ge_0_9999'] == 1136).any())
    log(f'\nMejor variante: {best["model"]} {best["mode"]} {best["padding"]} {best["pooling"]} '
        f'(coseno medio {best["cos_mean"]:.6f})')
    log('Conclusión: ' + ('alguna variante reproduce EMB_41 (coseno ≥ 0.9999 en las 1,136 filas)'
                          if reproducible else 'EMB_41 no es reproducible desde checkpoint-342'))

    log('\nLogReg sobre EMB_41 (protocolo §2), solo F1-Val:')
    y = {s: np.asarray(emb[k]).astype(int) for s, k in (('train', 'y_train'), ('val', 'y_val'))}
    sel = []
    for C in np.logspace(-4, 2, 13):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, solver='lbfgs', max_iter=5000,
                                                               random_state=SEED))
        m.fit(ref['train'], y['train'])
        f1 = f1_score(y['val'], m.predict(ref['val']), average='macro')
        sel.append({'C': C, 'f1_val': f1})
        log(f'  C = {C:.6g}  F1-Val = {f1:.6f}')
    sel = pd.DataFrame(sel)
    sel.to_csv(HERE / 'logreg_emb41_f1val.csv', index=False, lineterminator='\n')
    b = sel.loc[sel['f1_val'] == sel['f1_val'].max(), 'C'].min()
    log(f'  mejor: C = {b:.6g}, F1-Val = {sel["f1_val"].max():.6f} (empate → menor C)')
    log(f'\nDuración: {time.time() - t0:.0f} s')


if __name__ == '__main__':
    try:
        main()
    finally:
        log.write()
