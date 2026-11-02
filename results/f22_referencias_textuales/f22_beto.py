"""
f22_beto.py
───────────
F2.2 — Predicciones por muestra de BETO (Exp 4.1) con su cabeza de
clasificación (docs/especificacion_f22.md §2-§4).

Orden (§3): integridad de insumos → alineación CSV ↔ índice maestro → V1 →
V3 (si alguna falla, se detiene sin escribir predicciones) → inferencia
sobre val y test → V2 → métricas de §4 (val y test; excepción explícita
para test de BETO).

Uso (desde la raíz del repo, .venv, CPU):
    python results/f22_referencias_textuales/f22_beto.py

Salidas:
    results/predictions/referencia/text__beto__finetune__seed42.csv (+ manifiesto)
    results/f22_referencias_textuales/v3_embeddings.csv   coseno y vecino por fila
    results/f22_referencias_textuales/beto_metricas.json  V1-V3 y métricas
    results/f22_referencias_textuales/f22_beto.log
"""

import hashlib
import json
import math
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from common import (FASE, OUT_DIR, PRED_SPLITS, SEED, TASK_ID, Log, Stop, load_index, paths,
                    sha256, verify_dir, verify_file)
from src.canonical.build import confusion_counts, f1_macro_from_confusion
from src.predictions.write import write_predictions

MAX_LENGTH = 128
V3_BATCH = 16
V3_COS_MIN = 0.9999
REPORTED = {'f1_test': 0.7586, 'confusion_test': [[57, 13], [13, 31]]}  # [[TN, FP], [FN, TP]]
EMB_KEYS = {'train': 'X_train', 'val': 'X_val', 'test': 'X_test'}
LABEL_KEYS = {'train': 'y_train', 'val': 'y_val', 'test': 'y_test'}

log = Log(OUT_DIR / 'f22_beto.log')


def load_texts(index, log):
    """Textos de cada split en el orden del índice; verifica id, sha256 del texto y etiqueta."""
    texts = {}
    for s in paths.SPLITS:
        df = pd.read_csv(paths.csv_path(s))
        ix = index[s]
        if len(df) != len(ix):
            raise Stop(f'{s}: {len(df)} filas en el CSV y {len(ix)} en el índice')
        sha = df['text'].map(lambda t: hashlib.sha256(t.encode('utf-8')).hexdigest())
        bad = ((df['id'].values != ix['id'].values) | (sha.values != ix['text_sha256'].values)
               | (df['manual_classification'].values != ix['y_true'].values))
        if bad.any():
            raise Stop(f'{s}: {int(bad.sum())} filas del CSV no coinciden con el índice (id/texto/etiqueta)')
        texts[s] = df['text'].tolist()
        log(f'  {s}: {len(df)} filas alineadas con el índice (id, sha256 del texto, y_true)')
    return texts


def check_v1(ckpt, log):
    cfg = json.loads((ckpt / 'config.json').read_text(encoding='utf-8'))
    state = json.loads((ckpt / 'trainer_state.json').read_text(encoding='utf-8'))
    n_labels = len(cfg.get('id2label', {}))
    arch = cfg.get('architectures')
    best = state.get('best_model_checkpoint', '')
    ok = n_labels == 2 and arch == ['BertForSequenceClassification'] and best.endswith('checkpoint-342')
    detail = {'architectures': arch, 'id2label': cfg.get('id2label'), 'n_labels': n_labels,
              'best_model_checkpoint': best, 'best_metric': state.get('best_metric')}
    log(f'V1: {"OK" if ok else "FALLA"}  {detail}')
    if not ok:
        raise Stop('V1 no se cumple')
    epoch3 = [h for h in state['log_history'] if 'eval_f1_macro' in h and h.get('epoch') == 3.0]
    if len(epoch3) != 1:
        raise Stop('trainer_state.json: no hay un único eval_f1_macro de la época 3')
    return detail, epoch3[0]['eval_f1_macro'], [
        {'epoch': h['epoch'], 'eval_f1_macro': h['eval_f1_macro']} for h in state['log_history']
        if 'eval_f1_macro' in h]


@torch.no_grad()
def cls_embeddings(model, tok, texts):
    """Procedimiento de extraer_embeddings_finetuned: padding dinámico, lotes de 16, [CLS]."""
    out = []
    for i in range(0, len(texts), V3_BATCH):
        enc = tok(texts[i:i + V3_BATCH], padding=True, truncation=True, max_length=MAX_LENGTH,
                  return_tensors='pt')
        out.append(model(**enc).last_hidden_state[:, 0, :].numpy())
    return np.concatenate(out).astype(np.float64)


def check_v3(ckpt, texts, emb, index, log):
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt).eval()
    rows, ok = [], True
    for s in paths.SPLITS:
        t0 = time.time()
        new = cls_embeddings(model, tok, texts[s])
        ref = np.asarray(emb[EMB_KEYS[s]], dtype=np.float64)
        if ref.shape != new.shape:
            raise Stop(f'V3 {s}: forma {new.shape} != EMB_41 {ref.shape}')
        nn_ = new / np.linalg.norm(new, axis=1, keepdims=True)
        rn = ref / np.linalg.norm(ref, axis=1, keepdims=True)
        sim = nn_ @ rn.T
        cos = np.diag(sim)
        nearest = sim.argmax(axis=1)
        rank = (sim > cos[:, None]).sum(axis=1)  # filas de EMB_41 más parecidas que la propia
        is_self = nearest == np.arange(len(cos))
        # Desviación declarada (§6): si el vecino es otra fila con el mismo texto (sha256),
        # sus embeddings son idénticos y el empate de argmax no indica permutación
        text_sha = index[s]['text_sha256'].values
        dup_tie = ~is_self & (text_sha[nearest] == text_sha) & (rank == 0)
        n_cos = int((cos < V3_COS_MIN).sum())
        n_nn = int((~is_self & ~dup_tie).sum())
        ok &= n_cos == 0 and n_nn == 0
        log(f'V3 {s}: {len(cos)} filas, coseno mín {cos.min():.8f}, filas con coseno < {V3_COS_MIN}: {n_cos}, '
            f'vecino más cercano ≠ propia fila: {n_nn} (+ {int(dup_tie.sum())} empates con un texto idéntico)  '
            f'({time.time() - t0:.0f} s)')
        for i in np.flatnonzero(dup_tie):
            log(f'  empate declarado: {index[s]["sample_id"][i]} ↔ {index[s]["sample_id"][nearest[i]]} '
                f'(mismo text_sha256)')
        rows.append(pd.DataFrame({'sample_id': index[s]['sample_id'], 'split': s, 'cosine': cos,
                                  'nearest_is_self': is_self, 'nearest_same_text': dup_tie,
                                  'rank_of_self': rank}))
    detail = pd.concat(rows, ignore_index=True)
    detail.to_csv(OUT_DIR / 'v3_embeddings.csv', index=False, lineterminator='\n')
    summary = {'ok': bool(ok), 'cos_min': float(detail['cosine'].min()),
               'n_cos_below': int((detail['cosine'] < V3_COS_MIN).sum()),
               'n_nearest_not_self': int((~detail['nearest_is_self'] & ~detail['nearest_same_text']).sum()),
               'n_ties_identical_text': int(detail['nearest_same_text'].sum()), 'n_rows': len(detail)}
    log(f'V3: {"OK" if ok else "FALLA"}  {summary}')
    if not ok:
        raise Stop('V3 no se cumple: la correspondencia corpus ↔ embeddings queda en duda')
    return summary


@torch.no_grad()
def predict(ckpt, texts):
    """§2: tokenización max_length 128 con padding fijo; p_dep = softmax(logits)[:, 1] en float64."""
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()
    out = []
    for i in range(0, len(texts), V3_BATCH):
        enc = tok(texts[i:i + V3_BATCH], padding='max_length', truncation=True, max_length=MAX_LENGTH,
                  return_tensors='pt')
        out.append(torch.softmax(model(**enc).logits, dim=-1)[:, 1].double().numpy())
    return np.concatenate(out)


def metrics(y, p):
    y_pred = (p > 0.5).astype(int)
    cm = confusion_counts(y, y_pred)
    f1 = f1_macro_from_confusion(cm)
    sk = f1_score(y, y_pred, average='macro')
    if not math.isclose(f1, sk, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(f'F1 {f1} != sklearn {sk}')
    return {'f1_macro': f1, 'accuracy': accuracy_score(y, y_pred),
            'confusion': cm, 'confusion_matrix_tn_fp_fn_tp': [[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]],
            'fn_rate': cm['fn'] / (cm['fn'] + cm['tp'])}


def v2_report(m_val, ref, p_val, y_val, log):
    """V2: coincidencia a cuatro decimales; si no, distancia a la matriz más cercana que da ref."""
    ok = round(m_val['f1_macro'], 4) == round(ref, 4)
    out = {'ok': ok, 'f1_val_recalculado': m_val['f1_macro'], 'eval_f1_macro_epoca3': ref}
    log(f'V2: {"OK" if ok else "NO COINCIDE"}  F1-Val recalculado {m_val["f1_macro"]:.6f} '
        f'vs eval_f1_macro época 3 {ref:.6f}')
    if not ok:
        pos, neg = int(y_val.sum()), int(len(y_val) - y_val.sum())
        c = m_val['confusion']
        cands = [(abs(tp - c['tp']) + abs(tn - c['tn']), tp, tn)
                 for tp in range(pos + 1) for tn in range(neg + 1)
                 if abs(f1_macro_from_confusion({'tp': tp, 'fn': pos - tp, 'tn': tn, 'fp': neg - tn}) - ref) < 1e-9]
        dist = np.abs(p_val - 0.5)
        out.update({
            'matrices_que_dan_ref': [{'tp': tp, 'tn': tn, 'cambios_minimos': d} for d, tp, tn in sorted(cands)],
            'cambios_minimos': min(cands)[0] if cands else None,
            'distancias_a_0_5_mas_pequenas': sorted(dist.tolist())[:10],
            'n_con_distancia_menor_a_0_05': int((dist < 0.05).sum()),
            'n_con_distancia_menor_a_0_01': int((dist < 0.01).sum()),
        })
        log(f'  matrices (TP, TN) con F1 = ref: {[(tp, tn) for _, tp, tn in sorted(cands)]}; '
            f'cambios mínimos de predicción: {out["cambios_minimos"]}')
        log(f'  predicciones con |p−0.5| < 0.05: {out["n_con_distancia_menor_a_0_05"]}; < 0.01: '
            f'{out["n_con_distancia_menor_a_0_01"]}; las 5 más cercanas: '
            f'{[round(d, 5) for d in out["distancias_a_0_5_mas_pequenas"][:5]]}')
    return out


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    log(f'F2.2 — BETO (Exp 4.1). torch {torch.__version__}, hilos {torch.get_num_threads()}, CPU')
    ckpt = paths.ckpt_41_path()

    log('\n— Integridad de insumos (meta/data_manifest.csv)')
    inputs = {paths.REL['CSV'][s]: verify_file(paths.REL['CSV'][s], log) for s in paths.SPLITS}
    inputs[paths.REL['EMB_41']] = verify_file(paths.REL['EMB_41'], log)
    inputs[paths.REL['CKPT_41']] = verify_dir(paths.REL['CKPT_41'], log)
    inputs['meta/sample_index.csv (repo)'] = sha256(paths.SAMPLE_INDEX)

    log('\n— Alineación CSV ↔ índice maestro')
    index = load_index()
    texts = load_texts(index, log)
    emb = joblib.load(paths.emb_41_path())
    for s in paths.SPLITS:
        if not np.array_equal(np.asarray(emb[LABEL_KEYS[s]]).astype(int), index[s]['y_true'].values):
            raise Stop(f'{s}: etiquetas de EMB_41 ≠ índice')
    log('  etiquetas de EMB_41 = índice en los tres splits')

    log('\n— V1')
    v1, f1_epoch3, history = check_v1(ckpt, log)
    log('\n— V3')
    v3 = check_v3(ckpt, texts, emb, index, log)

    log('\n— Inferencia (cabeza de clasificación, fp32, CPU)')
    frames, results = [], {}
    for s in PRED_SPLITS:
        p = predict(ckpt, texts[s])
        y = index[s]['y_true'].values.astype(int)
        frames.append(pd.DataFrame({'sample_id': index[s]['sample_id'], 'split': s, 'y_true': y,
                                    'technique': 'text', 'arch': 'beto', 'strategy': 'finetune',
                                    'seed': SEED, 'p_dep': p}))
        results[s] = (y, p)
    df = pd.concat(frames, ignore_index=True)

    config = {'checkpoint': paths.REL['CKPT_41'], 'evaluator': 'trainer_head (BertForSequenceClassification)',
              'tokenizer': {'padding': 'max_length', 'truncation': True, 'max_length': MAX_LENGTH},
              'batch_size': V3_BATCH, 'precision': 'fp32', 'device': 'cpu',
              'p_dep': 'softmax(logits)[:, 1] → float64', 'decision': 'p_dep > 0.5', 'seed': SEED,
              'script': 'results/f22_referencias_textuales/f22_beto.py',
              'script_sha256': sha256(OUT_DIR / 'f22_beto.py')}
    selection = {'criterio': 'load_best_model_at_end por eval_f1_macro (Trainer, TT-I)',
                 'punto': 'checkpoint-342 (época 3)', 'eval_f1_macro_trainer_state': f1_epoch3}
    pred_inputs = {k: v for k, v in inputs.items() if k != paths.REL['CSV']['train']}
    csv_path = write_predictions(df, fase=FASE, task_id=TASK_ID, config=config, inputs=pred_inputs,
                                 model_selection=selection)
    back = pd.read_csv(csv_path, float_precision='round_trip')  # el lector por defecto se desvía 1 ulp
    if not np.array_equal(back['p_dep'].values, df['p_dep'].values):
        raise RuntimeError('p_dep no hace ida y vuelta exacta por el CSV')
    log(f'  escrito {csv_path.relative_to(paths.REPO).as_posix()} ({len(df)} filas)')

    log('\n— Métricas (§4)')
    m = {s: metrics(*results[s]) for s in PRED_SPLITS}
    for s in PRED_SPLITS:
        log(f'  {s}: F1-Macro {m[s]["f1_macro"]:.6f}, exactitud {m[s]["accuracy"]:.6f}, '
            f'[[TN, FP], [FN, TP]] = {m[s]["confusion_matrix_tn_fp_fn_tp"]}, tasa FN {m[s]["fn_rate"]:.6f}')

    log('\n— V2')
    v2 = v2_report(m['val'], f1_epoch3, results['val'][1], results['val'][0], log)

    cm_rep = REPORTED['confusion_test']
    f1_rep_cm = f1_macro_from_confusion({'tn': cm_rep[0][0], 'fp': cm_rep[0][1], 'fn': cm_rep[1][0],
                                         'tp': cm_rep[1][1]})
    comparison = {
        'f1_test_reportado': REPORTED['f1_test'], 'confusion_test_reportada': cm_rep,
        'f1_de_la_matriz_reportada': f1_rep_cm,
        'f1_test_recalculado': m['test']['f1_macro'],
        'confusion_test_recalculada': m['test']['confusion_matrix_tn_fp_fn_tp'],
        'matriz_coincide': m['test']['confusion_matrix_tn_fp_fn_tp'] == cm_rep,
        'diferencia_f1_vs_reportado': m['test']['f1_macro'] - REPORTED['f1_test'],
    }
    log(f'\n— Comparación con lo reportado: F1-Test reportado {REPORTED["f1_test"]}, '
        f'matriz reportada {cm_rep} (implica F1 {f1_rep_cm:.6f}); recalculado '
        f'{m["test"]["f1_macro"]:.6f}, matriz {m["test"]["confusion_matrix_tn_fp_fn_tp"]} '
        f'→ matriz {"COINCIDE" if comparison["matriz_coincide"] else "NO coincide"}')

    out = {'v1': v1, 'v2': v2, 'v3': v3, 'historial_eval_f1_macro': history, 'metricas': m,
           'comparacion_reportado': comparison, 'inputs': inputs, 'duracion_s': round(time.time() - t_start, 1)}
    (OUT_DIR / 'beto_metricas.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n',
                                                encoding='utf-8', newline='\n')
    log(f'\nDuración: {out["duracion_s"]} s')


if __name__ == '__main__':
    try:
        main()
    except Stop as e:
        log(f'\nDETENIDO: {e}')
        raise SystemExit(2)
    finally:
        log.write()
