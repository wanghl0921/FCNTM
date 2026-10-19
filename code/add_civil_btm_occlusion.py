# -*- coding: utf-8 -*-
'''Append the BTM curve to the civil partial-occlusion files, without retraining.

Why a separate script: occlusion_robustness.py puts the requested model list
into the identity payload of a result file, so re-running it with --models BTM
against the existing five-model civil curves would treat them as unpinned,
stash them and recompute every requested model. This script computes only the
BTM rows with the same masking realisation as the original run, merges them
into the existing CSV, and rewrites the identity sidecar so the payload names
every model actually present in the file.

    python add_civil_btm_occlusion.py --corpus civil --seeds 11,23,37 --k 30
'''

from __future__ import annotations

import argparse
import gc
import shutil

import numpy as np
import pandas as pd

import occlusion_robustness as O
import result_identity as RI

RERUN = O.OUT_DIR / 'rerun'


def masks_for(test, rates, seed):
    '''Reproduce occlusion_robustness.main's masking for one seed exactly.'''
    rng = np.random.RandomState(seed + 7)
    doc_seeds = rng.randint(0, 10 ** 6, size=len(test['reason_tokens']))
    visible, hidden = {}, {}
    for rate in rates:
        vis, hid = [], []
        for i, toks in enumerate(test['reason_tokens']):
            v, h = O.mask_tokens(toks, rate, np.random.RandomState(doc_seeds[i]))
            vis.append(v)
            hid.append(h)
        visible[rate], hidden[rate] = vis, hid
    return visible, hidden


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='civil')
    ap.add_argument('--seeds', default='11,23,37')
    ap.add_argument('--k', type=int, default=30)
    ap.add_argument('--rates', default='0,0.25,0.5,0.75,1.0')
    args = ap.parse_args()
    rates = [float(r) for r in args.rates.split(',') if r]

    train, test = O.load_corpus(args.corpus)
    word2id = test['dictionary'].token2id

    for seed in (int(s) for s in args.seeds.split(',') if s):
        path = O.OUT_DIR / f'occlusion_robustness_{args.corpus}_seed{seed}.csv'
        if not path.exists():
            raise SystemExit(f'{path.name} not found')
        prev = pd.read_csv(path)
        if 'BTM' in O.complete_models(prev, rates):
            print(f'[{args.corpus} seed {seed}] BTM already complete; skipping',
                  flush=True)
            continue
        ident_old = RI.stored(path)
        if not ident_old or 'payload' not in ident_old:
            raise SystemExit(f'{path.name}: no identity sidecar; refusing to '
                             f'overwrite')
        if 'BTM' in set(prev['model']):
            prev = prev[prev['model'] != 'BTM']
            print(f'[{args.corpus} seed {seed}] replacing an incomplete BTM '
                  f'block', flush=True)

        visible, hidden = masks_for(test, rates, seed)
        adapter = O.BTMAdapter(train, args.k, seed)
        adapter.dictionary = train['dictionary']
        rows = O.run_model(adapter, visible, test['item_tokens'], hidden,
                           rates, word2id, 'BTM')
        del adapter
        gc.collect()

        merged = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
        RERUN.mkdir(parents=True, exist_ok=True)
        backup = RERUN / f'backup_occlusion_{args.corpus}_seed{seed}_pre_btm.csv'
        shutil.copy2(path, backup)
        merged.to_csv(path, index=False, encoding='utf-8-sig')

        payload = dict(ident_old['payload'])
        models = payload.get('models', prev['model'].tolist())
        payload['models'] = sorted(set(models) | {'BTM'})
        RI.write(path, RI.identity(payload))
        count = len(payload['models'])
        print(f'[{args.corpus} seed {seed}] added BTM rows ({len(rows)}); '
              f'backup {backup.name}; identity now pins {count} models',
              flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())