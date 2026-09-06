"""Supervised Chinese clock conversion with held-out hour/minute combinations."""
from pathlib import Path
import json
import re

import numpy as np

from .checkpoint import load_checkpoint, save_checkpoint
from .model import TinyGPT, ModelConfig
from .tokenizer import CharacterTokenizer
from .training import loss_and_gradients


def chinese(n):
    digits = '零一二三四五六七八九'
    if n < 10:
        return digits[n]
    return (digits[n // 10] if n >= 20 else '') + '十' + (digits[n % 10] if n % 10 else '')


def examples():
    """Unambiguous scope: morning 1–11, noon 12, afternoon 1–6, evening 7–11."""
    result = []
    for hour in range(1, 24):
        period = '上午' if hour < 12 else '中午' if hour == 12 else '下午' if hour < 19 else '晚上'
        h = hour if hour <= 12 else hour - 12
        for minute in range(60):
            suffixes = [chinese(minute) + '分']
            if minute == 0:
                suffixes += ['', '整']
            if minute == 30:
                suffixes += ['半']
            for suffix in suffixes:
                result.append(dict(input=period + chinese(h) + '点' + suffix,
                                   output=f'{hour:02}:{minute:02}', group=f'{hour}:{minute}'))
    return result


def dataset(seed=42):
    rows = examples()
    groups = sorted({r['group'] for r in rows})
    np.random.default_rng(seed).shuffle(groups)
    n = len(groups)
    assignment = {g: 'train' if i < int(n*.8) else 'validation' if i < int(n*.9) else 'test'
                  for i, g in enumerate(groups)}
    return {split: [r for r in rows if assignment[r['group']] == split]
            for split in ('train', 'validation', 'test')}


def predict(model, tokenizer, text, *, use_cache=False):
    prompt = text + '='
    ids = tokenizer.encode(prompt)
    if tokenizer.UNK_ID in ids:
        raise ValueError('时间表达包含模型词表之外的字符')
    return tokenizer.decode(model.generate(ids, 5, use_cache=use_cache)[len(ids):])


def evaluate(model, tokenizer, rows):
    failures = []
    for row in rows:
        actual = predict(model, tokenizer, row['input'])
        if actual != row['output']:
            failures.append(dict(**row, actual=actual))
    return dict(total=len(rows), correct=len(rows)-len(failures),
                accuracy=1-len(failures)/len(rows), failures=failures)


class TrainingCancelled(Exception):
    """Raised at a cooperative training boundary."""


def validate_splits(splits):
    """Validate uploaded task rows and prevent alias leakage across datasets."""
    if not isinstance(splits, dict) or set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('请分别提供训练集、验证集和测试集')
    reference = {row['input']: row for row in examples()}
    seen_inputs, seen_groups = set(), set()
    clean = {}
    for name in ('train', 'validation', 'test'):
        rows = splits[name]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 2000:
            raise ValueError(f'{name} 需包含 1～2000 条样例')
        clean[name] = []
        groups = set()
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict) or not isinstance(row.get('input'), str):
                raise ValueError(f'{name} 第 {index} 行缺少 input 文本')
            expected = reference.get(row['input'])
            if expected is None or row.get('output') != expected['output']:
                raise ValueError(f'{name} 第 {index} 行不是支持的时间表达或答案不正确')
            if row['input'] in seen_inputs:
                raise ValueError(f'{name} 第 {index} 行与其他样例重复')
            if expected['group'] in seen_groups:
                raise ValueError('同一小时和分钟的不同表达不能跨训练、验证和测试集')
            seen_inputs.add(row['input'])
            groups.add(expected['group'])
            clean[name].append(expected.copy())
        seen_groups.update(groups)
    alphabet = set(''.join(r['input'] + '=' + r['output'] for r in clean['train']))
    if any(set(r['input'] + r['output']) - alphabet
           for name in ('validation', 'test') for r in clean[name]):
        raise ValueError('验证或测试集含训练词表中没有的字符，请补充训练集')
    return clean


def fit(directory, steps=4000, *, splits=None, progress=None, cancelled=None):
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("steps must be between 1 and 10000")
    splits = validate_splits(dataset() if splits is None else splits)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (directory / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    tokenizer = CharacterTokenizer(''.join(r['input']+'='+r['output'] for r in splits['train']))
    model = TinyGPT(ModelConfig(vocab_size=tokenizer.vocab_size, dimensions=64, heads=4, layers=2, context_length=32))
    encoded = []
    for row in splits['train']:
        prefix = row['input'] + '='
        ids = np.asarray(tokenizer.encode(prefix + row['output']))
        encoded.append((ids[:-1], ids[1:], len(prefix)-1))
    rng = np.random.default_rng(7)
    moments = None
    best, best_step = -1, 0
    history = []
    for step in range(1, steps+1):
        if cancelled and cancelled():
            raise TrainingCancelled('训练已停止；当前推理模型未改变')
        summed = None
        losses = []
        for index in rng.integers(len(encoded), size=16):
            x,y,start = encoded[index]
            loss, gradients = loss_and_gradients(model,x,y,answer_start=start)
            losses.append(loss)
            if summed is None:
                summed = [(p,g.copy()/16) for p,g in gradients]
            else:
                for (_,total),(_,g) in zip(summed,gradients): total += g/16
        if moments is None:
            moments = [(np.zeros_like(g),np.zeros_like(g)) for _,g in summed]
        norm = np.sqrt(sum(float(np.sum(g*g)) for _,g in summed))
        for (p,g),(m,v) in zip(summed,moments):
            if not np.isfinite(g).all(): raise ValueError('non-finite gradient')
            g = g/max(1.,norm)
            m *= .9; m += .1*g
            v *= .999; v += .001*g*g
            p -= .001 * (m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        if step % 10 == 0 or step == steps:
            row = dict(step=step, train_batch_loss=float(np.mean(losses)))
            if step % 100 == 0 or step == steps:
                score = evaluate(model,tokenizer,splits['validation'])
                row['validation_accuracy'] = score['accuracy']
                if score['accuracy'] > best:
                    best, best_step = score['accuracy'], step
                    save_checkpoint(directory/'model.npz',model,tokenizer)
            row.update(best_step=best_step, best_validation_accuracy=max(0, best))
            history.append(row)
            if progress:
                progress(**row)
            else:
                print(json.dumps(row),flush=True)
            if best >= .99:
                break
    if cancelled and cancelled():
        raise TrainingCancelled('训练已停止')
    model,tokenizer = load_checkpoint(directory/'model.npz')
    report = dict(task='time', seed=7, split_seed=42, best_step=best_step, history=history,
                  counts={name:len(rows) for name,rows in splits.items()},
                  validation=evaluate(model,tokenizer,splits['validation']),
                  test=evaluate(model,tokenizer,splits['test']))
    report['passed'] = report['test']['accuracy'] >= .95
    (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding='utf-8')
    if not progress:
        print(json.dumps({k:{a:b for a,b in report[k].items() if a!='failures'} for k in ('validation','test')}),flush=True)
    return model, tokenizer, report


def convert_model(model, tokenizer, text, *, use_cache=False):
    # Membership validates supported grammar only; the answer always comes from weights.
    if not isinstance(text, str) or text not in {r['input'] for r in examples()}:
        raise ValueError('不支持的时间表达；请使用上午一至十一点、中午十二点、下午一至六点、晚上七至十一点，搭配整/半/零至五十九分')
    answer = predict(model,tokenizer,text,use_cache=use_cache)
    if not re.fullmatch(r'(?:[01][0-9]|2[0-3]):[0-5][0-9]',answer):
        raise ValueError(f'模型未生成有效时间：{answer!r}')
    return answer


def convert(checkpoint, text):
    model, tokenizer = load_checkpoint(checkpoint)
    return convert_model(model, tokenizer, text)
