"""Request-scoped inference measurements, including prompt prefill."""
from statistics import median
from time import perf_counter


def timed_generate(model, ids, count, use_cache):
    started = perf_counter()
    result = model.generate(ids, count, use_cache=use_cache)
    elapsed = perf_counter() - started
    return result, {'elapsed_ms': elapsed * 1000,
                    'tokens_per_second': count / elapsed if count else 0,
                    'use_cache': use_cache}


def benchmark(model, tokenizer, ids, count):
    # Warm both paths; alternate order to reduce first-run and ordering effects.
    for cache in (False, True):
        model.generate(ids, count, use_cache=cache)
    times = {False: [], True: []}
    outputs = {}
    identical = True
    for repeat in range(3):
        for cache in ((False, True) if repeat % 2 == 0 else (True, False)):
            result, metrics = timed_generate(model, ids, count, cache)
            if cache in outputs:
                identical = identical and outputs[cache] == result
            outputs[cache] = result
            times[cache].append(metrics['elapsed_ms'])
        identical = identical and outputs[False] == outputs[True]
    measurements = {}
    for cache, label in ((False, 'uncached'), (True, 'cached')):
        elapsed = median(times[cache])
        measurements[label] = {'elapsed_ms': elapsed,
                               'tokens_per_second': count * 1000 / elapsed if count else 0,
                               'completion': tokenizer.decode(outputs[cache][len(ids):])}
    return {**measurements, 'identical': identical, 'repeats': 3,
            'speedup': measurements['uncached']['elapsed_ms'] / measurements['cached']['elapsed_ms'],
            'prompt_tokens': len(ids), 'generated_tokens': count,
            'window_rebuilds': max(0, len(ids) + count - 1 - model.config.context_length) if count else 0}
