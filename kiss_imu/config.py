"""YAML configs with dotted command-line overrides (`pgo.window=15 lo.registration.name=small_gicp`)."""

import copy

import yaml


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(cfg, overrides):
    for item in overrides or []:
        key, val = item.split('=', 1)
        node = cfg
        *path, last = key.split('.')
        for p in path:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    return cfg


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop('base', None)
    if base:                    # one path or a list, merged in order, then this file on top
        import os
        merged = {}
        for b in [base] if isinstance(base, str) else base:
            merged = merge(merged, load_config(os.path.join(os.path.dirname(path), b)))
        cfg = merge(merged, cfg)
    return apply_overrides(cfg, overrides)
