import json
import yaml
import numpy as np

from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel, AutoConfig
import datasets
import os
import contextlib
import io


def _truthy_env(name: str, default: str = "1") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() not in {"0", "false", "no", "off", ""}


def _maybe_quiet_hf_load() -> contextlib.AbstractContextManager:
    """
    Silence HuggingFace/transformers progress bars and noisy stdout during model load.

    Default behavior is quiet. Set ABENCH_QUIET_MODEL_LOAD=0 to re-enable.
    """
    quiet = _truthy_env("ABENCH_QUIET_MODEL_LOAD", default="1")
    if not quiet:
        return contextlib.nullcontext()

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    try:
        from transformers.utils import logging as _tlog

        _tlog.set_verbosity_error()
        _tlog.disable_progress_bar()
    except Exception:
        pass
    try:
        datasets.logging.set_verbosity_error()
    except Exception:
        pass

    return contextlib.redirect_stdout(io.StringIO())

@dataclass
class BaseConfig:
    """Base config class for all configs."""
    
    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as f:
            config = json.load(f)
        return cls.from_dict(config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config = yaml.load(f)
        return cls.from_dict(config)
    
    @classmethod
    def from_dict(cls, config: dict):
        # If cls is BaseConfig itself (not a subclass), it has no fields
        # so we can't use **config. Subclasses should override this if needed.
        # For dataclass subclasses, this will work correctly.
        import inspect
        if cls == BaseConfig:
            # BaseConfig has no fields, return empty instance
            return cls()
        # For subclasses (like FaissEnginConfig), use **config
        return cls(**config)


# a: score, i: index
def topk_merge(a1, a2, i1, i2, k):
    # 合并 value 和 index
    a_concat = np.concatenate([a1, a2])
    i_concat = np.concatenate([i1, i2])

    # 按照 value 排序（升序，值越小越相似）
    sorted_indices = np.argsort(a_concat)

    topk_indices = sorted_indices[:k]

    # 最终的 topk 值和对应索引
    a_topk = a_concat[topk_indices].astype(np.float32)
    i_topk = i_concat[topk_indices].astype(np.int64)
    return a_topk, i_topk

def fvec_inner_product(x, y):
    return np.dot(np.array(x), np.array(y))

def fvec_L2sqr(x, y):
    return np.linalg.norm(np.array(x) - np.array(y)) ** 2


def load_model(model_path: str, use_fp16: bool = False):
    buf_err = io.StringIO()
    try:
        quiet = _truthy_env("ABENCH_QUIET_MODEL_LOAD", default="1")
        if quiet:
            with _maybe_quiet_hf_load(), contextlib.redirect_stderr(buf_err):
                _ = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=True, trust_remote_code=True
                )
        else:
            _ = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=True, trust_remote_code=True
            )
    except Exception as exc:
        tail = (buf_err.getvalue() or "").strip()
        if tail:
            raise RuntimeError(f"failed to load model from {model_path}: {exc}\n{tail}") from exc
        raise RuntimeError(f"failed to load model from {model_path}: {exc}") from exc

    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        while True:
            new_line = f.readline()
            if not new_line:
                return
            new_item = json.loads(new_line)

            yield new_item


def load_corpus(corpus_path: str):
    import os
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_path!r}. "
            "Ensure the path exists or set corpus_path (e.g. via --corpus_path or CORPUS_PATH env) to a valid .jsonl or .parquet file."
        )
    if corpus_path.endswith(".jsonl"):
        return datasets.load_dataset("json", data_files=corpus_path, split="train")
    if corpus_path.endswith(".parquet"):
        return datasets.load_dataset("parquet", data_files=corpus_path, split="train")
    raise ValueError(
        f"Unsupported corpus format: {corpus_path!r}. Use .jsonl or .parquet."
    )


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]

    return results
