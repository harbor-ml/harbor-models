import rpc
import os
import sys
import json

import numpy as np
import cloudpickle
import torch
import importlib
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn import functional as F

IMPORT_ERROR_RETURN_CODE = 3


def predict_func(raw_text, next_words, num_tries, model, tokenizer):
    texts = [tokenizer.encode(raw_text) for _ in range(num_tries)]
    inp, past, logits = torch.tensor(texts), None, None
    inp = inp.to("cuda")
    text = []
    with torch.no_grad():
        for _ in range(next_words):
            logits, past = model(inp, past=past)
            inp = torch.multinomial(F.softmax(logits[:, -1]), 1)
            text.append(inp.to("cpu").numpy().squeeze())
    del inp, past, logits

    results = {}
    for i in range(num_tries):
        decoded_text = tokenizer.decode(np.vstack(text)[:, i])
        results[
            i+1
        ] = f"""
        {raw_text}{decoded_text}
        """
    return results


class PyTorchContainer(rpc.ModelContainerBase):
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model = model.to("cuda")

    def predict_strings(self, inputs):
        result = []
        for inp in inputs:
            request = json.loads(inp)
            resp = predict_func(
                request["text"],
                request["num_words"],
                request["num_tries"],
                self.model,
                self.tokenizer,
            )
            result.append(json.dumps(resp))
        return result


if __name__ == "__main__":
    print("Starting PyTorchContainer container")
    rpc_service = rpc.RPCService()
    try:
        model = PyTorchContainer()
        sys.stdout.flush()
        sys.stderr.flush()
    except ImportError:
        sys.exit(IMPORT_ERROR_RETURN_CODE)
    rpc_service.start(model)
