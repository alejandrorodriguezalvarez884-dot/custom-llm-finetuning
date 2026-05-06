"""
Local inference server for the fine-tuned LittleLamb model.
Exposes an OpenAI-compatible /v1/chat/completions endpoint.

Usage:
    python src/server.py

The server loads the base model and merges the LoRA adapters from
outputs/model/ at startup. If no adapters are found, the base model
is used (useful for smoke-testing without fine-tuning first).
"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).parent.parent
MODEL_ID = "MultiverseComputingCAI/LittleLamb"
ADAPTER_DIR = ROOT / "outputs" / "model"
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

_SYSTEM_PROMPT = (
    "You are a helpful assistant with specialized knowledge from a curated set of documents. "
    "Answer questions accurately and concisely based on your training. "
    "If you are unsure, say so clearly."
)

_state: dict = {}


def _load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    has_adapters = (ADAPTER_DIR / "adapter_config.json").exists()

    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_DIR) if has_adapters else MODEL_ID,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    if has_adapters:
        print("Loading and merging LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
        model = model.merge_and_unload()
        print("Adapters merged into base model.")
    else:
        print("Warning: no fine-tuned adapters found in outputs/model/. Using base model.")
        model = base_model

    model.eval()
    return model, tokenizer


@asynccontextmanager
async def lifespan(app):
    _state["model"], _state["tokenizer"] = _load_model()
    print(f"\nServer ready at http://{SERVER_HOST}:{SERVER_PORT}")
    yield


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="LittleLamb Local API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class _Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[_Message]
    max_tokens: int = 512
    temperature: float = 0.7


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    import torch

    model = _state.get("model")
    tokenizer = _state.get("tokenizer")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=max(request.temperature, 1e-7),
            do_sample=request.temperature > 0.01,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
