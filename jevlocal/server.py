"""HTTP server with Jev's /v1/systemone request and response shapes."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from jevlocal.calibration import Calibration
from jevlocal.engine import Engine


def build_app(engine: Engine, workers: int = 64) -> FastAPI:
    app = FastAPI(title="jevlocal")
    pool = ThreadPoolExecutor(workers)

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": engine.model_name}]}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        body = await request.json()
        options = body.get("jevlocal") or {}
        if not isinstance(body.get("questions"), dict) or not body["questions"]:
            raise HTTPException(422, "questions must be a non-empty map")
        try:
            return await asyncio.get_running_loop().run_in_executor(pool, lambda: engine.evaluate(
                body, permutations=options.get("permutations"), calibrate=options.get("calibrate", True),
                raw=options.get("raw", False)))
        except ValueError as err:
            raise HTTPException(422, str(err)) from None

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, help="Hugging Face id (hf, sglang) or Bedrock model id")
    parser.add_argument("--name", help="model name reported in responses (default: model id)")
    parser.add_argument("--backend", choices=["sglang", "hf", "bedrock"], default="sglang")
    parser.add_argument("--region", default="us-west-2", help="AWS region for the bedrock backend")
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    parser.add_argument("--calibration", help="JSON file with per-kind temperatures")
    parser.add_argument("--permutations", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.backend == "hf":
        from jevlocal.backends import HFBackend

        backend = HFBackend(args.model_id)
    elif args.backend == "bedrock":
        from jevlocal.backends import BedrockBackend

        backend = BedrockBackend(args.model_id, region=args.region)
    else:
        from transformers import AutoTokenizer

        from jevlocal.backends import SGLangBackend

        backend = SGLangBackend(AutoTokenizer.from_pretrained(args.model_id), args.sglang_url)
    engine = Engine(backend, args.name or args.model_id, Calibration.load(args.calibration), args.permutations)
    uvicorn.run(build_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
