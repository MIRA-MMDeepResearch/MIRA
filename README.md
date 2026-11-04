# MIRA

Inference code for [MIRA](https://arxiv.org/abs/2610.25926) and [MIRA-SFT-35B-A3B](https://huggingface.co/MIRA-MMDeepResearch/MIRA-SFT-35B-A3B): one ReAct agent with browser search, open, find and image-aware Python execution.

## Setup

Use Python 3.10 or later on Linux.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/secrets.example.yaml configs/secrets.yaml
export MODELS_ROOT=/path/to/models
```

Fill in your Serper and Jina keys in `configs/secrets.yaml`; set `llm.api_key` if your model server requires authentication. Configure the model URL and SandboxFusion endpoint in `configs/mira.yaml`.

Download the model to `$MODELS_ROOT/MIRA-SFT-35B-A3B`. For a local model server, run vLLM in a separate environment:

```bash
vllm serve "$MODELS_ROOT/MIRA-SFT-35B-A3B" --served-model-name MIRA \
  --tensor-parallel-size 8 --max-model-len 262144 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --host 127.0.0.1 --port 8000
```

Start [SandboxFusion](https://bytedance.github.io/SandboxFusion/) with Pillow and NumPy available:

```bash
docker run --rm -d -p 127.0.0.1:8080:8080 volcengine/sandbox-fusion:server-20250609
```

Reverse image search uploads images to the public host configured by `image.lens_upload_endpoint` (Uguu by default).

## Run

Create a JSONL file with one request per line:

```json
{"id":"example","question":"What is shown in these images?","images":["image1.jpg","image2.jpg"]}
```

Image paths are relative to the input file; HTTP(S) URLs and image data URLs are also supported. Omit `images` for text-only requests.

```bash
python -m mira.infer --input requests.jsonl --output workspace/run --workers 4
```

Answers and trajectories are written to `workspace/run/results.jsonl`. Use a new output directory for each run. Each request runs once; failures are recorded and cause a nonzero exit status.

The research system prompt is in [prompt.py](mira/agent/prompts/prompt.py). The paper's report-contract prompt is preserved in [harness.py](mira/agent/prompts/harness.py) for reference; it requires a `submit_report` gate, which this four-tool runner does not implement.

## License

Apache-2.0. See [THIRD_PARTY.md](THIRD_PARTY.md).
