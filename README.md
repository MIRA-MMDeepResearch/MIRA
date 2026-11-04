<div align="center">

  <img src="assets/mira-logo.png" alt="MIRA" width="22%">
  <h1>MIRA: A Multimodal Illustrated Deep Research Agent for Controllable Reports</h1>

  [![Paper](https://img.shields.io/badge/Paper-DA291C?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.25926)
  [![HF Papers](https://img.shields.io/badge/HF_Papers-8F1D14?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/papers/2610.25926)
  [![Project Page](https://img.shields.io/badge/Project_Page-000000?style=for-the-badge&logo=googlechrome&logoColor=white)](https://mira-mmdeepresearch.github.io/)
  [![Model](https://img.shields.io/badge/MIRA--SFT--35B--A3B-DA291C?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/MIRA-MMDeepResearch/MIRA-SFT-35B-A3B)
  [![Data](https://img.shields.io/badge/MIRA_Data-DA291C?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/MIRA-MMDeepResearch)

  [![Model License](https://img.shields.io/badge/Model_License-Apache_2.0-DA291C.svg)](https://huggingface.co/MIRA-MMDeepResearch/MIRA-SFT-35B-A3B)
  [![Data License](https://img.shields.io/badge/Data_License-CC_BY--NC_4.0-8F1D14.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
  [![arXiv](https://img.shields.io/badge/arXiv-2610.25926-B31B1B.svg)](https://arxiv.org/abs/2610.25926)

</div>

---

## Setup

Use Python 3.10 or later on Linux.

```bash
pip install -r requirements.txt
cp configs/secrets.example.yaml configs/secrets.yaml
```

Configure your model name, OpenAI-compatible API endpoint, tokenizer directory and sandbox endpoint in [configs/mira.yaml](configs/mira.yaml), and your API keys in `configs/secrets.yaml`.

Python execution uses [SandboxFusion](https://bytedance.github.io/SandboxFusion/) with Pillow and NumPy.

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
