# llm-brainstorm

This project provides a small interface for running "tournaments" between language model answers. It is built with Gradio and LiteLLM.

## Usage

1. Create a `.env` file in the repository root and define any API keys required by your model. You can also set defaults for:
   - `NUM_TOP_PICKS`
   - `POOL_SIZE`
   - `MAX_WORKERS`
   - `NUM_GENERATIONS`
2. Install dependencies (example with `pip`):
   ```bash
   pip install gradio litellm python-dotenv tqdm matplotlib
   ```
3. Run the app:
   ```bash
   python main.py
   ```
4. Open the displayed local URL to provide an instruction and evaluation criteria.

The interface will generate multiple answers, score them, and run a head-to-head tournament to find the best outputs.

