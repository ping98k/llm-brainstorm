# llm-brainstorm

This project provides a small interface for running "tournaments" between language model answers. It is built with Gradio and LiteLLM.

## Usage

1. Create a `.env` file in the repository root and define any API keys required by your model. You can also set defaults for:
   - `NUM_TOP_PICKS`
   - `POOL_SIZE`
   - `MAX_WORKERS`
   - `NUM_GENERATIONS`
   - `OPENAI_API_BASE`
   - `OPENAI_API_KEY`
   - `GENERATE_MODEL`
   - `SCORE_MODEL`
   - `PAIRWISE_MODEL`
   - `GENERATE_TEMPERATURE`
   - `SCORE_TEMPERATURE`
   - `PAIRWISE_TEMPERATURE`
   - `PASS_INSTRUCTION_TO_SCORE`
   - `PASS_INSTRUCTION_TO_PAIRWISE`
   - `ENABLE_SCORE_FILTER`
   - `ENABLE_PAIRWISE_FILTER`
   - `ENABLE_GENERATE_THINKING`
   - `ENABLE_SCORE_THINKING`
   - `ENABLE_PAIRWISE_THINKING`
   - `THINKING_BUDGET_TOKENS`
   
   When any of the thinking flags are enabled, the app sends
   `thinking={"type": "enabled", "budget_tokens": $THINKING_BUDGET_TOKENS}` with each
   `litellm.completion` call for that model. Otherwise it sends
   `thinking={"type": "disabled", "budget_tokens": 0}`.
2. Install dependencies (example with `pip`):
   ```bash
   pip install gradio litellm python-dotenv tqdm matplotlib
   ```
3. Run the app:
   ```bash
   python main.py
   ```
4. Open the displayed local URL. At the top of the page you can optionally override the API base path and token (the token field is blank by default). Additional settings let you configure score and pairwise filtering.

The interface will generate multiple answers, optionally filter them by score and run a pairwise tournament to select the best outputs.

## Terminology

- *Judge* refers to both the **Score Model** and **Pairwise Model**.
- When instructions mention **updating the LLM**, update the **Generation**, **Score**, and **Pairwise** models together.
- *Output*, *answer*, or *response* are all considered the same as a **player** in the tournament.