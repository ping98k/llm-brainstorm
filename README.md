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

   When any of the thinking flags are enabled, the app sends
   `chat_template_kwargs={"enable_thinking": True}` with each
   `litellm.completion` call for that model. Otherwise it sends
   `chat_template_kwargs={"enable_thinking": False}`.
2. Install dependencies (example with `pip`):
   ```bash
   pip install gradio litellm python-dotenv tqdm matplotlib
   ```
3. Run the app:
   ```bash
   python main.py
   ```
4. Open the displayed local URL. At the top of the page you can optionally override the API base path and token (the token field is blank by default). Additional settings let you configure score and pairwise filtering.

The interface will generate multiple answers, optionally filter them by score and run a pairwise tournament to select the best outputs. Results from previous pairwise comparisons are cached, so duplicate matches are skipped for faster tournaments. Pairwise results are aggregated using an Elo rating system to rank the players.

## Terminology

- *Judge* refers to both the **Score Model** and **Pairwise Model**.
- When instructions mention **updating the LLM**, update the **Generation**, **Score**, and **Pairwise** models together.
- *Output*, *answer*, or *response* are all considered the same as a **player** in the tournament.