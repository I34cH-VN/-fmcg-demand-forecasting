# Assumptions

- The default config uses the synthetic sample data so portfolio users can run the CLI without private raw data.
- Production or private runs should override `input.path` or set it empty and place CSV files under `data/raw/`.
- Weekly data is expected to use Monday-start weeks in `week_start`.
- Forecasting is one-week-ahead using historical lag and rolling features.
- The AI layer uses a deterministic `MockLLM` by default. `OpenAILLM` is optional and requires `OPENAI_API_KEY`.
