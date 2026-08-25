# Capability evaluations

These cases compare providers on the work this assistant actually needs: difficult explanations, tutoring, coding, tool selection, clarification, and memory personalization.

The runner records raw model output, tool-call fragments, time to first streamed event, first visible text, and total time. It does not use an automatic judge, mutate the workspace, execute model-selected tools, or commit API results.

Run one provider or one case at a time:

```bash
python evals/run.py --provider groq --output /tmp/groq-eval.json
python evals/run.py --provider gemini-gemma4-31b --case reasoning_tradeoff
```

Review the output against the rubric in `cases.json`. Use the results to decide routing; do not infer that the fastest model is the best model.
