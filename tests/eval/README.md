# Dataset Format (`.jsonl`)

- Each line in the file is a valid JSON object with the following fields:

```json
{
  "id": "unique string identifier (SHA-256 hash of the prompt",
  "prompt": "input text that the model sees",
  "completion": "expected answer or continuation (about 256 tokens)"
}
```

- To run

```bash
python eval.py -p ../data/input.txt -s ../data/output.txt -r refs_openai_gpt5.jsonl
```
