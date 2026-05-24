import json
import re

def extract_prediction(text):
    try:
        matches = re.findall(r"\{[\s\S]*?\}", text)
        if matches:
            last_json = matches[-1]
            return json.loads(last_json)

    except Exception:
        pass

    return {
    "score": -1,
    "rationale": "Parsing failed"
    }