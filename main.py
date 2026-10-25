import argparse
import json
from asr_audit import load, score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Offline call-transcript evaluation')
    parser.add_argument('input', nargs='?', default='synthetic_calls.jsonl')
    args = parser.parse_args()
    print(json.dumps(score(load(args.input)), ensure_ascii=False, indent=2))
