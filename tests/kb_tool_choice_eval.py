#!/usr/bin/env python3
"""Local KB-agent evaluation: model-only auto tool choice + actual Harness JSONL grading.

Standard library only. This script never executes a model-requested tool.
"""
import argparse
import csv
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

KB_TOOL = 'mcp__kb_agent__kb_search'
DEFAULT_PROMPT = '我之前是如何设计 DMA 任务批处理，将 1760 次提交减少到 55 次的？请解释 batch 划分和计算公式。'


def records(path):
    with path.open(encoding='utf-8') as f:
        for no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{no}: invalid JSON: {exc}') from exc


def request_json(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.load(res)


def prompt_items(args):
    if args.prompts_file:
        obj = json.loads(Path(args.prompts_file).read_text(encoding='utf-8'))
        if not isinstance(obj, list) or not obj or any(not isinstance(x, str) or not x.strip() for x in obj):
            raise ValueError('--prompts-file must contain a nonempty JSON array of nonempty strings')
        return obj
    return [args.prompt]


def direct(args):
    rows = list(records(Path(args.template)))
    headers = [r['data']['header'] for r in rows if r.get('type') == 'request/header']
    if not headers:
        raise ValueError('Template has no request/header. Use a successful KB-agent session.jsonl')
    header = headers[0]
    tools = header.get('tools', [])
    kb_descriptor = next((t for t in tools if t.get('name') == KB_TOOL), None)
    if kb_descriptor is None:
        raise ValueError(f'Template does not advertise {KB_TOOL}')
    kb_properties = kb_descriptor.get('parameters', {}).get('properties', {})
    if 'filter_expr' in kb_properties:
        raise ValueError(
            'Template advertises deprecated raw filter_expr; capture a fresh '
            'successful session after updating/restarting the kb-agent MCP server.'
        )
    # Harness's tool descriptors use {name, description, parameters}; wrap as OpenAI tools.
    api_tools = [{'type': 'function', 'function': {
        'name': t['name'], 'description': t.get('description', ''),
        'parameters': t.get('parameters', {'type': 'object', 'properties': {}}),
    }} for t in tools]
    base = args.base_url.rstrip('/')
    models = request_json(base + '/models', timeout=30)
    available = [m.get('id') for m in models.get('data', [])]
    if args.model not in available:
        raise ValueError(f'Model {args.model!r} not in {base}/models: {available}')
    prompts = prompt_items(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = kb = any_tool = failed = 0
    with out.open('w', encoding='utf-8') as f:
        for prompt_index, prompt in enumerate(prompts, 1):
            for repetition in range(1, args.repeat + 1):
                payload = {
                    'model': args.model, 'messages': [
                        {'role': 'system', 'content': header['system']},
                        {'role': 'user', 'content': prompt}],
                    'tools': api_tools, 'tool_choice': 'auto',
                    'max_tokens': args.max_tokens, 'stream': False,
                }
                if args.temperature is not None:
                    payload['temperature'] = args.temperature
                start = time.monotonic()
                row = {'prompt_index': prompt_index, 'repetition': repetition,
                       'model': args.model, 'prompt': prompt, 'kb_called': False,
                       'any_tool_called': False, 'error': '', 'tool_calls': [],
                       'finish_reason': '', 'response': '', 'seconds': None}
                try:
                    answer = request_json(base + '/chat/completions', payload, args.timeout)
                    choice = answer['choices'][0]
                    msg = choice['message']
                    calls = msg.get('tool_calls') or []
                    row['tool_calls'] = [{'name': c.get('function', {}).get('name'),
                                          'arguments': c.get('function', {}).get('arguments')}
                                         for c in calls]
                    row['kb_called'] = any(c['name'] == KB_TOOL for c in row['tool_calls'])
                    row['any_tool_called'] = bool(calls)
                    row['finish_reason'] = choice.get('finish_reason')
                    row['response'] = msg.get('content') or ''
                    row['usage'] = answer.get('usage', {})
                    completed += 1
                    kb += row['kb_called']
                    any_tool += row['any_tool_called']
                except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError) as exc:
                    row['error'] = str(exc)
                    failed += 1
                row['seconds'] = round(time.monotonic() - start, 2)
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                print(f"{prompt_index}:{repetition} kb={int(row['kb_called'])} "
                      f"any_tool={int(row['any_tool_called'])} error={bool(row['error'])} "
                      f"time={row['seconds']}s", flush=True)
    print(f'Completed={completed}, kb_search={kb}, any_tool={any_tool}, errors={failed}; '
          f'KB tool-choice rate={kb}/{completed} ({100*kb/completed:.1f}%)' if completed
          else f'No completed calls; errors={failed}')
    print(f'Results: {out}\nNOTE: model-only, no tool executed; not a Harness E2E result.')


def visible_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    return '\n'.join(b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text')


def tool_result_error(event):
    msg = event.get('data', {}).get('message', {})
    blocks = msg.get('content', [])
    errors = [b.get('isError') for b in blocks if isinstance(b, dict) and b.get('type') == 'tool-result']
    return any(x is True for x in errors) if errors else None


def normalized_answer(text):
    # Qwen3 reasoning was embedded in content in these sessions; exclude it from scoring.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S | re.I)
    return unicodedata.normalize('NFKC', text).strip()


def collect_paths(paths):
    found = []
    for supplied in paths:
        p = Path(supplied)
        if p.is_dir():
            found.extend(sorted(p.rglob('*.jsonl')))
        elif p.is_file():
            found.append(p)
        else:
            print(f'WARNING: missing {p}', file=sys.stderr)
    return list(dict.fromkeys(found))


def grade(args):
    paths = collect_paths(args.paths)
    if not paths:
        raise ValueError('No JSONL files found')
    lines = []
    for path in paths:
        rows = list(records(path))
        head = next((r['data']['header'] for r in rows if r.get('type') == 'request/header'), {})
        config = head.get('config', {})
        calls = [r for r in rows if r.get('type') == 'tool/call']
        kb_calls = [c for c in calls if c.get('data', {}).get('name') == KB_TOOL]
        results = [r for r in rows if r.get('type') == 'tool/result']
        results_by_id = {r.get('data', {}).get('message', {}).get('source', {}).get('callId'): r
                         for r in results}
        kb_results = [results_by_id.get(c.get('data', {}).get('callId')) for c in kb_calls]
        ok = sum(r is not None and tool_result_error(r) is False for r in kb_results)
        finish = next((r.get('data', {}).get('reason', {}).get('kind') for r in reversed(rows)
                       if r.get('type') == 'turn/end'), '')
        assistants = [r for r in rows if r.get('type') == 'assistant/message']
        final = normalized_answer(visible_text(assistants[-1]['data']['message'].get('content', []))) if assistants else ''
        requests = [r for r in rows if r.get('type') == 'user/message']
        user_prompt = visible_text(requests[0].get('data', {}).get('content', [])) if requests else ''
        usage = assistants[-1].get('data', {}).get('usage', {}) if assistants else {}
        # DMA-only mechanical coverage indicators, NOT a truthfulness/quality score.
        dma = '1760' in user_prompt and '55' in user_prompt
        coverage = {f'has_{n}': bool(re.search(r'(?<!\d){n}(?!\d)', final)) if dma else ''
                    for n in ('11', '5', '32', '1760', '55')}
        result = {
            'file': str(path), 'preset': rows[0].get('agentPreset', '') if rows else '',
            'provider': config.get('provider', ''), 'model': config.get('model', ''),
            'prompt': user_prompt, 'finish': finish, 'all_tool_calls': len(calls),
            'kb_calls': len(kb_calls), 'kb_tool_successes': ok,
            'kb_triggered': bool(kb_calls), 'kb_completed': ok > 0,
            'input_tokens_last': usage.get('inputTokens', ''),
            'output_tokens_last': usage.get('outputTokens', ''),
            'dma_test': dma, **coverage,
            'has_source_title': ('DMA计算方案设计' in final) if dma else '',
            'has_chunk_citation': ('chunk:' in final or 'chunk_id' in final) if dma else '',
            'mentions_other_baseline': bool(re.search(r'(?<!\d)56320(?!\d)', final)) if dma else '',
            'manual_grounded': '', 'manual_correct': '', 'manual_notes': '',
            'answer': final,
        }
        lines.append(result)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(lines[0]))
        writer.writeheader()
        writer.writerows(lines)
    complete = [x for x in lines if x['finish'] == 'completed']
    hit = sum(x['kb_triggered'] for x in complete)
    success = sum(x['kb_completed'] for x in complete)
    print(f'Logs={len(lines)}, completed={len(complete)}, kb_search triggered={hit}, '
          f'kb_search returned OK={success}')
    if complete:
        print(f'Automatic trigger rate (completed only): {hit}/{len(complete)} '
              f'= {100*hit/len(complete):.1f}%')
    print('Coverage flags only check strings; fill manual_grounded/manual_correct (0/1) '
          'after reviewing answer against known evidence.')
    print(f'CSV: {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    d = sub.add_parser('direct', help='20 model-only tool-choice attempts, no tools executed')
    d.add_argument('--template', required=True, help='Real KB-agent session.jsonl (for system and all tools)')
    d.add_argument('--base-url', default='http://127.0.0.1:8001/v1')
    d.add_argument('--model', required=True, help='Exact model id returned by /v1/models')
    d.add_argument('--prompt', default=DEFAULT_PROMPT)
    d.add_argument('--prompts-file', help='JSON array of questions; repeat applies to each question')
    d.add_argument('--repeat', type=int, default=20)
    d.add_argument('--max-tokens', type=int, default=4096)
    d.add_argument('--temperature', type=float, help='Omit to use the server/model default')
    d.add_argument('--timeout', type=int, default=600)
    d.add_argument('--out', default='kb_tool_choice_direct.jsonl')
    d.set_defaults(fn=direct)
    g = sub.add_parser('grade', help='Grade actual Harness session.jsonl files (one fresh session per file)')
    g.add_argument('paths', nargs='+', help='20 session.jsonl files or a directory containing them')
    g.add_argument('--out', default='kb_harness_eval.csv')
    g.set_defaults(fn=grade)
    a = p.parse_args()
    if a.mode == 'direct' and (a.repeat < 1 or a.max_tokens < 1):
        p.error('--repeat and --max-tokens must be positive')
    try:
        a.fn(a)
    except (ValueError, OSError) as e:
        p.error(str(e))


if __name__ == '__main__':
    main()
