from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def _to_text(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, list):
        return ' '.join(_to_text(item) for item in value if item is not None).strip()
    return str(value).strip()


def _build_legacy_node_text(node: dict) -> str:
    resource_id = _to_text(node.get('resource-id'))
    content_desc = _to_text(node.get('content-desc'))
    text = _to_text(node.get('text'))
    bounds = node.get('bounds', [])
    clickable = str(node.get('clickable', False))
    scroll_v = str(node.get('scrollable-vertical', False))
    scroll_h = str(node.get('scrollable-horizontal', False))
    ancestors = _to_text(node.get('ancestors'))

    parts = [
        f'rid={resource_id}',
        f'desc={content_desc}',
        f'text={text}',
        f'bounds={bounds}',
        f'clickable={clickable}',
        f'scroll_v={scroll_v}',
        f'scroll_h={scroll_h}',
        f'ancestors={ancestors}',
    ]
    return ' | '.join(parts)


def _build_semantic_node_text(node: dict) -> str:
    node_class = _to_text(node.get('class'))
    resource_id = _to_text(node.get('resource-id'))
    content_desc = _to_text(node.get('content-desc'))
    text = _to_text(node.get('text'))
    visibility = _to_text(node.get('visibility'))
    clickable = str(node.get('clickable', False))
    enabled = str(node.get('enabled', True))
    selected = str(node.get('selected', False))
    focused = str(node.get('focused', False))
    scroll_v = str(node.get('scrollable-vertical', False))
    scroll_h = str(node.get('scrollable-horizontal', False))
    visible_to_user = str(node.get('visible-to-user', True))

    parts = [
        f'class={node_class}',
        f'rid={resource_id}',
        f'text={text}',
        f'desc={content_desc}',
        f'visibility={visibility}',
        f'clickable={clickable}',
        f'enabled={enabled}',
        f'selected={selected}',
        f'focused={focused}',
        f'scroll_v={scroll_v}',
        f'scroll_h={scroll_h}',
        f'visible={visible_to_user}',
    ]
    return ' | '.join(parts)


def _walk_node(node: dict, tokens: list[str], profile: str) -> None:
    if profile == 'semantic_v2':
        tokens.append(_build_semantic_node_text(node))
    else:
        tokens.append(_build_legacy_node_text(node))

    for child in node.get('children', []):
        if isinstance(child, dict):
            _walk_node(child, tokens, profile)


@lru_cache(maxsize=131072)
def _linearize_tree_cached(json_path: str, max_nodes: int, profile: str) -> str:
    with Path(json_path).open('r', encoding='utf-8') as fp:
        payload = json.load(fp)

    activity_name = _to_text(payload.get('activity_name'))
    root = payload.get('activity', {}).get('root', {})
    tokens: list[str] = []
    if activity_name:
        prefix = 'screen' if profile == 'semantic_v2' else 'activity'
        tokens.append(f'{prefix}={activity_name}')
    if root:
        _walk_node(root, tokens, profile)

    return '\n'.join(tokens[:max_nodes])


def linearize_tree(json_path: str | Path, max_nodes: int = 256, profile: str = 'legacy') -> str:
    normalized_path = str(Path(json_path).resolve())
    return _linearize_tree_cached(normalized_path, max_nodes, profile)
