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


def _normalize_bounds(bounds: object) -> tuple[float, float, float, float, float, float] | None:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    try:
        left, top, right, bottom = [float(value) for value in bounds]
    except (TypeError, ValueError):
        return None

    width = max(right - left, 0.0)
    height = max(bottom - top, 0.0)
    return left, top, right, bottom, width, height


def _bucket_3way(value: float, low: float = 1 / 3, high: float = 2 / 3) -> str:
    if value < low:
        return 'low'
    if value < high:
        return 'mid'
    return 'high'


def _area_bucket(area_ratio: float) -> str:
    if area_ratio < 0.01:
        return 'tiny'
    if area_ratio < 0.05:
        return 'small'
    if area_ratio < 0.2:
        return 'medium'
    if area_ratio < 0.5:
        return 'large'
    return 'full'


def _shape_bucket(width: float, height: float) -> str:
    if width <= 0 or height <= 0:
        return 'flat'
    ratio = width / max(height, 1e-6)
    if ratio > 1.6:
        return 'wide'
    if ratio < 0.625:
        return 'tall'
    return 'balanced'


def _simplify_identifier(value: object) -> str:
    text = _to_text(value)
    if not text:
        return ''
    return text.split('/')[-1]


def _spatial_tokens(node: dict, root_bounds: tuple[float, float, float, float, float, float] | None) -> list[str]:
    node_bounds = _normalize_bounds(node.get('bounds'))
    if root_bounds is None or node_bounds is None:
        return []

    root_left, root_top, _, _, root_width, root_height = root_bounds
    left, top, right, bottom, width, height = node_bounds
    if root_width <= 0 or root_height <= 0:
        return []

    center_x = ((left + right) * 0.5 - root_left) / root_width
    center_y = ((top + bottom) * 0.5 - root_top) / root_height
    area_ratio = (width * height) / max(root_width * root_height, 1e-6)

    return [
        f'pos_x={_bucket_3way(center_x)}',
        f'pos_y={_bucket_3way(center_y)}',
        f'size={_area_bucket(area_ratio)}',
        f'shape={_shape_bucket(width, height)}',
    ]


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


def _build_semantic_v3_node_text(
    node: dict,
    depth: int,
    root_bounds: tuple[float, float, float, float, float, float] | None,
) -> str:
    node_class = _to_text(node.get('class'))
    resource_id = _to_text(node.get('resource-id'))
    resource_tail = _simplify_identifier(resource_id)
    content_desc = _to_text(node.get('content-desc'))
    text = _to_text(node.get('text'))
    visibility = _to_text(node.get('visibility'))
    package = _to_text(node.get('package'))
    package_tail = package.split('.')[-1] if package else ''
    class_tail = node_class.split('.')[-1] if node_class else ''
    clickable = str(node.get('clickable', False))
    enabled = str(node.get('enabled', True))
    selected = str(node.get('selected', False))
    focused = str(node.get('focused', False))
    focusable = str(node.get('focusable', False))
    long_clickable = str(node.get('long-clickable', False))
    visible_to_user = str(node.get('visible-to-user', True))
    scroll_v = str(node.get('scrollable-vertical', False))
    scroll_h = str(node.get('scrollable-horizontal', False))
    adapter_view = str(node.get('adapter-view', False))
    draw = str(node.get('draw', False))
    child_count = len(node.get('children', [])) if isinstance(node.get('children'), list) else 0

    parts = [
        f'class={node_class}',
        f'class_short={class_tail}',
        f'package={package}',
        f'package_short={package_tail}',
        f'rid={resource_id}',
        f'rid_short={resource_tail}',
        f'text={text}',
        f'desc={content_desc}',
        f'visibility={visibility}',
        f'clickable={clickable}',
        f'enabled={enabled}',
        f'focusable={focusable}',
        f'long_clickable={long_clickable}',
        f'selected={selected}',
        f'focused={focused}',
        f'visible={visible_to_user}',
        f'scroll_v={scroll_v}',
        f'scroll_h={scroll_h}',
        f'adapter={adapter_view}',
        f'draw={draw}',
        f'depth={depth}',
        f'child_count={child_count}',
    ]
    parts.extend(_spatial_tokens(node, root_bounds))
    return ' | '.join(parts)


def _walk_node(
    node: dict,
    tokens: list[str],
    profile: str,
    root_bounds: tuple[float, float, float, float, float, float] | None,
    depth: int,
) -> None:
    if profile == 'semantic_v3':
        tokens.append(_build_semantic_v3_node_text(node, depth, root_bounds))
    elif profile == 'semantic_v2':
        tokens.append(_build_semantic_node_text(node))
    else:
        tokens.append(_build_legacy_node_text(node))

    for child in node.get('children', []):
        if isinstance(child, dict):
            _walk_node(child, tokens, profile, root_bounds, depth + 1)


@lru_cache(maxsize=131072)
def _linearize_tree_cached(json_path: str, max_nodes: int, profile: str) -> str:
    with Path(json_path).open('r', encoding='utf-8') as fp:
        payload = json.load(fp)

    activity_name = _to_text(payload.get('activity_name'))
    root = payload.get('activity', {}).get('root', {})
    root_bounds = _normalize_bounds(root.get('bounds')) if root else None
    tokens: list[str] = []
    if activity_name:
        prefix = 'screen' if profile.startswith('semantic') else 'activity'
        tokens.append(f'{prefix}={activity_name}')
        activity_package = activity_name.split('/')[0]
        if profile == 'semantic_v3' and activity_package:
            tokens.append(f'screen_pkg={activity_package}')
    if profile == 'semantic_v3' and root_bounds is not None:
        orientation = 'portrait' if root_bounds[5] >= root_bounds[4] else 'landscape'
        tokens.append(f'orientation={orientation}')
    if root:
        _walk_node(root, tokens, profile, root_bounds, depth=0)

    return '\\n'.join(tokens[:max_nodes])


def linearize_tree(json_path: str | Path, max_nodes: int = 256, profile: str = 'legacy') -> str:
    normalized_path = str(Path(json_path).resolve())
    return _linearize_tree_cached(normalized_path, max_nodes, profile)
