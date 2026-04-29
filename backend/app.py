"""PDF to Markdown converter using Vision LLM (Ollama / Azure OpenAI)."""

import base64
import io
import json
import os
import re
import sys
import time
import uuid

import fitz  # PyMuPDF
import requests
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont

from providers import get_all_models, get_provider

app = Flask(__name__, static_folder="static")
CORS(app)

OUTPUT_FORMAT = "md"

BASE_RULES = {
    "ja": [
        "- 画像に見えるテキストだけを書き起こすこと。説明や補足は一切加えない",
        "- 見出し、段落、箇条書きなど、元の文書構造をMarkdownで再現する（表の扱いは後述の表ルールに必ず従う）",
        "- ページ冒頭にある本文より明らかに大きい番号なし文書タイトルは `# タイトル` として出力する",
        "- **本文より大きい・太字・独立行** の番号付き節見出しは、番号付き箇条書きではなく必ず Markdown 見出しとして出力する。**元の番号は絶対に削除・省略せず、見出しテキストの先頭にそのまま残す**。`#` の個数は **番号のドット区切り段数** で機械的に決める:",
        "    - 1 段（`1` / `2` / `第3章`）→ `## 1. 概要`、`## 第3章 実験`",
        "    - 2 段（`1.1` / `2.3`）→ `### 1.1 方式`",
        "    - 3 段（`1.1.1`）→ `#### 1.1.1 詳細`",
        "    - 4 段以上はさらに `#` を 1 つずつ追加",
        "    - 悪い例: `## 概要`（番号 `1.` が消えている）。正: `## 1. 概要`",
        "- 通常の箇条書き（本文と同サイズで `1.`/`2.`/`・` が並ぶもの）は `1.` や `-` のリストとして出力する。見出しと混同しないこと",
        "- 数式はLaTeX記法で記述する。インライン数式は必ず $...$、ブロック数式は必ず $$...$$ を使う",
        "- ブロック数式では \\begin{equation}...\\end{equation}、\\[...\\]、その他の表示数式環境を使わない",
        "- ヘッダー・フッター（ページ番号、章タイトル等）は省略する",
        "- コードフェンスで囲まないこと。Markdownの内容だけを直接出力する",
    ],
    "en": [
        "- Transcribe only the visible text. Do not add explanations or commentary",
        "- Reproduce the document structure (headings, paragraphs, lists) in Markdown. Tables MUST follow the dedicated table rule stated below",
        "- The large un-numbered document title at the top of the page must be emitted as `# Title`",
        "- Numbered section titles that appear as **larger/bold standalone lines** must be emitted as Markdown headings. **Never strip or omit the original number** — keep it verbatim at the start of the heading text. The `#` count is determined mechanically by **the number of dot-separated segments**:",
        "    - 1 segment (`1`, `2`, `Chapter 3`) → `## 1. Introduction`, `## Chapter 3 Results`",
        "    - 2 segments (`1.1`, `2.3`) → `### 1.1 Method`",
        "    - 3 segments (`1.1.1`) → `#### 1.1.1 Details`",
        "    - 4+ segments add one more `#` per level",
        "    - Bad: `## Introduction` (missing number). Good: `## 1. Introduction`",
        "- Ordinary numbered lists (body-size text with `1.`/`2.` running inline) stay as `1.` / `-` list items — do not confuse them with headings",
        "- Write math in LaTeX notation. Inline math must use $...$ and block math must use $$...$$",
        "- Never use \\begin{equation}...\\end{equation}, \\[...\\], or other display-math environments for block equations",
        "- Omit running headers/footers (page numbers, chapter titles, etc.)",
        "- Do not wrap output in code fences. Output raw Markdown directly",
    ],
}

SYSTEM_HEADER = {
    "ja": (
        "あなたはOCR専門のアシスタントです。"
        "与えられたPDFページの画像を読み取り、内容をMarkdownとして正確に書き起こしてください。\nルール:"
    ),
    "en": (
        "You are an OCR assistant. "
        "Read the given PDF page image and transcribe its content accurately as Markdown.\nRules:"
    ),
}

FIGURE_RULE = {
    "ja": (
        "- 図・写真・ダイアグラム・イラスト本体を見つけた場合、その位置で次の2行を順に出力する:\n"
        "  <!-- asset: kind=figure id=N -->\n"
        "  ![簡潔な説明](assets/PLACEHOLDER)\n"
        "  Nはページ内の図ごとに1から連番。説明は1行で簡潔に。座標やbboxは書かない。"
        "  マーカーは図が見えるその場の読順位置に置き、ページ末尾へ回さない。"
        "  図番号・図注・注記・脚注・凡例説明など、図の外側に印字されたテキストは通常のMarkdown本文として別に書き起こし、このマーカーに含めない"
    ),
    "en": (
        "- When you encounter the body of a figure/photo/diagram/illustration, emit these two lines in place:\n"
        "  <!-- asset: kind=figure id=N -->\n"
        "  ![brief caption](assets/PLACEHOLDER)\n"
        "  N is a per-page counter starting at 1. Keep the caption short. Do not include any coordinates or bbox. "
        "Keep the marker exactly at the figure's reading-order position; do not move it to the end of the page. "
        "Keep printed figure numbers/captions/notes/footnotes as normal Markdown text outside the marker"
    ),
}

FIGURE_RULE_PLAIN = {
    "ja": "- 図や画像がある場合は ![簡潔な説明]() で示す",
    "en": "- For figures/images, use ![brief description]()",
}

TABLE_RULE = {
    "md": {
        "ja": (
            "- 【表ルール / tables=md】表はすべて Markdown テーブル（`| ... |` 記法）で書き起こす。画像化はしない"
        ),
        "en": (
            "- [Table rule / tables=md] Transcribe every table as a Markdown table (`| ... |` syntax). Do not emit image markers"
        ),
    },
    "complex": {
        "ja": (
            "- 【表ルール / tables=complex】以下の判定を表ごとに行い、**必ず** どちらか一方だけを出力する:\n"
            "  - 単純な表（概ね 2〜4 列・結合セルなし・明瞭な罫線・本文と同サイズのテキスト）は Markdown テーブルで書き起こす\n"
            "  - 結合セルがある／セル数が多い／スキャン画像的にかすれている／複雑な罫線や入れ子構造がある表は、Markdown テーブル化を禁止し、次の2行だけを出力する:\n"
            "    <!-- asset: kind=table id=N -->\n"
            "    ![簡潔な説明](assets/PLACEHOLDER)\n"
            "  Nはページ内の表ごとに1から連番。座標やbboxは書かない。迷ったら画像側を選ぶ。"
            "  マーカーは表が見えるその場の読順位置に置き、ページ末尾へ回さない。"
            "  表番号・表題・注記・脚注は通常のMarkdown本文として別に書き起こし、このマーカーに含めない"
        ),
        "en": (
            "- [Table rule / tables=complex] For every table choose exactly ONE of the following:\n"
            "  - Simple tables (roughly 2–4 columns, no merged cells, clean ruling, body-size text) → transcribe as a Markdown table\n"
            "  - Tables with merged cells, many cells, scan-like rendering, complex ruling, or nested headers → DO NOT write a Markdown table; emit only these two lines:\n"
            "    <!-- asset: kind=table id=N -->\n"
            "    ![brief caption](assets/PLACEHOLDER)\n"
            "  N is a per-page counter starting at 1. No coordinates or bbox. If uncertain, prefer the image marker. "
            "Keep the marker exactly at the table's reading-order position; do not move it to the end of the page. "
            "Keep printed table numbers/titles/notes/footnotes as normal Markdown text outside the marker"
        ),
    },
    "all": {
        "ja": (
            "- 【表ルール / tables=all】**どんな表でも** Markdown テーブル（`| ... |` 記法）を書いてはならない。"
            "表が単純か複雑かに関わらず、表の位置に次の2行だけを出力する:\n"
            "  <!-- asset: kind=table id=N -->\n"
            "  ![簡潔な説明](assets/PLACEHOLDER)\n"
            "  Nはページ内の表ごとに1から連番。`|` を使った行や罫線の再現、表の中身のテキスト書き起こしは一切しない。座標やbboxも書かない。"
            "  マーカーは表が見えるその場の読順位置に置き、ページ末尾へ回さない。"
            "  表番号・表題・注記・脚注は通常のMarkdown本文として別に書き起こす"
        ),
        "en": (
            "- [Table rule / tables=all] You MUST NOT emit any Markdown table (no `| ... |` rows) for ANY table, "
            "regardless of whether it looks simple or complex. In place of each table emit ONLY these two lines:\n"
            "  <!-- asset: kind=table id=N -->\n"
            "  ![brief caption](assets/PLACEHOLDER)\n"
            "  N is a per-page counter starting at 1. Never transcribe the cells, never draw pipe characters, never include coordinates or bbox. "
            "Keep the marker exactly at the table's reading-order position; do not move it to the end of the page. "
            "Keep printed table numbers/titles/notes/footnotes as normal Markdown text outside the marker"
        ),
    },
}


TABLE_ALSO_IMAGE_RULE = {
    "ja": (
        "- 【追加ルール / 表を画像としても保存】Markdown テーブルとして書き起こした表については、"
        "そのテーブルの**直後の空行の次**に次の2行も加えること:\n"
        "    <!-- asset: kind=table id=N -->\n"
        "    ![簡潔な説明](assets/PLACEHOLDER)\n"
        "  N はページ内の表ごとに 1 から連番。"
        "既に画像マーカーだけで出力した表（MD テーブルを書いていない表）には重複して加えない。"
        "座標や bbox は書かない"
    ),
    "en": (
        "- [Extra rule / Also keep tables as images] For every table you transcribed as a Markdown table, "
        "append these two lines **right after** the table (separated by a blank line):\n"
        "    <!-- asset: kind=table id=N -->\n"
        "    ![brief caption](assets/PLACEHOLDER)\n"
        "  N is a per-page counter starting at 1. "
        "Do NOT add this marker for tables that were already emitted as image markers only. "
        "No coordinates or bbox"
    ),
}


def _build_system_prompt(lang, extract_figures, tables_mode, table_also_image=False):
    lang = lang if lang in BASE_RULES else "ja"
    rules = list(BASE_RULES[lang])
    rules.append(FIGURE_RULE[lang] if extract_figures else FIGURE_RULE_PLAIN[lang])
    table_rule = TABLE_RULE.get(tables_mode, TABLE_RULE["md"])[lang]
    if table_rule:
        rules.append(table_rule)
    if table_also_image and tables_mode != "all":
        rules.append(TABLE_ALSO_IMAGE_RULE[lang])
    return SYSTEM_HEADER[lang] + "\n" + "\n".join(rules)

USER_PROMPTS = {
    "ja": "このPDFの{page}/{total}ページ目を書き起こしてください。文書の言語は日本語です。",
    "en": "Transcribe page {page}/{total} of this PDF. The document is in English.",
}

CONTINUATION = {
    "ja": "\n\n前ページ末尾:\n{tail}\n\n途切れた文があれば続きから始めてください。前ページの内容は繰り返さないこと。",
    "en": "\n\nPrevious page ended with:\n{tail}\n\nIf a sentence was cut off, continue from where it left off. Do not repeat previous content.",
}

MAX_TOKENS = int(os.environ.get("PDF2MD_MAX_TOKENS", "8192"))
JPEG_QUALITY = int(os.environ.get("PDF2MD_JPEG_QUALITY", "75"))
DEFAULT_MODEL = os.environ.get("PDF2MD_MODEL", "qwen2.5vl:32b")
DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
DEFAULT_DPI = int(os.environ.get("PDF2MD_DEFAULT_DPI", "150"))
DEFAULT_THINKING = os.environ.get("PDF2MD_DEFAULT_THINKING", "off") == "on"
MAX_IMAGE_EDGE = int(os.environ.get("PDF2MD_MAX_IMAGE_EDGE", "1800"))
OLLAMA_TIMEOUT = int(os.environ.get("PDF2MD_OLLAMA_TIMEOUT", "600"))
OLLAMA_LOAD_TIMEOUT = int(os.environ.get("PDF2MD_OLLAMA_LOAD_TIMEOUT", "300"))
OLLAMA_KEEP_ALIVE = os.environ.get("PDF2MD_OLLAMA_KEEP_ALIVE", "15m")
GRID_COLS = int(os.environ.get("PDF2MD_GRID_COLS", "20"))
GRID_ROWS = int(os.environ.get("PDF2MD_GRID_ROWS", "26"))
GRID_PAD_CELLS = int(os.environ.get("PDF2MD_GRID_PAD_CELLS", "1"))
GRID_FINE_COLS = int(os.environ.get("PDF2MD_GRID_FINE_COLS", "24"))
GRID_FINE_ROWS = int(os.environ.get("PDF2MD_GRID_FINE_ROWS", "24"))
GRID_FINE_PAD_CELLS = int(os.environ.get("PDF2MD_GRID_FINE_PAD_CELLS", "0"))
LOCALIZE_ENABLED = os.environ.get("PDF2MD_LOCALIZE_ENABLED", "on") == "on"
LOCALIZE_FINE_ENABLED = os.environ.get("PDF2MD_LOCALIZE_FINE_ENABLED", "on") == "on"
LOCALIZE_NUM_PREDICT = int(os.environ.get("PDF2MD_LOCALIZE_NUM_PREDICT", "1024"))
LOCALIZE_FINE_MARGIN_PX = int(os.environ.get("PDF2MD_LOCALIZE_FINE_MARGIN_PX", "48"))
DEBUG_DIR = os.environ.get("PDF2MD_DEBUG_DIR", "").strip()
DISPLAY_MATH_RE = re.compile(
    r"\\begin\{equation\*?\}\s*(.*?)\s*\\end\{equation\*?\}", flags=re.DOTALL
)
BRACKET_MATH_RE = re.compile(r"\\\[\s*(.*?)\s*\\\]", flags=re.DOTALL)
INLINE_PAREN_MATH_RE = re.compile(r"\\\(\s*(.*?)\s*\\\)", flags=re.DOTALL)
ASSET_MARKER_RE = re.compile(
    r"<!--\s*asset:\s*kind=(figure|table)(?:\s+id=(\d+))?\s*-->"
    r"(?:[ \t]*\n?[ \t]*!\[([^\]]*)\]\([^)]*\))?",
    flags=re.IGNORECASE,
)
ASSET_CROP_DPI = int(os.environ.get("PDF2MD_ASSET_DPI", "200"))
PAGE_ASSET_DPI = int(os.environ.get("PDF2MD_PAGE_ASSET_DPI", "200"))
PAGE_ASSET_JPEG_QUALITY = int(os.environ.get("PDF2MD_PAGE_ASSET_JPEG_QUALITY", "90"))
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
FIGURE_CAPTION_RE = re.compile(
    r"^(?:図|fig(?:ure)?\.?)\s*[-.:]?\s*(\d+)\b", flags=re.IGNORECASE
)
TABLE_CAPTION_RE = re.compile(
    r"^(?:表|table)\s*[-.:]?\s*(\d+)\b", flags=re.IGNORECASE
)


def _page_to_jpeg(page, dpi, quality=None):
    """Render a PDF page to JPEG bytes plus image metadata."""
    page_width = page.rect.width * dpi / 72
    page_height = page.rect.height * dpi / 72
    max_dim = max(page_width, page_height, 1)
    downscale = min(1.0, MAX_IMAGE_EDGE / max_dim)
    render_scale = dpi / 72 * downscale

    pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), alpha=False)
    jpg_quality = JPEG_QUALITY if quality is None else quality
    img_bytes = pix.tobytes("jpeg", jpg_quality=jpg_quality)
    return img_bytes, {
        "requested_dpi": dpi,
        "effective_dpi": round(72 * render_scale),
        "width": pix.width,
        "height": pix.height,
        "image_bytes": len(img_bytes),
    }


def _wrap_block_math(match):
    body = match.group(1).strip()
    return f"\n$$\n{body}\n$$\n"


def _crop_asset(page, x1, y1, x2, y2, img_w, img_h):
    """bbox is in pixel coords of the rendered image (img_w x img_h)."""
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    nx1 = max(0.0, x1 / img_w - 0.005)
    ny1 = max(0.0, y1 / img_h - 0.005)
    nx2 = min(1.0, x2 / img_w + 0.005)
    ny2 = min(1.0, y2 / img_h + 0.005)
    if nx2 - nx1 < 0.01 or ny2 - ny1 < 0.01:
        return None
    pw = page.rect.width
    ph = page.rect.height
    clip = fitz.Rect(nx1 * pw, ny1 * ph, nx2 * pw, ny2 * ph)
    scale = ASSET_CROP_DPI / 72
    try:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False
        )
        return pix.tobytes("png")
    except Exception:
        return None


def _parse_asset_markers(content):
    """Find all asset markers, assign per-kind sequential ids in reading order."""
    markers = []
    counters = {"figure": 0, "table": 0}
    for m in ASSET_MARKER_RE.finditer(content):
        kind = m.group(1).lower()
        counters[kind] += 1
        assigned_id = counters[kind]
        explicit_id = None
        if m.group(2):
            try:
                explicit_id = int(m.group(2))
            except ValueError:
                explicit_id = None
        markers.append(
            {
                "kind": kind,
                "assigned_id": assigned_id,
                "asset_key": f"{kind}-{assigned_id}",
                "explicit_id": explicit_id,
                "caption": (m.group(3) or "").strip(),
                "span": m.span(),
                "raw": m.group(0),
            }
        )
    return markers


def _cell_label(col_idx):
    """0-based column index to 'A'..'Z','AA','AB',... label."""
    if col_idx < 26:
        return chr(ord("A") + col_idx)
    hi = col_idx // 26 - 1
    lo = col_idx % 26
    return chr(ord("A") + hi) + chr(ord("A") + lo)


def _parse_cell_label(label, cols, rows):
    """Parse 'C4' / 'AB12' to (col_idx, row_idx) 0-based, or None if invalid."""
    if not label:
        return None
    m = re.match(r"^([A-Za-z]{1,2})(\d{1,3})$", label.strip())
    if not m:
        return None
    letters = m.group(1).upper()
    if len(letters) == 1:
        col = ord(letters) - ord("A")
    else:
        col = (ord(letters[0]) - ord("A") + 1) * 26 + (ord(letters[1]) - ord("A"))
    row = int(m.group(2)) - 1
    if col < 0 or col >= cols or row < 0 or row >= rows:
        return None
    return (col, row)


def _grid_cells_to_pixels(tl, br, img_w, img_h, cols, rows, pad_cells=1):
    tl_c = _parse_cell_label(tl, cols, rows)
    br_c = _parse_cell_label(br, cols, rows)
    if tl_c is None or br_c is None:
        return None
    c1, r1 = tl_c
    c2, r2 = br_c
    c1, c2 = sorted((c1, c2))
    r1, r2 = sorted((r1, r2))
    c1 = max(0, c1 - pad_cells)
    r1 = max(0, r1 - pad_cells)
    c2 = min(cols - 1, c2 + pad_cells)
    r2 = min(rows - 1, r2 + pad_cells)
    x1 = c1 * img_w / cols
    y1 = r1 * img_h / rows
    x2 = (c2 + 1) * img_w / cols
    y2 = (r2 + 1) * img_h / rows
    return (x1, y1, x2, y2)


def _normalize_caption_text(text):
    return re.sub(r"\s+", " ", (text or "").translate(FULLWIDTH_DIGITS)).strip()


def _rect_to_tuple(rect):
    if rect is None:
        return None
    if isinstance(rect, (list, tuple)):
        if len(rect) != 4:
            return None
        return tuple(float(v) for v in rect)
    try:
        return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    except Exception:
        return None


def _sort_rect(rect):
    x1, y1, x2, y2 = rect
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (x1, y1, x2, y2)


def _rect_width(rect):
    return max(0.0, rect[2] - rect[0])


def _rect_height(rect):
    return max(0.0, rect[3] - rect[1])


def _rect_area(rect):
    return _rect_width(rect) * _rect_height(rect)


def _rect_union(a, b):
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _rects_overlap_or_close(a, b, pad=0.0):
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _rect_reading_order(rect):
    return (round(rect[1], 1), round(rect[0], 1))


def _dedupe_rects(rects, tol=2.0):
    out = []
    ordered = sorted(
        (_sort_rect(r) for r in rects if r is not None), key=_rect_reading_order
    )
    for rect in ordered:
        if any(
            max(abs(rect[i] - existing[i]) for i in range(4)) <= tol for existing in out
        ):
            continue
        out.append(rect)
    return out


def _filter_graphic_rects(rects, page_rect):
    page_box = _sort_rect(_rect_to_tuple(page_rect))
    page_area = max(_rect_area(page_box), 1.0)
    min_area = max(page_area * 0.0015, 250.0)
    out = []
    for rect in _dedupe_rects(rects):
        w = _rect_width(rect)
        h = _rect_height(rect)
        area = _rect_area(rect)
        if w < 12 or h < 12:
            continue
        if area < min_area or area > page_area * 0.90:
            continue
        aspect = w / max(h, 1.0)
        if aspect > 25.0 or aspect < 0.04:
            continue
        out.append(rect)
    return out


def _merge_candidates(candidates, pad=6.0):
    merged = []
    ordered = sorted(candidates, key=lambda item: _rect_reading_order(item["bbox"]))
    for cand in ordered:
        bbox = _sort_rect(cand["bbox"])
        sources = {cand["source"]}
        changed = True
        while changed:
            changed = False
            kept = []
            for existing in merged:
                if _rects_overlap_or_close(existing["bbox"], bbox, pad):
                    bbox = _rect_union(existing["bbox"], bbox)
                    sources.update(existing["sources"])
                    changed = True
                else:
                    kept.append(existing)
            merged = kept
        merged.append({"bbox": bbox, "sources": sources})
    merged.sort(key=lambda item: _rect_reading_order(item["bbox"]))
    return merged


def _extract_text_lines(page):
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return []
    lines = []
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            parts = [span.get("text", "") for span in line.get("spans", [])]
            text = _normalize_caption_text("".join(parts))
            bbox = _rect_to_tuple(line.get("bbox"))
            if text and bbox is not None:
                lines.append({"text": text, "bbox": _sort_rect(bbox)})
    lines.sort(key=lambda item: _rect_reading_order(item["bbox"]))
    return lines


def _extract_caption_anchors(page):
    anchors = {"figure": [], "table": []}
    for line in _extract_text_lines(page):
        for kind, pattern in (("figure", FIGURE_CAPTION_RE), ("table", TABLE_CAPTION_RE)):
            m = pattern.match(line["text"])
            if not m:
                continue
            caption_id = None
            try:
                caption_id = int(m.group(1))
            except ValueError:
                caption_id = None
            anchors[kind].append(
                {
                    "caption_id": caption_id,
                    "bbox": line["bbox"],
                    "text": line["text"],
                }
            )
            break
    for kind in anchors:
        anchors[kind].sort(key=lambda item: _rect_reading_order(item["bbox"]))
    return anchors


def _extract_image_candidates(page):
    rects = []
    try:
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") == 1:
                bbox = _rect_to_tuple(block.get("bbox"))
                if bbox is not None:
                    rects.append(bbox)
    except Exception:
        pass
    try:
        for info in page.get_images(full=True):
            for rect in page.get_image_rects(info[0]):
                bbox = _rect_to_tuple(rect)
                if bbox is not None:
                    rects.append(bbox)
    except Exception:
        pass
    return [
        {"bbox": rect, "source": "native-image"}
        for rect in _filter_graphic_rects(rects, page.rect)
    ]


def _extract_drawing_candidates(page):
    rects = []
    if hasattr(page, "cluster_drawings"):
        try:
            for rect in page.cluster_drawings():
                bbox = _rect_to_tuple(rect)
                if bbox is not None:
                    rects.append(bbox)
        except Exception:
            rects = []
    if not rects and hasattr(page, "get_drawings"):
        try:
            for item in page.get_drawings():
                bbox = _rect_to_tuple(item.get("rect"))
                if bbox is not None:
                    rects.append(bbox)
        except Exception:
            rects = []
    return [
        {"bbox": rect, "source": "native-drawing"}
        for rect in _filter_graphic_rects(rects, page.rect)
    ]


def _extract_table_candidates(page):
    if not hasattr(page, "find_tables"):
        return []
    try:
        table_finder = page.find_tables()
    except Exception:
        return []
    tables = getattr(table_finder, "tables", None) or []
    rects = []
    for table in tables:
        bbox = _rect_to_tuple(getattr(table, "bbox", None))
        if bbox is not None:
            rects.append(bbox)
    return [
        {"bbox": rect, "source": "native-table"}
        for rect in _filter_graphic_rects(rects, page.rect)
    ]


def _pdf_rect_to_image_pixels(page, rect, img_w, img_h):
    pw = max(page.rect.width, 1.0)
    ph = max(page.rect.height, 1.0)
    sx = img_w / pw
    sy = img_h / ph
    return (
        max(0.0, rect[0] * sx),
        max(0.0, rect[1] * sy),
        min(float(img_w), rect[2] * sx),
        min(float(img_h), rect[3] * sy),
    )


def _score_candidate_for_anchor(kind, candidate_bbox, anchor_bbox, page_height):
    page_height = max(page_height, 1.0)
    cand_cx = (candidate_bbox[0] + candidate_bbox[2]) / 2.0
    anchor_cx = (anchor_bbox[0] + anchor_bbox[2]) / 2.0
    x_penalty = abs(cand_cx - anchor_cx) * 0.15
    if kind == "figure":
        if candidate_bbox[3] <= anchor_bbox[1] + 6.0:
            y_penalty = max(0.0, anchor_bbox[1] - candidate_bbox[3])
            relation_penalty = 0.0
        elif candidate_bbox[1] >= anchor_bbox[3] - 6.0:
            y_penalty = max(0.0, candidate_bbox[1] - anchor_bbox[3])
            relation_penalty = page_height * 0.25
        else:
            y_penalty = 0.0
            relation_penalty = page_height * 0.10
    else:
        if candidate_bbox[1] >= anchor_bbox[3] - 6.0:
            y_penalty = max(0.0, candidate_bbox[1] - anchor_bbox[3])
            relation_penalty = 0.0
        elif candidate_bbox[3] <= anchor_bbox[1] + 6.0:
            y_penalty = max(0.0, anchor_bbox[1] - candidate_bbox[3])
            relation_penalty = page_height * 0.06
        else:
            y_penalty = 0.0
            relation_penalty = page_height * 0.10
    return relation_penalty + y_penalty + x_penalty


def _assign_native_candidates(kind_markers, candidates, anchors, kind, page_height):
    assigned = {}
    used_candidate_idxs = set()
    used_anchor_idxs = set()
    by_id = {}
    for idx, anchor in enumerate(anchors):
        by_id.setdefault(anchor["caption_id"], []).append((idx, anchor))

    for marker in kind_markers:
        anchor = None
        anchor_idx = None
        explicit_id = marker.get("explicit_id")
        if explicit_id is not None:
            for idx, candidate_anchor in by_id.get(explicit_id, []):
                if idx not in used_anchor_idxs:
                    anchor = candidate_anchor
                    anchor_idx = idx
                    break
        if anchor is None:
            for idx, candidate_anchor in enumerate(anchors):
                if idx not in used_anchor_idxs:
                    anchor = candidate_anchor
                    anchor_idx = idx
                    break
        if anchor is None:
            continue

        best_idx = None
        best_score = None
        for idx, candidate in enumerate(candidates):
            if idx in used_candidate_idxs:
                continue
            score = _score_candidate_for_anchor(
                kind, candidate["bbox"], anchor["bbox"], page_height
            )
            if best_score is None or score < best_score:
                best_idx = idx
                best_score = score
        if best_idx is None or best_score is None or best_score > page_height * 0.45:
            continue
        used_candidate_idxs.add(best_idx)
        used_anchor_idxs.add(anchor_idx)
        assigned[marker["asset_key"]] = candidates[best_idx]

    remaining_markers = [m for m in kind_markers if m["asset_key"] not in assigned]
    remaining_candidates = [
        candidate
        for idx, candidate in enumerate(candidates)
        if idx not in used_candidate_idxs
    ]
    if remaining_markers and len(remaining_markers) == len(remaining_candidates):
        remaining_markers.sort(key=lambda item: item["assigned_id"])
        remaining_candidates.sort(key=lambda item: _rect_reading_order(item["bbox"]))
        for marker, candidate in zip(remaining_markers, remaining_candidates):
            assigned[marker["asset_key"]] = candidate
    return assigned


def _detect_pdf_native_assets(page, markers, img_w, img_h):
    """Best-effort native PDF detection for image/table regions."""
    if not markers:
        return {}, {}, {}

    anchors = _extract_caption_anchors(page)
    figure_candidates = _merge_candidates(
        _extract_image_candidates(page) + _extract_drawing_candidates(page), pad=8.0
    )
    figure_candidates = [
        {
            "bbox": candidate["bbox"],
            "source": (
                "native-mixed"
                if len(candidate["sources"]) > 1
                else next(iter(candidate["sources"]))
            ),
        }
        for candidate in figure_candidates
    ]
    table_candidates = _extract_table_candidates(page)

    resolved = {}
    sources = {}
    meta = {
        "figure_candidates": len(figure_candidates),
        "table_candidates": len(table_candidates),
        "figure_anchors": len(anchors["figure"]),
        "table_anchors": len(anchors["table"]),
    }

    for kind, candidates in (("figure", figure_candidates), ("table", table_candidates)):
        kind_markers = [m for m in markers if m["kind"] == kind]
        if not kind_markers or not candidates:
            continue
        assigned = _assign_native_candidates(
            kind_markers, candidates, anchors.get(kind, []), kind, page.rect.height
        )
        for asset_key, candidate in assigned.items():
            resolved[asset_key] = _pdf_rect_to_image_pixels(
                page, candidate["bbox"], img_w, img_h
            )
            sources[asset_key] = candidate["source"]
    return resolved, sources, meta


_GRID_FONT = None


def _load_grid_font(size):
    global _GRID_FONT
    if _GRID_FONT is not None and _GRID_FONT.size == size:
        return _GRID_FONT
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            _GRID_FONT = ImageFont.truetype(path, size)
            return _GRID_FONT
        except Exception:
            continue
    _GRID_FONT = ImageFont.load_default()
    return _GRID_FONT


def _draw_label(draw, text, x, y, font, halo=(255, 255, 255), fill=(180, 20, 20)):
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, fill=halo, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def _render_image_with_grid(jpeg_bytes, cols, rows):
    """Overlay a labeled grid on an image. Returns (new_jpeg_bytes, (img_w, img_h))."""
    im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img_w, img_h = im.size
    m_top = max(36, int(img_h * 0.03))
    m_left = max(40, int(img_w * 0.03))
    canvas_w = img_w + m_left
    canvas_h = img_h + m_top
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(im, (m_left, m_top))
    draw = ImageDraw.Draw(canvas, "RGBA")
    cw = img_w / cols
    ch = img_h / rows
    font_size = max(16, int(min(cw, ch) * 0.42))
    font = _load_grid_font(font_size)
    line_color = (200, 30, 30, 90)
    for i in range(cols + 1):
        x = int(m_left + i * cw)
        draw.line([(x, m_top), (x, m_top + img_h)], fill=line_color, width=1)
    for j in range(rows + 1):
        y = int(m_top + j * ch)
        draw.line([(m_left, y), (m_left + img_w, y)], fill=line_color, width=1)
    for i in range(cols):
        label = _cell_label(i)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = m_left + int(i * cw + cw / 2 - tw / 2)
        y = max(1, (m_top - th) // 2 - bbox[1])
        _draw_label(draw, label, x, y, font)
    for j in range(rows):
        label = str(j + 1)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = max(1, (m_left - tw) // 2 - bbox[0])
        y = m_top + int(j * ch + ch / 2 - th / 2 - bbox[1])
        _draw_label(draw, label, x, y, font)
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue(), (img_w, img_h)


def _parse_localize_key(obj):
    key = obj.get("key")
    if key:
        return str(key).strip().lower()
    try:
        kind = str(obj["kind"]).strip().lower()
        asset_id = int(obj["id"])
    except (KeyError, ValueError, TypeError):
        return None
    return f"{kind}-{asset_id}"


def _localize_assets_via_grid(
    page_num,
    grid_jpeg_b64,
    markers,
    img_w,
    img_h,
    model,
    thinking,
    cols,
    rows,
    pad_cells,
    scope_label,
    provider=None,
):
    """Ask the vision model to localize asset bodies on a gridded image."""
    last_col = _cell_label(cols - 1)
    sys_prompt = (
        "You are given an image overlaid with a coordinate grid.\n"
        "The image may be a full PDF page or a cropped region.\n"
        f"Columns are labeled A through {last_col} (left to right), written in the top white margin.\n"
        f"Rows are labeled 1 through {rows} (top to bottom), written in the left white margin.\n"
        "For each asset listed by the user, output ONE JSON line giving the smallest "
        "grid-cell rectangle that FULLY contains only the visual body of the asset.\n"
        'Format: {"key": "figure-1", "tl": "<Col><Row>", "br": "<Col><Row>"}\n'
        "Rules:\n"
        "- tl = top-left cell, br = bottom-right cell, INCLUSIVE.\n"
        "- Exclude printed figure/table captions, titles, notes, footnotes, and surrounding paragraph text.\n"
        "- Include legends or labels that are visually inside the figure/table itself.\n"
        "- When uncertain, prefer including one extra cell over clipping.\n"
        "- Output ONLY JSON lines, one per asset. No prose, no code fences."
    )
    listing = []
    for m in markers:
        cap = m["caption"] or "(no caption)"
        listing.append(f"  {m['asset_key']} ({m['kind']}) — visual hint: \"{cap}\"")
    user_text = (
        f"Page {page_num}, scope={scope_label}. Locate the following {len(markers)} asset body/bodies:\n"
        + "\n".join(listing)
    )
    try:
        result = provider.chat(
            model,
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_text, "images": [grid_jpeg_b64]},
            ],
            stream=False,
            thinking=thinking,
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={"num_predict": LOCALIZE_NUM_PREDICT, "temperature": 0.0},
            timeout=OLLAMA_TIMEOUT,
        )
        body = result["content"]
    except Exception as e:
        print(
            f"[pdf2md] page {page_num}: localize-pass ({scope_label}) failed: {e}",
            file=sys.stderr,
            flush=True,
        )
        return {}, ""
    result = {}
    for obj in _iter_json_objects(body):
        asset_key = _parse_localize_key(obj)
        if asset_key is None:
            continue
        try:
            tl = str(obj["tl"])
            br = str(obj["br"])
        except (KeyError, TypeError):
            continue
        px = _grid_cells_to_pixels(
            tl, br, img_w, img_h, cols, rows, pad_cells=pad_cells
        )
        if px is None:
            continue
        result[asset_key] = px
    return result, body


def _iter_json_objects(text):
    """Yield parsed JSON objects from mixed text (one-per-line, or braces embedded in prose)."""
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                yield obj
                continue
        except Exception:
            pass
        for m in re.finditer(r"\{[^{}]*\}", s):
            try:
                obj = json.loads(m.group(0))
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _crop_image_region(jpeg_bytes, bbox, margin_px):
    im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img_w, img_h = im.size
    x1 = max(0, int(bbox[0] - margin_px))
    y1 = max(0, int(bbox[1] - margin_px))
    x2 = min(img_w, int(bbox[2] + margin_px))
    y2 = min(img_h, int(bbox[3] + margin_px))
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = im.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue(), (x1, y1, x2, y2)


def _safe_debug_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _write_debug_bytes(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    except Exception:
        pass


def _write_debug_text(path, text):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def _write_debug_json(path, data):
    _write_debug_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def _render_bbox_overlay(jpeg_bytes, markers, bboxes, sources=None):
    im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    font = _load_grid_font(max(16, int(min(im.size) * 0.018)))
    for marker in markers:
        bbox = bboxes.get(marker["asset_key"])
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle(
            [(int(x1), int(y1)), (int(x2), int(y2))],
            outline=(40, 200, 60, 255),
            width=4,
        )
        label = marker["asset_key"]
        if sources and marker["asset_key"] in sources:
            label += f" [{sources[marker['asset_key']]}]"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        lx = int(x1)
        ly = max(0, int(y1) - text_h - 10)
        draw.rectangle(
            [(lx, ly), (lx + text_w + 10, ly + text_h + 6)],
            fill=(0, 0, 0, 170),
        )
        draw.text((lx + 5, ly + 3), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _refine_grid_boxes(page_num, page_jpeg, markers, coarse_boxes, model, thinking, debug_ctx, provider=None):
    refined = {}
    raw_fine = {}
    fine_grid_ms = 0
    fine_localize_ms = 0
    if not LOCALIZE_FINE_ENABLED:
        return refined, raw_fine, fine_grid_ms, fine_localize_ms

    for marker in markers:
        asset_key = marker["asset_key"]
        coarse_bbox = coarse_boxes.get(asset_key)
        if coarse_bbox is None:
            continue
        margin_px = max(
            LOCALIZE_FINE_MARGIN_PX,
            int(max(_rect_width(coarse_bbox), _rect_height(coarse_bbox)) * 0.12),
        )
        crop_bytes, crop_region = _crop_image_region(page_jpeg, coarse_bbox, margin_px)
        if crop_bytes is None or crop_region is None:
            continue
        grid_started = time.perf_counter()
        fine_grid_bytes, (crop_w, crop_h) = _render_image_with_grid(
            crop_bytes, GRID_FINE_COLS, GRID_FINE_ROWS
        )
        fine_grid_ms += round((time.perf_counter() - grid_started) * 1000)
        fine_started = time.perf_counter()
        fine_boxes, raw = _localize_assets_via_grid(
            page_num,
            base64.b64encode(fine_grid_bytes).decode(),
            [marker],
            crop_w,
            crop_h,
            model,
            thinking,
            GRID_FINE_COLS,
            GRID_FINE_ROWS,
            GRID_FINE_PAD_CELLS,
            f"fine:{asset_key}",
            provider=provider,
        )
        fine_localize_ms += round((time.perf_counter() - fine_started) * 1000)
        raw_fine[asset_key] = raw
        fine_bbox = fine_boxes.get(asset_key)
        if fine_bbox is None:
            continue
        refined[asset_key] = (
            fine_bbox[0] + crop_region[0],
            fine_bbox[1] + crop_region[1],
            fine_bbox[2] + crop_region[0],
            fine_bbox[3] + crop_region[1],
        )
        if debug_ctx:
            base_name = _safe_debug_name(asset_key)
            _write_debug_bytes(
                os.path.join(debug_ctx["page_dir"], f"{base_name}.fine-crop.jpg"),
                crop_bytes,
            )
            _write_debug_bytes(
                os.path.join(debug_ctx["page_dir"], f"{base_name}.fine-grid.jpg"),
                fine_grid_bytes,
            )
            _write_debug_text(
                os.path.join(debug_ctx["page_dir"], f"{base_name}.fine-raw.txt"),
                raw,
            )
    return refined, raw_fine, fine_grid_ms, fine_localize_ms


def _summarize_sources(source_by_key):
    unique = sorted(set(source_by_key.values()))
    return "+".join(unique) if unique else None


def _extract_assets(
    content, page, page_num, page_jpeg, img_w, img_h, model, thinking, debug_ctx=None, provider=None
):
    """Locate and crop figure/table markers. Returns (new_content, assets, info)."""
    markers = _parse_asset_markers(content)
    info = {
        "markers": len(markers),
        "fast_path": False,
        "extracted": 0,
        "source": None,
        "sources": {},
        "bboxes": {},
        "raw_pass2": {"coarse": "", "fine": {}},
        "timings": {
            "native_ms": 0,
            "grid_render_ms": 0,
            "coarse_localize_ms": 0,
            "fine_grid_ms": 0,
            "fine_localize_ms": 0,
        },
        "native_hits": 0,
        "debug_dir": debug_ctx.get("page_dir") if debug_ctx else None,
    }
    if not markers:
        return content, [], info

    if debug_ctx:
        _write_debug_bytes(os.path.join(debug_ctx["page_dir"], "page.jpg"), page_jpeg)
        _write_debug_text(
            os.path.join(debug_ctx["page_dir"], "content.markers.md"), content
        )

    bboxes = {}
    source_by_key = {}
    native_meta = {}
    if LOCALIZE_ENABLED:
        native_started = time.perf_counter()
        native_boxes, native_sources, native_meta = _detect_pdf_native_assets(
            page, markers, img_w, img_h
        )
        info["timings"]["native_ms"] = round(
            (time.perf_counter() - native_started) * 1000
        )
        bboxes.update(native_boxes)
        source_by_key.update(native_sources)
        info["native_hits"] = len(native_boxes)

    unresolved_markers = [m for m in markers if m["asset_key"] not in bboxes]
    if LOCALIZE_ENABLED and unresolved_markers:
        grid_started = time.perf_counter()
        coarse_grid_bytes, _ = _render_image_with_grid(page_jpeg, GRID_COLS, GRID_ROWS)
        info["timings"]["grid_render_ms"] = round(
            (time.perf_counter() - grid_started) * 1000
        )
        if debug_ctx:
            _write_debug_bytes(
                os.path.join(debug_ctx["page_dir"], "page.coarse-grid.jpg"),
                coarse_grid_bytes,
            )
        coarse_started = time.perf_counter()
        coarse_boxes, raw = _localize_assets_via_grid(
            page_num,
            base64.b64encode(coarse_grid_bytes).decode(),
            unresolved_markers,
            img_w,
            img_h,
            model,
            thinking,
            GRID_COLS,
            GRID_ROWS,
            GRID_PAD_CELLS,
            "page",
            provider=provider,
        )
        info["timings"]["coarse_localize_ms"] = round(
            (time.perf_counter() - coarse_started) * 1000
        )
        info["raw_pass2"]["coarse"] = raw
        for asset_key, bbox in coarse_boxes.items():
            bboxes[asset_key] = bbox
            source_by_key[asset_key] = "grid"
        refined_boxes, fine_raw, fine_grid_ms, fine_localize_ms = _refine_grid_boxes(
            page_num,
            page_jpeg,
            unresolved_markers,
            coarse_boxes,
            model,
            thinking,
            debug_ctx,
            provider=provider,
        )
        info["timings"]["fine_grid_ms"] = fine_grid_ms
        info["timings"]["fine_localize_ms"] = fine_localize_ms
        info["raw_pass2"]["fine"] = fine_raw
        for asset_key, bbox in refined_boxes.items():
            bboxes[asset_key] = bbox
            source_by_key[asset_key] = "grid-fine"
        if debug_ctx and raw:
            _write_debug_text(
                os.path.join(debug_ctx["page_dir"], "page.coarse-raw.txt"), raw
            )

    if debug_ctx and native_meta:
        _write_debug_json(
            os.path.join(debug_ctx["page_dir"], "native-meta.json"), native_meta
        )

    info["fast_path"] = bool(bboxes) and not any(
        source.startswith("grid") for source in source_by_key.values()
    )
    assets = []
    filenames = {}
    if bboxes:
        counters = {"figure": 0, "table": 0}
        for m in markers:
            bb = bboxes.get(m["asset_key"])
            if bb is None:
                continue
            png = _crop_asset(page, bb[0], bb[1], bb[2], bb[3], img_w, img_h)
            if png is None:
                continue
            counters[m["kind"]] += 1
            idx = counters[m["kind"]]
            filename = f"p{page_num}-{m['kind']}{idx}.png"
            assets.append({"filename": filename, "png_bytes": png, "kind": m["kind"]})
            filenames[m["asset_key"]] = filename
            info["bboxes"][m["asset_key"]] = [round(v, 1) for v in bb]
            info["sources"][m["asset_key"]] = source_by_key.get(
                m["asset_key"], "unknown"
            )
            if debug_ctx:
                _write_debug_bytes(
                    os.path.join(debug_ctx["page_dir"], "assets", filename), png
                )
    info["extracted"] = len(assets)
    info["source"] = _summarize_sources(info["sources"])

    def sub_marker(match):
        m_id_counter = sub_marker.counter
        kind = match.group(1).lower()
        m_id_counter[kind] += 1
        assigned_id = m_id_counter[kind]
        asset_key = f"{kind}-{assigned_id}"
        caption = (match.group(3) or "").strip()
        fn = filenames.get(asset_key)
        if fn:
            return f"![{caption}](assets/{fn})"
        if caption:
            return f"![{caption}](assets/MISSING-p{page_num}-{kind}{assigned_id})"
        return ""
    sub_marker.counter = {"figure": 0, "table": 0}
    new_content = ASSET_MARKER_RE.sub(sub_marker, content)

    if debug_ctx:
        _write_debug_text(
            os.path.join(debug_ctx["page_dir"], "content.final.md"), new_content
        )
        _write_debug_json(
            os.path.join(debug_ctx["page_dir"], "boxes.json"),
            {"bboxes": info["bboxes"], "sources": info["sources"]},
        )
        _write_debug_bytes(
            os.path.join(debug_ctx["page_dir"], "page.boxes.png"),
            _render_bbox_overlay(page_jpeg, markers, bboxes, source_by_key),
        )

    print(
        f"[pdf2md] page {page_num}: markers={info['markers']} "
        f"extracted={info['extracted']} source={info['source']} "
        f"fast_path={info['fast_path']} bboxes={info['bboxes']}",
        file=sys.stderr,
        flush=True,
    )
    return new_content, assets, info


def _normalize_markdown(content):
    # Strip wrapping code fences if the model adds them.
    content = re.sub(
        r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$",
        r"\1",
        content,
        flags=re.DOTALL,
    )
    content = DISPLAY_MATH_RE.sub(_wrap_block_math, content)
    content = BRACKET_MATH_RE.sub(_wrap_block_math, content)
    content = INLINE_PAREN_MATH_RE.sub(lambda m: f"${m.group(1).strip()}$", content)
    return re.sub(r"\n{3,}", "\n\n", content).strip()


@app.route("/api/pdf2md/models", methods=["GET"])
def list_models():
    """Return vision-capable models from all configured providers."""
    try:
        models = get_all_models()
        if not models:
            models = [{"name": DEFAULT_MODEL, "provider": DEFAULT_PROVIDER}]
        return jsonify({"models": models, "default": DEFAULT_MODEL})
    except Exception:
        return jsonify(
            {"models": [{"name": DEFAULT_MODEL, "provider": DEFAULT_PROVIDER}], "default": DEFAULT_MODEL}
        )


@app.route("/api/pdf2md/convert", methods=["POST"])
def convert():
    pdf_file = request.files.get("pdf")
    if not pdf_file:
        return jsonify({"error": "No PDF file provided"}), 400

    lang = request.form.get("lang", "ja")
    model = request.form.get("model", DEFAULT_MODEL)
    dpi = int(request.form.get("dpi", str(DEFAULT_DPI)))
    thinking = request.form.get("thinking", "on" if DEFAULT_THINKING else "off") == "on"
    pages_str = request.form.get("pages", "").strip()
    extract_figures = request.form.get("extract_figures", "off") == "on"
    tables_mode = request.form.get("tables", "md")
    if tables_mode not in TABLE_RULE:
        tables_mode = "md"
    table_also_image = request.form.get("table_also_image", "off") == "on"
    provider_name = request.form.get("provider", DEFAULT_PROVIDER)
    try:
        provider = get_provider(provider_name)
    except ValueError:
        provider = get_provider(DEFAULT_PROVIDER)

    pdf_bytes = pdf_file.read()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return jsonify({"error": f"Failed to open PDF: {e}"}), 400

    total_pages = len(doc)
    page_indices = _parse_pages(pages_str, total_pages)

    system_prompt = _build_system_prompt(
        lang, extract_figures, tables_mode, table_also_image
    )
    user_template = USER_PROMPTS.get(lang, USER_PROMPTS["ja"])
    extract_enabled = extract_figures or tables_mode != "md" or table_also_image
    print(
        f"[pdf2md] convert: lang={lang} model={model} "
        f"extract_figures={extract_figures} tables={tables_mode} "
        f"table_also_image={table_also_image} "
        f"extract_enabled={extract_enabled}",
        file=sys.stderr,
        flush=True,
    )

    def get_tail(text, n_lines=3):
        lines = [l for l in text.strip().splitlines() if l.strip()]
        return "\n".join(lines[-n_lines:])

    def generate():
        convert_pages = len(page_indices)
        yield f"data: {json.dumps({'type': 'start', 'total_pages': convert_pages, 'format': OUTPUT_FORMAT})}\n\n"

        model_load_checked = False
        prev_tail = ""
        debug_job_dir = None
        debug_job_id = None
        if DEBUG_DIR:
            try:
                os.makedirs(DEBUG_DIR, exist_ok=True)
                debug_job_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
                debug_job_dir = os.path.join(DEBUG_DIR, debug_job_id)
                os.makedirs(debug_job_dir, exist_ok=True)
            except Exception as e:
                print(
                    f"[pdf2md] debug dir init failed: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                debug_job_dir = None
                debug_job_id = None
        try:
            for seq, idx in enumerate(page_indices):
                page_num = idx + 1
                page = doc[idx]
                debug_ctx = None
                if debug_job_dir:
                    try:
                        page_dir = os.path.join(debug_job_dir, f"page-{page_num:04d}")
                        os.makedirs(page_dir, exist_ok=True)
                        debug_ctx = {
                            "job_id": debug_job_id,
                            "job_dir": debug_job_dir,
                            "page_dir": page_dir,
                            "page_num": page_num,
                        }
                    except Exception:
                        debug_ctx = None

                render_started = time.perf_counter()
                page_jpeg, image_meta = _page_to_jpeg(page, dpi)
                b64 = base64.b64encode(page_jpeg).decode()
                render_ms = round((time.perf_counter() - render_started) * 1000)

                user_text = user_template.format(page=page_num, total=total_pages)
                if prev_tail:
                    user_text += CONTINUATION.get(lang, CONTINUATION["ja"]).format(
                        tail=prev_tail
                    )

                metrics = {
                    **image_meta,
                    "render_ms": render_ms,
                }

                try:
                    call_timeout = OLLAMA_TIMEOUT
                    if not model_load_checked:
                        model_load_checked = True
                        if hasattr(provider, "is_model_loaded") and not provider.is_model_loaded(model):
                            call_timeout = OLLAMA_TIMEOUT + OLLAMA_LOAD_TIMEOUT
                            print(
                                f"[pdf2md] page {page_num}: model not loaded, using extended timeout {call_timeout}s",
                                file=sys.stderr,
                                flush=True,
                            )
                    llm_started = time.perf_counter()
                    result = provider.chat(
                        model,
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_text, "images": [b64]},
                        ],
                        stream=False,
                        thinking=thinking,
                        keep_alive=OLLAMA_KEEP_ALIVE,
                        options={"num_predict": MAX_TOKENS, "temperature": 0.1},
                        timeout=call_timeout,
                    )
                    metrics.update(
                        {
                            "ollama_ms": round(
                                (time.perf_counter() - llm_started) * 1000
                            ),
                            "prompt_eval_count": result.get("prompt_eval_count"),
                            "eval_count": result.get("eval_count"),
                            "prompt_eval_ms": round(
                                (result.get("prompt_eval_duration") or 0) / 1_000_000
                            ),
                            "eval_ms": round(
                                (result.get("eval_duration") or 0) / 1_000_000
                            ),
                            "total_ms": round(
                                (result.get("total_duration") or 0) / 1_000_000
                            ),
                        }
                    )
                    content = _normalize_markdown(result["content"])
                except requests.exceptions.Timeout:
                    content = f"<!-- Page {page_num}: timeout -->"
                    metrics["error"] = "timeout"
                except Exception as e:
                    content = f"<!-- Page {page_num}: error - {e} -->"
                    metrics["error"] = str(e)

                page_assets = []
                if extract_enabled and not metrics.get("error"):
                    pass2_started = time.perf_counter()
                    content, page_assets, asset_info = _extract_assets(
                        content,
                        page,
                        page_num,
                        page_jpeg,
                        image_meta["width"],
                        image_meta["height"],
                        model,
                        thinking,
                        debug_ctx,
                        provider=provider,
                    )
                    if asset_info.get("markers"):
                        metrics["pass2_ms"] = round(
                            (time.perf_counter() - pass2_started) * 1000
                        )
                        metrics["asset_source"] = asset_info.get("source")
                        metrics["asset_markers"] = asset_info.get("markers")
                        metrics["asset_extracted"] = asset_info.get("extracted")
                        metrics["asset_native_hits"] = asset_info.get("native_hits")
                        for key, value in (asset_info.get("timings") or {}).items():
                            if value:
                                metrics[key] = value
                        if asset_info.get("debug_dir"):
                            metrics["asset_debug_dir"] = asset_info.get("debug_dir")

                prev_tail = get_tail(content)

                page_asset_started = time.perf_counter()
                page_asset_jpeg, page_asset_meta = _page_to_jpeg(
                    page, PAGE_ASSET_DPI, quality=PAGE_ASSET_JPEG_QUALITY
                )
                page_asset_filename = f"p{page_num}-page.jpg"
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "asset",
                            "filename": page_asset_filename,
                            "data": base64.b64encode(page_asset_jpeg).decode(),
                        }
                    )
                    + "\n\n"
                )
                metrics["page_asset_ms"] = round(
                    (time.perf_counter() - page_asset_started) * 1000
                )
                metrics["page_asset_width"] = page_asset_meta["width"]
                metrics["page_asset_height"] = page_asset_meta["height"]
                metrics["page_asset_bytes"] = page_asset_meta["image_bytes"]

                for asset in page_assets:
                    b64_png = base64.b64encode(asset["png_bytes"]).decode()
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "type": "asset",
                                "filename": asset["filename"],
                                "data": b64_png,
                            }
                        )
                        + "\n\n"
                    )
                assets_meta = [
                    {
                        "filename": page_asset_filename,
                        "size_bytes": page_asset_meta["image_bytes"],
                        "kind": "page",
                        "page_num": page_num,
                        "width": page_asset_meta["width"],
                        "height": page_asset_meta["height"],
                    }
                ] + [
                    {
                        "filename": a["filename"],
                        "size_bytes": len(a["png_bytes"]),
                        "kind": a.get("kind", "figure"),
                        "page_num": page_num,
                    }
                    for a in page_assets
                ]
                yield f"data: {json.dumps({'type': 'page', 'page': seq + 1, 'page_label': page_num, 'content': content, 'metrics': metrics, 'assets': assets_meta})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        finally:
            doc.close()

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _parse_pages(pages_str, total):
    """Parse page specification like '1-3,5,8-10' into sorted 0-based indices."""
    if not pages_str:
        return list(range(total))
    indices = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = max(1, int(start))
            end = min(total, int(end))
            indices.update(range(start - 1, end))
        else:
            p = int(part)
            if 1 <= p <= total:
                indices.add(p - 1)
    return sorted(indices) if indices else list(range(total))


@app.route("/api/pdf2md/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3200)
