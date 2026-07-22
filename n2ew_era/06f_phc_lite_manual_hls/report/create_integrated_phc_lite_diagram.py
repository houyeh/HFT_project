from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "PHC_lite_nine_step_flow_integrated.png"

FONT_BOLD = Path(r"C:\Windows\Fonts\msjhbd.ttc")
FONT_REG = Path(r"C:\Windows\Fonts\msjh.ttc")


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size=size)


W, H = 2600, 1800
M = 22
BG = "#fbfcfd"
INK = "#162033"
MUTED = "#5d697a"
TEAL = "#0b7189"
GRID = "#d9e1ea"
RED = "#d54b4b"
GRAY = "#60666d"


img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

title_f = font(42)
subtitle_f = font(22)
node_title_f = font(26)
node_body_f = font(22)
small_f = font(18)
card_title_f = font(25)
card_body_f = font(23)
formula_f = font(21)
legend_f = font(24)


def text_size(text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    if not text:
        return 0, 0
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(text: str, fnt: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        cur = ""
        for ch in para:
            test = cur + ch
            if text_size(test, fnt)[0] <= max_w or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    return lines


def draw_multiline(
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str,
    max_w: int,
    line_gap: int = 7,
    anchor: str = "la",
) -> int:
    x, y = xy
    lines = wrap_text(text, fnt, max_w)
    _, h0 = text_size("字", fnt)
    line_h = h0 + line_gap
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_h), line, font=fnt, fill=fill, anchor=anchor)
    return y + len(lines) * line_h


def rounded_box(
    box: tuple[int, int, int, int],
    fill: str,
    outline: str,
    width: int = 3,
    radius: int = 12,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def node(
    key: str,
    x: int,
    y: int,
    w: int,
    h: int,
    fill: str,
    outline: str,
    title: str,
    body: str,
    title_color: str = INK,
) -> dict[str, int]:
    rounded_box((x, y, x + w, y + h), fill, outline, radius=12, width=3)
    title_end = draw_multiline((x + 18, y + 16), title, node_title_f, title_color, w - 36, 8)
    draw_multiline((x + 18, title_end + 12), body, node_body_f, INK, w - 36, 11)
    return {"x": x, "y": y, "w": w, "h": h, "cx": x + w // 2, "cy": y + h // 2}


def point(n: dict[str, int], side: str) -> tuple[int, int]:
    if side == "top":
        return n["cx"], n["y"]
    if side == "bottom":
        return n["cx"], n["y"] + n["h"]
    if side == "left":
        return n["x"], n["cy"]
    if side == "right":
        return n["x"] + n["w"], n["cy"]
    raise ValueError(side)


def arrow(
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = GRAY,
    width: int = 4,
    dashed: bool = False,
    label: str | None = None,
    label_pos: float = 0.5,
    label_offset: tuple[int, int] = (0, -18),
) -> None:
    sx, sy = start
    ex, ey = end
    if dashed:
        dx, dy = ex - sx, ey - sy
        dist = math.hypot(dx, dy)
        if dist == 0:
            return
        ux, uy = dx / dist, dy / dist
        dash, gap = 15, 11
        d = 0.0
        while d < dist:
            d2 = min(d + dash, dist)
            draw.line(
                (sx + ux * d, sy + uy * d, sx + ux * d2, sy + uy * d2),
                fill=color,
                width=width,
            )
            d += dash + gap
    else:
        draw.line((sx, sy, ex, ey), fill=color, width=width)

    ang = math.atan2(ey - sy, ex - sx)
    size = 13
    pts = [
        (ex, ey),
        (ex - size * math.cos(ang - 0.45), ey - size * math.sin(ang - 0.45)),
        (ex - size * math.cos(ang + 0.45), ey - size * math.sin(ang + 0.45)),
    ]
    draw.polygon(pts, fill=color)

    if label:
        lx = sx + (ex - sx) * label_pos + label_offset[0]
        ly = sy + (ey - sy) * label_pos + label_offset[1]
        tw, th = text_size(label, small_f)
        draw.rounded_rectangle(
            (lx - 8, ly - 4, lx + tw + 8, ly + th + 6),
            radius=8,
            fill=BG,
            outline=None,
        )
        draw.text((lx, ly), label, font=small_f, fill=color)


def poly_arrow(points: list[tuple[int, int]], color: str = GRAY, width: int = 4, dashed: bool = False) -> None:
    for a, b in zip(points, points[1:]):
        arrow(a, b, color=color, width=width, dashed=dashed)


def routed_arrow(points: list[tuple[int, int]], color: str = GRAY, width: int = 4) -> None:
    for a, b in zip(points, points[1:-1]):
        draw.line((a[0], a[1], b[0], b[1]), fill=color, width=width)
    arrow(points[-2], points[-1], color=color, width=width)


def card(num: str, x: int, y: int, w: int, h: int, title: str, body: str, formula: str, color: str) -> None:
    rounded_box((x, y, x + w, y + h), "#ffffff", color, width=3, radius=14)
    draw.text((x + 16, y + 12), f"{num} {title}", font=card_title_f, fill=color)
    y2 = draw_multiline((x + 16, y + 46), body, card_body_f, INK, w - 32, 5)
    draw_multiline((x + 16, y2 + 4), formula, formula_f, MUTED, w - 32, 4)


# Frame and title
draw.rectangle((3, 3, W - 4, H - 4), outline=TEAL, width=6)
title = "PHC-lite 單步推論核心：九步資料流與遞迴回授"
tw, _ = text_size(title, title_f)
draw.text(((W - tw) // 2, 18), title, font=title_f, fill=INK)
subtitle = "輸入：x_t（16維市場特徵）+ state_prev（16維前一步記憶）  /  輸出：readout（2維訊號分數）+ state_next（16維）"
sw, _ = text_size(subtitle, subtitle_f)
draw.text(((W - sw) // 2, 72), subtitle, font=subtitle_f, fill=MUTED)

# Flow layout
flow_x = 32
top_y = 125
nodes: dict[str, dict[str, int]] = {}

nodes["x"] = node("x", flow_x + 20, top_y, 300, 105, "#eaf2ff", "#5d8ddd", "x_t", "市場特徵（16維）")
nodes["prev"] = node("prev", flow_x + 1190, top_y, 340, 105, "#eaf2ff", "#5d8ddd", "state_prev", "前一步隱藏狀態（16維）")

nodes["s1"] = node(
    "s1",
    flow_x + 20,
    top_y + 150,
    430,
    150,
    "#fff3d6",
    "#e4a534",
    "STEP1 ENC_LINEAR",
    "LayerNorm → SiLU\nencoded = SiLU(LN(ENC_LINEAR(x_t)))",
)
nodes["s2"] = node(
    "s2",
    flow_x + 20,
    top_y + 350,
    330,
    135,
    "#e6f7ec",
    "#41a56d",
    "STEP2 GATE_LINEAR",
    "Sigmoid → gate_t\n控制新資訊採用比例",
)
nodes["s3"] = node(
    "s3",
    flow_x + 415,
    top_y + 350,
    400,
    135,
    "#fff0ed",
    "#e7685c",
    "STEP3 CANDIDATE_LINEAR",
    "Tanh → candidate_t\n新記憶候選草稿",
)
nodes["s5"] = node(
    "s5",
    flow_x + 890,
    top_y + 350,
    510,
    165,
    "#e6f7ec",
    "#41a56d",
    "STEP5 GAIN_LINEAR",
    "+ GAIN_PREV_W\nSigmoid → gain_t\n決定參考外部訊息比例",
)
nodes["s4"] = node(
    "s4",
    flow_x + 200,
    top_y + 550,
    700,
    170,
    "#f0e2ff",
    "#9c57e5",
    "STEP4 state_temporal",
    "state_temporal = DECAY ⊙ state_prev\n+ gate_t ⊙ candidate_t\n打折舊記憶 + 篩選新草稿",
)
nodes["s6"] = node(
    "s6",
    flow_x + 930,
    top_y + 550,
    520,
    145,
    "#fff3d6",
    "#e4a534",
    "STEP6 SYNAPSE_LINEAR",
    "SiLU → message\n把暫存記憶整理成訊息",
)
nodes["next"] = node(
    "next",
    flow_x + 1135,
    top_y + 740,
    390,
    125,
    "#e8e8e8",
    "#7a7a7a",
    "state_next",
    "= state_temporal\n回授下一步",
)
nodes["s7"] = node(
    "s7",
    flow_x + 430,
    top_y + 870,
    760,
    170,
    "#f0e2ff",
    "#9c57e5",
    "STEP7 loop_input",
    "loop_input = state_temporal\n+ ALPHA ⊙ gain_t ⊙ message\n真正往下送的內容",
)
nodes["s8"] = node(
    "s8",
    flow_x + 490,
    top_y + 1100,
    390,
    135,
    "#eaf2ff",
    "#5d8ddd",
    "STEP8 LayerNorm",
    "z = LayerNorm(loop_input)\n自動調音量，穩定範圍",
)
nodes["s9"] = node(
    "s9",
    flow_x + 1010,
    top_y + 1100,
    470,
    135,
    "#fff3d6",
    "#e4a534",
    "STEP9 READOUT_LINEAR",
    "readout = READOUT_LINEAR(z)\n輸出 2 維訊號分數",
)
nodes["out"] = node(
    "out",
    flow_x + 1135,
    top_y + 1335,
    390,
    115,
    "#e8e8e8",
    "#7a7a7a",
    "輸出 readout",
    "2維訊號分數",
)

# Flow arrows
arrow(point(nodes["x"], "bottom"), point(nodes["s1"], "top"))
arrow(point(nodes["s1"], "bottom"), point(nodes["s2"], "top"))
arrow((nodes["s1"]["cx"] + 40, nodes["s1"]["y"] + nodes["s1"]["h"]), point(nodes["s3"], "top"))
arrow((nodes["s1"]["x"] + nodes["s1"]["w"], nodes["s1"]["cy"]), point(nodes["s5"], "left"), label="encoded")

arrow(point(nodes["s2"], "bottom"), (nodes["s4"]["x"] + 120, nodes["s4"]["y"]))
arrow(point(nodes["s3"], "bottom"), (nodes["s4"]["x"] + 340, nodes["s4"]["y"]))
arrow(point(nodes["s4"], "right"), point(nodes["s6"], "left"))
arrow((nodes["s4"]["cx"], nodes["s4"]["y"] + nodes["s4"]["h"]), (nodes["s7"]["x"] + 170, nodes["s7"]["y"]))
arrow((nodes["s6"]["cx"], nodes["s6"]["y"] + nodes["s6"]["h"]), (nodes["s7"]["x"] + 470, nodes["s7"]["y"]))
routed_arrow(
    [
        point(nodes["s5"], "bottom"),
        (nodes["s5"]["x"] + nodes["s5"]["w"] + 70, nodes["s5"]["y"] + nodes["s5"]["h"]),
        (nodes["s5"]["x"] + nodes["s5"]["w"] + 70, nodes["s7"]["y"] - 16),
        (nodes["s7"]["x"] + 705, nodes["s7"]["y"]),
    ],
    color=GRAY,
    width=4,
)
arrow(point(nodes["s7"], "bottom"), point(nodes["s8"], "top"))
arrow(point(nodes["s8"], "right"), point(nodes["s9"], "left"))
arrow(point(nodes["s9"], "bottom"), point(nodes["out"], "top"))

# state_prev dependency routing
draw.line(
    (nodes["prev"]["cx"], nodes["prev"]["y"] + nodes["prev"]["h"], nodes["prev"]["cx"], nodes["s4"]["y"] - 18),
    fill="#9aa1aa",
    width=4,
)
arrow((nodes["prev"]["cx"], nodes["s4"]["y"] - 18), (nodes["s4"]["x"] + 515, nodes["s4"]["y"]), color="#9aa1aa", label="state_prev × DECAY")
arrow((nodes["prev"]["cx"], nodes["s5"]["y"] - 42), (nodes["s5"]["x"] + 250, nodes["s5"]["y"]), color="#9aa1aa", label="state_prev × GAIN_PREV_W", label_offset=(-40, -24))

# state_temporal to state_next and feedback path
arrow((nodes["s4"]["x"] + nodes["s4"]["w"], nodes["s4"]["y"] + 48), point(nodes["next"], "left"))
poly_arrow(
    [
        point(nodes["next"], "right"),
        (flow_x + 1530, nodes["next"]["cy"]),
        (flow_x + 1530, nodes["prev"]["cy"]),
        point(nodes["prev"], "right"),
    ],
    color=RED,
    width=4,
    dashed=True,
)
# Legend
legend_y = H - 170
legend_items = [
    ("#fff3d6", "#e4a534", "線性層 / 訊息轉換"),
    ("#e6f7ec", "#41a56d", "Sigmoid 閘控"),
    ("#fff0ed", "#e7685c", "Tanh 候選"),
    ("#f0e2ff", "#9c57e5", "狀態更新 / 遞迴核心"),
    ("#eaf2ff", "#5d8ddd", "LayerNorm / 輸入狀態"),
    ("#e8e8e8", "#7a7a7a", "輸出 / 回授狀態"),
]
def legend_item(x: int, y: int, fill: str, outline: str, txt: str) -> int:
    rounded_box((x, y, x + 32, y + 32), fill, outline, width=2, radius=4)
    draw.text((x + 44, y - 1), txt, font=legend_f, fill=INK)
    return x + text_size(txt, legend_f)[0] + 112


lx = flow_x + 10
for fill, outline, txt in legend_items[:4]:
    lx = legend_item(lx, legend_y, fill, outline, txt)

row2_y = legend_y + 62
lx = flow_x + 10
for fill, outline, txt in legend_items[4:]:
    lx = legend_item(lx, row2_y, fill, outline, txt)

draw.line((lx + 10, row2_y + 18, lx + 88, row2_y + 18), fill=GRAY, width=5)
draw.text((lx + 102, row2_y + 2), "同時間依賴", font=legend_f, fill=INK)
draw.line((lx + 285, row2_y + 18, lx + 363, row2_y + 18), fill=RED, width=5)
for dx in range(285, 363, 30):
    draw.line((lx + dx + 12, row2_y + 18, lx + dx + 26, row2_y + 18), fill=BG, width=6)
draw.text((lx + 382, row2_y + 2), "跨時間步回授", font=legend_f, fill=INK)

# Right integrated text panel
panel_x = 1600
panel_y = 125
panel_w = 960
panel_h = 1600
rounded_box((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), "#f6f9fb", "#c5d4df", width=3, radius=18)
draw.text((panel_x + 22, panel_y + 18), "九步說明與精確數學定義", font=font(31), fill=INK)

cards = [
    ("①", "輸入整理", "把原始輸入整理成模型好處理的格式。", "encoded = SiLU(LayerNorm(ENC_LINEAR(x_t)))", "#e4a534"),
    ("②", "採用比例", "決定這筆新資訊有多少比例可被採用，如水龍頭控制流量。", "gate_t = Sigmoid(GATE_LINEAR(encoded))", "#41a56d"),
    ("③", "候選記憶", "草擬如果採用這筆新資訊，記憶會更新成什麼樣子。", "candidate_t = Tanh(CANDIDATE_LINEAR(encoded))", "#e7685c"),
    ("④", "暫存記憶", "把打折後舊記憶與篩選過的新草稿依比例混合。", "state_temporal = DECAY ⊙ state_prev + gate_t ⊙ candidate_t", "#9c57e5"),
    ("⑤", "外部訊息權重", "決定這次要多認真參考其他來源，而非只看自己的記憶。", "gain_t = Sigmoid(GAIN_LINEAR(encoded) + GAIN_PREV_W × state_prev)", "#41a56d"),
    ("⑥", "訊息整理", "把目前記憶狀態整理成要傳遞給下一階段的訊息。", "message = SiLU(SYNAPSE_LINEAR(state_temporal))", "#e4a534"),
    ("⑦", "真正送出的內容", "把④的暫存記憶，加上⑤決定要聽多少的⑥訊息。", "loop_input = state_temporal + ALPHA ⊙ gain_t ⊙ message", "#9c57e5"),
    ("⑧", "穩定範圍", "把組合後訊號正規化，避免訊號忽大忽小，如同自動調音量。", "z = LayerNorm(loop_input)", "#5d8ddd"),
    ("⑨", "輸出與回授", "由穩定訊號算出最終分數；同時把④的 state_temporal 存為下一步記憶。", "readout = READOUT_LINEAR(z)；state_next = state_temporal", "#e4a534"),
]

y = panel_y + 68
card_hs = [118, 128, 128, 134, 142, 128, 134, 134, 148]
for (num, title_c, body, formula, color), ch in zip(cards, card_hs):
    card(num, panel_x + 18, y, panel_w - 36, ch, title_c, body, formula, color)
    y += ch + 12

feedback_h = panel_y + panel_h - y - 16
rounded_box((panel_x + 18, y, panel_x + panel_w - 18, y + feedback_h), "#fff5f5", RED, width=3, radius=14)
draw.text((panel_x + 38, y + 16), "虛線回授路徑的工程意義", font=card_title_f, fill=RED)
feedback = (
    "圖中的紅色虛線表示 state_next 必須回頭餵給下一個時間步的 state_prev。"
    "因此「這一步沒算完，下一步就不能開始」；這是模型數學定義上的遞迴依賴，"
    "也是硬體上無法跨時間步管線化、II = Latency 的直接來源（對應設計概念 3.4 節）。"
)
draw_multiline((panel_x + 38, y + 55), feedback, card_body_f, INK, panel_w - 90, 7)

footer = "圖一：PHC-lite 九步運算流程（整合口語說明、數學定義與跨時間步回授限制）"
fw, _ = text_size(footer, subtitle_f)
draw.text(((W - fw) // 2, H - 52), footer, font=subtitle_f, fill=MUTED)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, quality=95)
print(OUT)
