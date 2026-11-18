"""Build the INT4096 Group 08 "Detecting the Detector" draft deck in the user's retro-desktop template.

Usage (run with python3 -I, paths as arguments):
    python3 -I build_deck.py <template.pptx> <figure_dir> <work_dir> <out.pptx>

- <template.pptx>  the user's template (read only; only its master, layouts, embedded fonts and icon images are reused)
- <figure_dir>     汇报材料包-20261117/01_图-16比9 (read only)
- <work_dir>       scratch dir under /tmp for cropped figure copies
- <out.pptx>       output deck

Canvas: 1440 x 810 pt (18288000 x 10287000 EMU). All coordinates below are in points.
"""
import os
import sys
import textwrap
import zipfile

from lxml import etree
from PIL import Image, ImageChops
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.oxml.ns import qn
from pptx.util import Emu

TEMPLATE, FIGDIR, WORK, OUT = sys.argv[1:5]
os.makedirs(WORK, exist_ok=True)

# ------------------------------------------------------------------ constants
SW, SH = 1440, 810
BLUE = "0057CC"          # template accent (title bar, headlines)
DESK = "CACACA"          # template desktop grey
FACE = "D2D2D2"          # bevel face (from template SVG)
BAR = "C0C0C0"           # scrollbar grey (template)
DARK = "939393"          # bevel shadow (30% black over D2D2D2, as in template SVG)
INK = "000000"
NOTE_INK = "333333"
YELLOW = "FFF200"        # placeholder highlight, used ONLY for team-to-do markers

WIN_X, WIN_W = 16, 1408
TB_Y, TB_H = 12, 44                 # title bar
MENU_Y = 56                         # menu bar row (on desktop grey, as in template)
WIN_Y, WIN_B = 98, 800              # white window
SCR_H = 26                          # scrollbar strip height (thinned from template)
SCR_Y = WIN_B - SCR_H
CX0, CX1 = 56, 1384                 # content margins inside window
BOX_X, BOX_Y, BOX_W, BOX_H = 58, 118, 540, 62   # bevel title box (identical on all content slides)
TAKE_X = BOX_X + BOX_W + 28
FIG_TOP = 192
FOOT_B = 766                        # bottom of footnote block
FOOT_SIZE = 20
FOOT_LH = 23

TABS = ["Intro", "Audit", "ML", "DL", "LoRA", "Wrap-up"]
WINDOW_TITLE = "Detecting the Detector - INT4096 Group 08"
DL_TAG = "DL extension, instructor-approved"


def E(pt):
    return Emu(int(round(pt * 12700)))


# ------------------------------------------------------------------ template icons (png + svg pairs)
ICONS = {  # name: (png, svg) inside the template package
    "close": ("image1.png", "image2.svg"), "max": ("image3.png", "image4.svg"),
    "min": ("image5.png", "image6.svg"), "arrow": ("image7.png", "image8.svg"),
    "cursor": ("image9.png", "image10.svg"), "info": ("image15.png", "image16.svg"),
    "tile": ("image17.png", "image18.svg"), "keyboard": ("image19.png", "image20.svg"),
    "pc": ("image21.png", "image22.svg"), "folder": ("image23.png", "image24.svg"),
    "cd": ("image25.png", "image26.svg"), "bulb": ("image29.png", "image30.svg"),
    "progress": ("image31.png", "image32.svg"), "warning": ("image33.png", "image34.svg"),
    "magnifier": ("image41.png", "image42.svg"),
}
with zipfile.ZipFile(TEMPLATE) as z:
    ICON_BLOBS = {k: (z.read("ppt/media/" + p), z.read("ppt/media/" + s)) for k, (p, s) in ICONS.items()}
ICON_PNG = {}
for k, (png, _svg) in ICON_BLOBS.items():
    path = os.path.join(WORK, "icon_%s.png" % k)
    with open(path, "wb") as f:
        f.write(png)
    ICON_PNG[k] = path


def icon_ratio(name):
    with Image.open(ICON_PNG[name]) as im:
        return im.size[0] / im.size[1]


# ------------------------------------------------------------------ figure cropping (copies only)
CROP_OVERRIDE = {"fig20_dl_method": (244, 482), "fig22_lora_method": (232, 489), "fig18_ml_method": (244, 482),
                 "fig01_hook_toefl": (158, 548)}
CROP_SOURCE = {"fig01_hook_toefl": "fig01_hook"}


def crop_figure(name):
    """Crop the title+subtitle band (top) and the footnote/source lines (bottom) of a 1920x1080 figure.
    Bands are found from the pixels so a regenerated figure is still cropped correctly."""
    src = os.path.join(FIGDIR, CROP_SOURCE.get(name, name) + ".png")
    if name in CROP_OVERRIDE:
        top, bottom = CROP_OVERRIDE[name]
        out = os.path.join(WORK, name + "_crop.png")
        c = Image.open(src).convert("RGB").crop((60, top, 1860, bottom))
        bg = c.getpixel((2, 2))
        r, g, b_ = c.split()
        near = [ch.point(lambda v, t=t: 255 if abs(v - t) <= 3 else 0) for ch, t in zip((r, g, b_), bg)]
        c.paste((255, 255, 255), mask=ImageChops.multiply(ImageChops.multiply(near[0], near[1]), near[2]))
        c.save(out, optimize=True)
        return out, (1800, bottom - top), (top, bottom)
    im = Image.open(src).convert("RGB")
    w, h = im.size
    bg = im.getpixel((5, 5))
    diff = ImageChops.difference(im, Image.new("RGB", im.size, bg)).convert("L").point(lambda v: 255 if v > 12 else 0)
    ink = [diff.crop((0, y, w, y + 1)).getbbox() is not None for y in range(h)]
    bands, start = [], None
    for y, v in enumerate(ink):
        if v and start is None:
            start = y
        if not v and start is not None:
            bands.append((start, y - 1))
            start = None
    # band 0 = title (~60-101), band 1 = subtitle (~129-154); content starts with the next band
    assert bands[0][0] < 110 and 120 < bands[1][0] < 140, (name, bands[:3])
    sub_end = bands[1][1]
    nxt = bands[2][0]
    top = max(sub_end + 2, nxt - 10)
    # footnote block = bands starting at y >= 975 (two caveat lines + source line)
    content = [b for b in bands[2:] if b[0] < 975]
    bottom = min(content[-1][1] + 10, 978)
    x0, x1 = 60, 1860
    out = os.path.join(WORK, name + "_crop.png")
    c = im.crop((x0, top, x1, bottom))
    # the figures sit on a near-white (252,252,251) page; map exactly that background to the window's white
    # so no faint box shows. Only pixels within 3 levels of the page colour change; data colours are untouched.
    r, g, b_ = c.split()
    near = [ch.point(lambda v, t=t: 255 if abs(v - t) <= 3 else 0) for ch, t in zip((r, g, b_), bg)]
    mask = ImageChops.multiply(ImageChops.multiply(near[0], near[1]), near[2])
    c.paste((255, 255, 255), mask=mask)
    c.save(out, optimize=True)
    return out, (x1 - x0, bottom - top), (top, bottom)


# ------------------------------------------------------------------ low-level shape helpers
def _strip_style(shape):
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def rect(slide, x, y, w, h, fill=None, line=None, line_w=2.25, dash=None, name=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, E(x), E(y), E(w), E(h))
    _strip_style(shp)
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb(fill)
    else:
        shp.fill.background()
    if line:
        shp.line.color.rgb = _rgb(line)
        shp.line.width = E(line_w)
        if dash:
            ln = shp._element.spPr.find(qn("a:ln"))
            pd = etree.SubElement(ln, qn("a:prstDash"))
            pd.set("val", dash)
    else:
        shp.line.fill.background()
    if name:
        shp.name = name
    return shp


def _rgb(hexstr):
    from pptx.dml.color import RGBColor
    return RGBColor.from_string(hexstr)


def bevel(slide, x, y, w, h, b=4.0, face=FACE, raised=True, outline=True):
    """Win98 bevel built from flat rectangles (crisp at any size), like the template's SVG boxes."""
    if outline:
        rect(slide, x, y, w, h, fill=INK)
        x, y, w, h = x + b, y + b, w - 2 * b, h - 2 * b
    hi, lo = ("FFFFFF", DARK) if raised else ("808080", "FFFFFF")
    rect(slide, x, y, w, h, fill=hi)
    rect(slide, x + b, y + b, w - b, h - b, fill=lo)
    rect(slide, x + b, y + b, w - 2 * b, h - 2 * b, fill=face)
    return (x + b, y + b, w - 2 * b, h - 2 * b)


_svg_parts = {}


def picture(slide, prs, name, x, y, w=None, h=None, flip_h=False):
    r = icon_ratio(name)
    if w is None:
        w = h * r
    if h is None:
        h = w / r
    pic = slide.shapes.add_picture(ICON_PNG[name], E(x), E(y), E(w), E(h))
    pic.name = "icon_" + name
    # attach the template's SVG so PowerPoint renders the icon crisply (PNG stays as fallback)
    if name not in _svg_parts:
        _svg_parts[name] = Part(PackURI("/ppt/media/tpl_%s.svg" % name), "image/svg+xml",
                                prs.part.package, ICON_BLOBS[name][1])
    rid = slide.part.relate_to(_svg_parts[name], RT.IMAGE)
    blip = pic._element.find(".//" + qn("a:blip"))
    ext_lst = etree.SubElement(blip, qn("a:extLst"))
    ext = etree.SubElement(ext_lst, qn("a:ext"))
    ext.set("uri", "{96DAC541-7B7A-43D3-8B79-37D633B846F1}")
    svg = etree.SubElement(ext, "{http://schemas.microsoft.com/office/drawing/2016/SVG/main}svgBlip",
                           nsmap={"asvg": "http://schemas.microsoft.com/office/drawing/2016/SVG/main"})
    svg.set(qn("r:embed"), rid)
    if flip_h:
        pic._element.spPr.find(qn("a:xfrm")).set("flipH", "1")
    return pic, w, h


FACES = {("cousine", False): "Cousine", ("cousine", True): "Cousine Bold",
         ("anon", True): "Anonymous Pro Bold", ("anon", False): "Anonymous Pro Bold"}


def _style_run(r, text, size, font="cousine", bold=False, color=INK, spc=None):
    r.text = text
    rPr = r._r.get_or_add_rPr()
    rPr.set("lang", "en-US")
    rPr.set("sz", str(int(round(size * 100))))
    if bold or font == "anon":
        rPr.set("b", "1")
    if spc is not None:
        rPr.set("spc", str(int(round(spc * 100))))
    fill = etree.SubElement(rPr, qn("a:solidFill"))
    etree.SubElement(fill, qn("a:srgbClr")).set("val", color)
    face = FACES[(font, bool(bold or font == "anon"))]
    for tag in ("a:latin", "a:ea", "a:cs", "a:sym"):
        etree.SubElement(rPr, qn(tag)).set("typeface", face)


def text(slide, x, y, w, h, paras, size=22, font="cousine", bold=False, color=INK, align="l",
         anchor="t", lh=None, spc=None, space_after=0, name=None, fill=None, line=None, dash=None,
         inset=0, marL=0, indent=0):
    """paras: list of paragraphs; a paragraph is a str or a list of runs (str or (str, {opts}))."""
    tb = slide.shapes.add_textbox(E(x), E(y), E(w), E(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    bp = tf._txBody.find(qn("a:bodyPr"))
    for k in ("lIns", "tIns", "rIns", "bIns"):
        bp.set(k, str(int(inset * 12700)))
    bp.set("anchor", {"t": "t", "m": "ctr", "b": "b"}[anchor])
    if fill or line:
        if fill:
            tb.fill.solid()
            tb.fill.fore_color.rgb = _rgb(fill)
        if line:
            tb.line.color.rgb = _rgb(line)
            tb.line.width = E(1.75)
            if dash:
                ln = tb._element.spPr.find(qn("a:ln"))
                etree.SubElement(ln, qn("a:prstDash")).set("val", dash)
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        if lh:
            p.line_spacing = E(lh)
        if space_after:
            p.space_after = E(space_after)
        if marL or indent:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(marL * 12700)))
            pPr.set("indent", str(int(indent * 12700)))
        runs = para if isinstance(para, list) else [para]
        for run in runs:
            if isinstance(run, tuple):
                t, o = run
            else:
                t, o = run, {}
            r = p.add_run()
            _style_run(r, t, o.get("size", size), o.get("font", font), o.get("bold", bold),
                       o.get("color", color), o.get("spc", spc))
    return tb


def nlines(s, width, size, cw=0.6):
    cpl = max(1, int(width / (cw * size)))
    return len(textwrap.wrap(s, cpl, break_long_words=False)) or 1


# ------------------------------------------------------------------ chrome
def chrome(slide, prs, active, frac, extra_tab=None, status=None):
    bgfill = slide.background.fill
    bgfill.solid()
    bgfill.fore_color.rgb = _rgb(DESK)
    # title bar
    rect(slide, WIN_X, TB_Y, WIN_W, TB_H, fill=BLUE, line=INK, name="title bar")
    text(slide, WIN_X + 16, TB_Y, 900, TB_H, WINDOW_TITLE, size=22, font="cousine", bold=True,
         color="F2F2F2", anchor="m", name="window title")
    bx = WIN_X + WIN_W - 8 - 34
    for nm in ("close", "max", "min"):
        picture(slide, prs, nm, bx, TB_Y + 5, w=34, h=34)
        bx -= 40
    # menu bar (text on the desktop grey, active tab bold, exactly like the template)
    tabs = TABS + ([extra_tab] if extra_tab else [])
    x = WIN_X + 24
    for t in tabs:
        on = (t == active)
        w = len(t) * (0.6 * 22 + 1.5) + 8
        text(slide, x, MENU_Y + 8, w + 10, 32, [[(t, {"bold": on})]], size=22, spc=1.5, anchor="m",
             name="tab " + t)
        x += w + 40
    if status:
        sw = len(status) * 0.6 * 20 + 36
        sx = WIN_X + WIN_W - sw
        inner = bevel(slide, sx, MENU_Y + 7, sw, 34, b=2.0, raised=False, outline=False)
        text(slide, inner[0], inner[1], inner[2], inner[3], status, size=20, align="c", anchor="m",
             name="status field")
    # window
    rect(slide, WIN_X, WIN_Y, WIN_W, WIN_B - WIN_Y, fill="FFFFFF", line=INK, name="window")
    # scrollbar strip; the thumb position shows progress through the deck
    rect(slide, WIN_X, SCR_Y, WIN_W, SCR_H, fill=BAR, line=INK, name="scrollbar")
    bw = 34
    for i, xx in enumerate((WIN_X, WIN_X + WIN_W - bw)):
        bevel(slide, xx, SCR_Y, bw, SCR_H, b=2.0, outline=False)
        picture(slide, prs, "arrow", xx + bw / 2 - 5, SCR_Y + 6, w=10, h=14, flip_h=(i == 0))
    tw = 110
    track0, track1 = WIN_X + bw, WIN_X + WIN_W - bw - tw
    bevel(slide, track0 + (track1 - track0) * frac, SCR_Y, tw, SCR_H, b=2.0, outline=False)


def header(slide, prs, label, take=None, icon=None, take_size=24):
    """Bevel title box with the 'i' bubble + takeaway sentence (identical position on every content slide)."""
    bevel(slide, BOX_X, BOX_Y, BOX_W, BOX_H, b=4.0)
    lsize = 30 if len(label) <= 29 else 26
    text(slide, BOX_X + 10, BOX_Y, BOX_W - 20, BOX_H, label, size=lsize, font="anon", align="c",
         anchor="m", spc=-0.5, name="title")
    picture(slide, prs, "info", BOX_X - 22, BOX_Y - 17, h=48)
    right = CX1
    if icon:
        ih = 66 if icon != "progress" else 30
        r = icon_ratio(icon)
        iw = ih * r
        if iw > 130:
            iw, ih = 130, 130 / r
        picture(slide, prs, icon, CX1 - iw, BOX_Y + BOX_H / 2 - ih / 2, w=iw, h=ih)
        right = CX1 - iw - 24
    if take:
        text(slide, TAKE_X, BOX_Y - 8, right - TAKE_X, BOX_H + 16, take, size=take_size, bold=True,
             color=BLUE, anchor="m", lh=take_size * 1.17, name="takeaway")


def footnote(slide, s, top):
    return text(slide, CX0, top, CX1 - CX0, FOOT_B - top, s, size=FOOT_SIZE, color=NOTE_INK,
                lh=FOOT_LH, name="footnote")


def figure(slide, crop, area_top, area_bottom, max_w=CX1 - CX0 + 30):
    path, (pw, ph), _ = crop
    s = min(max_w / pw, (area_bottom - area_top) / ph)
    w, h = pw * s, ph * s
    x = (SW - w) / 2
    y = area_top + ((area_bottom - area_top) - h) / 2
    pic = slide.shapes.add_picture(path, E(x), E(y), E(w), E(h))
    pic.name = "figure"
    return w, h, s


def foot_height(s):
    return nlines(s, CX1 - CX0 - 4, FOOT_SIZE) * FOOT_LH


# ------------------------------------------------------------------ deck content
FIG = {}  # filled lazily

MAIN = []  # (kind, kwargs)

FOOT = {
 "fig01_hook": "Illustrative pair, not a random sample: 19 of 91 TOEFL essays were flagged by all seven detectors, 72 of 88 native essays by none. Scores archived from Liang et al. (2023); flagged = score of 0.5 or more.",
 "fig03_opensource": "Same Liang benchmark. Gains: far fewer false flags and much better AI detection. On bias (AUC) the open-source detectors are no better than the best commercial ones. Official thresholds, no training on our data.",
 "fig12_hotspots": "Shading: falcon-7b word probabilities (the model behind Binoculars and Fast-DetectGPT). Illustrative pair; the medians on the right use all essays. Hewlett is not used: its placeholders (@CAPS1) inflate perplexity.",
 "fig18_ml_method": "The classical method from the course: simple, fast, and it shows which words push the score up. Default regularisation; settings were not tuned (see the backup slide).",
 "fig19_ml_results": "Threshold 0.5, mean of 5 seeds. Unfamiliar native essays (mean of CollegeEssay, CS224N, LOCNESS): 85% flagged in arm A, 43% in arm C. Arm D adds other native essays and does not help TOEFL (47%).",
 "fig20_dl_method": "All 125 million parameters are fine-tuned. roberta-base; learning rate 2e-5, batch 16, 512 tokens, 2 epochs; settings not tuned. Deep learning is an extension the instructor approved verbally.",
 "fig21_dl_results": "Threshold 0.5, mean of 5 seeds; arm C varies by 8 points. Unfamiliar native essays: 69% flagged in arm A, 45% in arm C. School-age non-native essays (B) barely help; exam-style essays (C) do.",
 "fig22_lora_method": "A 27B model cannot be fully retrained on two 46 GB GPUs, so LoRA trains only about 0.5% of it. LoRA rank 16, alpha 32; settings not tuned. Instructor approved deep learning verbally.",
 "fig23_lora_results": "Only 2 seeds (threshold 0.5): in arm A the TOEFL rate was 56% in one run, 23% in the other. Unfamiliar native essays: 29% in arm A, 9% in arm C. Effect size is indicative; the direction matches the other models.",
 "fig04_shift": "Arm A: trained only on US school essays; threshold 0.5. Logistic regression and RoBERTa flag native essays of a new style (CollegeEssay, CS224N) about as often as TOEFL essays. LOCNESS has only 50 essays.",
 "fig05_trend": "Threshold 0.5. Error bars: SD across seeds (Qwen: 2 seeds, so arm A varies a lot). Arm C was trained on essays like TOEFL; it is not a general fix. Arm D adds other native essay types and barely helps.",
 "fig05_trend_before_plainwords": "Threshold 0.5. Error bars: SD across seeds (Qwen: 2 seeds, so arm A varies a lot). Arm C was trained on essays like TOEFL; it is not a general fix. Arm D adds other native essay types and barely helps.",
 "fig14_pipeline_before_plainwords": "Test essays: 13,672 in total, minus the half of the FCE and W&I essays that arm C trains on (10,171 left). The checks behind steps 2 and 3 are in the reproducibility folder of the package.",
 "fig03_opensource_before_plainwords": "Same Liang benchmark. Gains: far fewer false flags and much better AI detection. On bias (AUC) the open-source detectors are no better than the best commercial ones. Official thresholds, no training on our data.",
 "fig15_pitfalls_detail": "Sources: the archived first runs and the final runs (arm A and B, threshold 0.5). 0.888 was estimated from the per-corpus flag rates of the first analysis.",
 "fig15_pitfalls": "Sources: the archived first runs and the final runs (arm A and B, threshold 0.5). 0.888 was estimated from the per-corpus flag rates of the first analysis.",
 # backups
 "fig02_commercial": "Left: share of real human essays flagged as AI. Right: AUC = how well each detector tells Liang's 352 AI essays from the 179 real TOEFL + Hewlett essays; 1.0 = perfect, 0.5 = chance.",
 "fig06_scoreboard": "Logistic regression and RoBERTa: 5 seeds, Qwen: 2 seeds; threshold 0.5. Bold = best in row (ties within noise both bold). With the right data, logistic regression (8%, arm C) beats Qwen without it (40%, arm A).",
 "fig07_notfixed": "Numbers are % flagged (grey = arm A, bold = arm C). CS224N is still flagged 97% / 70% / 23% in arm C: fixing the data for one kind of writing does not make a detector fair in general.",
 "fig08_learning_backup": "Threshold 0.5; 5 seeds. Quadrupling the data does not lower arm A (it even rises for logistic regression, 70% to 80%); changing what the data contains moves the rate far more.",
 "fig09_length_backup": "Bias AUC: 0.5 = none. TOEFL essays are about 104 words, Hewlett about 378. The self-trained models do not drop when length is matched; the open-source detectors drop by 0.06-0.08.",
 "fig10_prf1": "Liang test set: 394 human + 352 AI essays; AI = positive class; threshold 0.5. F1 on TOEFL / Hewlett counts AI and that human group equally. Bold = best of the six models; means over seeds (5; Qwen 2).",
 "fig11_prf1_full": "Liang test set: 394 human + 352 AI essays; AI = positive class; threshold 0.5; mean ± SD over seeds (Qwen: 2 seeds). Quil and Sapling: no scores for 145 CS224N human essays (tested on 249).",
 "fig13_polish": "Same essay as slide 6; the polished text comes from Liang et al. (2023), matched to the original by word overlap. Shading: falcon-7b word probabilities; darker = more predictable.",
 "fig16_models": "Same input and output for all three: an essay goes in, the probability that it is AI comes out (flagged at 0.5). Deep learning is an extension the instructor approved verbally.",
 "fig17_settings": "Why not tune: it keeps the four training sets (A to D) as the only difference. Qwen weights were converted from FP8 to bf16 before training (about 2.7% weight error, checked against a bf16 shard).",
}

NOTES = {}
NOTES[1] = """Presenter: A | ~0.3 min
Hello everyone, we are Group 08.
We start with one question.
If you write your own essay in English, will an AI detector believe you?
(Optional: ask "Has anyone here ever worried that a detector might flag their own writing?" Pause for two seconds, then go to the next slide.)"""
NOTES[2] = """Presenter: A | ~1.0 min
Let the audience read the essay for a few seconds before talking.
Likely question: did you pick this essay on purpose? Yes. It is a hand-picked pair to make the point, not a random sample. In the whole benchmark, 19 of 91 TOEFL essays were flagged by all seven detectors, and 72 of 88 native essays by none.
Likely question: did you run these detectors yourselves? No. These are the archived 2023 scores from Liang et al.
The native essay is on the backup slide. "@CAPS1" in it is a name the dataset anonymised."""
NOTES[3] = """Presenter: A | ~1.3 min
We chose this topic for four reasons.
First, the stakes are real. A detector score can start a misconduct case against a student. [TEAM: add an example you know.]
Second, it affects people like us. [TEAM: say your own reason here.]
Third, the evidence is old and incomplete. Liang and colleagues showed in 2023 that commercial detectors flag non-native writing far more often. On their archived scores we count 61.9 percent of TOEFL essays flagged, against 5.4 percent of native essays. But nobody tested why: is it because the writer is non-native, or because the writing is unfamiliar to the detector?
Fourth, it fits this course. If machine learning supports decisions about students, we must test it for bias.
So we asked three questions: is there bias, what causes it, and can training data fix it?
Here is our short answer first. Yes, detectors flag non-native exam essays more. Mainly because they have not seen that kind of writing. Training on the right essays cuts false flags in all three of our models, but not everywhere."""
NOTES[4] = """Presenter: B | ~1.2 min
Example: an essay that appears in both PERSUADE and ELLIPSE was counted once, not twice. We removed 452 such PERSUADE essays.
Likely question: why 10,171 and not 13,672? Set C trains on half of the FCE and Write and Improve essays, so we took that half out of the test.
Likely question: what is new compared with Liang et al.? Seven extra data sources, three model families trained on the same data, and a comparison of non-native with native essays of several kinds."""
NOTES[5] = """Presenter: B | ~1.1 min
Point at the gap between the grey bars and the two purple bars.
Likely question: so the open-source detectors are fair? No. In 87% and 82% of TOEFL and native essay pairs they still scored the TOEFL essay as more AI-like. The best commercial detector on that measure was at 84%.
Likely question: are these numbers from 2023? The commercial scores are the archived 2023 ones. We ran Binoculars and Fast-DetectGPT ourselves with their official thresholds."""
NOTES[6] = """Presenter: B | ~1.3 min
Why does this happen? Let us look at the words.
Perplexity-based detectors measure how predictable the wording is. Darker red means a language model could easily predict that word.
On top is the TOEFL essay from the start. It uses common, safe phrases like "I believe" and "looking forward to". Its perplexity is 8.8. Its most predictable sentence, "That's why it's so important to me", has a perplexity of 3.
Below is a native college essay with rarer words. Its perplexity is 21.0.
This is only one pair, so the real evidence is on the right. We do not use the Hewlett essays here: 85 of 88 contain placeholders like @CAPS1, and these inflate perplexity.
Over all essays, GPTZero itself reported a median perplexity of 42 for TOEFL essays and 86 for native college essays. And falcon-7b agrees: 89 percent of TOEFL essays are more predictable than the median native essay.
Simple, correct English tends to be predictable, and these detectors read predictable wording as a sign of AI."""
NOTES[7] = """Presenter: C | ~1.0 min
Example: each essay becomes a long list of counts, such as how often "I believe" or "very important" appears, and the model learns a weight for each one.
Likely question: why count word pairs too? Pairs like "looking forward" carry more style than single words.
Likely question: did you tune it? No. We used the settings from the course notebook, so the training essays are the only thing that changes between sets."""
NOTES[8] = """Presenter: C | ~1.3 min
The slide explains itself. Read the headline, then point at the arrow from 80% to 8%.
Likely question: why is 43% still high? Those native essays are in styles that none of our training sets contain, for example the CS224N essays.
Likely question: does the detector still catch AI essays? Yes. F1 goes from 0.72 with set A to 0.80 with set C. The table is on the backup slides.
Likely question: did you tune the model? No. We kept the settings fixed, so the training essays are the only difference between the sets."""
NOTES[9] = """Presenter: D | ~0.9 min
Example: RoBERTa reads a sentence like "I believe the most important decision I've ever made would be going to engineering school" as a whole, with each word in its context.
Likely question: why only 2 passes over the data? We cut the passes from 4 to 2 when the training data grew to 8,000 essays. All settings are on the backup slide.
Likely question: is deep learning allowed? The instructor approved it verbally as an extension. Logistic regression stays our main model."""
NOTES[10] = """Presenter: D | ~1.2 min
Point at the arrow from 99% to 26%.
Likely question: why does set B help so little? Our reading is that school-age non-native essays are a different kind of writing from TOEFL exam essays.
Likely question: how stable is 26%? It moves by about 8 points between runs.
Limitation to mention if asked: our settings were not tuned, and our seeds only show run-to-run spread."""
NOTES[11] = """Presenter: E | ~1.1 min
Example: LoRA is a bit like fitting a small adapter to a big machine instead of rebuilding the machine.
Likely question: why only 2 runs? Each run took about 4 hours on two GPUs.
Likely question: did you change Qwen itself? We converted its weights from FP8 to bf16 before training, with about 2.7% weight error. The weights stay frozen during training."""
NOTES[12] = """Presenter: E | ~1.3 min
Point at the arrow from 40% to 5%.
Likely question: is Qwen the best model? It catches AI best, F1 0.94 with set C. But with only 2 runs its numbers are less certain: set A gave 56% in one run and 23% in the other.
Likely question: why does set D not bring TOEFL down? Set D adds other native essays; it stays at 35%."""
NOTES[13] = """Presenter: C | ~1.0 min
Point at the purple bars.
Likely question: what are these native essays? College application essays, CS224N student essays and LOCNESS essays from the Liang et al. benchmark and our added data. We show their average.
Likely question: why is Qwen different? Its average for these essays is below its TOEFL rate, but CS224N alone is flagged more than TOEFL: 61% against 40%. Qwen has only 2 runs."""
NOTES[14] = """Presenter: D | ~1.8 min
This is the main result. Point at the three end labels, then at the faded set D dots.
Likely question: is this a general fix? No. Set C was trained on essays like TOEFL, and native essays of other new styles are still flagged. See the backup slide on what set C misses.
Likely question: are your detectors better than the commercial ones? We do not claim that. Set C saw essays like TOEFL, so it is not a fair comparison. The full chart with the commercial range and error bars is on a backup slide.
Likely question: how much does set D help? 47%, 91% and 35%, against 80%, 99% and 40% with set A."""
NOTES[15] = """Presenter: E | ~1.5 min
Go through the four rows quickly; the blue column says what each mistake means.
Likely question: which mistake mattered most? Number 3. Qwen looked perfect, 0% false flags, only because it was tested on essays it had trained on.
Likely question: what did you change for mistake 2? We made every text look the same and added a check that no formatting difference is left. The backup slide lists the change for each mistake.
Likely question: did you redo mistake 1 inside one corpus? No, we did not. We stopped using that model as evidence and tested on seven new data sources instead."""
NOTES[16] = """Presenter: A | ~1.2 min
Read the three recommendations, then the conclusion box.
Likely question: should schools stop using detectors? We do not say that. We say a score should start a review, and the detector should be tested on writing like the students' own.
Likely question: does this hold for 2025 or 2026 detectors? We did not test them. Our human essays are from before ChatGPT, and the AI essays and detector scores are from 2019 to 2023.
The AI-use statement is a draft. [TEAM: confirm the wording before the final version.]"""
NOTES["thanks"] = """Presenter: A | Q&A
Thank you for listening. We are happy to take your questions.
Backup slides, if needed: commercial detectors in detail, the model scoreboard, what arm C does not fix, more training data, essay length, precision/recall/F1, the full table, the GPT-4 polishing example, the three models, and all settings."""
NOTES["refs"] = """Presenter: not presented | reference list
Shown at the end for completeness.
[TEAM: citation details for Fast-DetectGPT, ELLIPSE, ArguGPT, HC-Var and daigt_v4 still need to be found, checked and added. Check the Stowe et al. (2026) entry against the original before the final version.]"""
NOTES["divider"] = """Presenter: any | backup
Backup slides. Show one only if someone asks a question about it."""
NOTES["fig05_trend_before_plainwords"] = """Presenter: any | backup, only if asked
The full version of our main chart. Error bars show the spread across seeds; Qwen has only two seeds, so its arm A result varies a lot.
The grey band is the seven commercial detectors from 2023, 52 to 76 percent. The purple lines are Binoculars, 7 percent, and Fast-DetectGPT, 35 percent.
We do not claim our detectors are better than the commercial ones: arm C was trained on essays like TOEFL, so this is not a like-for-like comparison."""
NOTES["fig15_pitfalls_detail"] = """Presenter: any | backup, only if asked
The same four mistakes with what we changed for each one. We kept the failed runs and re-ran everything; the numbers are from the final runs."""
NOTES["fig01_hook"] = """Presenter: any | backup, only if asked
The full pair from slide 2. Top: the TOEFL essay, flagged by all seven commercial detectors. Bottom: a native US school essay, flagged by none. "@CAPS1" is a name the dataset anonymised. The pair is hand-picked, not a random sample."""
NOTES["fig14_pipeline_before_plainwords"] = """Presenter: any | backup, only if asked
The detailed pipeline. Cleaning made line breaks, quote marks and look-alike letters identical in every text. Duplicates were checked by full text and by the first 200 characters. 452 PERSUADE essays that are also in ELLIPSE were removed."""
NOTES["fig03_opensource_before_plainwords"] = """Presenter: any | backup, only if asked
All three measures per detector. Left: TOEFL essays flagged. Middle: bias AUC, how often a TOEFL essay looks more AI-like than a native essay (0.5 = no bias). Right: how well each detector catches AI essays (0.5 = coin flip). The open-source detectors catch AI far better, but their bias AUC (0.87 and 0.82) is no lower than the best commercial one (0.84)."""
NOTES["fig02_commercial"] = """Presenter: any | backup, only if asked
Commercial detectors in detail. Red bars: share of real TOEFL essays flagged as AI, 52 to 76 percent. Grey bars: native Hewlett essays, 0 to 12 percent.
On the right, the AUC for catching AI essays: 0.44 to 0.60. 0.5 is a coin flip, so these detectors barely tell AI from human essays here."""
NOTES["fig06_scoreboard"] = """Presenter: any | backup, only if asked
Does a bigger model solve the problem? Partly. Qwen catches AI best on the same prompts, AUC 0.99, and it wrongly flags unfamiliar native essays the least, 29 percent.
But with the right data, in arm C, logistic regression flags 8 percent of TOEFL essays. That beats Qwen without the right data: 40 percent in arm A.
Logistic regression trains in seconds on a laptop CPU. Qwen needs about four hours on two GPUs: 2.9 hours of training and 74 minutes of scoring. Qwen has only two seeds."""
NOTES["fig07_notfixed"] = """Presenter: any | backup, only if asked
Arm C fixed TOEFL. Did it fix everything? Grey dots are arm A, coloured dots are arm C.
CollegeEssay and LOCNESS also drop. But CS224N, native essays in a style that is still unseen, is still flagged: 97 percent by logistic regression, 70 percent by RoBERTa and 23 percent by Qwen.
So fixing the data for one kind of writing does not make a detector fair in general."""
NOTES["fig08_learning_backup"] = """Presenter: any | backup, only if asked
Does more training data help? Not really. We trained with 1,000, 2,000 and 4,000 essays per class, with five seeds.
For logistic regression in arm A, the TOEFL false-flag rate even rises, from 70 to 80 percent. RoBERTa stays between 91 and 99 percent.
Changing what the data contains, like in arm B, moves the rate far more than adding more of the same."""
NOTES["fig09_length_backup"] = """Presenter: any | backup, only if asked
Is it just essay length? TOEFL essays are about 104 words, Hewlett essays about 378.
We repeated the bias AUC using only pairs of essays with similar length, within 30 percent of word count.
The open-source detectors drop by 0.06 to 0.08, so length explains a small part. Our self-trained models do not drop at all."""
NOTES["fig10_prf1"] = """Presenter: any | backup, only if asked
Precision, recall and F1 for all detectors on the Liang test set. The commercial detectors reach an F1 of only 0.25 to 0.52. Binoculars and Fast-DetectGPT reach 0.82 and 0.87.
From arm A to arm C, F1 rises for every model: logistic regression 0.72 to 0.80, RoBERTa 0.70 to 0.80, Qwen 0.83 to 0.94, with two seeds.
The last column is the gap between F1 on Hewlett and F1 on TOEFL; arm C makes it smaller for all three models."""
NOTES["fig11_prf1_full"] = """Presenter: any | backup, only if asked
This is the full table: every detector and every arm, mean plus or minus standard deviation over seeds. Qwen has only two seeds."""
NOTES["fig13_polish"] = """Presenter: any | backup, only if asked
This is the same TOEFL essay as on slide 6, after GPT-4 polished it. The polished text comes from Liang et al.
GPT-4 replaced common phrases with rarer ones: "important" became "crucial", and "always been looking forward to since I was a little kid" became "a lifelong dream since my childhood".
The original was flagged by 7 of 7 detectors. The polished version by 0 of 7.
So the detectors react to how plain the wording is, not to who wrote it."""
NOTES["fig16_models"] = """Presenter: any | backup, only if asked
All three models do the same job. An essay goes in. A probability that it is AI comes out. We call it AI at 0.5.
Logistic regression is the classical machine learning from class, and our main model. RoBERTa is deep learning: all 125 million parameters are updated. Qwen has 27 billion parameters; LoRA freezes it and trains a small add-on, 125 million numbers, less than half a percent.
Deep learning is an extension; our instructor approved it verbally.
We did not tune any settings. We fixed them in advance, so the only thing that changes between our training sets is the essays. The full list of settings is in the next backup slide."""
NOTES["fig17_settings"] = """Presenter: any | backup, only if asked
This table lists every setting. None of them was tuned on validation data.
The settings came from the course notebook and from common values.
When our training data grew, we cut the number of passes for RoBERTa from four to two, and for Qwen from three to one.
The validation set of 600 essays was only used for a second threshold that catches 90 percent of AI essays. The main results use 0.5.
We chose not to tune, so that only the training essays differ between arms. The cost is that results could change with tuning."""


# ------------------------------------------------------------------ build
prs = Presentation(TEMPLATE)
sld_lst = prs.slides._sldIdLst
for sid in list(sld_lst):
    prs.part.drop_rel(sid.rId)
    sld_lst.remove(sid)
blank = [l for l in prs.slide_layouts if l.name == "Blank"][0]

BACKUP_ORDER = ["fig01_hook", "fig14_pipeline_before_plainwords", "fig03_opensource_before_plainwords",
                "fig05_trend_before_plainwords", "fig15_pitfalls_detail", "fig02_commercial", "fig06_scoreboard", "fig07_notfixed", "fig08_learning_backup",
                "fig09_length_backup", "fig10_prf1", "fig11_prf1_full", "fig13_polish", "fig16_models",
                "fig17_settings"]
ORDER = (["title"] + list(range(2, 17)) + ["thanks", "refs", "divider"] + ["b:" + b for b in BACKUP_ORDER])
TOTAL = len(ORDER)
SECTION = {"title": "Intro", 2: "Intro", 3: "Intro", 4: "Audit", 5: "Audit", 6: "Audit", 7: "ML", 8: "ML",
           9: "DL", 10: "DL", 11: "LoRA", 12: "LoRA", 13: "Wrap-up", 14: "Wrap-up", 15: "Wrap-up",
           16: "Wrap-up", "thanks": "Wrap-up", "refs": None}

FIGSLIDES = {  # main figure slides: fig, label, takeaway, icon, dl_tag
 6: ("fig12_hotspots", "WHY: PREDICTABLE WORDS", "Which words make an essay look like AI? The ones a language model can predict", "magnifier", False),
}
PLAIN = {  # stand-alone layout = finding headline + figure + "what we did / saw / why it matters"
 2: dict(fig="fig01_hook_toefl", label="ONE ESSAY, SEVEN VERDICTS", dl=False, icon="pc",
         take="A student wrote this TOEFL essay, and seven AI detectors all said it was AI",
         did="We took real essays from the Liang et al. (2023) benchmark and looked at the scores that seven "
             "commercial detectors gave them in 2023.",
         saw="All seven called this 107-word essay AI, while a native student's essay about computers, spelling "
             "mistakes and all, was flagged by none.",
         why="If a detector score can start a misconduct case, a student could be accused over an essay they wrote "
             "themselves.",
         tech="Hand-picked pair, not a random sample: 19 of 91 TOEFL essays were flagged by all seven detectors, 72 of "
              "88 native essays by none. Flagged = score of 0.5 or more. The full pair: backup."),
 4: dict(fig="fig14_pipeline", label="HOW WE BUILT THE TEST", dl=False, icon="folder",
         take="We built a test of 10,171 essays that the detectors never saw in training",
         did="We combined Liang et al.'s 925 essays with seven newer data sources, cleaned every text the same way "
             "and removed duplicates.",
         saw="That left 10,171 test essays with no overlap with training, and four training sets (A to D) of 8,000 "
             "essays each.",
         why="Because no test essay was used in training, a detector cannot pass our test by remembering essays.",
         tech="New sources: PERSUADE 2.0, ELLIPSE, daigt_v4, HC-Var, W&I + LOCNESS, FCE, ArguGPT. Main measures were "
              "written down before the GPU runs. Step details: backup."),
 5: dict(fig="fig03_opensource", label="THE PROBLEM IS REAL", dl=False, icon=None,
         take="Newer open-source detectors flag fewer TOEFL essays, but still find them more AI-like than native essays",
         did="We took the 2023 scores of seven commercial detectors on the Liang benchmark and ran two 2024 "
             "open-source detectors, Binoculars and Fast-DetectGPT, on the same essays.",
         saw="The commercial detectors flagged 52% to 76% of real TOEFL essays and the open-source ones 7% and 35%.",
         why="Even the newer tools scored the TOEFL essay as more AI-like in 87% and 82% of TOEFL and native pairs, so "
             "the problem is smaller but still there.",
         tech="Threshold 0.5 for commercial detectors; official thresholds for the open-source ones, no training on our "
              "data. The open-source ones also catch AI essays far better (AUC 0.93 and 0.94). Detail: backup."),
 7: dict(fig="fig18_ml_method", label="ML: HOW IT WORKS", dl=False, icon="keyboard",
         take="Our main model, logistic regression, counts words and word pairs",
         did="We turned each essay into counts of words and word pairs (20,000 features) and trained logistic "
             "regression to give each one a weight.",
         saw="It trains in seconds on a laptop, and we can read which words push an essay toward AI.",
         why="It is the method from our course and our main model, so the two deep-learning models are compared "
             "against it.",
         tech="8,000 essays per training set (4,000 human, 4,000 AI), 5 runs each; course-notebook settings, not tuned."),
 8: dict(fig="fig19_ml_results", label="ML: RESULTS", dl=False, icon=None,
         take="Trained only on native essays, the detector calls 8 in 10 TOEFL essays AI",
         terms=["False flag = a real essay, written by a person, that the detector calls AI.",
                "Training sets = A: native only. B: + school-age non-native. C: + exam-style non-native. D: + other native."],
         did="We trained a word-counting detector (logistic regression) four times, once on each training set, and "
             "tested it on the same real essays.",
         saw="With set A it flagged 80% of TOEFL essays. With set C it flagged 8%, but still 43% of native essays in "
             "a new style.",
         why="A student can be flagged for an essay they really wrote, just because the detector never learned from "
             "essays like theirs.",
         tech="Threshold 0.5, mean of 5 runs. New-style native = CollegeEssay, CS224N, LOCNESS. Catching AI essays "
              "(F1): 0.72 with set A, 0.80 with set C (backup)."),
 9: dict(fig="fig20_dl_method", label="DL: HOW IT WORKS", dl=True, icon="pc",
         take="Our deep-learning model, RoBERTa, reads whole sentences instead of counting words",
         did="We took RoBERTa, a language model already trained on a lot of English, and retrained all 125 million "
             "of its numbers on our essays.",
         saw="One training run took about 4 minutes on one GPU, and we ran it 5 times for each training set.",
         why="It shows what reading the context adds, but it cannot tell us which words pushed its score.",
         tech="roberta-base; learning rate 2e-5, batch 16, up to 512 word pieces, 2 passes; settings not tuned. The "
              "instructor approved deep learning verbally."),
 10: dict(fig="fig21_dl_results", label="DL: RESULTS", dl=True, icon=None,
          take="RoBERTa calls almost every TOEFL essay AI until it learns from exam essays",
          did="We trained RoBERTa on each training set (A to D) and tested it on the same real TOEFL and native essays.",
          saw="With set A it flagged 99% of TOEFL essays. With exam-style essays added (set C) it flagged 26%, and 45% "
              "of new-style native essays.",
          why="A bigger model still makes this mistake when its training essays do not match the writing it has to "
              "judge.",
          tech="Threshold 0.5, mean of 5 runs; the set C number moves by about 8 points between runs. Catching AI "
               "essays (F1): 0.70 → 0.80 (backup)."),
 11: dict(fig="fig22_lora_method", label="LORA: HOW IT WORKS", dl=True, icon="cd",
          take="Qwen is too big to retrain, so we froze it and trained a small add-on called LoRA",
          did="We kept all 27 billion numbers in Qwen fixed and trained only an add-on of 125 million numbers, less "
              "than half a percent of the model.",
          saw="One run took about 4 hours on two GPUs, so we could only afford 2 runs per training set.",
          why="It let us test a large language model on the two GPUs we had, but 2 runs make its results less certain.",
          tech="Qwen3.8-27B; LoRA rank 16, alpha 32, learning rate 1e-4; settings not tuned. The instructor approved "
               "deep learning verbally."),
 12: dict(fig="fig23_lora_results", label="LORA: RESULTS", dl=True, icon=None,
          take="Qwen starts with fewer false flags, and exam essays bring it down to 1 in 20",
          did="We trained Qwen with LoRA on each training set (A to D), twice each, and tested it on the same real "
              "essays.",
          saw="With set A it flagged 40% of TOEFL essays. With set C it flagged 5%, and 9% of new-style native essays.",
          why="The large model shows the same pattern as the smaller ones, but its two runs differ a lot (56% and 23% "
              "with set A).",
          tech="Threshold 0.5, mean of 2 runs. Catching AI essays (F1): 0.83 → 0.94 (backup)."),
 13: dict(fig="fig04_shift", label="ALL THREE: UNSEEN WRITING", dl=True, icon=None,
          take="Native writers get flagged too when their style is new to the detector",
          did="We trained all three detectors on US school essays only (set A) and tested them on other native essays "
              "(college applications, CS224N, LOCNESS) and on TOEFL essays.",
          saw="The word-counting model flagged 85% of these native essays, about as often as TOEFL essays (80%). "
              "RoBERTa flagged 69% and Qwen 29%.",
          why="The detectors react to writing they have not seen, so being a non-native writer is only part of the "
              "problem.",
          tech="Threshold 0.5. New-style native = mean of CollegeEssay, CS224N, LOCNESS. Qwen has 2 runs; for it, "
               "CS224N alone is flagged more than TOEFL (61% vs 40%)."),
 14: dict(fig="fig05_trend", label="THE FIX: A → B → C", dl=True, icon="progress",
          take="Adding exam-style essays to training cut TOEFL false flags in all three models",
          terms=["False flag = a real essay, written by a person, that the detector calls AI.",
                 "Training sets = A: native only. B: + school-age non-native. C: + exam-style non-native. D: + other native."],
          did="We trained each of the three detectors on sets A, B, C and D and counted how many real TOEFL essays "
              "it called AI.",
          saw="From set A to set C, false flags fell from 80% to 8%, from 99% to 26% and from 40% to 5%. Set D "
              "helped much less.",
          why="The essays a detector learns from change who it wrongly flags, so schools should test it on their own "
              "students' writing first.",
          tech="Threshold 0.5; Qwen has 2 runs. Set C was trained on essays like TOEFL, so it does not fix other new "
               "styles. Full chart with error bars: backup."),
 15: dict(fig="fig15_pitfalls", label="FOUR MISTAKES WE FIXED", dl=False, icon="warning",
          take="We found four mistakes in our own tests and re-ran everything after fixing them",
          did="Before trusting our numbers we went back over our own pipeline, found four mistakes and kept the "
              "failed runs as a record.",
          saw="Each one changed a key number. Qwen went from 0% to 40% of TOEFL essays flagged once we stopped "
              "testing it on its own training essays.",
          why="A detector can look good for the wrong reason, so anyone testing one should check how the test was "
              "built.",
          tech="Sources: archived first runs and final runs (sets A and B, threshold 0.5). 0.888 was estimated from "
               "per-corpus flag rates. What we changed for each mistake: backup."),
}
TERM_SIZE, TERM_LH, TECH_SIZE, TECH_LH = 20, 24, 14, 17
ROW_SIZE, ROW_LH, ROW_LABEL_W = 21, 25, 190

BACKUPS = {
 "fig01_hook": ("BACKUP: THE ESSAY PAIR", "The TOEFL essay and the native essay side by side, scored by the same seven detectors"),
 "fig14_pipeline_before_plainwords": ("BACKUP: PIPELINE DETAILS", "How the experiment was built: from raw essays to a test with no overlap"),
 "fig03_opensource_before_plainwords": ("BACKUP: DETECTORS IN DETAIL", "Open-source detectors from 2024 are far better, but still flag TOEFL more"),
 "fig15_pitfalls_detail": ("BACKUP: MISTAKES AND FIXES", "Four mistakes we made, what each one means and what we changed"),
 "fig05_trend_before_plainwords": ("BACKUP: FULL TREND CHART", "Training on the right kind of essays cuts TOEFL false flags, in all three models"),
 "fig02_commercial": ("BACKUP: COMMERCIAL 2023", "2023 commercial detectors flag TOEFL essays, and barely spot AI"),
 "fig06_scoreboard": ("BACKUP: SCOREBOARD", "Bigger models detect better, but the right training data matters more than size"),
 "fig07_notfixed": ("BACKUP: WHAT C MISSES", "Exam-style data fixes TOEFL and helps elsewhere, but not everywhere"),
 "fig08_learning_backup": ("BACKUP: MORE DATA?", "More training data barely helps; the kind of data does"),
 "fig09_length_backup": ("BACKUP: ESSAY LENGTH", "Essay length explains only a small part of the TOEFL bias"),
 "fig10_prf1": ("BACKUP: P/R/F1 SUMMARY", "Precision, recall and F1: the right training data lifts every model"),
 "fig11_prf1_full": ("BACKUP: FULL P/R/F1", "Accuracy, precision, recall and F1 for every detector and arm"),
 "fig13_polish": ("BACKUP: GPT-4 POLISHING", "The same essay, polished by GPT-4: from 7 of 7 detectors to 0 of 7"),
 "fig16_models": ("BACKUP: THREE MODELS", "How we used machine learning, deep learning and LoRA"),
 "fig17_settings": ("BACKUP: ALL SETTINGS", "Every setting, and what we did not tune"),
}
DIVIDER_ITEMS = ["Essay pair", "Pipeline", "Detectors", "Full trend", "Mistakes, fixes", "Commercial", "Scoreboard", "What C misses", "More data?", "Essay length",
                 "P/R/F1", "Full table", "GPT-4 polish", "3 models", "Settings"]

crop_log = {}


def get_crop(name):
    if name not in FIG:
        FIG[name] = crop_figure(name)
        crop_log[name] = FIG[name][2]
    return FIG[name]


def figure_slide(slide, figname, label, take, icon, extra_bottom=0):
    header(slide, prs, label, take, icon)
    fn = FOOT[figname]
    fh = foot_height(fn)
    ftop = FOOT_B - fh - extra_bottom
    footnote(slide, fn, ftop)
    figure(slide, get_crop(figname), FIG_TOP, ftop - 10)
    return ftop


def plain_slide(slide, spec):
    """Stand-alone slide: finding headline, optional terms strip, figure, then three labelled one-sentence lines
    (what we did / what we saw / why it matters, >= 20 pt) and a 14 pt technical line."""
    header(slide, prs, spec["label"], spec["take"], spec["icon"])
    w = CX1 - CX0
    tech_h = nlines(spec["tech"], w - 4, TECH_SIZE) * TECH_LH
    tech_top = FOOT_B - tech_h
    text(slide, CX0, tech_top, w, tech_h, spec["tech"], size=TECH_SIZE, color=NOTE_INK, lh=TECH_LH, name="technical note")
    rows = [("What we did", spec["did"]), ("What we saw", spec["saw"]), ("Why it matters", spec["why"])]
    tw = w - ROW_LABEL_W - 36
    heights = [nlines(t, tw - 6, ROW_SIZE) * ROW_LH for _, t in rows]
    gap = 9
    box_h = sum(heights) + gap * (len(rows) - 1) + 20
    box_top = tech_top - 8 - box_h
    inner = bevel(slide, CX0, box_top, w, box_h, b=2.0, raised=False, outline=False, face="F4F4F4")
    y = inner[1] + 8
    for (lab, t), h in zip(rows, heights):
        text(slide, inner[0] + 14, y, ROW_LABEL_W, ROW_LH, [[(lab, {"bold": True, "color": BLUE})]], size=20, lh=ROW_LH)
        text(slide, inner[0] + 14 + ROW_LABEL_W, y, tw, h, t, size=ROW_SIZE, lh=ROW_LH, name=lab)
        y += h + gap
    fig_top = FIG_TOP
    if spec.get("terms"):
        n = sum(nlines(t, w - 24, TERM_SIZE) for t in spec["terms"])
        th = n * TERM_LH + 10
        rect(slide, CX0, FIG_TOP - 2, w, th, fill="F4F4F4")
        paras = []
        for t in spec["terms"]:
            k, _, v = t.partition(" = ")
            paras.append([(k + ": ", {"bold": True, "color": BLUE}), v])
        text(slide, CX0 + 12, FIG_TOP + 3, w - 24, th - 6, paras, size=TERM_SIZE, lh=TERM_LH, name="terms")
        fig_top = FIG_TOP + th + 6
    figure(slide, get_crop(spec["fig"]), fig_top, box_top - 8)


def add_notes(slide, key):
    slide.notes_slide.notes_text_frame.text = NOTES[key]


for idx, key in enumerate(ORDER):
    slide = prs.slides.add_slide(blank)
    frac = idx / (TOTAL - 1)
    if isinstance(key, str) and key.startswith("b:"):
        name = key[2:]
        chrome(slide, prs, "Backup", frac, extra_tab="Backup", status="BACKUP - only if asked")
        label, take = BACKUPS[name]
        figure_slide(slide, name, label, take, None)
        add_notes(slide, name)
        continue
    if key == "divider":
        chrome(slide, prs, "Backup", frac, extra_tab="Backup", status="BACKUP - only if asked")
        text(slide, 120, 250, 1200, 190, "BACKUP", size=180, font="anon", color=BLUE, align="c", anchor="m",
             spc=-6, name="headline")
        text(slide, 170, 460, 1100, 44, "Extra slides, shown only if someone asks", size=28, align="c",
             spc=1.5, name="subtitle")
        items = DIVIDER_ITEMS
        bwid, gap, per_row = 200, 16, 6
        for i, it in enumerate(items):
            row = i // per_row
            n_row = min(per_row, len(items) - row * per_row)
            x0 = (SW - (n_row * bwid + (n_row - 1) * gap)) / 2
            inner = bevel(slide, x0 + (i % per_row) * (bwid + gap), 546 + row * 70, bwid, 54, b=3.0)
            text(slide, inner[0], inner[1], inner[2], inner[3], it, size=20, bold=True, align="c", anchor="m")
        picture(slide, prs, "folder", 150, 262, h=120)
        picture(slide, prs, "cd", 1180, 262, h=120)
        add_notes(slide, "divider")
        continue
    chrome(slide, prs, SECTION.get(key), frac,
           status=(DL_TAG if ((key in FIGSLIDES and FIGSLIDES[key][4]) or (key in PLAIN and PLAIN[key]["dl"]))
                   else None))
    if key == "title":
        text(slide, 120, 186, 1200, 40, "Auditing AI-text detectors for bias against non-native English writers",
             size=24, align="c", name="kicker")
        text(slide, 60, 236, 1320, 270, ["DETECTING THE", "DETECTOR"], size=128, font="anon", color=BLUE,
             align="c", anchor="m", lh=128, spc=-4, name="headline")
        text(slide, 100, 520, 1240, 44, "Would an AI-text detector believe you wrote your own essay?",
             size=29, align="c", spc=0.5, name="question")
        bw_, bh_ = 420, 66
        inner = bevel(slide, (SW - bw_) / 2, 612, bw_, bh_, b=4.0)
        text(slide, inner[0], inner[1], inner[2], inner[3], "INT4096 | Group 08", size=26, bold=True,
             align="c", anchor="m", name="button")
        picture(slide, prs, "cursor", (SW + bw_) / 2 - 26, 650, h=62)
        add_notes(slide, 1)
        continue
    if key == "thanks":
        text(slide, 100, 230, 1240, 220, "THANK YOU", size=190, font="anon", color=BLUE, align="c", anchor="m",
             spc=-6, name="headline")
        bw_, bh_ = 400, 74
        inner = bevel(slide, (SW - bw_) / 2, 500, bw_, bh_, b=4.0)
        text(slide, inner[0], inner[1], inner[2], inner[3], "Questions?", size=32, bold=True, align="c",
             anchor="m", name="button")
        picture(slide, prs, "cursor", (SW + bw_) / 2 - 30, 542, h=66)
        add_notes(slide, "thanks")
        continue
    if key == "refs":
        header(slide, prs, "REFERENCES", None, None)
        picture(slide, prs, "keyboard", CX1 - 230, BOX_Y - 4, h=64)
        picture(slide, prs, "pc", CX1 - 70, BOX_Y - 10, h=74)
        refs = [
            "Crossley, Baffour, Tian, Franklin, Benner & Boser (2024). A large-scale corpus for assessing written argumentation: PERSUADE 2.0. Assessing Writing, 61.",
            "Granger (1998). LOCNESS. CECL, Université catholique de Louvain.",
            "Hans et al. (2024). Spotting LLMs With Binoculars. ICML 2024 (PMLR 235). arXiv:2401.12070.",
            "Hu et al. (2021). LoRA. arXiv:2106.09685.",
            "Liang, Yuksekgonul, Mao, Wu & Zou (2023). GPT detectors are biased against non-native English writers. Patterns, 4(7), 100779. doi:10.1016/j.patter.2023.100779",
            "Liang et al. (2023). Data set. Zenodo. doi:10.5281/zenodo.7893958",
            "Liu et al. (2019). RoBERTa. arXiv:1907.11692.",
            "Stowe et al. (2026). Identifying Bias in Machine-generated Text Detection. arXiv:2512.09292.",
            "Weber-Wulff et al. (2023). Testing of detection tools for AI-generated text. Int. J. for Educational Integrity, 19(1), 26. doi:10.1007/s40979-023-00146-z",
            "Yannakoudakis, Briscoe & Medlock (2011). FCE corpus. ACL.",
            "Yannakoudakis, Andersen, Geranpayeh, Briscoe & Nicholls (2018). Write & Improve. Applied Measurement in Education, 31(3).",
        ]
        text(slide, CX0 + 6, 208, CX1 - CX0 - 6, 556, refs, size=22, lh=26, space_after=9, marL=34, indent=-34,
             name="references")
        add_notes(slide, "refs")
        continue
    if key in PLAIN:
        plain_slide(slide, PLAIN[key])
        add_notes(slide, key)
        continue
    if key in FIGSLIDES:
        figname, label, take, icon, _dl = FIGSLIDES[key]
        figure_slide(slide, figname, label, take, icon)
        add_notes(slide, key)
        continue
    if key == 4:
        header(slide, prs, "HOW WE BUILT THE TEST", "From raw essays to a test with no overlap", "folder")
        # "What is new" strip (from slide03_novelty.md) instead of the figure footnote
        cols = [("+7 data sources", "925 → 10,171 held-out test essays"),
                ("3 model families", "trained on identical data"),
                ("A/B/C/D design", "only the human essays change"),
                ("New comparison", "non-native vs native (not ESL vs corrected)")]
        ph = 92
        ptop = FOOT_B - ph
        inner = bevel(slide, CX0, ptop, CX1 - CX0, ph, b=2.0, raised=False, outline=False, face="EFEFEF")
        text(slide, inner[0] + 14, inner[1], 140, inner[3], [[("What is", {"bold": True})], [("new?", {"bold": True})]],
             size=24, color=BLUE, anchor="m", lh=27)
        cw = (inner[2] - 150) / 4
        for i, (h1, h2) in enumerate(cols):
            text(slide, inner[0] + 150 + i * cw, inner[1] + 8, cw - 14, inner[3] - 12,
                 [[(h1, {"bold": True})], h2], size=20, lh=23)
        figure(slide, get_crop("fig14_pipeline"), FIG_TOP, ptop - 10)
        add_notes(slide, 4)
        continue
    if key == 3:
        header(slide, prs, "WHY THIS TOPIC",
               "Detector scores can start misconduct cases, and they may not treat every student the same", None)
        # left: why it matters
        lx, lw = CX0 + 4, 676
        text(slide, lx, 200, lw, 34, "Why it matters", size=26, bold=True, name="col head")
        y = 244
        reasons = [
            ("1", "High stakes: a detector score can start a misconduct case against a student.",
             "[TEAM: confirm + add an example you know]"),
            ("2", "It affects people like us.", "[TEAM: add your own reason]"),
            ("3", "Old evidence: 61.9% of TOEFL vs 5.4% of native essays flagged (our count, Liang et al.'s 2023 scores). The cause was never tested.", None),
            ("4", "It fits this course: ML that supports decisions about students must be tested for bias.", None),
        ]
        for num, body, ph_ in reasons:
            n = nlines(body, lw - 40, 22)
            text(slide, lx, y, 34, 30, [[(num, {"bold": True, "color": BLUE})]], size=22)
            text(slide, lx + 38, y, lw - 38, n * 26, body, size=22, lh=26)
            y += n * 26 + 4
            if ph_:
                pw_ = len(ph_) * 0.6 * 20 + 20
                text(slide, lx + 38, y, pw_, 30, [[(ph_, {"bold": True})]], size=20, fill=YELLOW, line=INK,
                     dash="dash", anchor="m", inset=4, name="TEAM placeholder")
                y += 34
            y += 12
        # right: research questions, in a sunken panel
        rx, rw = 772, CX1 - 772
        inner = bevel(slide, rx, 198, rw, 390, b=2.0, raised=False, outline=False, face="F4F4F4")
        text(slide, inner[0] + 18, inner[1] + 10, inner[2] - 30, 34, "Our three questions", size=26, bold=True)
        rqs = [("RQ1", "Do detectors flag non-native exam essays more than native ones, and are newer open-source detectors better?"),
               ("RQ2", "Is that bias about the writer's background, or about writing the detector has not seen?"),
               ("RQ3", "Can training data fix it, and does a bigger model help?")]
        yy = inner[1] + 58
        for tag, q in rqs:
            n = nlines(q, inner[2] - 108, 24)
            text(slide, inner[0] + 18, yy, 70, 30, [[(tag, {"bold": True, "color": BLUE})]], size=24)
            text(slide, inner[0] + 88, yy, inner[2] - 104, n * 28, q, size=24, lh=28)
            yy += n * 28 + 20
        # answer-first dialog
        dy, dh = 606, 156
        rect(slide, CX0, dy, CX1 - CX0, dh, fill=BAR, line=INK)
        rect(slide, CX0, dy, CX1 - CX0, 34, fill=BLUE, line=INK)
        text(slide, CX0 + 12, dy, 600, 34, "Short answer", size=20, bold=True, color="F2F2F2", anchor="m")
        picture(slide, prs, "close", CX1 - 32, dy + 4, w=26, h=26)
        picture(slide, prs, "info", CX0 + 24, dy + 34 + (dh - 34) / 2 - 35, h=70)
        ans = ("Yes, detectors flag non-native exam essays more than native ones, mainly because they have not "
               "seen that kind of writing. Training on the right essays cuts false flags in all three models, "
               "but not everywhere.")
        text(slide, CX0 + 112, dy + 34, CX1 - CX0 - 136, dh - 34, ans, size=24, bold=True, anchor="m", lh=29,
             name="answer first")
        add_notes(slide, 3)
        continue
    if key == 16:
        header(slide, prs, "WHAT WE RECOMMEND",
               "Detector scores can help teachers, but the decision should stay with the teacher", None)
        tile_s = 150
        picture(slide, prs, "tile", CX0 + 10, 206, w=tile_s, h=tile_s)
        picture(slide, prs, "bulb", CX0 + 10 + 22, 206 + 12, h=tile_s - 24)
        recs = ["Use a high score to start a review, and look for other evidence.",
                "Test a detector on your own students' writing before using it.",
                "Check that its training essays match the writing it will judge."]
        yy = 214
        for i, r_ in enumerate(recs):
            text(slide, 244, yy, 50, 36, [[(str(i + 1), {"bold": True, "color": BLUE})]], size=25)
            text(slide, 290, yy, CX1 - 290, 36, r_, size=25, name="rec")
            yy += 50
        # conclusion dialog
        dy, dh = 392, 140
        rect(slide, CX0, dy, CX1 - CX0, dh, fill=BAR, line=INK)
        rect(slide, CX0, dy, CX1 - CX0, 32, fill=BLUE, line=INK)
        text(slide, CX0 + 12, dy, 600, 32, "Conclusion", size=20, bold=True, color="F2F2F2", anchor="m")
        picture(slide, prs, "close", CX1 - 30, dy + 4, w=24, h=24)
        concl = ("The detectors we tested flag writing they have not seen. Adding the right kind of essays to training "
                 "cut false flags in all three models, but other new styles were still flagged.")
        text(slide, CX0 + 20, dy + 32, CX1 - CX0 - 40, dh - 32, concl, size=25, bold=True, anchor="m", lh=30,
             name="conclusion")
        lead_in = "Scope and limits: "
        scope = (lead_in + "human essays written before ChatGPT (30 Nov 2022); AI essays and detector scores from "
                 "the 2019-2023 generation; 2025-26 systems not tested. Settings not tuned; Qwen: 2 seeds.")
        sh_ = nlines(scope, CX1 - CX0, 20) * 23
        text(slide, CX0, dy + dh + 16, CX1 - CX0, sh_, [[(lead_in, {"bold": True}), scope[len(lead_in):]]],
             size=20, color=NOTE_INK, lh=23, name="scope")
        ai = ("We used AI assistants (Claude via Claude Code, and OpenCode) to help plan experiments, write "
              "analysis code and draw figures. The team reviewed and checked all results and is responsible for "
              "the content. Qwen, RoBERTa and the open-source detectors are the subjects of our experiments, "
              "not authors.")
        ay = dy + dh + 16 + sh_ + 18
        ai_h = 42 + nlines(ai, CX1 - CX0 - 36, 21) * 25 + 16
        inner = bevel(slide, CX0, ay, CX1 - CX0, ai_h, b=2.0, raised=False, outline=False, face="F4F4F4")
        tag = "(draft - team to confirm wording)"
        text(slide, inner[0] + 14, inner[1] + 8, 360, 28, [[("AI-use statement", {"bold": True})]], size=20)
        text(slide, inner[0] + 14 + 230, inner[1] + 6, len(tag) * 12 + 16, 28, [[(tag, {"bold": True})]], size=20,
             fill=YELLOW, line=INK, dash="dash", anchor="m", inset=3, name="TEAM placeholder")
        text(slide, inner[0] + 14, inner[1] + 42, inner[2] - 28, inner[3] - 46, ai, size=21, lh=25,
             name="AI statement")
        add_notes(slide, 16)
        continue
    raise SystemExit("unhandled slide %r" % key)

# replace the template's thumbnail (it shows the template's own title) with a render of our title slide, if provided
thumb = os.path.join(WORK, "thumb.jpeg")
if os.path.exists(thumb):
    for part in prs.part.package.iter_parts():
        if str(part.partname).endswith("thumbnail.jpeg"):
            with open(thumb, "rb") as f:
                part._blob = f.read()

import datetime  # noqa: E402
cp = prs.core_properties
cp.title = "Detecting the Detector - INT4096 Group 08 (draft 18 Nov 2026)"
cp.subject = "Are AI-text detectors biased against non-native English writers?"
cp.author = "INT4096 Group 08"
cp.last_modified_by = "INT4096 Group 08"
cp.revision = 2
cp.modified = datetime.datetime(2026, 11, 18, 0, 0, 0)
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
for k, v in crop_log.items():
    print("crop", k, v)
