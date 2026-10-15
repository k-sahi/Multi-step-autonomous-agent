"""Render docs/question_bank/questions.py to Question_Bank.pdf.

  python docs/question_bank/build_pdf.py [output.pdf]
"""
import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

sys.path.insert(0, str(Path(__file__).parent))
from questions import FORMULAS, NOTES, SECTIONS  # noqa: E402

INK = colors.HexColor("#1d2433")
MUTED = colors.HexColor("#5b6475")
ACCENT = colors.HexColor("#0f6e6e")
RULE = colors.HexColor("#d9dee6")
SOFT = colors.HexColor("#f2f5f7")
LEVEL = {"E": ("Easy", colors.HexColor("#2f7d4a")), "M": ("Medium", colors.HexColor("#b26a00")),
         "H": ("Hard", colors.HexColor("#b03a2e"))}

S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=28, leading=34, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=13, leading=18, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=22, textColor=ACCENT,
                         spaceBefore=4, spaceAfter=4),
    "intro": ParagraphStyle("intro", fontName="Helvetica-Oblique", fontSize=9.5, leading=13, textColor=MUTED,
                            spaceAfter=10),
    "q": ParagraphStyle("q", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=INK),
    "a": ParagraphStyle("a", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK, alignment=TA_LEFT),
    "fu": ParagraphStyle("fu", fontName="Helvetica-Oblique", fontSize=9, leading=12.5, textColor=MUTED,
                         leftIndent=10, spaceBefore=3),
    "ref": ParagraphStyle("ref", fontName="Helvetica", fontSize=8, leading=10, textColor=MUTED, spaceBefore=3),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14.5, textColor=INK, spaceAfter=6),
    "toc1": ParagraphStyle("toc1", fontName="Helvetica", fontSize=10.5, leading=17, textColor=INK, leftIndent=0),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=12, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=INK),
}


def md(text: str) -> str:
    """Escape, then apply **bold** and `code` markup."""
    t = html.escape(text, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r'<font face="Courier" size="8.5" color="#0f4f4f">\1</font>', t)
    return t


class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(path, pagesize=LETTER, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                         topMargin=0.8 * inch, bottomMargin=0.75 * inch, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=lambda c, d: None),
                               PageTemplate("body", [frame], onPage=self._decorate)])

    def _decorate(self, c, d):
        c.saveState()
        c.setFont("Helvetica", 8)
        c.setFillColor(MUTED)
        c.drawString(self.leftMargin, LETTER[1] - 0.5 * inch, "Guarded Multi-Step Support Agent  |  Interview question bank")
        c.drawRightString(LETTER[0] - self.rightMargin, 0.45 * inch, f"{d.page}")
        c.setStrokeColor(RULE)
        c.line(self.leftMargin, LETTER[1] - 0.56 * inch, LETTER[0] - self.rightMargin, LETTER[1] - 0.56 * inch)
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name == "h1":
            key = f"s{id(f)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(f.getPlainText(), key, level=0)
            self.notify("TOCEntry", (0, f.getPlainText(), self.page, key))


def pill(level):
    name, col = LEVEL[level]
    return f'<font color="{col.hexval().replace("0x", "#")}" size="8"><b>{name.upper()}</b></font>'


def qa_block(num, item):
    level, q, a, fu, ref = item
    parts = [Paragraph(f'<font color="#0f6e6e">Q{num}</font>&nbsp;&nbsp;{md(q)}&nbsp;&nbsp;{pill(level)}', S["q"]),
             Spacer(1, 3), Paragraph(md(a), S["a"])]
    if fu:
        parts.append(Paragraph("<b>Follow-up:</b> " + md(fu), S["fu"]))
    if ref:
        parts.append(Paragraph("In the repo: " + md(f"`{ref}`"), S["ref"]))
    parts.append(Spacer(1, 11))
    return KeepTogether(parts)


def cover(story):
    total = sum(len(s["items"]) for s in SECTIONS)
    counts = {k: sum(1 for s in SECTIONS for i in s["items"] if i[0] == k) for k in "EMH"}
    story += [Spacer(1, 1.6 * inch),
              Paragraph("Guarded Multi-Step Support Agent", S["title"]), Spacer(1, 8),
              Paragraph("Interview question bank with model answers", S["subtitle"]), Spacer(1, 26)]
    rows = [["Questions", str(total)], ["Sections", str(len(SECTIONS))],
            ["Difficulty", f"{counts['E']} easy  /  {counts['M']} medium  /  {counts['H']} hard"],
            ["Covers", "ReAct agents, LangGraph state and interrupts, MCP tools, guardrails, human sign-off, "
                       "tau2-bench, pass^k, tool accuracy, escalation, trace debugging, Temporal, Cloud Run and Arize"]]
    t = Table([[Paragraph(r[0], S["cellb"]), Paragraph(r[1], S["cell"])] for r in rows],
              colWidths=[1.2 * inch, 4.9 * inch])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -2), 0.5, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [t, Spacer(1, 30),
              Paragraph("<b>How to use this.</b> Cover the answer and say yours out loud first, then compare. "
                        "Model answers are written to be spoken in 30-90 seconds. Follow-ups are what a good "
                        "interviewer asks next. 'In the repo' points to the code that backs the answer, so you can "
                        "cite it. No benchmark numbers are claimed: "
                        "fill them in from your own tau2 runs before an interview.", S["body"]),
              NextPageTemplate("body"), PageBreak()]


def toc(story):
    story.append(Paragraph("Contents", ParagraphStyle("h1toc", parent=S["h1"])))
    t = TableOfContents()
    t.levelStyles = [S["toc1"]]
    t.dotsMinLevel = 0
    story += [t, PageBreak()]


def formulas(story):
    story.append(Paragraph("Appendix: formula sheet", S["h1"]))
    data = [[Paragraph("Metric", S["cellb"]), Paragraph("Definition", S["cellb"])]]
    data += [[Paragraph(k, S["cellb"]), Paragraph(v, S["cell"])] for k, v in FORMULAS]
    t = Table(data, colWidths=[1.6 * inch, 5.1 * inch], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), SOFT), ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [t, Spacer(1, 18)] + [Paragraph(f"<b>{k}.</b> {v}", S["body"]) for k, v in NOTES]


def build(out: Path):
    doc = Doc(str(out), title="Guarded Multi-Step Support Agent - Question Bank", author="supportagent")
    story = []
    cover(story)
    toc(story)
    n = 0
    for sec in SECTIONS:
        story += [Paragraph(sec["title"], S["h1"]), Paragraph(md(sec["intro"]), S["intro"])]
        for item in sec["items"]:
            n += 1
            story.append(qa_block(n, item))
        story.append(PageBreak())
    formulas(story)
    doc.multiBuild(story)
    return n


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "Question_Bank.pdf"
    print(f"{build(out)} questions -> {out}")
