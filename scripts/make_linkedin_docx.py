"""Turn linkedin/article.html into a Word document for review.

Reads:   linkedin/article.html, linkedin/images/*.png
Writes:  linkedin/Kanto by the Numbers - LinkedIn article.docx

A4, 2.54 cm margins, Georgia.
Images sit inline with their captions; publishing notes go at the end.

Usage:  ~/.venvs/pokemon-ml/bin/python scripts/make_linkedin_docx.py
"""
from html.parser import HTMLParser
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "linkedin" / "article.html"
OUT = ROOT / "linkedin" / "Kanto by the Numbers - LinkedIn article.docx"
INK, GREY, RED = RGBColor(0x16, 0x15, 0x14), RGBColor(0x4A, 0x48, 0x44), RGBColor(0x9B, 0x1C, 0x1C)
FONT = "Georgia"


def set_font(style, size, bold=False, italic=False, color=INK):
    style.font.name = FONT
    style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    style.font.size, style.font.bold, style.font.italic, style.font.color.rgb = Pt(size), bold, italic, color


def add_hyperlink(paragraph, url, text):
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink"); link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r"); rpr = OxmlElement("w:rPr")
    for tag, val in (("w:color", "9B1C1C"), ("w:u", "single")):
        e = OxmlElement(tag); e.set(qn("w:val"), val); rpr.append(e)
    run.append(rpr); t = OxmlElement("w:t"); t.text = text; t.set(qn("xml:space"), "preserve"); run.append(t)
    link.append(run); paragraph._p.append(link)


class Builder(HTMLParser):
    """Walks the article's simple HTML: h1, h2, p, figure/img/figcaption, ol/ul/li,
    strong, em, a, code. Everything inside <main> and the notes section."""

    def __init__(self, doc):
        super().__init__()
        self.doc, self.p, self.fmt, self.href = doc, None, set(), None
        self.list_style, self.on, self.in_caption, self.skip = None, False, False, 0
        self.refs, self.ref_n = False, 0

    def new_para(self, style=None):
        self.p = self.doc.add_paragraph(style=style)
        return self.p

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("main", "section"): self.on = True; return
        if not self.on: return
        if tag in ("style", "script", "head"): self.skip += 1
        elif tag == "h1": self.new_para("Title")
        elif tag == "h2": self.new_para("Heading 1")
        elif tag == "p": self.new_para("Caption" if "refs" in a.get("class", "") and False else None)
        elif tag == "ol":
            self.refs = "refs" in a.get("class", ""); self.ref_n = 0
            self.list_style = None if self.refs else "List Number"
        elif tag == "ul": self.list_style = "List Bullet"
        elif tag == "li":
            self.new_para(self.list_style)
            if self.refs:
                self.ref_n += 1; self.p.add_run(f"{self.ref_n}. ")
                self.p.paragraph_format.left_indent = Cm(.6); self.p.paragraph_format.first_line_indent = Cm(-.6)
        elif tag == "img":
            p = self.new_para(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(SRC.parent / a["src"]), width=Cm(15.9))
            p._p.xpath(".//wp:docPr")[0].set("descr", a.get("alt", ""))   # alt text, as in Word's Alt Text pane
        elif tag == "figcaption": self.new_para("Caption"); self.in_caption = True
        elif tag in ("strong", "b"): self.fmt.add("b")
        elif tag in ("em", "i"): self.fmt.add("i")
        elif tag == "code": self.fmt.add("code")
        elif tag == "a": self.href = a.get("href")

    def handle_endtag(self, tag):
        if tag in ("main", "section"): self.on = False; return
        if tag in ("style", "script", "head"): self.skip = max(0, self.skip - 1)
        elif tag in ("strong", "b"): self.fmt.discard("b")
        elif tag in ("em", "i"): self.fmt.discard("i")
        elif tag == "code": self.fmt.discard("code")
        elif tag == "a": self.href = None
        elif tag == "figcaption": self.in_caption = False
        elif tag in ("ol", "ul"): self.list_style = None; self.refs = False

    def handle_data(self, data):
        if not self.on or self.skip or self.p is None: return
        text = " ".join(data.split())
        if not text: return
        if data[:1].isspace() and self.p.text and not self.p.text.endswith(" "): text = " " + text
        if data[-1:].isspace(): text += " "
        if self.href:
            add_hyperlink(self.p, self.href, text.strip()); return
        r = self.p.add_run(text)
        r.bold = "b" in self.fmt or None
        r.italic = "i" in self.fmt or None
        if "code" in self.fmt: r.font.name = "Menlo"


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.54))
    st = doc.styles
    set_font(st["Normal"], 11.5)
    st["Normal"].paragraph_format.space_after = Pt(8); st["Normal"].paragraph_format.line_spacing = 1.25
    set_font(st["Title"], 24)
    st["Title"].paragraph_format.space_before = Pt(12); st["Title"].paragraph_format.space_after = Pt(12)
    # remove the Title style's default bottom border
    pPr = st["Title"].element.get_or_add_pPr()
    for b in pPr.findall(qn("w:pBdr")): pPr.remove(b)
    set_font(st["Heading 1"], 16)
    h = st["Heading 1"].paragraph_format; h.space_before, h.space_after, h.keep_with_next = Pt(18), Pt(6), True
    set_font(st["Caption"], 9.5, italic=True, color=GREY)
    st["Caption"].paragraph_format.space_after = Pt(14)
    for name in ("List Number", "List Bullet"):
        set_font(st[name], 11.5); st[name].paragraph_format.space_after = Pt(4)

    Builder(doc).feed(SRC.read_text(encoding="utf-8"))

    # The notes section's heading arrives as Heading 1; mark it clearly as not for publishing
    for p in doc.paragraphs:
        if p.text.startswith("Alt text:"):
            for r in p.runs[1:]: r.text = ""
            p.runs[0].text = "Alt text: "; p.runs[0].bold = True
            p.add_run("every image in this document has alt text attached (in Word, right-click an image and choose View Alt Text). "
                      "LinkedIn lets you add alt text when you upload an image; copy it from there.")
        if p.text.startswith("Publishing notes"):
            p.insert_paragraph_before().add_run().add_break(__import__("docx").enum.text.WD_BREAK.PAGE)
            for r in p.runs: r.font.color.rgb = RED
    doc.core_properties.title = "Kanto by the Numbers: A Statistical Look at Gen 1 Pokémon"
    doc.save(OUT)
    print("wrote", OUT.name, round(OUT.stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
