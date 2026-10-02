"""The project-page bar shared by the generated pages in docs/ (src/report.py, src/samples.py).

It is the same markup the hand-maintained pages carry: a "Project page" link on the left and
the tensorViz mark, linked to https://tensorviz.dev, on the right. It sits as the first row
inside <main>, so the head/body markers that src/make_audit_page.py cuts on stay intact.
"""
from __future__ import annotations

_STYLE = """<style>
  /* project-page bar: the first row inside main, so the page generators' head/body markers stay intact */
  .pp-bar{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px 16px;
    margin:-28px 0 28px;padding:0 0 12px;border-bottom:1px solid #D9DEDE;
    font:15px/1.4 Geist,system-ui,-apple-system,"Segoe UI",sans-serif}
  .pp-bar a{text-decoration:none}
  .pp-home{color:#23479C;font-weight:600}
  .pp-home:hover{text-decoration:underline}
  .pp-tv{display:inline-flex;align-items:center;gap:7px;color:#1C2420;font-size:18px;font-weight:690;
    letter-spacing:-.048em;line-height:1}
  .pp-tv svg{width:22px;height:22px;fill:#F5F5EF;stroke:#1C2420;stroke-width:2.2;stroke-linecap:round}
  .pp-l{font-weight:380}
"""
_WIDE_TABLES = """  /* phones: wide tables scroll inside their own box instead of widening the page */
  @media (max-width:640px){main table{display:block;overflow-x:auto}}
"""


def style(wide_tables: bool = False) -> str:
    """The bar's <style> block, placed before the page's own stylesheet."""
    return _STYLE + (_WIDE_TABLES if wide_tables else "") + "</style>\n"


def nav(home: str = "index.html") -> str:
    """The bar itself; `home` is the project page relative to the generated page."""
    return (
        f'<nav class="pp-bar" aria-label="Site"><a class="pp-home" href="{home}">&larr; Project page</a>'
        '<a class="pp-tv" href="https://tensorviz.dev" aria-label="tensorViz (tensorviz.dev)">'
        '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M8 7h16M8 7v18m0-9h16m0-9v18M8 25h16"/>'
        '<circle cx="8" cy="7" r="3"/><circle cx="24" cy="7" r="3"/><circle cx="8" cy="25" r="3"/>'
        '<circle cx="24" cy="25" r="3"/><circle cx="24" cy="16" r="3"/></svg>'
        '<span>tensor<span class="pp-l">Viz</span></span></a></nav>\n'
    )
