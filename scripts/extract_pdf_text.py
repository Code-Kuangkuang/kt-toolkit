from pathlib import Path
from typing import Optional

import typer
from pypdf import PdfReader


app = typer.Typer(add_completion=False)


def extract_pdf_text(pdf_path: Path, out_path: Optional[Path] = None) -> tuple[int, int, Path]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Encrypted PDF is not supported: {pdf_path}") from exc

    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        parts.append(f"\n\n===== PAGE {i} =====\n\n{text}")

    content = "".join(parts)
    output = out_path if out_path is not None else pdf_path.with_suffix(".txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")

    return len(reader.pages), len(content), output


@app.command()
def main(
    pdf: Path = typer.Option(..., "--pdf", help="Path to the source PDF file."),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Optional output txt path. Defaults to PDF path with .txt suffix.",
    ),
) -> None:
    pages, chars, out_path = extract_pdf_text(pdf_path=pdf, out_path=out)
    typer.echo(f"pages={pages}")
    typer.echo(f"out={out_path}")
    typer.echo(f"chars={chars}")


if __name__ == "__main__":
    app()
