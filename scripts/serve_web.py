from pathlib import Path
import sys

import typer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

app = typer.Typer(add_completion=False)


@app.command()
def main(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto reload"),
):
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter(
            "WebUI requires optional dependencies. Install them with: "
            "pip install -r requirements-web.txt"
        ) from exc
    uvicorn.run("webui.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()

