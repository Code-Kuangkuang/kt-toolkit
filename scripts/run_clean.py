# 入口脚本；读取配置并调度清洗流程
import sys
from pathlib import Path

import yaml
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cleaning.pipeline import run_cleaning

app = typer.Typer(add_completion=False)

@app.command()
def main(
    config: str = typer.Option(..., "--config", "-c", help="Path to the config file"),
):
    cfg = yaml.safe_load(open(config, 'r', encoding='utf-8'))
    run_cleaning(cfg)

if __name__ == '__main__':
    app()
