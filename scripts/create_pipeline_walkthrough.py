from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "plots" / "pipeline_walkthrough.html"


def render_html():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Five Step AI Pipeline</title>
  <style>
    :root {
      --bg: #f7f8f5;
      --ink: #151c18;
      --muted: #69766f;
      --line: #dce3dc;
      --panel: #ffffff;
      --blue: #285f8f;
      --green: #198754;
      --gold: #bd8428;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    main {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: clamp(28px, 5vw, 72px);
    }

    .wrap {
      width: min(1200px, 100%);
    }

    h1 {
      margin: 0 0 34px;
      font-size: clamp(34px, 5vw, 68px);
      line-height: 1;
      letter-spacing: 0;
      text-align: center;
    }

    .pipeline {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      align-items: stretch;
    }

    .step {
      min-height: 210px;
      padding: 22px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 18px 54px rgba(21, 28, 24, 0.07);
      position: relative;
    }

    .step:not(:last-child)::after {
      content: "→";
      position: absolute;
      right: -17px;
      top: 50%;
      transform: translateY(-50%);
      z-index: 2;
      color: var(--blue);
      font-size: 30px;
      font-weight: 900;
    }

    .number {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 40px;
      height: 40px;
      margin-bottom: 18px;
      border-radius: 8px;
      background: var(--blue);
      color: #fff;
      font-weight: 900;
      font-size: 18px;
    }

    .step:nth-child(4) .number { background: var(--green); }
    .step:nth-child(5) .number { background: var(--gold); }

    h2 {
      margin: 0;
      font-size: clamp(20px, 2vw, 28px);
      line-height: 1.08;
    }

    .tag {
      margin-top: 16px;
      color: var(--muted);
      font-size: 14px;
      font-weight: 800;
      text-transform: uppercase;
    }

    @media (max-width: 920px) {
      .pipeline {
        grid-template-columns: 1fr;
      }

      .step {
        min-height: 132px;
      }

      .step:not(:last-child)::after {
        content: "↓";
        right: 22px;
        top: auto;
        bottom: -26px;
        transform: none;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="wrap">
      <h1>AI Prediction Pipeline</h1>

      <div class="pipeline">
        <article class="step">
          <div class="number">1</div>
          <h2>Collect News</h2>
          <div class="tag">Headlines</div>
        </article>

        <article class="step">
          <div class="number">2</div>
          <h2>Clean Text</h2>
          <div class="tag">Preprocessing</div>
        </article>

        <article class="step">
          <div class="number">3</div>
          <h2>FinBERT Reads Context</h2>
          <div class="tag">Transfer Learning</div>
        </article>

        <article class="step">
          <div class="number">4</div>
          <h2>Predict UP or DOWN</h2>
          <div class="tag">AI Signal</div>
        </article>

        <article class="step">
          <div class="number">5</div>
          <h2>Backtest Strategy</h2>
          <div class="tag">Market Simulation</div>
        </article>
      </div>
    </section>
  </main>
</body>
</html>
"""


def main():
    OUTPUT_PATH.write_text(render_html(), encoding="utf-8")
    print(f"Pipeline walkthrough saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
