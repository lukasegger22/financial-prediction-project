import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "plots" / "interactive_news_quiz.html"


def load_examples():
    return [
        {
            "headline": "ECB does not rule out further rate cuts amid inflation panic.",
            "actual": 1,
            "aiPrediction": 1,
        },
        {
            "headline": "US unemployment falls as consumer confidence reaches a yearly high.",
            "actual": 1,
            "aiPrediction": 1,
        },
        {
            "headline": "Global banks warn of recession risks after weak manufacturing data.",
            "actual": 0,
            "aiPrediction": 0,
        },
    ]


def render_html(examples):
    examples_json = json.dumps(examples, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI News Quiz</title>
  <style>
    :root {{
      --bg: #f7f8f5;
      --ink: #141a17;
      --muted: #6a756f;
      --line: #dce2dc;
      --panel: #fff;
      --green: #198754;
      --red: #c43c35;
      --blue: #285f8f;
      --gold: #bd8428;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    main {{
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: clamp(28px, 6vw, 72px);
    }}

    .quiz {{
      width: min(1100px, 100%);
      margin: 0 auto;
    }}

    .news-card {{
      padding: clamp(28px, 5vw, 56px);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 20px 70px rgba(20, 26, 23, 0.08);
    }}

    .headline {{
      margin: 0;
      font-size: clamp(34px, 5.2vw, 72px);
      line-height: 1.04;
      letter-spacing: 0;
      font-weight: 900;
    }}

    .actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 24px;
    }}

    button {{
      min-height: 76px;
      border: 0;
      border-radius: 8px;
      color: #fff;
      font: inherit;
      font-size: clamp(22px, 2.8vw, 34px);
      font-weight: 900;
      cursor: pointer;
      transition: transform 140ms ease, opacity 140ms ease;
    }}

    button:hover {{ transform: translateY(-2px); }}
    button:disabled {{ opacity: 0.5; transform: none; cursor: default; }}
    .up {{ background: var(--green); }}
    .down {{ background: var(--red); }}
    .secondary {{
      min-height: 46px;
      padding: 0 18px;
      background: transparent;
      border: 1px solid var(--line);
      color: var(--ink);
      font-size: 15px;
    }}

    .result {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}

    .result-box {{
      min-height: 132px;
      padding: 20px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}

    .result-box small {{
      display: block;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 900;
      text-transform: uppercase;
    }}

    .result-box strong {{
      display: block;
      font-size: clamp(25px, 3vw, 40px);
      line-height: 1;
    }}

    .hidden {{
      color: var(--muted);
    }}

    .green {{ color: var(--green); }}
    .red {{ color: var(--red); }}

    .controls {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-top: 18px;
    }}

    .status {{
      color: var(--muted);
      font-size: 15px;
      font-weight: 800;
    }}

    @media (max-width: 780px) {{
      main {{ padding: 20px; }}
      .actions, .result {{ grid-template-columns: 1fr; }}
      .headline {{ font-size: 36px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="quiz">
      <div class="news-card">
        <h1 class="headline" id="headline"></h1>
      </div>

      <div class="actions">
        <button class="up" id="guess-up">UP</button>
        <button class="down" id="guess-down">DOWN</button>
      </div>

      <div class="result">
        <div class="result-box">
          <small>Your vote</small>
          <strong id="your-vote" class="hidden">Choose</strong>
        </div>
        <div class="result-box">
          <small>AI predicted</small>
          <strong id="ai-prediction" class="hidden">Hidden</strong>
        </div>
        <div class="result-box">
          <small>Actual result</small>
          <strong id="actual-result" class="hidden">Hidden</strong>
        </div>
      </div>

      <div class="controls">
        <button class="secondary" id="next">Next headline</button>
        <div class="status" id="status"></div>
      </div>
    </section>
  </main>

  <script>
    const examples = {examples_json};
    let index = 0;
    let guessed = null;

    const directionText = value => value === 1 ? "UP" : "DOWN";
    const directionClass = value => value === 1 ? "green" : "red";

    function setDirection(id, value) {{
      const element = document.getElementById(id);
      element.textContent = directionText(value);
      element.className = directionClass(value);
    }}

    function resetDirection(id, text) {{
      const element = document.getElementById(id);
      element.textContent = text;
      element.className = "hidden";
    }}

    function render() {{
      const item = examples[index];
      guessed = null;
      document.getElementById("headline").textContent = item.headline;
      resetDirection("your-vote", "Choose");
      resetDirection("ai-prediction", "Hidden");
      resetDirection("actual-result", "Hidden");
      document.getElementById("status").textContent = `${{index + 1}} / ${{examples.length}}`;
      document.getElementById("guess-up").disabled = false;
      document.getElementById("guess-down").disabled = false;
      document.getElementById("next").disabled = index === examples.length - 1;
    }}

    function makeGuess(value) {{
      if (guessed !== null) return;
      guessed = value;
      const item = examples[index];
      setDirection("your-vote", guessed);
      setDirection("ai-prediction", item.aiPrediction);
      setDirection("actual-result", item.actual);
      document.getElementById("status").textContent =
        item.aiPrediction === item.actual ? "AI was correct." : "AI was wrong.";
      document.getElementById("guess-up").disabled = true;
      document.getElementById("guess-down").disabled = true;
    }}

    function next() {{
      if (index < examples.length - 1) {{
        index += 1;
        render();
      }}
    }}

    document.getElementById("guess-up").addEventListener("click", () => makeGuess(1));
    document.getElementById("guess-down").addEventListener("click", () => makeGuess(0));
    document.getElementById("next").addEventListener("click", next);
    render();
  </script>
</body>
</html>
"""


def main():
    examples = load_examples()
    OUTPUT_PATH.write_text(render_html(examples), encoding="utf-8")
    print(f"Interactive demo saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
