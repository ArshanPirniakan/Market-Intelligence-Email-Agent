import os
import json
import textwrap
import pandas as pd
from pathlib import Path
from datasets import load_dataset

import google.genai as genai
from google.genai import types
from exa_py import Exa

MODEL = "gemini-2.5-flash-preview-05-20"


def build_clients():
    gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    exa    = Exa(api_key=os.environ.get("EXA_API_KEY", ""))
    return gemini, exa


def load_huggingface_dataset(max_rows: int = 80) -> str:
    print("  -> Loading HuggingFace ad-copy dataset...")
    try:
        ds = load_dataset("jaykin01/advertisement-copy", split="train")
        df = ds.to_pandas().head(max_rows)
        lines = ["=== AD COPY EXAMPLES (HuggingFace) ==="]
        for _, row in df.iterrows():
            entry = "---\n"
            for c in row.index:
                val = str(row[c]).strip()
                if val and val != "nan":
                    entry += f"{c}: {val}\n"
            lines.append(entry)
        return "\n".join(lines)
    except Exception as e:
        print(f"  !  HuggingFace dataset error: {e}")
        return "=== AD COPY EXAMPLES (HuggingFace) ===\n(dataset unavailable)"


def load_kaggle_dataset(csv_path: str = "data/marketing_campaign.csv", max_rows: int = 60) -> str:
    print("  -> Loading Kaggle campaign dataset...")
    path = Path(csv_path)
    if not path.exists():
        return f"=== CAMPAIGN PERFORMANCE DATA (Kaggle) ===\n(file not found at {csv_path})"
    try:
        df = pd.read_csv(path).head(max_rows)
        lines = ["=== CAMPAIGN PERFORMANCE DATA (Kaggle) ==="]
        lines.append(f"Columns: {', '.join(df.columns.tolist())}\n")
        if "conversion_rate" in df.columns:
            top = df.nlargest(10, "conversion_rate")
            lines.append("Top 10 campaigns by conversion rate:")
            lines.append(top.to_string(index=False))
        else:
            lines.append(df.head(20).to_string(index=False))
        return "\n".join(lines)
    except Exception as e:
        print(f"  !  Kaggle dataset error: {e}")
        return "=== CAMPAIGN PERFORMANCE DATA (Kaggle) ===\n(parse error)"


def load_local_rules(path: str = "data/my_copywriting_rules.txt") -> str:
    p = Path(path)
    if p.exists():
        return f"=== CUSTOM COPYWRITING RULES ===\n{p.read_text(encoding='utf-8')}"
    return "=== CUSTOM COPYWRITING RULES ===\n(no local rules file found)"


def build_knowledge_base() -> str:
    return "\n\n".join([load_local_rules(), load_huggingface_dataset(), load_kaggle_dataset()])


def generate_search_queries(gemini, niche: str, audience: str, product: str) -> list[str]:
    prompt = textwrap.dedent(f"""
        You are a market research specialist.
        Generate exactly 3 short, targeted web search strings to find raw, unfiltered
        human sentiment (forum posts, reviews, Reddit threads) about the following:

        Niche: {niche}
        Target audience: {audience}
        Product/service: {product}

        Focus on real pain points, frustrations, desires, and vocabulary this audience uses.
        Return ONLY a JSON array of 3 strings, nothing else.
        Example: ["string1", "string2", "string3"]
    """).strip()

    response = gemini.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.4)
    )
    raw = response.text.strip().strip("```json").strip("```").strip()
    return json.loads(raw)


def fetch_exa_research(exa, queries: list[str]) -> str:
    print("  -> Fetching web research via Exa...")
    all_text = []
    for q in queries:
        print(f"     . {q}")
        result = exa.search_and_contents(
            q,
            num_results=3,
            use_autoprompt=True,
            text={"max_characters": 800}
        )
        for r in result.results:
            chunk = f"SOURCE: {r.url}\n{getattr(r, 'text', '') or ''}\n"
            all_text.append(chunk)
    return "\n---\n".join(all_text)


def synthesize_persona(gemini, raw_research: str, niche: str, audience: str) -> str:
    print("  -> Synthesizing customer persona with Gemini...")
    prompt = textwrap.dedent(f"""
        You are a direct-response marketing strategist.
        Based on the raw web research below about [{niche}] targeting [{audience}],
        write a structured Customer Persona & Sentiment Report with these sections:

        1. CORE DESIRES
        2. PRIMARY FEARS & FRICTION POINTS
        3. CUSTOMER VOCABULARY
        4. EMOTIONAL TRIGGERS

        Raw research:
        {raw_research[:6000]}

        Be specific and blunt. No filler. Do not use em dashes anywhere in your response.
    """).strip()

    response = gemini.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.5)
    )
    return response.text


def write_marketing_email(gemini, knowledge_base: str, persona_report: str,
                          niche: str, audience: str, product: str, cta: str) -> str:
    print("  -> Writing marketing email with Gemini...")
    prompt = textwrap.dedent(f"""
        You are a world-class direct-response email copywriter.

        TRAINING KNOWLEDGE BASE:
        {knowledge_base[:4000]}

        LIVE CUSTOMER PERSONA REPORT:
        {persona_report}

        TASK:
        Write ONE complete marketing email for:
        Niche: {niche}
        Audience: {audience}
        Product/service: {product}
        Call to action: {cta}

        STRUCTURE:
        1. Subject line: curiosity-driven, under 9 words, no clickbait cliches
        2. Preview text: 1 sentence, adds to subject without repeating it
        3. Opening hook: 1-2 sentences, hits the reader's core desire or fear immediately
        4. Agitate the pain: 2-3 sentences, make the problem feel real and urgent
        5. Solution bridge: 2-3 sentences, position the product as the natural answer
        6. Proof / credibility: 1-2 sentences (stat, social proof, or concrete outcome)
        7. CTA: one sentence, low friction, direct action

        RULES:
        - Use the customer's exact vocabulary from the persona report.
        - No fluff. Every sentence earns its place.
        - No exclamation marks. No ALL CAPS.
        - Write at a 7th-grade reading level.
        - Do not start with "I" or the brand name.
        - Do not use em dashes anywhere. Use commas, colons, or periods instead.
        - Total word count: 150-250 words.

        Output ONLY the email. No preamble.
    """).strip()

    response = gemini.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7)
    )
    return response.text


class MarketIntelligenceAgent:
    def __init__(self):
        print("\nInitialising Market Intelligence Email Agent...")
        self.gemini, self.exa = build_clients()
        print("Loading datasets...")
        self.knowledge_base = build_knowledge_base()
        print("Ready.\n")

    def run(self, niche: str, audience: str, product: str, cta: str) -> dict:
        print(f"Research phase: {niche}")
        queries = generate_search_queries(self.gemini, niche, audience, product)
        print(f"  Queries: {queries}")

        raw_research = fetch_exa_research(self.exa, queries)
        persona      = synthesize_persona(self.gemini, raw_research, niche, audience)

        print("\nCopywriting phase...")
        email = write_marketing_email(
            self.gemini, self.knowledge_base, persona,
            niche, audience, product, cta
        )

        return {"queries": queries, "persona_report": persona, "email": email}


def main():
    agent = MarketIntelligenceAgent()

    print("=" * 60)
    print("  MARKET INTELLIGENCE EMAIL AGENT")
    print("=" * 60)
    print("Type 'quit' at any prompt to exit.\n")

    while True:
        niche    = input("Niche / industry: ").strip()
        if niche.lower() == "quit": break

        audience = input("Target audience: ").strip()
        if audience.lower() == "quit": break

        product  = input("Product / service description: ").strip()
        if product.lower() == "quit": break

        cta      = input("Call to action: ").strip()
        if cta.lower() == "quit": break

        result = agent.run(niche, audience, product, cta)

        print("\n" + "=" * 60)
        print("CUSTOMER PERSONA REPORT")
        print("=" * 60)
        print(result["persona_report"])

        print("\n" + "=" * 60)
        print("GENERATED EMAIL")
        print("=" * 60)
        print(result["email"])

        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        safe_niche = niche.replace(" ", "_")[:30]
        out_path = out_dir / f"email_{safe_niche}.txt"
        out_path.write_text(
            f"PERSONA REPORT\n{'='*40}\n{result['persona_report']}\n\n"
            f"EMAIL\n{'='*40}\n{result['email']}",
            encoding="utf-8"
        )
        print(f"\nSaved to {out_path}")

        again = input("\nRun again? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
