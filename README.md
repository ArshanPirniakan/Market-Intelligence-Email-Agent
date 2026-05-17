# Market Intelligence Email Agent

A two-stage AI pipeline that researches your target audience on the web and writes high-converting marketing emails based on what it finds.

**Stack:** Google Gemini 2.5 Flash, Exa.ai, HuggingFace, Kaggle

---

## How it works

1. You provide a niche, target audience, product, and desired call to action.
2. The research agent uses Exa to find real forum posts and reviews from your audience, then synthesizes them into a customer persona report.
3. The copywriter agent combines that report with ad-copy training data to write a structured marketing email.
4. The output is printed to your terminal and saved to the `outputs/` folder.

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies**
```bash
pip install google-genai exa-py datasets pandas requests
```

**3. Set your API keys**

You need two API keys:
- Gemini: https://aistudio.google.com/app/apikey
- Exa: https://exa.ai

```bash
export GEMINI_API_KEY="your_gemini_key"
export EXA_API_KEY="your_exa_key"
```

On Windows:
```bash
set GEMINI_API_KEY=your_gemini_key
set EXA_API_KEY=your_exa_key
```

**4. Add the Kaggle dataset (optional)**

Download the Marketing Campaign Performance dataset from Kaggle and place the CSV at:
```
data/marketing_campaign.csv
```
https://www.kaggle.com/datasets/manishabhatt22/marketing-campaign-performance-dataset

The agent works without it. The HuggingFace dataset loads automatically.

**5. Add your own copywriting rules (optional)**

Create a file at `data/my_copywriting_rules.txt` with your own good/bad copy examples and guidelines. This file takes priority over the public datasets.

---

## Run

```bash
python agent.py
```

Follow the prompts in your terminal.

---

## Folder structure

```
agent.py
data/
  my_copywriting_rules.txt   (optional)
  marketing_campaign.csv     (optional, from Kaggle)
outputs/
  email_your_niche.txt       (auto-generated)
```

