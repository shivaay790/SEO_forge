# SEO Forge

> **AI-Assisted SEO Analysis — keyword gaps, search intent, and featured-snippet generation**

SEO Forge analyses a web page against its competitors and reports what is holding it back in search. It extracts and compares keywords, classifies search intent with BERT, scores content similarity, and generates a featured-snippet candidate with BART — then turns all of it into a list of concrete recommendations.

---

## 🏗️ Architecture

```
        --url                        --competitors
          │                                │
          ▼                                ▼
   ┌──────────────┐                ┌──────────────┐
   │ extract_     │                │ extract_     │
   │ content()    │  BeautifulSoup │ content()    │
   └──────┬───────┘                └──────┬───────┘
          │                               │
          ▼                               ▼
   ┌─────────────────────────────────────────────┐
   │              Analysis pipeline              │
   │                                             │
   │  analyze_intent()      BERT classifier      │
   │  extract_keywords()    KeyBERT + DistilBERT │
   │  calculate_similarity() cosine similarity   │
   │  generate_snippet()    BART-large-CNN       │
   └──────────────────┬──────────────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │ OnDemand.io Chat API  │  (GPT-4o endpoint)
          └───────────┬───────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  Summary report       │
          │  • intent + score     │
          │  • keyword gaps       │
          │  • snippet candidate  │
          │  • recommendations    │
          └───────────────────────┘
```

---

## 🚀 Key Components

1. **Content extraction (`extract_content`)**
   Fetches a URL and strips it to readable body text with BeautifulSoup.

2. **Intent classification (`analyze_intent`)**
   Runs a BERT classifier over the page text and returns the dominant search intent with a confidence score.

3. **Keyword extraction (`extract_keywords`)**
   Uses KeyBERT on `distilbert-base-nli-mean-tokens` with a relevance `filter_threshold` to pull the top-N keywords for a page.

4. **Similarity scoring (`calculate_similarity`)**
   Cosine similarity between BERT embeddings of your page and each competitor page.

5. **Snippet generation (`generate_snippet`)**
   Summarises the page with `facebook/bart-large-cnn` into a featured-snippet candidate.

6. **Recommendation engine (`generate_recommendations`, `mobile_optimization_suggestions`)**
   Diffs your keywords against competitor keywords to surface content gaps, plus a static mobile-optimisation checklist.

---

## 🛠️ Installation & Usage

### 1. Requirements

Python 3.9+:

```bash
git clone https://github.com/shivaay790/SEO_forge.git
cd SEO_forge
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> First run downloads ~2 GB of model weights (BERT, DistilBERT, BART) from Hugging Face.

### 2. Credentials

```bash
cp .env.example .env
```

Then edit `.env`:

```
ONDEMAND_API_KEY=your_ondemand_api_key_here
ONDEMAND_EXTERNAL_USER_ID=your_external_user_id_here
```

Get both from [on-demand.io](https://on-demand.io). `.env` is gitignored — keep your keys out of the repo.

### 3. Run an analysis

```bash
python Model.py \
  --url https://en.wikipedia.org/wiki/Health \
  --competitors https://www.wikihow.com/Be-Healthy https://www.wikihow.com/Category:Health
```

`--competitors` is optional; with none supplied you still get intent, keywords, and a snippet for your own page.

### 4. Sample output

```
==== Analyzing User Intent ====
User Intent: informational (Confidence: 0.87)

==== Extracting User Keywords ====
User Keywords: health, wellness, nutrition, disease, lifestyle, ...

==== Comparing with Competitors ====
Similarity with Competitor 1: 0.64
Similarity with Competitor 2: 0.58

==== SEO Recommendations ====
- Add coverage for: preventive care, mental wellbeing
- Strengthen thin sections around: exercise routines
```

---

## 📁 Project Structure

```
SEO_forge/
├── Model.py           # CLI analysis pipeline (entry point)
├── requirements.txt   # Python dependencies
├── .env.example       # Credential template
├── index.html         # Static UI prototype
├── app.js             # UI event handlers
├── style.css          # UI styling
├── LICENSE            # Apache-2.0
└── README.md
```

---

## ⚠️ Status of the web UI

`index.html`, `app.js`, and `style.css` are a **design prototype only**. The page calls endpoints (`/api/analyze-intent`, `/api/analyze-voice`, …) that no backend in this repo currently serves — opening `index.html` gives you the layout, not working analysis.

The working tool is the `Model.py` CLI described above. Wiring the prototype to a real Flask or FastAPI server that exposes those routes is the natural next step for this project.

---

## 📄 License

Apache-2.0 — see [LICENSE](LICENSE).
