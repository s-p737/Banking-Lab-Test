"""
7_extractbanks_llm.py
Uses Google Gemini Flash (free) to extract structured bank partnership
info from fintech terms pages.
"""

import json
import os
import re
import time
from pathlib import Path

from google import genai
import pandas as pd

# =====================================================
# CONFIG
# =====================================================
FINTECH_ROOT = Path("/yen/projects/faculty/nazkoont-baas/Data/Fintechs")
OUTPUT_CSV    = Path("fintech_bank_product_pairs_llm.csv")
API_KEY       = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MAX_FILES     = 10
MAX_CHARS     = 8000
SLEEP_BETWEEN = 1.0
SAVE_EVERY_N  = 50

BAAS_BANKS = {
    "evolve bank", "bancorp", "cross river", "green dot", "sutton bank",
    "column bank", "coastal community", "lineage bank", "blue ridge bank",
    "pathward", "stride bank", "goldman sachs", "webbank", "celtic bank",
}

client = genai.Client(api_key=API_KEY)

EXTRACTION_PROMPT = """You are an expert at reading fintech legal disclosures and terms of service pages.
Extract all bank partnership information from the text below.

Return ONLY a valid JSON object with this exact structure — no markdown, no explanation, nothing else:
{
  "partnerships": [
    {
      "bank_name_raw": "exact name as written on the page, e.g. Cross River Bank, Member FDIC",
      "bank_name_clean": "normalized name without Member FDIC etc, e.g. Cross River Bank",
      "product_type": "one of: deposit account | credit card | loan | sweep account | prepaid card | brokerage | payment | other",
      "product_subtype": "more specific description if available, else null",
      "apy_or_rate": "e.g. 4.50% APY, or null if not mentioned",
      "fdic_insured": true or false or null,
      "notes": "any other relevant terms or context worth noting",
      "confidence": "high | medium | low"
    }
  ]
}

Rules:
- Include ALL bank partnerships mentioned, even if mentioned only once
- If one bank offers multiple products, create one entry per product
- Extract the exact legal name as written (include N.A., FSB, National Association if present)
- If no bank partnership info is found, return {"partnerships": []}
- Do NOT invent anything not in the text
- Return ONLY the JSON object, nothing else

Text to analyze:
"""

def call_gemini(text: str) -> list:
    text = text[:MAX_CHARS]
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=EXTRACTION_PROMPT + text
        )
        raw = response.text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        return parsed.get("partnerships", [])
    except json.JSONDecodeError as e:
        print(f"      [WARN] JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"      [WARN] Gemini API error: {e}")
        return []

def boost_confidence(bank_name_clean: str, confidence: str) -> str:
    if not bank_name_clean:
        return confidence
    name_lower = bank_name_clean.lower()
    if any(known in name_lower for known in BAAS_BANKS):
        return "high"
    return confidence

def filename_to_year(filename: str):
    try:
        y = int(str(filename)[:4])
        if 1900 <= y <= 2200:
            return y
    except Exception:
        pass
    return None

def save_rows(rows: list):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV, dtype=str)
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"  [saved {len(rows)} rows to {OUTPUT_CSV}]")

processed_keys = set()
if OUTPUT_CSV.exists():
    existing_df = pd.read_csv(OUTPUT_CSV, dtype=str)
    for _, row in existing_df.iterrows():
        processed_keys.add((row["fintech"], row["source_type"], row["filename"]))
    print(f"Resuming: {len(processed_keys)} files already processed\n")

def main():
    rows_buffer = []
    processed_files = 0
    total_partnerships = 0

    for fintech_dir in sorted(FINTECH_ROOT.iterdir()):
        if not fintech_dir.is_dir():
            continue
        fintech = fintech_dir.name

        for source_type in ["Terms", "Website"]:
            subdir = fintech_dir / source_type
            if not subdir.exists():
                subdir = fintech_dir / source_type.lower()
                if not subdir.exists():
                    continue

            for txt_file in sorted(subdir.glob("*.txt")):
                key = (fintech, source_type.lower(), txt_file.name)
                if key in processed_keys:
                    continue

                if MAX_FILES is not None and processed_files >= MAX_FILES:
                    save_rows(rows_buffer)
                    print(f"\nDone (hit MAX_FILES={MAX_FILES})")
                    print(f"Processed {processed_files} files, found {total_partnerships} partnerships")
                    print(f"Output: {OUTPUT_CSV}")
                    return

                year = filename_to_year(txt_file.name)
                try:
                    text = txt_file.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    print(f"  [WARN] could not read {txt_file}: {e}")
                    continue

                processed_files += 1
                print(f"[{processed_files}] {fintech} / {source_type} / {txt_file.name}  (year={year})")

                partnerships = call_gemini(text)
                time.sleep(SLEEP_BETWEEN)

                if partnerships:
                    print(f"      found {len(partnerships)} partnership(s):")
                    for p in partnerships:
                        print(f"        - {p.get('bank_name_clean')} | {p.get('product_type')} | {p.get('apy_or_rate')} | confidence={p.get('confidence')}")
                    total_partnerships += len(partnerships)
                else:
                    print(f"      no partnerships found")

                for p in partnerships:
                    rows_buffer.append({
                        "fintech":         fintech,
                        "source_type":     source_type.lower(),
                        "filename":        txt_file.name,
                        "year":            year,
                        "bank_name_raw":   p.get("bank_name_raw", ""),
                        "bank_name_clean": p.get("bank_name_clean", ""),
                        "product_type":    p.get("product_type", ""),
                        "product_subtype": p.get("product_subtype", ""),
                        "apy_or_rate":     p.get("apy_or_rate", ""),
                        "fdic_insured":    p.get("fdic_insured", ""),
                        "notes":           p.get("notes", ""),
                        "confidence":      boost_confidence(
                                               p.get("bank_name_clean", ""),
                                               p.get("confidence", "low")
                                           ),
                        "llm_used":        "gemini-2.0-flash",
                    })

                if not partnerships:
                    rows_buffer.append({
                        "fintech": fintech, "source_type": source_type.lower(),
                        "filename": txt_file.name, "year": year,
                        "bank_name_raw": "", "bank_name_clean": "", "product_type": "",
                        "product_subtype": "", "apy_or_rate": "", "fdic_insured": "",
                        "notes": "", "confidence": "", "llm_used": "gemini-2.0-flash",
                    })

                processed_keys.add(key)

                if processed_files % SAVE_EVERY_N == 0:
                    save_rows(rows_buffer)
                    rows_buffer = []

    save_rows(rows_buffer)
    print(f"\nDone. Processed {processed_files} files, found {total_partnerships} partnerships.")
    print(f"Output: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
