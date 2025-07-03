from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO
import os
import re
import json

load_dotenv()

app = FastAPI()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode()))

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Format transactions into readable string
    data_string = ""
    for i, row in df.iterrows():
        data_string += (
            f"{i+1}. Date: {row['date']}, "
            f"Amount: ${row['amount']}, "
            f"Vendor: {row['vendor']}, "
            f"Description: {row['description']}\n"
        )

    # Updated prompt with all required fields
    prompt = f"""
You are a financial anomaly detector. Analyze the following transactions and return a JSON list of anomalies. 
Each anomaly must be an object with these fields:

- transaction_id (index of the transaction in the list)
- amount (as a number)
- timestamp (use the date from the transaction)
- vendor (from the transaction)
- anomaly_type (brief name like "Duplicate", "Suspicious Amount", etc.)
- severity ("low", "medium", or "high")
- confidence_score (between 0 and 1)
- description (brief explanation of the anomaly)

Return only the JSON list inside a markdown code block with ```json.

Transactions:
{data_string}
""".strip()

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    content = response.choices[0].message.content

    # Parse the JSON from the response
    try:
        json_match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
        if json_match:
            clean_json = json.loads(json_match.group(1))
        else:
            clean_json = json.loads(content)
    except Exception as e:
        return {"error": "Failed to parse LLM response", "raw": content, "exception": str(e)}

    # Enrich and clean up each anomaly object
    for anomaly in clean_json:
        try:
            idx = int(anomaly.get("transaction_id", 0)) - 1
            row = df.iloc[idx] if 0 <= idx < len(df) else {}

            anomaly["amount"] = float(row.get("amount", 0))
            anomaly["timestamp"] = row.get("date", "")
            anomaly["vendor"] = row.get("vendor", "")
            anomaly["description"] = anomaly.get("description") or anomaly.get("explanation", "")
            anomaly["anomaly_type"] = anomaly.get("anomaly_type") or anomaly.get("reason", "Unknown")
            anomaly["severity"] = anomaly.get("severity", "medium")
            anomaly["confidence_score"] = float(anomaly.get("confidence_score", 0.8))

        except Exception as e:
            anomaly["amount"] = 0
            anomaly["timestamp"] = ""
            anomaly["vendor"] = ""
            anomaly["description"] = "Unknown"
            anomaly["anomaly_type"] = "Unknown"
            anomaly["severity"] = "medium"
            anomaly["confidence_score"] = 0.5

    # Compute stats
    anomaly_count = len(clean_json)

    try:
        highest_amount = max([a["amount"] for a in clean_json])
    except:
        highest_amount = 0

    try:
        reason_counts = {}
        for a in clean_json:
            reason = a.get("anomaly_type", "Unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        most_common_reason = max(reason_counts.items(), key=lambda x: x[1])[0]
    except:
        most_common_reason = "Unknown"

    return {
        "anomaly_count": anomaly_count,
        "highest_amount": highest_amount,
        "most_common_reason": most_common_reason,
        "results": clean_json,
        "total_transactions": len(df)
    }
