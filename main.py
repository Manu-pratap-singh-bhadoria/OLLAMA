from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import chat
import json
import re

app = FastAPI()


class InvoiceRequest(BaseModel):
    text: str


class InvoiceResponse(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str


@app.post("/extract", response_model=InvoiceResponse)
def extract(req: InvoiceRequest):

    # Empty input
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Empty input")

    prompt = f"""
Extract invoice information.

Return ONLY a JSON object.

Required format:

{{
    "vendor": "",
    "amount": 0,
    "currency": "",
    "date": ""
}}

Rules:
- vendor = company name
- amount = total due as a number
- currency = 3-letter uppercase currency code
- date = YYYY-MM-DD

Invoice Text:

{req.text}
"""

    try:
        response = chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json"
        )

        result = response["message"]["content"]

        print("\n===== MODEL RESPONSE =====")
        print(result)
        print("==========================\n")

        # Remove markdown if present
        result = result.replace("```json", "").replace("```", "").strip()

        # Extract JSON object
        match = re.search(r"\{.*\}", result, re.DOTALL)

        if not match:
            raise HTTPException(
                status_code=500,
                detail="Model did not return valid JSON."
            )

        json_text = match.group(0)

        data = json.loads(json_text)

        return InvoiceResponse(
            vendor=data["vendor"],
            amount=float(data["amount"]),
            currency=data["currency"].upper(),
            date=data["date"]
        )

    except HTTPException:
        raise

    except Exception:
        return InvoiceResponse(
            vendor="",
            amount=0,
            currency="",
            date=""
        )