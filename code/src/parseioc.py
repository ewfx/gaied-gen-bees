import requests

# Parsio API Configuration
PARSIO_API_KEY = "jjzs2eoj07pfgb977kl6n4hqans7y7iid5knf6my48z7vg7x"
PARSIO_MAILBOX_ID = "customtextparsermailbox@io.parsio.io"

# Send email content to Parsio for extraction
def extract_banking_context_with_parsio(email_text: str):
    try:
        headers = {
            "Authorization": f"Bearer {PARSIO_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "mailbox_id": PARSIO_MAILBOX_ID,
            "content": email_text
        }

        response = requests.post("https://api.parsio.io/v1/emails", json=payload, headers=headers)
        
        if response.status_code != 200:
            logging.error(f"Parsio API error: {response.text}")
            raise HTTPException(status_code=500, detail="Error extracting context from Parsio API")

        parsed_data = response.json()

        # Extract banking contexts from Parsio response
        banking_contexts = []
        for field in parsed_data.get("data", {}):
            value = parsed_data["data"][field]
            if value and isinstance(value, str):
                banking_contexts.append(value)

        if not banking_contexts:
            banking_contexts.append("general banking inquiry")
        
        return banking_contexts

    except Exception as e:
        logging.error(f"Error during Parsio context extraction: {e}")
        raise HTTPException(status_code=500, detail="Error extracting banking context using Parsio API")
