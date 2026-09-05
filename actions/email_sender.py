"""
Encrypted email dispatch over TLS using smtplib + MIME.
Credentials are pulled from environment variables (.env) — never hardcoded.
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config.settings import EMAIL_ADDRESS, EMAIL_APP_PASSWORD, SMTP_SERVER, SMTP_PORT

# Simple contact book — extend or swap for a proper contacts file/DB.
CONTACTS = {
    "rahul": "rahul@example.com",
    "sarah": "sarah@example.com",
    "professor": "professor@example.com",
}


def send_email(contact_name: str, subject: str = "Message from SIRA", body: str = ""):
    if not contact_name:
        return "I didn't catch who to email."

    contact_key = contact_name.lower().strip()
    recipient = CONTACTS.get(contact_key)

    if not recipient:
        return f"I don't have a saved email address for {contact_name}."

    if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
        return "Email credentials aren't configured. Check your .env file."

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body or "This is an automated message sent via SIRA.", "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # upgrade to encrypted TLS connection
            server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
            server.send_message(msg)
        return f"Email sent to {contact_name}."
    except Exception as e:
        return f"Failed to send email: {e}"
