import smtplib
from email.mime.text import MIMEText

EMAIL_ADDRESS = "saniagemmathew@gmail.com"
EMAIL_PASSWORD = "ndwtuhlkvyjgcqqv"  # Use the app password here
RELATIVE_EMAIL = "saniagemmathew@gmail.com"

subject = "Test Email"
body = "This is a test email to check SMTP functionality."

msg = MIMEText(body)
msg['From'] = EMAIL_ADDRESS
msg['To'] = RELATIVE_EMAIL
msg['Subject'] = subject

try:
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    server.sendmail(EMAIL_ADDRESS, RELATIVE_EMAIL, msg.as_string())
    server.quit()
    print("✅ Test Email Sent Successfully!")
except smtplib.SMTPAuthenticationError as auth_err:
    print(f"❌ Authentication Error: {auth_err}")
except smtplib.SMTPException as smtp_err:
    print(f"❌ SMTP Error: {smtp_err}")
except Exception as e:
    print(f"❌ General Error: {e}")
