# VisionFind local interface

This is a local chat interface for the running Kaggle API. No models are downloaded to your PC. The API token is stored in `.env` and used only by the local Python server.

## Windows / VS Code setup

1. Extract this ZIP and open `visionfind_local` in VS Code.
2. Open Terminal → New Terminal. Run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe configure.py
```

Paste your current HTTPS tunnel URL when asked. The token prompt is hidden.

To retrieve your own token, run `print(API_TOKEN)` in a PRIVATE Kaggle cell. Copy it to the configuration prompt on your own computer. Clear that Kaggle output afterward and do not publish the notebook output or share the token in chat.

3. Start:

```powershell
.\.venv\Scripts\python.exe run.py
```

4. Open http://127.0.0.1:7860 in your browser. Keep the terminal and Kaggle running.

No virtual-environment activation is required. If `py` is unavailable, install Python 3.10 or newer, or use `python` instead.

## Features

- Minimal chat tab; separate Help tab.
- Enter sends; Shift+Enter creates a newline.
- Up to ten image attachments, or one video (30 seconds/100 MB backend limit).
- Small attachment thumbnails; click to enlarge. Annotated output images and processed videos appear in the conversation.
- Queued jobs with waiting/working status. Refresh resumes a pending job in the same browser tab.
- Local server proxies authenticated media, including video range requests.
- Token stays out of browser code and requests.
- Microphone: browser speech recognition → editable text → Send. Browser-dependent and may use an external recognition service. Kaggle Whisper transcription is NOT connected in this version.

## When the connection changes

Stop the local server with Ctrl+C. Run configure.py again with the new URL/token, then run run.py. If Kaggle restarted, click New chat. Old sessions are no longer present on the backend. The local UI and backend must both remain active.

## Testing and limits

Run `python test_proxy.py` to check the local proxy with a fake backend. Tests do not establish visual/model accuracy. The live token was not supplied to the package author, so PC-to-Kaggle connectivity must be confirmed on your computer.

This is a personal demo. The server binds to 127.0.0.1, not a public network interface. Sessions and jobs depend on the Kaggle process. Video explanations can hallucinate; object detections can miss or mislabel objects. Video processing is not live inference. No fine-tuning is included.

Do not commit `.env`. No token is included in this ZIP. The package is standalone; it does not change your model handlers.

Technical sources: https://fastapi.tiangolo.com/tutorial/static-files/ and https://www.python-httpx.org/async/.
