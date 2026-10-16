# FindIt AI — API deployment

This folder is the standalone deployable server: image uploads + multilingual CLIP search. Do not upload `.env`, user photos, tokens, or credentials to GitHub.

## Render free deployment

1. Sign in to GitHub and Render. Create a new repository containing only the source files in the clean deployment ZIP; the ZIP excludes `.env` and API secrets.
2. In Render create a Blueprint from that repository; `render.yaml` selects the **Free** web-service plan. Confirm Free is selected. No paid upgrade is authorized.
3. Set `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, and `CLOUDINARY_API_SECRET` in Render's environment settings. Set the Firebase project ID to `findit-ai-d5212`. Never put these secrets in Dart.
4. Docker downloads pinned quantized ONNX models at build time. Check the build and deployment logs. GET `/health` must return `status: ok` and `images_configured: true`.
5. Update the Flutter run/build setting: `--dart-define=IMAGE_API_URL=https://YOUR-SERVICE.onrender.com`. The same URL handles images and search. Rebuild the app. Release builds reject HTTP URLs.
6. With the computer's local image server stopped, sign in, add a photo report, then search by Arabic description, by photo, and by both. Verify from a second account.

Render Free currently sleeps after 15 minutes without requests. The next request can be slow. It is a prototype environment, not an uptime guarantee. Firestore and Cloudinary remain the durable stores; embeddings are rebuilt in memory after restarts.

## Search behavior and limits

- Real multilingual CLIP text embeddings, and CLIP vision embeddings, with quantized ONNX on CPU; no fake similarity results and no training from scratch.
- Text mode compares descriptions. Image mode compares available images. Hybrid combines normalized text and image vectors in a shared 512-dimensional space. Arabic quality must be evaluated on project-specific data.
- Reads the newest 150 Firestore reports using the caller's Firebase ID token, respecting Firestore rules. Keeps only found, unreturned items and the selected category. If the dataset exceeds this window, the response/UI disclose the limit. No location/date ranking is implemented yet.
- Scores are cosine similarities presented on a 0–100 scale, not probabilities or ownership proof. The initial weights are equal, not tuned on a labeled dataset.
- The server only fetches image URLs from the configured Cloudinary account, without redirects, limits photo bytes/dimensions, bounds caches and serializes search requests. Unavailable candidate images are skipped and disclosed.
- Uploaded query images stay in request memory/caches and are not uploaded to Cloudinary or Firestore. Restart clears caches. Public Cloudinary item-photo URLs are not private document storage.
- Concurrent search returns HTTP 429 for retry. No billing credentials or paid deployment is configured by this package.

## Local development

Create `.env` locally from `.env.example`; install `requirements.txt`, then run:

```powershell
python -m uvicorn app:app --env-file .env --host 127.0.0.1 --port 8000
```

Set `MODEL_CACHE` to a folder on D before model downloads to avoid filling C. `python embeddings.py` downloads pinned models (about 228 MB). Debug Flutter on the Android emulator defaults to `http://10.0.2.2:8000`.

## Validation

Run `pytest -q`. Unit tests mock external services; real model smoke testing validates Arabic/English embeddings and image vectors. Full cloud deployment and independent-account image/text searches require a live account and public HTTPS endpoint.
