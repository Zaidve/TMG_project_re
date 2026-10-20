# Mobile version: notes for future work

Status (2026-10-20): **parked**. The desktop extension is the current deliverable. This file
records what was learned and what is already in place, so mobile work can resume later.

## What is already in the code

- **Service on the local network** (`service/app.py`):
  `python -m service.app --host 0.0.0.0 --api-key <key>` accepts connections from other
  devices on the same Wi-Fi. In that mode it refuses to start without an access key, and
  every request must send it in the `X-API-Key` header. Tested: no key or a wrong key gets
  401; the right key works.
- **Extension settings** (`extension/popup.*`, `background.js`): the "Dịch vụ mô hình"
  section takes a service address and access key; **Lưu** asks the browser for permission to
  reach that address (`optional_host_permissions`).
- **Store package**: `extension/` zipped (see `dist/clickbait-detector-<version>.zip`) is
  what Microsoft Edge Add-ons and the Chrome Web Store accept.

Desktop behaviour is unchanged by all of this: by default the service listens on
`127.0.0.1` only and needs no key.

## Which phone browsers can run it

| Browser | Extensions | Route |
|---|---|---|
| Chrome (Android, iOS) | No | Not possible |
| Edge for Android | Limited, Canary channel | Publish to Edge Add-ons (visibility *Hidden*), then in Edge Canary: `edge://flags` → enable *Android Extension* → Settings → About → tap the version 5 times → *Extension install by ID* |
| Firefox for Android | Yes | Small manifest changes (`browser_specific_settings`, background scripts), then test over USB with Mozilla's `web-ext run --target=firefox-android` without publishing; publish on addons.mozilla.org for others |
| Safari (iOS) | Yes (web extensions) | Package with Xcode as an iOS app; needs a Mac and an Apple developer account |

The Edge Canary steps come from a December 2024 report (Windows Latest); recheck them
before relying on them. Edge for Android's extension support is new, so APIs such as
`scripting.registerContentScripts` and `permissions.request` may behave differently there.

## The model service for phones

A phone cannot reach `127.0.0.1` on a computer. Options:

1. **Same Wi-Fi, for testing**: run the service with `--host 0.0.0.0 --api-key ...`, allow it
   through Windows Firewall on private networks only, and enter `http://<computer-ip>:8000`
   plus the key in the extension popup. Switch back to the default mode afterwards.
2. **Hosted, for real use**: a small server or cloud VM with HTTPS. The model needs about
   1 GB of RAM and runs on CPU (about 70 ms per headline). Needs an access key, rate limiting
   and a privacy note, since headlines would then leave the phone.
3. **On the phone itself**: export to ONNX and quantise (540 MB → about 135 MB), run with
   onnxruntime. Hard part: PhoBERT needs `pyvi` word segmentation and its BPE tokenizer,
   which would have to be reimplemented in JavaScript; the model was trained on segmented
   text, so skipping segmentation needs retraining with `segment=False` first.

## Suggested order when resuming

1. Decide where the model runs (hosted service is the most practical).
2. Firefox for Android first: closest to the current code, testable over USB without a store.
3. Edge for Android via a hidden Edge Add-ons listing.
4. Safari/iOS last (needs a Mac).
