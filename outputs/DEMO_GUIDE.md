# Demo Guide — Sepsis Early Detection Prototype

## Start from a cold boot

Open **Terminal 1 (backend)** and run from the project root:

```powershell
Set-Location "C:\Users\Kritika Chaudhari\OneDrive\Documents\ChatGPT\aiml\sepsis-prototype"
.\venv\Scripts\python.exe -m uvicorn src.api:app --reload --port 8000
```

Wait for `Sepsis API ready` and the endpoint list. Leave this terminal open.

Open **Terminal 2 (frontend)** and run:

```powershell
Set-Location "C:\Users\Kritika Chaudhari\OneDrive\Documents\ChatGPT\aiml\sepsis-prototype\frontend"
npm install
npm run dev
```

Open the Vite URL printed in Terminal 2 — normally **http://localhost:5173**. You should see the “Sepsis Early Detection — Prototype” dashboard: six vital inputs on the left and an empty Risk Assessment card on the right.

## Demo walkthrough

1. Click **Load High-Risk Example**. The six inputs fill from a real held-out sepsis-labelled window; the small line below the panel says `reference label: Sepsis`.
2. Click **Predict Sepsis Risk**. Point out the percentage, cohort-relative coloured risk badge, and the five red/blue SHAP bars explaining which inputs pushed the estimate up or down.
3. Click **Load Low-Risk Example**, then **Predict Sepsis Risk**. The note changes to `reference label: No Sepsis`; compare its score, badge, and explanation with the first example.
4. For a controlled manual contrast, enter HR **150**, Temp **39.5**, Resp **32**, MAP **55**, O2Sat **88**, and Lactate **6.0**. Click **Predict Sepsis Risk** and note the score/badge and SHAP bars.
5. Replace those with HR **75**, Temp **37**, Resp **14**, MAP **85**, O2Sat **98**, and Lactate **1.0**, then click **Predict Sepsis Risk** again. The displayed score should decrease. Predictions update when the button is pressed, not while a slider is moving.

The colours are cohort-relative presentation bands for this imbalanced research prototype — not clinical treatment thresholds.

## Quick troubleshooting

- **Port 8000 is already in use:** find the process with `Get-NetTCPConnection -LocalPort 8000`, stop its displayed PID with `Stop-Process -Id <PID>`, then re-run the backend command. Alternatively, use `--port 8001` and change `frontend/vite.config.js` to the same port.
- **`npm run dev` fails:** confirm Node.js is installed with `node --version`, run `npm install` again inside `frontend`, and check that the terminal is in the `frontend` folder rather than the project root.
- **Blank browser page:** use the Vite URL printed by Terminal 2 (not a bookmarked address), check Terminal 2 for a compile error, and refresh with `Ctrl+F5`. If the page loads but requests fail, open `http://localhost:8000/health`; it should return `{"status":"ok"}`.

## If asked why AUC dropped after removing ICULOS

“The earlier model was getting a big advantage from knowing how long someone had already been in the ICU. That can look impressive in a graph, but it is closer to a timing clue than a real early-warning signal from the patient’s condition. We removed it on purpose so the model has to rely more on vitals and laboratory patterns. The score dropped a little because the test became more honest; that is a better foundation for improving a genuinely useful early-detection tool.”
