# Prompt: Verify Implementation Against README.md and Deploy

You are tasked with verifying that `main.py` correctly implements the web application as described in `README.md`, then pushing the code to GitHub and running the app via Streamlit.  
Follow the steps below in order. Be thorough and provide evidence at each step.

---

## Step 1: Read `README.md` — Extract Requirements
- Carefully read `README.md` to understand the **application’s purpose**, **functional requirements**, **UI structure**, and **technical constraints**.  
- Build a **requirement checklist**, including:
  - Core features.
  - Expected inputs/outputs.
  - UI structure and components.
  - Data flow and processing steps.
  - Libraries and coding conventions.
  - Deployment instructions (if included).

👉 *Goal*: Create a structured “expectation map” for later verification.

---

## Step 2: Inspect `main.py` — Static Verification
- Review the `main.py` source code without executing it.
- Check that:
  - All required features are present.
  - The structure matches README’s description (functions, UI layout, logic).
  - Only allowed libraries are imported.
  - Functions have docstrings, type hints, and proper error handling.
  - Caching and data handling are implemented correctly.
- Map each requirement from Step 1 to the relevant part of the code.
- Record **evidence (function names, line numbers)** and mark each as **Pass**, **Partial**, or **Fail**.

👉 *Goal*: Verify code coverage and structure statically.

---

## Step 3: Run `main.py` — Dynamic Functional Testing
- Launch the app locally (e.g., `streamlit run main.py`).
- Test all UI components and workflows:
  - Verify inputs, sliders, file uploads, and model parameters work as described.
  - Check that outputs (plots, metrics, tables, reports) match the expected behavior.
  - Intentionally try **edge cases** (e.g., invalid CSV, empty inputs) to confirm error handling.
- Record evidence (screenshots, logs, output snippets).

👉 *Goal*: Ensure the application behaves correctly in practice.

---

## Step 4: Cross-Check README vs Code
- Compare the **requirement checklist** with the actual implementation and behavior.
- For each item, record:
  - **Requirement**
  - **Evidence** (line number, function, or screenshot)
  - **Status** (Pass / Partial / Fail)
  - **Notes** (any differences or suggestions)
- Identify any gaps or deviations between README and code.

👉 *Goal*: Provide a clear mapping and highlight discrepancies.

---

## Step 5: Push Code to GitHub
- Initialize a Git repository if needed.
- Commit all relevant files (`main.py`, `README.md`, requirements, etc.).
- Create a **clean branch** (e.g., `feature/streamlit-app`) or push to `main`.
- Use appropriate commit messages, e.g.:

---

## Step 6: Deploy & Run via Streamlit
- Use Streamlit to run the application either locally or via Streamlit Cloud:
- **Local**: Run `streamlit run main.py` and verify functionality again.
- **Streamlit Cloud**:
  1. Log in to [https://streamlit.io](https://streamlit.io).
  2. Deploy the GitHub repository.
  3. Set Python version and `requirements.txt`.
  4. Run the deployed app.
- Check:
- UI renders correctly in a browser.
- All interactive elements work as expected in the deployed environment.
- Links/download buttons (e.g., model file, markdown report) function correctly.
- No library or path errors appear in Streamlit Cloud logs.

👉 *Goal*: Confirm successful deployment and public access to the working app.

---

## Step 7: Final Verification Report
- Summarize the overall verification results:
- ✅ **Confirmed** — if all requirements match, with evidence.
- ⚠️ **Partially Confirmed** — list missing or incomplete items and recommendations.
- ❌ **Not Confirmed** — list critical missing functionality or failures.
- Include:
- A requirement vs implementation table.
- Screenshots and evidence links.
- GitHub repo URL and Streamlit deployment URL.

👉 *Goal*: Deliver a structured, evidence-based verification and deployment summary.
