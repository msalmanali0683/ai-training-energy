# Collection log: `reported_energy.csv`

Collected 2026-11-07. Every value in the CSV was read from the primary source fetched in this session: WebFetch of the arXiv PDF or HTML, the GitHub raw model card, or the HF page. PDFs were converted with `pdftotext -layout`. Where the text layer was misaligned, tables were re-extracted with PyMuPDF. No value was filled in from memory.

## Conventions
- **Units.** Values are kept as the source states them. Values given in kWh were converted to MWh, values in kg to t, and values in lbs to t (Strubell). Every conversion is noted in `notes`.
- **Energy basis is not uniform.** The `notes` column says whether a reported MWh includes PUE:
  - Includes PUE: Patterson 2021, LLaMA 1, Granite, Strubell, Morrison 2025.
  - Excludes PUE: OLMo 2024, OLMo 2, BLOOM, StarCoderBase.
- **Derived values are kept out of the numeric columns** unless the source states the components. Three cases have computed sums; each is flagged in `notes`:
  - Llama 3.2 Vision GPU-hours = sum of the stated stages.
  - StarCoder GPU-hours = 320,256 + 11,208.
  - StarCoder CO2 = 16.68 + 0.58 t.
- Other derived quantities appear only in `notes`, never as values: PaLM chip-hours and MWh, Gopher MWh and chip-hours, Meta implied MWh.
- **Third-party values.** Some rows hold values estimated by someone other than the developer, but always by the authors of the cited source: GPT-3 (Patterson), Strubell rows, OPT notes. These are marked `estimated_by_authors`, and the notes say who made the estimate.

## Sources tried

| # | Source | Opened? | Used? | Notes |
|---|---|---|---|---|
| 1 | Patterson et al. 2021, arXiv:2104.10350 (PDF) | Yes | Yes, 6 rows | Table 4: ET-NAS, T5, Meena, GShard-600B, Switch, GPT-3 |
| 1b | Patterson et al. 2022, arXiv:2204.05149 (PDF) | Yes | No | GLaM and GPT-3 appear only in Fig. 3 (graphic), plus CI 0.088 vs 0.429. No tabulated GLaM/PaLM energy. GLaM taken from its own paper instead |
| 1c | GLaM, Du et al., arXiv:2112.06905 (PDF) | Yes | Yes, 1 row | §6.4 and App. F |
| 1d | PaLM, arXiv:2204.02311 v5 (PDF) | Yes | Yes, 1 row | App. B. CO2 only; energy not stated |
| 1e | Gopher, arXiv:2112.11446 v2 (PDF) | Yes | Yes, 1 row | App. F. CO2 only |
| 2 | Luccioni et al. BLOOM, arXiv:2211.02001 v1 (PDF) | Yes | Yes, 1 row | Table 1, §4.2, Table 4. The JMLR version was not opened separately |
| 3a | Llama 2 MODEL_CARD.md (GitHub raw) | Yes | Yes, 3 rows | |
| 3b | Llama 2 paper arXiv:2307.09288 | **No** | n/a | PDF >10 MB (WebFetch limit), and one timeout. Model card used instead (it carries the same CO2 table) |
| 3c | Llama 3 MODEL_CARD.md | Yes | Yes, 2 rows | |
| 3d | Llama 3.1 MODEL_CARD.md (llama-models) | Yes | Yes, 3 rows | |
| 3e | Llama 3.2 MODEL_CARD.md + MODEL_CARD_VISION.md | Yes | Yes, 4 rows | SpinQuant/QLoRA quantization rows excluded (post-hoc quantization, not training a model) |
| 3f | Llama 3.3 MODEL_CARD.md | Yes | Yes, 1 row, **flagged** | Its 12.9 t for 7.0M GPU-h looks like a card error |
| 3g | Llama 4 MODEL_CARD.md | Yes | Yes, 2 rows | Behemoth: no values |
| 4 | LLaMA, arXiv:2302.13971 (PDF) | Yes | Yes, 4 rows | Table 15. Its OPT and BLOOM rows are LLaMA-authors' re-estimates at 0.385 kg/kWh. They are mentioned in notes, not used as rows |
| 5 | OPT, arXiv:2205.01068 (PDF) | Yes | Yes, 1 row | §6. 75 t CO2 only |
| 6a | Gemma, arXiv:2403.08295 (PDF) | Yes | **Excluded** | Only a combined figure (~131 tCO2eq for 2B + 7B). Not per model |
| 6b | Gemma 2, arXiv:2408.00118 (HTML) | Yes | **Excluded** | Combined 1247.61 tCO2eq for all sizes. Not per model |
| 6c | Gemma 3, arXiv:2503.19786 (HTML) | Yes | **Excluded** | No carbon/energy statement found. Only TPU type and chip counts |
| 6d | OLMo, arXiv:2402.00838 v4 (PDF) | Yes | Yes, 2 rows | Table 6. Its rows for Gopher/BLOOM/OPT/T5/LLaMA are secondary and were not used as rows |
| 6e | OLMo 2, arXiv:2501.00656 v3 (PDF) | Yes | Yes, 2 rows | Table 19 |
| 6f | Morrison et al. 2025 (AI2), arXiv:2503.05804 (PDF) | Yes | Yes, 3 rows | Table 2: OLMo 1B, OLMo 7B 0424/0724, OLMoE. Rows marked † (unreleased internal models) were excluded. Its values for OLMo 7B and OLMo 2 7B/13B appear in notes only |
| 6g | Olmo 3, arXiv:2512.13961 v2 (PDF) | Yes | **Excluded** | §2.4 gives only wall-clock (~56 days, 1024 H100) and a $ figure. No energy, CO2 or explicit GPU-hours |
| 6h | Falcon series, arXiv:2311.16867 (HTML, first 100k chars) | Partly | Hardware only | Compute given in PF-days |
| 6i | Falcon-180B HF model card | Yes | No numbers | No GPU-h |
| 6j | HF blog "Falcon 180B" | Yes | Yes, 1 row | "~7,000,000 GPU hours" |
| 6k | StarCoder, arXiv:2305.06161 v2 (PDF) + HF model card | Yes | Yes, 2 rows | |
| 6l | StarCoder2, arXiv:2402.19173 (PDF; only v1 was served) | Yes | Yes, 2 rows | The 15B estimate is absent in v1. The HF card for starcoder2-15b gives only "1024 x H100" |
| 6m | Granite 3.0 tech report (GitHub paper.pdf) | Yes | Yes, 4 rows | Table 7 |
| 6n | Granite Code, arXiv:2405.04324 (PDF) | Yes | **Excluded** | Only a combined 455 tCO2eq for all Granite Code models |
| 6o | DeepSeek-V3, arXiv:2412.19437 (HTML) | Yes | Yes, 1 row | Table 1 GPU-h |
| 6p | DeepSeek-R1, arXiv:2501.12948v2 (HTML) | Yes | **Excluded** | Training cost is in Supplementary B.4.4, which is not in the HTML |
| 6q | DeepSeek-R1, Nature s41586-025-09422-z | **No** | n/a | Redirected to an auth page. Press reports (512/648 H800, ~80 h, $294k) could not be verified in the primary source |
| 6r | BloombergGPT, arXiv:2303.17564 (PDF) | Yes | **Excluded** | 1.3M GPU-h is a planned budget, not actual. Actual is only 512 A100 x ~53 days. No explicit actual GPU-h or energy |
| 6s | Phi, Mistral, Qwen | Not fetched | n/a | Skipped for time and not checked in this session. **Worth a later check** |
| 7 | Strubell et al. 2019, arXiv:1906.02243 (PDF) | Yes | Yes, 6 rows | Table 3. TPU rows (BERT-base TPU, NAS TPU) lack power/CO2. GPT-2 is kept as hardware x time only |
| 8 | Luccioni & Hernandez-Garcia 2023, arXiv:2302.08476 (PDF) | Yes | **Excluded** | Per-model values are not in the paper text (figures only). They come from author surveys plus TDP estimates by the survey authors, with data in a GitHub repo. Not primary |
| 9 | Morand, Ligozat, Névéol 2025, arXiv:2510.09022 | Yes | **Excluded** | Estimates derived from the Epoch database itself. Circular for validation |

## Doubts / caveats
1. **Llama 3.3 70B.** The card states 7.0M GPU-h with only 12.9 t location-based, against 2,040 t for Llama 3.1 70B at the same GPU-h. Probably a typo. Recommend excluding this row.
2. **OLMo 2 13B.** The OLMo 2 paper (257 MWh pre-PUE, 101 t) and Morrison 2025 (230 MWh, 101 t) disagree on energy. The OLMo 2 paper is used.
3. **The "energy" column mixes PUE-inclusive and GPU-only values.** BLOOM, StarCoderBase and OLMo are GPU-only. Normalise using `pue_reported` before comparing.
4. **Meta model cards report no MWh**, only GPU-h x peak W "adjusted for PUE". The implied MWh is a TDP upper bound.
5. **Strubell values are scaled-up estimates** from short runs on different hardware, and are in lbs. Treat them as low-quality ground truth.
6. **Third-party estimate.** GPT-3 energy/CO2 is Patterson's estimate, not OpenAI's.
7. **Epoch name mapping:**
   - `Switch` (1.571T) is assumed to equal Patterson's "Switch Transformer 1500B".
   - `Transformer` (213M) is mapped to Strubell's Transformer-big.
   - `StarCoder` is mapped to the cumulative StarCoder row, not StarCoderBase.
   - `Llama 3.2 11B` is mapped to the 11B Vision row.
   - GShard-600B is not matched: Epoch has only the dense 2.3B variant.
8. **Two OLMo-7B rows exist** because there were two independent full training runs on different hardware. Neither is in Epoch.
