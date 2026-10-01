# Answer grounding

Retrieval finds potentially relevant passages; it does not prove that those passages answer a question.
For example, DAFI 36-2903 refers readers to AFI 34-1201 for general customs and courtesies. That referral
does not supply the contents of AFI 34-1201. Adjacent uniform-travel permissions do not establish
saluting exemptions.

The chatbot now:

1. Joins overlapping passages from the same source page to restore split sentences.
2. Requests supporting quotations or an insufficient-evidence decision.
3. Checks quotation text against the source, permitting PDF whitespace repairs and Markdown emphasis.
4. Restores sentence context so a cropped referral cannot be used as evidence for another publication's rules.
5. Checks the proposed answer for relevance and support. An unsupported summary can fall back to verified
   source quotations; missing, irrelevant, or unverifiable evidence produces an insufficient-evidence response.

The internal decisions use [LM Studio's structured JSON output](https://lmstudio.ai/docs/developer/openai-compat/structured-output).
Answerable questions can require three generation calls, increasing latency. The same model performs
the semantic checks, so they reduce errors rather than guarantee correctness. Verify consequential answers
with `/sources`. Some valid questions can receive conservative refusals.

To answer general customs-and-courtesies questions, add the applicable AFI 34-1201 document to `documents/`
and run `ingest.py` again. Verify the edition you intend to use. DAFI 36-2903 itself contains some narrower
uniform-specific saluting provisions, which can still be answered when the retrieved text directly supports them.

Run the regression checks from this folder:

```powershell
& '..\.venv\Scripts\python.exe' -B -m unittest test_grounding -v
```

Live checks performed: the broad saluting question declined and cited the AFI 34-1201 referral;
the five dress-and-appearance elements, maternity mess-dress saluting exception, and publication scope
returned supported answers or exact source quotations.
