# Gemini Instructions

## 1. Blog Writing Style & Structure (`_posts/`)
- **Deep Technical Rigor:** When writing or expanding on systems engineering topics (like ELF, ABIs, memory management, or hardware interactions), always prioritize deep, technically accurate explanations over surface-level overviews. Discuss the *why* and the *how* (e.g., compiler assumptions, memory layouts).
- **Engaging Tone:** Maintain a professional yet engaging tone suitable for a senior engineering audience.

## 2. Citations and References
- When linking to external documentation or references, prefer using Markdown footnotes (e.g., `[^1]`) in the body text.
- Define these footnotes in a dedicated `## References` section at the very end of the file (e.g., `[^1]: **Title:** Description. ([Link](url))`).

## 3. Code Snippets
- Always use correct language tags for syntax highlighting (e.g., `c`, `cpp`, `assembly`).
- When providing explanatory code (like host vs. device execution), include clear, numbered, instructional comments to walk the reader through the logic.

## 4. General Code Auditing & Systems Work
- When debugging or auditing systems code (like standard libraries, memory allocators, or hardware interfaces), be pedantic about low-level constraints like endianness, alignment, word sizes, and ABI calling conventions.

## 5. Disclaimer
At the end of every article write "*Disclaimer: This article was generated using the Gemini 3.1 Pro model.*"

## 6. Repo
- This is a git repo.
- Commit often.
- Format with `mdformat`.