PLAN = """You lead a small research team. Break the question into 2-{max_sub} focused sub-questions that
together answer it. Each must be searchable on the web on its own.

Question: {question}

Reply with JSON only: {{"sub_questions": ["...", "..."]}}"""

EXTRACT = """You are a researcher. From the page below, extract facts that help answer the sub-question.
Copy numbers, dates and names exactly. Do not add anything the page does not say.

Sub-question: {sub_question}
Page title: {title}
Page text:
{text}

Reply with JSON only: {{"facts": ["short factual statement", "..."]}}  (empty list if nothing relevant)"""

WRITE = """You are the writer. Using ONLY the numbered sources below, write a concise research brief that
answers the question.
- Start with a 2-3 sentence direct answer, then short sections with headings.
- Cite every factual sentence with source numbers like [1] or [2][4].
- If the sources disagree or leave something open, say so.

Question: {question}

Sources:
{sources}
{feedback}
Brief:"""

CRITIQUE = """You are the critic. Check the brief against the sources.

Question: {question}

Sources:
{sources}

Brief:
{draft}

Find (a) claims not supported by the cited sources and (b) important parts of the question left unanswered.
Reply with JSON only:
{{"ok": true or false,
  "issues": ["specific problems, empty if ok"],
  "follow_ups": ["up to 2 new sub-questions to research if information is missing, else empty"]}}"""

FEEDBACK = """
The critic flagged these problems in your previous draft. Fix them; drop any claim you cannot support:
{issues}
"""
