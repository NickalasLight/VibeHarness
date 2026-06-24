# Job application — agent instruction prompt

You are completing a job application on behalf of a candidate (a real user pulled
from our database). The candidate's data is in `jason_statham.json` and their
documents are `jason_statham_resume.docx` and `jason_statham_cover_letter.docx`
(all in this `test_assets/` folder).

**Task:** Apply to the posting at
`http://localhost:3000/careers/senior-net-engineer-dallas-tx-FT-2024-8842`.

Steps:
1. Open the job posting and its application form.
2. Fill out EVERY field of the application form accurately using the candidate's
   data (name, contact, address, work authorization / sponsorship, relocation,
   availability, work preference, current title/company, experience, skills,
   education, links, salary expectation, the "why interested" question, voluntary
   disclosures, signature, etc.).
3. Where the form allows file uploads, attach the candidate's resume and cover
   letter.
4. Review the completed form for accuracy, then submit the application.
5. Confirm the application was submitted successfully.

Rules:
- Use ONLY the candidate's provided data; do NOT invent personal details.
- If a required form field has no corresponding value in the candidate record,
  do not fabricate one — note it as a gap.
