# test_assets — synthetic job-applicant fixtures

These files are **synthetic test fixtures**. They simulate a real job applicant
that the agent applies on behalf of, so the job-application flow can be exercised
end-to-end. **None of this is real personal data.**

## The applicant: Jason Statham (fictional)

A believable senior .NET engineer based in Dallas, TX, tailored to a specific
internal job posting.

## Source posting

Built from the FlashTec careers portal posting:

- URL: `http://localhost:3000/careers/senior-net-engineer-dallas-tx-FT-2024-8842`
- Role: **Senior .NET Engineer** (Job ID `FT-2024-8842`)
- Company: FlashTec — real-time payments infrastructure
- Location: Dallas, TX (Hybrid)
- Key requirements: 5+ yrs C# / .NET Core, ASP.NET Core Web APIs at scale,
  SQL Server, Azure + containers, SOLID / clean architecture / testing,
  event-driven systems (Kafka / RabbitMQ / Azure Service Bus), CS degree.

The role data was fetched from the portal's `GET /api/job` endpoint (the page
itself is a client-rendered SPA) and the applicant was tailored to match those
requirements. `jason_statham.json` top-level fields intentionally mirror the
portal application form fields (firstName, lastName, email, phone, address,
workAuthorization, currentTitle, totalExperienceYears, skills, salaryExpectation,
education, etc.).

## Files

| File | Description |
|------|-------------|
| `jason_statham.json` | Applicant data: contact, location, work history, skills, education, certifications, voluntary disclosures. |
| `jason_statham_resume.docx` | Resume generated from the JSON, tailored to the role. |
| `jason_statham_cover_letter.docx` | Cover letter referencing the FlashTec posting. |
| `generate_docx.py` | Regenerates the two `.docx` files from `jason_statham.json` (requires `python-docx`). |

## Regenerating the .docx files

```bash
pip install python-docx
python test_assets/generate_docx.py
```
