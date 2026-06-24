# Job application — agent instruction prompt

You are filling out a job application on behalf of a candidate. ALL of the
candidate's information is written out below in plain text — use it directly to
fill the form. You do NOT need to read any file for the candidate's data.

**Apply at:** http://localhost:3000/careers/senior-net-engineer-dallas-tx-FT-2024-8842
(open the application form at `/careers/senior-net-engineer-dallas-tx-FT-2024-8842/apply`).

## Candidate information (use these exact values)

**Personal**
- First name: Jason
- Last name: Statham
- Email: jason.statham.dev@example.com
- Phone: +1 (214) 555-0199
- Address: 1820 McKinney Avenue, Apt 14C
- City: Dallas — State: TX — ZIP/postal code: 75201 — Country: United States
- LinkedIn: https://www.linkedin.com/in/jason-statham-dotnet
- GitHub: https://github.com/jstatham-dev
- Portfolio: https://jstatham.dev

**Work eligibility**
- Work authorization: U.S. Citizen
- Requires visa sponsorship: No
- Willing to relocate: No — the candidate already lives in the Dallas, TX area
- Work preference: Hybrid
- Available start date: 2026-07-21 (July 21, 2026)

**Experience & compensation**
- Current title: Senior Software Engineer
- Current company: Lone Star Payments Inc.
- Total years of experience: 9
- Years of .NET experience: 9
- Salary expectation: $165,000 (if a plain number is required, enter 165000)

**Work history**
1. Senior Software Engineer — Lone Star Payments Inc., Dallas TX (Mar 2021–Present): C#/.NET 8 microservices processing 3M+ transactions/day at sub-100ms p99; re-architected the ledger with DDD (−70% reconciliation defects); event-driven pipelines on Azure Service Bus + Kafka; SQL tuning cut settlement batch 45m→6m; mentors 4 engineers; on-call.
2. Software Engineer II — Brazos Software Group, Austin TX (Jun 2018–Feb 2021): ASP.NET Core Web APIs for a B2B invoicing platform (12,000 merchants); migrated a .NET Framework monolith to containerized .NET Core on Azure AKS; gRPC contracts; Azure DevOps CI/CD.
3. Software Engineer — Trinity Web Solutions, Dallas TX (Jul 2016–May 2018): C#/ASP.NET MVC logistics apps; normalized SQL Server schemas + stored procedures; xUnit tests.

**Skills:** C#, .NET 8 / .NET Core, ASP.NET Core Web API, Domain-Driven Design, Clean Architecture, SOLID, Microsoft SQL Server, Entity Framework Core, Dapper, Azure (App Services/Functions/AKS), Docker, Kubernetes, Azure Service Bus, Apache Kafka, RabbitMQ, gRPC, xUnit/automated testing, CI/CD (Azure DevOps, GitHub Actions), Terraform/Bicep, OpenTelemetry.

**Education**
- Degree: Bachelor of Science in Computer Science (if a degree dropdown is shown, pick "Bachelor's Degree")
- School: The University of Texas at Dallas
- Field of study: Computer Science
- Graduation: May 2016

**Additional questions**
- Why interested: "FlashTec's bet on .NET, Azure, and event-driven architecture to move real money at five-nines reliability is exactly the problem space I have spent my career in. I want to bring my payments and ledger experience to a Dallas-based platform team where I can own high-throughput services end-to-end and mentor the next generation of engineers."
- How did you hear about us: Company careers page (if a dropdown, pick "Company Website")

**Voluntary self-identification**
- Gender: Decline to self-identify
- Ethnicity: Decline to self-identify
- Veteran status: I am not a protected veteran
- Disability status: Decline to answer

**Certification / signature**
- Certify the information is true: Yes
- Consent to data processing: Yes
- Signature name: Jason Statham

**Documents to upload** (where the form has a file upload):
- Resume: `test_assets/jason_statham_resume.docx`
- Cover letter: `test_assets/jason_statham_cover_letter.docx`

## Steps
1. Open the application form.
2. Fill EVERY field using the candidate information above.
3. Upload the resume and cover letter where the form allows.
4. Review the form for accuracy, then submit; confirm it was submitted.

Rules: use ONLY the information above; do NOT invent details. If a required field
has no value here, note it rather than making one up.
