# SnapQC

Read `../COORDINATION.md` and all session notes in `../.agent-coordination/`.
Claim scope under the registry lock before making shared changes.

Read `../icxhnk-labs/README.md` before onboarding, backend implementation, or
deployment. Its Definition of Done is mandatory for production integration:
- Use `https://accounts.icxhnk.com` as production issuer with canonical `IDENTITY_*` env names.
- Never display `ICXHNK` as product branding in user-facing UI; always use **SnapQC**.
- Maintain `/whats-new` in desktop sidebar and mobile navigation for every user-facing change.
- Never write secrets to Git, logs, docs, or agent output.
- Report real blockers honestly; never claim trained ML accuracy or live production readiness from design prototypes.

Use `✅` for verified success and `❌` for failures/blockers, and end updates with `🟢 Progress: NN%`.
