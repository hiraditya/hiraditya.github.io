---
title: "Where Should Your Code Live?"
date: 2026-07-14 12:00:00 -0700
categories: [Systems, Infrastructure]
tags: [git, code-hosting, open-source, infrastructure]
---

*A survey of code repository hosting in 2026*

Most developers do not think about where their Git repository lives until something goes wrong. An account gets suspended. A DMCA notice takes down a project overnight. A platform changes its terms of service to train AI models on your code. Then it becomes urgent.

This post is specifically about **code repository hosting** — where your Git history, branches, and collaboration happen. I am not covering CI/CD, deployment, or compute hosting. Those are separate concerns with separate providers. The question here is narrow: where should your source code live, and what are the tradeoffs?

## The Incumbents

### GitHub

GitHub is the default. Most open-source projects live there. Most hiring managers look there. Most tooling integrates with it first. If you have no strong reason to be elsewhere, inertia keeps you on GitHub.

But there are reasons to think carefully about this default:

- **DMCA compliance.** GitHub is a US-based entity and complies aggressively with DMCA takedown requests. The `youtube-dl` takedown in 2020 was the most visible example, but smaller projects get hit regularly. The process is automated and weighted toward the claimant. Your project can go offline before you have a chance to respond.
- **AI training.** In April 2026, GitHub updated its policy to use interaction data from Copilot Free, Pro, and Pro+ users to train its AI models by default[^1]. This includes code snippets and surrounding context from files you are actively editing. Enterprise users are exempt. Individual developers are opted in unless they manually disable the setting. This is a meaningful policy distinction.
- **Account suspension.** GitHub can suspend accounts for terms-of-service violations, and the appeals process is opaque. If your account is suspended, your repositories become inaccessible. For a solo developer, this can mean losing years of work history, issue discussions, and release artifacts in a single action.
- **Microsoft ownership.** This is a matter of individual judgment. Some developers are comfortable with Microsoft stewardship. Others are not. The concern is not about today's policies but about the structural fact that a single corporation controls the platform where the majority of the world's open-source code lives.

None of these are reasons to leave GitHub in a panic. They are reasons to understand what you are trading for the convenience.

### GitLab.com

GitLab is the most feature-complete alternative to GitHub. It offers repository hosting, issue tracking, merge requests, and a comprehensive DevOps platform. The self-managed edition (GitLab CE/EE) can be installed on your own infrastructure, which gives you full control over the data.

On the code hosting side specifically:

- GitLab has publicly stated that it does not train AI models on customer code at any tier[^2].
- DMCA handling follows standard legal compliance, but GitLab has historically been more responsive to due diligence than GitHub.
- The interface is heavier than GitHub's. For pure code hosting, it can feel like overkill.

### Bitbucket

Owned by Atlassian. The natural choice if your team already uses Jira, Confluence, and the rest of the Atlassian stack. As a standalone code hosting platform, it has few distinguishing features over GitHub or GitLab. Atlassian has been deprioritizing Bitbucket's standalone offering in favor of tighter integration with its project management tools.

## The Alternatives

### Codeberg

Codeberg[^3] is a non-profit code hosting platform based in Germany, run by the Codeberg e.V. association. It is powered by Forgejo (see below) and is the closest thing to a "drop-in replacement" for GitHub in terms of user experience.

Why it matters:

- **Non-profit governance.** There is no board of directors optimizing for revenue. The platform exists to serve its users. This changes the incentive structure around every policy decision.
- **GDPR jurisdiction.** Hosted in Germany, subject to European data protection law. DMCA-style takedowns require rigorous legal proof before action is taken. The bar is higher than in the US.
- **No AI training.** Your code is not scraped for model training.
- **Familiar interface.** If you know GitHub, you know Codeberg. Pull requests, issues, wikis, releases — the workflow is nearly identical.

The tradeoff is scale. Codeberg runs on non-profit infrastructure. It can be slower than GitHub during peak traffic. The community is smaller, which means fewer eyeballs on your project if discoverability matters to you.

### SourceHut

SourceHut[^4] (sr.ht) is the most opinionated platform on this list. Created by Drew DeVault, it is built around a philosophy of minimalism, privacy, and email-based workflows.

The distinguishing characteristics:

- **Email-native collaboration.** There are no pull requests. Contributors send patches via `git send-email` to the project's mailing list. Code review happens in email threads. This is how the Linux kernel and many other large projects have worked for decades. It is unfamiliar to developers raised on GitHub's web UI, but it is well-tested at scale.
- **No JavaScript required.** The web interface is fast, lightweight, and works in any browser. No tracking, no analytics, no ads.
- **Paid model.** SourceHut charges a subscription (approximately $5–$15/month, with financial aid available). This means you are the customer, not the product. The business model is not advertising or data monetization.
- **CI/CD.** SourceHut has a built-in build system (`builds.sr.ht`) that supports multiple Linux distributions and BSDs natively.

The tradeoff is the learning curve. If your contributors are accustomed to clicking "Create Pull Request" on a web page, asking them to learn `git format-patch` and `git send-email` is a real barrier to contribution. This is a deliberate choice by the platform, not an oversight.

### Forgejo and Gitea (Self-Hosted)

Forgejo[^5] and Gitea are lightweight, self-hosted Git forges written in Go. Forgejo is a community-driven fork of Gitea that split off to ensure non-profit governance after concerns about Gitea's commercial direction.

Self-hosting gives you:

- **Total data sovereignty.** No third party can suspend your account, take down your repository, or change the terms of service. Your code lives on hardware you control.
- **Policy control.** You define the moderation policy, the access controls, and the data retention rules.
- **Low resource requirements.** Both Forgejo and Gitea run comfortably on a Raspberry Pi or a $5/month VPS.

The tradeoff is operational responsibility. You are responsible for backups, updates, TLS certificates, uptime, and security patches. For a solo developer, this is manageable. For a team, it requires someone to be on-call for infrastructure.

If you want the Forgejo/Gitea experience without self-hosting, Codeberg is exactly that — a managed Forgejo instance run by a non-profit.

### Radicle (Peer-to-Peer)

Radicle[^6] is a fundamentally different architecture. It is a peer-to-peer code collaboration network with no central server. Repositories are replicated across nodes in a P2P network, and identity is managed via public-key cryptography.

Why this matters:

- **No single point of takedown.** There is no server to serve a DMCA notice to. Code propagates across the network. Removing it from one node does not remove it from others.
- **Local-first.** Your repository lives on your machine first. Network replication is secondary. You can work offline indefinitely.
- **Sovereignty by design.** Social artifacts like issues and code reviews are stored as Git objects, not in a proprietary database controlled by a platform.

The tradeoff is maturity. Radicle is functional but lacks the polish and feature breadth of established platforms. There is no integrated CI/CD, no complex project management, and the contributor onboarding experience is steeper than any server-based forge.

Radicle is best suited for projects that prioritize censorship resistance above all else, or for developers who want to decouple their collaboration workflow from any platform provider entirely.

## Comparison

| | **GitHub** | **GitLab.com** | **Codeberg** | **SourceHut** | **Forgejo** (self-hosted) | **Radicle** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Governance** | Microsoft (corp.) | GitLab Inc. (corp.) | Non-profit (Germany) | Indie (paid) | You | P2P network |
| **Jurisdiction** | US | US (self-host anywhere) | Germany (GDPR) | US | Yours | None |
| **AI training** | Yes (opt-out) | No | No | No | N/A | N/A |
| **DMCA risk** | High (automated) | Medium | Low (legal review) | Low | None | None |
| **Workflow** | Web-based PRs | Web-based MRs | Web-based PRs | Email patches | Web-based PRs | P2P/local |
| **Cost** | Free / paid tiers | Free / paid tiers | Free (donations) | ~$5–15/mo | Server cost | Free |
| **Self-hostable** | No | Yes (CE/EE) | Via Forgejo | Yes | Yes | Yes (node) |

## A Practical Strategy

For most developers, the answer is not "leave GitHub" — it is "do not depend exclusively on GitHub."

1. **Keep GitHub for discoverability.** If you maintain open-source projects, GitHub is where potential contributors find you. The network effect is real.
2. **Mirror to a second home.** Push to Codeberg, a self-hosted Forgejo instance, or even a bare Git repo on a VPS. This takes five minutes to set up (`git remote add mirror ...`) and protects against account suspension or policy changes.
3. **Own your critical repositories.** For anything proprietary or sensitive, self-hosting with Forgejo is the most defensible choice. A $5/month VPS running Forgejo gives you a private Git forge with a web UI, access controls, and zero dependence on any platform provider.
4. **Evaluate your exposure to AI training.** If you use GitHub Copilot, check your settings. If you do not want your code used for model training, opt out explicitly. If this is a non-negotiable concern, Codeberg or SourceHut are structurally aligned with your preference.

The underlying principle is straightforward: Git is a distributed version control system. Your hosting strategy should be distributed too.

## References

[^1]: **GitHub Copilot data usage policy update (April 2026):** GitHub began using interaction data from Copilot Free/Pro/Pro+ users for AI model training by default. ([Link](https://github.blog/changelog/2025-04-24-copilot-will-use-interaction-data-for-model-training/))
[^2]: **GitLab AI data policy:** GitLab does not train AI models on customer code at any tier. ([Link](https://about.gitlab.com/blog/2025/04/28/gitlab-does-not-train-ai-models-on-customer-code/))
[^3]: **Codeberg:** Non-profit code hosting powered by Forgejo. ([Link](https://codeberg.org/))
[^4]: **SourceHut:** Minimalist, privacy-focused code hosting with email-based workflows. ([Link](https://sourcehut.org/))
[^5]: **Forgejo:** Community-driven, self-hosted Git forge (fork of Gitea). ([Link](https://forgejo.org/))
[^6]: **Radicle:** Peer-to-peer code collaboration network. ([Link](https://radicle.xyz/))

*Disclaimer: This article was generated using the Gemini 3.1 Pro and Claude Opus 4.8 models.*
