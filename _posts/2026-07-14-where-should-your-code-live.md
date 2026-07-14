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

- **DMCA compliance.** GitHub is a US-based entity and complies aggressively with DMCA takedown requests. The `youtube-dl` takedown in 2020[^7] was the most visible example, but smaller projects get hit regularly. The process is automated and weighted toward the claimant. Your project can go offline before you have a chance to respond.
- **AI training.** In April 2026, GitHub updated its policy to use interaction data from Copilot Free, Pro, and Pro+ users to train its AI models by default[^1]. This includes code snippets and surrounding context from files you are actively editing. Enterprise users are exempt. Individual developers are opted in unless they manually disable the setting. This is a meaningful policy distinction.
- **Account suspension.** GitHub can suspend accounts for terms-of-service violations, and the appeals process is opaque. If your account is suspended, your repositories become inaccessible. For a solo developer, this can mean losing years of work history, issue discussions, and release artifacts in a single action.
- **Microsoft ownership.** This is a matter of individual judgment. Some developers are comfortable with Microsoft stewardship[^8]. Others are not. The concern is not about today's policies but about the structural fact that a single corporation controls the platform where the majority of the world's open-source code lives.

None of these are reasons to leave GitHub in a panic. They are reasons to understand what you are trading for the convenience.

It is also worth noting that any agreement you have with a hosting provider can develop loopholes over time. Emerging technologies create new categories of use that existing terms of service never anticipated. LLM training on hosted code is the current example, but it will not be the last. And there is a subtler problem with the opt-out model: there is always a gap between when a new policy takes effect and when you discover it and change your settings. During that window, the provider is operating under the new default. If the new default allows training on your code, your code has already been ingested by the time you toggle the setting. Opting out after the fact does not un-train a model. The damage — if you consider it damage — is done before you had a chance to make a choice.

GitHub publishes all DMCA takedown notices in a public repository[^11], which makes for instructive reading. The scale of enforcement is visible there: in March 2026, Anthropic filed a DMCA against a single repository and GitHub took down the entire fork network of 8,100 repositories in a single action[^12]. Whether or not the takedown was justified, the mechanism itself — one filing, 8,100 repos offline — is worth understanding if your project depends on the platform.

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

SourceHut[^4] (sr.ht) is the most opinionated platform on this list. Created by [Drew DeVault](https://drewdevault.com/), it is built around a philosophy of minimalism, privacy, and email-based workflows.

The distinguishing characteristics:

- **Email-native collaboration.** There are no pull requests. Contributors send patches via `git send-email` to the project's mailing list. Code review happens in email threads. This is how the Linux kernel[^10] and many other large projects have worked for decades. It is unfamiliar to developers raised on GitHub's web UI, but it is well-tested at scale.
- **No JavaScript required.** The web interface is fast, lightweight, and works in any browser. No tracking, no analytics, no ads.
- **Paid model.** SourceHut charges a subscription (approximately $5–$15/month, with financial aid available). This means you are the customer, not the product. The business model is not advertising or data monetization.
- **CI/CD.** SourceHut has a built-in build system (`builds.sr.ht`) that supports multiple Linux distributions and BSDs natively.

The traditional tradeoff was the learning curve. If your contributors are accustomed to clicking "Create Pull Request" on a web page, asking them to learn `git format-patch` and `git send-email` felt like a real barrier. In practice, LLM-based coding tools have largely eliminated this friction — any AI assistant can generate a properly formatted patch and compose the email for you. What was once an arcane workflow is now a prompt away, even for newcomers.

### Forgejo and Gitea (Self-Hosted)

Forgejo[^5] and Gitea are lightweight, self-hosted Git forges written in Go. Forgejo is a community-driven fork of Gitea that split off in late 2022 to ensure non-profit governance after Gitea's domain and trademark were transferred to a for-profit company without community consultation[^9].

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

### Decentralized Storage: The Permanence Layer

The platforms above handle the *workflow* of code collaboration — commits, patches, code review. A separate class of emerging protocols addresses a different problem: ensuring that code, once published, cannot be removed by any single entity.

These are not Git forges. They are storage networks. You pay for availability, and in return you get data that no centralized actor can take down.

**Walrus**[^17] is a decentralized storage protocol built on the Sui blockchain. It distributes data across a global network of independent storage nodes using a 2D erasure-coding scheme (RedStuff), which means the data can be reconstructed even if a significant fraction of nodes go offline. You interact with Walrus via a CLI or HTTP API to store and retrieve blobs. Storage is paid in WAL tokens. It also supports hosting fully decentralized static websites (Walrus Sites), which could serve as a censorship-resistant landing page for a project's documentation and release artifacts.

**Arweave**[^18] takes a different approach: permanent storage with a one-time fee. When you store data on Arweave, it is written to an append-only structure (the "blockweave") and replicated across the network indefinitely. There is no recurring cost and no expiry. This makes Arweave well-suited for archiving snapshots of a repository — a periodic `git bundle` pushed to Arweave creates an immutable backup that survives regardless of what happens to your primary hosting.

**IPFS and Filecoin**[^19] provide content-addressed storage. IPFS gives every piece of data a unique hash; anyone who has the hash can retrieve the data from any node that hosts it. The limitation is that data only stays available as long as at least one node "pins" it. Filecoin adds an incentive layer: storage providers are paid to keep data online, creating a market for reliable long-term hosting. You can push a bare Git repository to IPFS and distribute the hash. The repository becomes globally addressable and difficult to censor, though the workflow is not interactive — you cannot `git push` to IPFS directly.

These protocols share a common tradeoff: you pay for availability (in tokens, fees, or operational complexity), and in return you get storage that no single entity — no corporation, no government, no DMCA filing — can remove. They are not replacements for a Git forge. They are a complementary layer for projects that need guarantees beyond what any server-based platform can offer.

| Protocol | Model | Cost | Best For |
| :--- | :--- | :--- | :--- |
| **Walrus** | Erasure-coded blob storage (Sui) | WAL tokens | Live data, decentralized sites, API-accessible storage |
| **Arweave** | Permanent append-only storage | One-time fee | Immutable archival snapshots, "permaweb" |
| **IPFS + Filecoin** | Content-addressed + incentivized pinning | Filecoin tokens | Globally addressable backups, large dataset distribution |

## Encrypted Pushes: Staying on GitHub Without Trusting It

There is a middle path for developers who want to keep GitHub's network effects and tooling but do not want the platform to have access to their plaintext source: encrypt before you push.

Git supports this natively through **clean/smudge filters**[^15]. A clean filter runs when content is staged for commit (working directory → repository). A smudge filter runs when content is checked out (repository → working directory). By configuring a clean filter that encrypts and a smudge filter that decrypts, your local working copy remains plaintext while the blobs stored on GitHub are ciphertext. From GitHub's perspective, your repository contains opaque binary data.

The practical tool for this is **git-crypt**[^16]. Setup looks roughly like this:

```bash
# Install git-crypt and initialize it in your repo
git-crypt init

# Specify which files to encrypt via .gitattributes
echo "*.cpp filter=git-crypt diff=git-crypt" >> .gitattributes
echo "*.h filter=git-crypt diff=git-crypt" >> .gitattributes

# Export the symmetric key (back this up securely)
git-crypt export-key /path/to/keyfile

# Collaborators unlock with the key
git-crypt unlock /path/to/keyfile
```

You can also do this with a custom script in `.gitconfig` if you prefer not to depend on `git-crypt`:

```bash
# .gitconfig
[filter "encrypt"]
    clean = openssl enc -aes-256-cbc -salt -pbkdf2 -pass file:$HOME/.git-encrypt-key
    smudge = openssl enc -aes-256-cbc -d -salt -pbkdf2 -pass file:$HOME/.git-encrypt-key
```

Either approach works. The tradeoffs are real:

- **No server-side diffs.** Pull requests on GitHub will show binary gibberish. Code review must happen locally or through a separate channel.
- **No GitHub CI/CD.** GitHub Actions cannot read your source, so any CI pipelines must run on infrastructure you control. This is outside the scope of this post, but it is the most significant practical limitation.
- **Key management.** Every collaborator needs the decryption key. Losing the key means losing access to the repository history. This is a hard operational requirement.
- **Selective encryption.** You do not have to encrypt everything. `git-crypt` lets you specify patterns in `.gitattributes`, so you can encrypt source files while leaving `README.md`, `LICENSE`, and build scripts in plaintext.

The result is a repository that appears on GitHub as a normal project — it has commits, branches, history — but the actual source code is unreadable to the platform, to Copilot, and to any model training pipeline. Your code lives on GitHub's infrastructure, but GitHub cannot read it.

This is not a clean solution. It adds friction to collaboration, breaks web-based code review, and introduces key management overhead. But for developers or teams who need GitHub's ecosystem while treating the platform as an untrusted storage layer, it works.

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
## Jurisdictional Risk

There is a dimension to code hosting that individual developers rarely think about but that matters at the institutional and national level: jurisdiction.

GitHub, GitLab.com, and Bitbucket are all US-based entities subject to US trade sanctions enforced by the Office of Foreign Assets Control (OFAC). In July 2019, GitHub restricted access to private repositories, paid features, and the GitHub Marketplace for developers in Iran, Syria, Crimea, Cuba, and North Korea[^13]. Developers in those regions found themselves locked out of their own private repositories overnight. GitHub later secured a license from OFAC to restore full access for Iranian developers in January 2021[^14], but the precedent was set: a policy decision in Washington can make your codebase inaccessible regardless of what your terms of service say.

This is not a hypothetical concern for governments, universities, defense contractors, and companies in non-US jurisdictions. If your country's relationship with the US changes, your access to a US-hosted platform can change with it. The code is still there. You simply cannot reach it.

For entities that need to account for this risk, several providers operate outside US jurisdiction:

| Provider | Jurisdiction | Type | Notes |
| :--- | :--- | :--- | :--- |
| **Codeberg** | Germany (EU) | Non-profit SaaS | GDPR-compliant, community-governed |
| **Framagit** | France (EU) | Non-profit SaaS | Operated by Framasoft, focused on software freedom |
| **GNU Savannah** | France (EU) | Non-profit SaaS | Exclusively for GNU and free software projects |
| **Gitee** | China | Commercial SaaS | Domestic Chinese alternative to GitHub, government-backed |
| **Forgejo / GitLab CE** | Your choice | Self-hosted | Host on infrastructure in any jurisdiction you control |

The self-hosted option is the most defensible for sovereign entities, but it is not absolute. A Forgejo or GitLab CE instance running on domestic infrastructure — a European VPS from Hetzner or Exoscale, or a government-operated data center — places the code under the legal framework of the hosting country, not the platform provider's country. No foreign platform can suspend your account or change the terms of service. However, self-hosting still has dependencies: your DNS registrar can suspend your domain, your cloud or VPS provider can terminate your account, and a country-level firewall can block your server's IP entirely. True sovereignty requires owning each layer of the stack — hardware, network, DNS — which is feasible for a government data center but not for a solo developer on a rented VPS.

For individual developers, jurisdictional risk may feel abstract. For a university research group, a defense contractor, or a national open-source initiative, it is a concrete operational concern. The question is not whether your current access will be disrupted. The question is whether your hosting strategy survives a change in geopolitical conditions that you cannot predict.

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
[^7]: **Standing up for developers: youtube-dl is back.** GitHub Blog, November 2020. GitHub reinstated the youtube-dl repository after the EFF filed a counter-notice to the RIAA's DMCA takedown. ([Link](https://github.blog/news-insights/policy-news-and-insights/standing-up-for-developers-youtube-dl-is-back/))
[^8]: **Microsoft completes acquisition of GitHub.** Microsoft News, October 2018. ([Link](https://news.microsoft.com/2018/10/26/microsoft-completes-acquisition-of-github/))
[^9]: **Forgejo — beyond coding. We forge.** Forgejo's founding statement explaining the fork from Gitea after its domain and trademark were transferred to a for-profit entity. ([Link](https://forgejo.org/faq/))
[^10]: **Submitting patches: the essential guide to getting your code into the kernel.** Linux kernel documentation on the email-based patch workflow. ([Link](https://www.kernel.org/doc/html/latest/process/submitting-patches.html))
[^11]: **GitHub DMCA repository:** Public archive of all DMCA takedown notices and counter-notices received by GitHub. ([Link](https://github.com/github/dmca))
[^12]: **Anthropic DMCA takedown notice (March 2026):** Anthropic filed a DMCA against a single repository; GitHub processed the takedown against the entire fork network of 8.1K repositories. ([Link](https://github.com/github/dmca/blob/master/2026/03/2026-03-31-anthropic.md))
[^13]: **GitHub blocked developers in Iran, Syria, and Crimea.** TechMonitor, July 2019. GitHub restricted access to private repositories and paid features for developers in US-sanctioned countries. ([Link](https://www.techmonitor.ai/technology/software/github-blocked-developers-iran-syria-crimea))
[^14]: **GitHub is now free for teams and available in Iran.** Developer Tech, January 2021. GitHub secured an OFAC license to restore full access for Iranian developers. ([Link](https://www.developer-tech.com/news/github-now-free-teams-available-iran/))
[^15]: **Git attributes — clean/smudge filters.** Git documentation on configuring filters that transform content on checkout (smudge) and staging (clean). ([Link](https://git-scm.com/book/en/v2/Customizing-Git-Git-Attributes))
[^16]: **git-crypt:** Transparent file encryption in Git using clean/smudge filters with GPG or symmetric key support. ([Link](https://github.com/AGWA/git-crypt))
[^17]: **Walrus:** Decentralized storage and data availability protocol built on Sui, using erasure coding for resilient blob storage. ([Link](https://docs.wal.app))
[^18]: **Arweave:** Permanent, decentralized data storage with a one-time fee. Data is written to an append-only "blockweave" and replicated indefinitely. ([Link](https://www.arweave.org/))
[^19]: **IPFS and Filecoin:** Content-addressed storage (IPFS) with an incentivized pinning layer (Filecoin) for reliable long-term data availability. ([IPFS](https://ipfs.tech/), [Filecoin](https://filecoin.io/))

*Disclaimer: This post is for informational purposes only and does not constitute legal advice. Consult a qualified attorney for guidance on intellectual property, licensing, or jurisdictional matters. This article was generated using the Gemini 3.1 Pro and Claude Opus 4.8 models.*
