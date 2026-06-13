---
title: "Hardening the ELF: Understanding RELRO and GOT Overwrites"
date: 2026-06-12 08:00:00 -0700
categories: [Security, Systems]
tags: [elf, relro, exploit-mitigation, linux, toolchain]
---

In our [previous post](https://hiraditya.github.io/posts/the-hidden-complexity-of-hello-world/), we took a deep dive into the hidden complexities of the simplest C program. We discussed how modern Position Independent Executables (PIE) rely on the **PLT (Procedure Linkage Table)** and **GOT (Global Offset Table)** to dynamically resolve shared library functions like `puts()`.

We noted that under "lazy binding", the dynamic linker looks up the true memory address of `puts` on the fly and dynamically overwrites the GOT entry with that exact address.

But there is a glaring issue with this mechanism: **if the dynamic linker can overwrite the GOT at runtime, so can an attacker.** 

Let's explore the classic GOT Overwrite attack, and how modern OS vendors use a mitigation called **RELRO (Relocation Read-Only)** to harden ELF binaries.

## The GOT Overwrite Vulnerability

For lazy binding to work, the memory section containing the PLT-specific GOT entries (the `.got.plt` section) *must* be mapped into memory with write permissions (`RW-`). The dynamic linker (`ld-linux.so`) needs to write the resolved function addresses there.

However, write permissions are a double-edged sword. If an attacker discovers an arbitrary memory write vulnerability in your program—perhaps a classic buffer overflow, a format string vulnerability, or a use-after-free bug—they can target the `.got.plt` section.

### The Exploit Flow:
1. The attacker exploits the memory corruption bug to overwrite the GOT entry for a commonly called function, such as `printf` or `exit`.
2. Instead of pointing to the real `printf` in `libc`, the attacker overwrites the GOT entry with the memory address of the `system()` function, or a pointer to their own malicious shellcode.
3. The next time the program attempts to log a message using `printf(user_input)`, it actually jumps directly to `system(user_input)`, granting the attacker a root shell.

### A First-Principles Example: The Format String Bug
One of the most classic ways to achieve a GOT overwrite is via a **Format String Vulnerability**. 

Imagine a poorly written C program that logs user input like this:
```c
char user_input[100];
gets(user_input);
// VULNERABLE: No format specifier (like "%s") is used!
printf(user_input); 
exit(0);
```

Because the user controls the format string, they can input format specifiers like `%x` to leak memory addresses off the stack. More dangerously, they can use the **`%n`** specifier. 

In C, `%n` does not print anything. Instead, it **writes** the number of bytes printed so far into the memory address provided by the corresponding argument. 

By carefully crafting a payload, an attacker can:
1. Pass the exact memory address of `exit@got` (the GOT entry for the `exit` function).
2. Pad the output with spaces or characters until exactly `X` bytes have been printed (where `X` is the memory address of `system()` or their shellcode).
3. Use `%n` to write that `X` value directly into the `exit@got` address.

When the program subsequently calls `exit(0);`, the CPU looks up the GOT entry, finds the attacker's newly written address, and executes it. The program doesn't exit; it spawns a malicious shell!

### The Relationship to CVEs
A "GOT Overwrite" itself does not have a specific Common Vulnerabilities and Exposures (CVE) ID because it is a **binary exploitation technique**, not a specific software bug. 

However, countless CVEs have been exploited *using* this exact technique. For example, historical vulnerabilities in network daemons or even glibc itself (like the famous CVE-2015-0235 "GHOST" vulnerability) often culminated in attackers utilizing a GOT overwrite as the final payload delivery mechanism to hijack control flow.

This threat is not just historical. A deep search of recent exploits confirms that modern memory corruption bugs still frequently fall back to GOT manipulation if Full RELRO isn't strictly enforced. Notable modern examples include:
- **CVE-2026-23479 (Redis):** A complex Use-After-Free that was successfully escalated into an out-of-bounds write. Exploit developers specifically targeted the GOT to repoint the `strcasecmp()` function to `system()`, turning a standard Redis command into a remote root shell.
- **CVE-2026-24872 (SkyFire_548 Engine):** An unchecked pointer arithmetic bug that granted an out-of-bounds write primitive, which was subsequently weaponized to corrupt GOT entries and hijack the execution flow of the engine.

This technique is incredibly reliable because, without mitigations, the GOT is situated at a predictable offset in memory and is always writable, making it a prime target for exploitation.

## Enter RELRO (Relocation Read-Only)

To mitigate this attack surface, compiler engineers and security researchers developed **RELRO**. The core concept is simple: if data doesn't *need* to be written to after the program starts, the loader should strip away the write permissions and make it Read-Only (`R--`).

There are two levels of RELRO implementation:

### 1. Partial RELRO
Partial RELRO forces the loader to resolve certain ELF internal data sections (like `.dynamic` and the standard `.got`) at load time, and then immediately marks them as read-only. 

However, Partial RELRO **preserves lazy binding**. Because lazy binding requires updating `.got.plt` during program execution, the `.got.plt` section is left entirely writable. While Partial RELRO protects some internal structures, the classic GOT overwrite attack vector remains wide open.

### 2. Full RELRO
To completely close the vulnerability, modern systems (such as Fedora, Ubuntu, and RedHat) enforce **Full RELRO** on security-critical binaries and network daemons. 

Full RELRO takes a drastic approach: **it completely disables lazy binding.**

When a binary is compiled with Full RELRO (`-Wl,-z,relro,-z,now`), it instructs the dynamic linker to resolve *all* external symbols at load time, before `_start` or `main` is ever executed. This is equivalent to running the program with the `LD_BIND_NOW=1` environment variable.

Once the dynamic linker has traversed the entire symbol table and populated every single GOT entry, it leverages the `mprotect()` system call to mark the entire GOT (both `.got` and `.got.plt`) as strictly Read-Only. 

If an attacker attempts a GOT overwrite against a Full RELRO binary, the CPU will immediately trigger a Segmentation Fault (SIGSEGV), stopping the exploit dead in its tracks.

## The Tradeoff: Security vs. Performance

Despite the immense security benefits of Full RELRO, it was not adopted as the default compiler behavior immediately.

The tradeoff is **startup performance**. To understand this overhead, we can classify the "pre-main" execution time of an application into a few distinct segments:
1. **Load Time:** The time taken by the OS to map the binary and its dependencies from disk into memory.
2. **Time to Resolve Symbols:** The time the dynamic linker spends resolving references to shared libraries and populating the GOT.
3. **Time to Launch Global Constructors:** The execution of `.init_array` functions and C++ static initializers before `main` starts.

Lazy binding was invented specifically to optimize the **"Time to Resolve Symbols"** segment. Large graphical applications or massive monolithic binaries might link against hundreds of shared libraries containing thousands of functions, most of which are never called during a standard execution. Resolving all of them upfront causes a noticeable delay in program startup time. Full RELRO intentionally sacrifices this optimization, front-loading the entire symbol resolution cost to guarantee security.

However, as CPUs have gotten drastically faster and security threats have become vastly more sophisticated, the industry consensus has shifted. The millisecond startup penalty of Full RELRO is now widely considered a mandatory price to pay for robust memory safety. In fact, major Linux distributions like Fedora have already moved to enable Full RELRO globally by default for all packages.

### Combating Overhead on Mobile: Caches and Warm Starts

While desktop and server environments have largely absorbed this startup cost, launch time overhead remains incredibly critical on mobile platforms like iOS and Android, where users expect instantaneous application response.

To combat the overhead of exhaustive symbol resolution during a "Cold Start" (where the application process is launched from scratch), mobile operating systems have engineered sophisticated caching mechanisms to enable blazing-fast "Warm Starts" and "Hot Starts":
- **iOS (`dyld3` Closure Caches):** Apple's `dyld3` dynamic linker completely reimagined symbol resolution. During an app's first launch (or at install time), it exhaustively resolves all symbols and pre-calculates the necessary memory addresses. It then serializes this information into a "closure cache" on disk. On subsequent launches, `dyld3` simply reads the pre-calculated closure, completely bypassing the expensive symbol search overhead while still maintaining the security benefits of Full RELRO.
- **Android (Zygote and Profiles):** Android utilizes a special daemon called "Zygote," which pre-loads and pre-links common framework libraries. When a new app is launched, it simply forks from the Zygote process, inheriting the already-resolved symbols for shared core libraries. Android also leverages Baseline Profiles to pre-compile critical code paths, avoiding both JIT compilation and symbol resolution overhead during the delicate startup window.

Through these innovations, mobile systems can enforce strict RELRO protections without sacrificing the instantaneous feel of a warm start.

### The Security Cost of Caching

However, as is often the case in systems engineering, these optimizations introduce entirely new classes of security vulnerabilities:

1. **Android's ASLR Weakness:** Because every Android app is `fork()`ed from the exact same Zygote template, they all inherit an identical memory layout. This drastically weakens **Address Space Layout Randomization (ASLR)**. If an attacker manages to leak the memory address of `libc` in one sandboxed app (like a web browser tab), they instantly know the exact ASLR offsets for *every other app on the device* until it reboots, vastly simplifying cross-process exploitation.
2. **iOS Closure Tampering:** Because `dyld3` closure caches serialize complex linking data to the filesystem, they have become high-value targets. Historically, if an attacker could bypass integrity checks and replace a valid closure with a maliciously crafted one, they could hijack the execution flow of highly privileged "entitled" processes without ever needing a runtime memory corruption bug.

## Life After RELRO: Where Do Attackers Pivot?

If Full RELRO perfectly secures the GOT, what does a modern attacker do when faced with a memory corruption bug?

When the GOT is marked read-only, exploit developers are forced to look for other writable function pointers. As highlighted in a [Hacker News discussion](https://news.ycombinator.com/item?id=29186252) on ELF hardening, the typical response is to target other data pointers not secured by RELRO. 

If the application itself stores function pointers in its writable `.data` or `.bss` segments (such as callback arrays or vtables), those become the primary targets. If the binary lacks such pointers, attackers will often try to leak the base address of `libc` and target internal library hooks—historically, the `__malloc_hook` and `__free_hook` were favorite pivot points because they are frequently called and resided in writable memory, though modern glibc versions have since removed them to close this exact loophole.

## Checking Your Binaries

You can easily check if your binaries are hardened with RELRO using the popular `checksec` tool, or by inspecting the ELF headers directly with `readelf`.

A binary with Full RELRO will exhibit a `GNU_RELRO` program header, and the `BIND_NOW` dynamic flag:

```bash
$ readelf -d my_program | grep BIND_NOW
 0x0000000000000018 (BIND_NOW)           
```

---

## References

1. **Pre-Main Time Classification & Optimizations:** Kumar, Aditya. *App Startup compiler optimizations and techniques for embedded systems.* 
2. **Fedora and Full RELRO:** "Hardening ELF binaries using Relocation Read-Only (RELRO)." *Red Hat Developer Blog*. ([Link](https://www.redhat.com/en/blog/hardening-elf-binaries-using-relocation-read-only-relro))
3. **Life After RELRO:** Community discussion on post-RELRO exploitation targets. *Hacker News*. ([Link](https://news.ycombinator.com/item?id=29186252))
4. **Android Zygote ASLR Weakness:** For an in-depth look at how Zygote's `fork()` model compromises Address Space Layout Randomization, see architectural analyses by security teams like Google Project Zero. ([Link](https://googleprojectzero.blogspot.com/2016/12/bitunmap-attacking-android-ashmem.html))
5. **iOS dyld3 Cache Tampering:** Apple's evolution of the dynamic linker and closure caches has been well-documented in WWDC sessions such as "App Startup Time: Past, Present, and Future" and subsequent security research. ([Link](https://developer.apple.com/videos/play/wwdc2017/413/))

---

*Disclaimer: This article was generated by prompting Gemini, an AI developed by Google.*
