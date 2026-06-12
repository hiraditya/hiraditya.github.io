---
title: "Hardening the ELF: Understanding RELRO and GOT Overwrites"
date: 2026-06-12 08:00:00 -0700
categories: [Security, Systems]
tags: [elf, relro, exploit-mitigation, linux, toolchain]
---

In our [previous post](https://hiraditya.github.io/posts/the-hidden-complexity-of-hello-world/), we took a deep dive into the hidden complexities of the simplest C program. We discussed how modern Position Independent Executables (PIE) rely on the **PLT (Procedure Linkage Table)** and **GOT (Global Offset Table)** to dynamically resolve shared library functions like `puts()`.

We noted that under "lazy binding", the dynamic linker looks up the true memory address of `puts` on the fly and dynamically overwrites the GOT entry with that exact address.

But there is a glaring, terrifying issue with this mechanism: **if the dynamic linker can overwrite the GOT at runtime, so can an attacker.** 

Let's explore the classic GOT Overwrite attack, and how modern OS vendors use a mitigation called **RELRO (Relocation Read-Only)** to harden ELF binaries.

## The GOT Overwrite Vulnerability

For lazy binding to work, the memory section containing the PLT-specific GOT entries (the `.got.plt` section) *must* be mapped into memory with write permissions (`RW-`). The dynamic linker (`ld-linux.so`) needs to write the resolved function addresses there.

However, write permissions are a double-edged sword. If an attacker discovers an arbitrary memory write vulnerability in your program—perhaps a classic buffer overflow, a format string vulnerability, or a use-after-free bug—they can target the `.got.plt` section.

### The Exploit Flow:
1. The attacker exploits the memory corruption bug to overwrite the GOT entry for a commonly called function, such as `printf` or `exit`.
2. Instead of pointing to the real `printf` in `libc`, the attacker overwrites the GOT entry with the memory address of the `system()` function, or a pointer to their own malicious shellcode.
3. The next time the program attempts to log a message using `printf(user_input)`, it actually jumps directly to `system(user_input)`, granting the attacker a root shell.

This technique is incredibly reliable because the GOT is situated at a predictable offset in memory, making it a prime target for exploitation.

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

If Full RELRO is so secure, why wasn't it the default from the beginning?

The tradeoff is **startup performance**. Lazy binding was invented because large graphical applications or massive monolithic binaries might link against hundreds of shared libraries containing thousands of functions, most of which are never called during a standard execution. Resolving all of them upfront causes a noticeable delay in program startup time.

However, as CPUs have gotten drastically faster and security threats have become vastly more sophisticated, the industry consensus has shifted. The millisecond startup penalty of Full RELRO is now widely considered a mandatory price to pay for robust memory safety.

## Checking Your Binaries

You can easily check if your binaries are hardened with RELRO using the popular `checksec` tool, or by inspecting the ELF headers directly with `readelf`.

A binary with Full RELRO will exhibit a `GNU_RELRO` program header, and the `BIND_NOW` dynamic flag:

```bash
$ readelf -d my_program | grep BIND_NOW
 0x0000000000000018 (BIND_NOW)           
```

---

*Many thanks to Elliott Hughes for highlighting this critical security perspective following our deep-dive into the C toolchain!*
