---
layout: post 
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
title:  "Opportunities of performance improvements in the C++ Standard Library"
date:   2020-04-25 19:11:59 -0700
categories: Performance
author: Aditya Kumar
tags: [C++ Standard Library, Performance]
---

## Introduction to the C++ Standard Library

The C++ standard library is a collection of classes and functions. These are written in C++ and are part of the C++ standard itself. All popular compiler toolchains come with a C++ standard library. The popular ones are *libstdc++ (GNU)*, *libc++(LLVM)* also popularly known as *libcxx*, *msvc-stl*(appears to be derived from *dinkumware C++ library* and *libc++* and was upstreamed in 2019). Needless to say, the standard library plays a very important role in the runtime performance of many systems.

Over a period of time I have collected a list of performance opportunities. Some of them I found online from the mailing list and bugzilla. Others by reading source code, and previous experience with the performance analysis of *libstdc++* and *libc++*. Disclaimer: For some of the items below I do not have experimental numbers, and I’m mostly relying on what was reported in the references.

### Performance opportunities in the C++ standard library

The algorithmic requirements on STL and their compliance by all the popular libraries makes it easy to believe that the performance couldn’t be improved further. The requirements are based on computational complexity theory [big O and friends](https://en.wikipedia.org/wiki/Big_O_notation). Because the constant factor isn’t taken into account (for good reasons), the realized performance depends on the actual implementation and the workload. The `iostream` library is just slow, like almost all the interfaces are slow (probably except for the ones I optimized ;) ). The source code looks very much like a typical Java program (too many indirections and virtual methods).

There are four subsections to classify performance opportunities.
- Standard Library Containers
    - like STL Containers, `iostream` library
- Standard Library Algorithms
    - `sort`, `find`
- Source code annotations to improve performance
    - to help compiler make better optimization decisions


### Standard Library Containers
#### STL:
**std::vector**
- Allocator in `std::vector` has perf issues (See: [Really bad codegen for libc++ vector][9]). Use realloc whenever appropriate. Some improvements were proposed in (See: [Improving std::vector<char> and std::deque<char> perfomance][10].

**std::string**
- `string::find.*of`, and `string::rfind` are still suboptimal (See [string][16])

**std::map**
- This is in general slow because of pointer chasing. There might be other issues with the implementation of rb-tree. Needs more investigation.

**std::iostream**
- Istream is very slow ([istream][1])
- `std::ostream` is very slow ([ostream][8])

### Standard library algorithms

- `std::sort`: of clang/gcc may be slow depending on the workload ([sort][14]).
- `std::find` of *libcxx* is very slow compared to *libstdc++* because llvm does not unroll the loop automatically ([find][13]).

### Source code annotations to improve performance
#### Annotating non-returning functions:
This will help the compiler reorganize basic blocks in the function.

Annotating pointers with restrict:
- Adding restrict will help alias analysis and many other PRE type optimizations.

Annotating branches with `builtin_likely`:
- This will help basic block reordering and hence help locality. In some cases it can save a branch.

### Compiler optimizations that can help improve the performance of C++ standard library:
#### Whole program devirtualization
Devirtualization will help the iostream library because it has too many virtual methods.

#### Inlining important functions:
Constructors and destructors of STL containers like `string`, `vector` etc.

#### Vectorization
Vectorize `memcpy`, `memset` etc style loops. Also related to Loop Idiom Recognition

#### Loop idiom recognition
Detect `memcpy`, `memset`, `memchr`, `memcmp` style loops

#### Jump Threading
Even better, jump threading with auto-FDO. (See: [Jump Threading Bug][12])

#### Profile guided optimization of C++ standard library (Auto-FDO)
Helps bring micro-optimization and code-layout tuned according to our workloads

#### Loop unrolling

`std::find` will benefit from:
    - Loop unrolling (See [Loop unroll opportunities][13])
    - Needs loop rotation to make loop-unroll more effective (See: [Loop rotation bug][15])

### Miscellaneous:
CoroFrame does not pay attention to the lifetime markers (See [CoroFrame][11])


[1]: https://github.com/hiraditya/std-benchmark/blob/master/docs/slides/slide-cppnow.pdf

[2]: https://devblogs.microsoft.com/cppblog/improving-the-performance-of-standard-library-functions/

[3]: https://www.dre.vanderbilt.edu/~schmidt/PDF/perf4.pdf

[4]: https://stackoverflow.com/questions/4340396/does-the-c-standard-mandate-poor-performance-for-iostreams-or-am-i-just-deali

[5]: https://lists.llvm.org/pipermail/cfe-dev/2016-July/049814.html

[6]: https://stackoverflow.com/questions/38624468/clang-fstreams-10x-slower-than-g

[7]: http://www.stroustrup.com/Performance-TR.pdf

[8]: https://bugs.llvm.org/show_bug.cgi?id=40763

[9]: https://bugs.llvm.org/show_bug.cgi?id=35637

[10]: https://reviews.llvm.org/D44823

[11]: https://bugs.llvm.org/show_bug.cgi?id=41877

[12]: https://bugs.llvm.org/show_bug.cgi?id=43276

[13]: https://bugs.llvm.org/show_bug.cgi?id=19708

[14]: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=82739

[15]: https://bugs.llvm.org/show_bug.cgi?id=27360

[16]: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=93584
