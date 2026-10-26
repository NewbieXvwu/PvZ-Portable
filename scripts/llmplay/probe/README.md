# 探路包：先确认沙箱能不能把引擎编出来

这个包只做一件事：在目标沙箱里**从源码编译引擎**，然后重放 4 局人类示范，
看编出来的引擎是不是和我们的一致。**不包含**任何 LLM 接口或任务书。

## 怎么跑

    cd llmplay_probe
    bash build.sh          # 报告环境 + 依赖，缺了就 apt 装，然后编译
    python3 selfcheck.py   # 重放 4 局示范，打印一段可复制的报告

把 `selfcheck.py` 最后那一段（`=== llmplay probe report ===` 到 `=== end ===`）
整段复制回来即可。里面包含编译器版本、glibc、CPU、构建结果、每局结果、
第一个不一致的操作序号。

## 包里有什么

    src/                引擎源码（deploy4 源码树）
    vendor/mimalloc/    mimalloc 源码（离线，供 CMake 的 FetchContent 用）
    res/                游戏资源（main.pak + properties）
    demos/              4 局人类示范的数值存档（ticks/frames/meta，不含画面）
    build.sh            构建脚本
    selfcheck.py        重放自检
    build_prebuilt/     预编的静态二进制（兜底；build.sh 失败时 selfcheck 会用它）

## 判定标准

**存档里记录的每一步都复现一致、整局仍然通关**，就算通过。具体说：

- 存档记 `ok=true` 的操作必须被接受，记 `ok=false` 的必须同样被拒绝
  （人类示范里有少量暂停时重复点种被引擎拒绝的操作，存档按原样记了）；
- 每个操作结束的 tick 必须与存档一致；
- 每个"人类做决定"的时刻（`frames.jsonl` 里那 800–1300 个点），阳光、波次、
  植物数、僵尸数、结果都必须一致；
- 终局 tick / 波数 / 结果一致。

`state_hash` **不参与判定**：它随构建变化（官方 `human_demo.py verify` 发现二进制
SHA 不同时也会自动跳过哈希比较）。自检只在开局取一次作诊断打印。

## 编译依赖（build.sh 会自动检查）

- cmake ≥ 3.16
- C++20 编译器，**必须支持 `<format>`**（GCC 13+ / Clang 17+；GCC 11/12 编不过）
- SDL2、libopenmpt、zlib、libpng、libjpeg、BLAS（OpenBLAS）的开发包
- 不需要网络（mimalloc 已随包提供）
