# task256：GPT 包静态引擎关闭运行时动态加载

## 触发条件与根因

238d 与台式机的 ~/pvz-logs/llmplay-build/pvz-portable-238c 相同，SHA256 为
6fafbb265fddc0e25304c6e1139cb54bcd6f6643bb007f6749093dd8c358df5e。
它虽然没有 ELF 解释器，SDL 2.32.10 的通用初始化仍会加载 DBus。
SexyAppBase 构造函数在解析 -env 之前调用 SDL_Init(SDL_INIT_TIMER)，
SDL 的 SDL_InitSubSystem() 无条件进入编译启用的 SDL_DBus_Init()。

台式机系统 glibc 为 2.43，原有桌面库下旧引擎正常启动。将 Ubuntu 24.04 noble
发布版的 libc 2.39、libdbus 1.14.10、libsystemd 255.4 及依赖解包到任务目录，
仅给引擎设置该目录的 LD_LIBRARY_PATH，即可稳定触发 SIGFPE。
有 DISPLAY 和真实 dbus-run-session 时会触发，清空 DISPLAY、DBus 会话地址与
XDG_RUNTIME_DIR 后同样触发。会话连接成功与否并非必要条件，库在连接之前加载。

在台式机可用下面的命令重现，复现目录中的 noble 库无需安装到系统：

    ulimit -v 8000000
    W=$HOME/pvz-logs/llmplay-build/task256
    env -u DISPLAY -u DBUS_SESSION_BUS_ADDRESS -u XDG_RUNTIME_DIR \
      LD_LIBRARY_PATH="$W/noble/root/usr/lib/x86_64-linux-gnu" \
      "$HOME/pvz-logs/llmplay-build/pvz-portable-238c" -env \
      -resdir "$HOME/.cache/pvz-research-resources" -savedir "$W/repro-save"

GDB 使用同一构建的未剥离符号版本取得调用链，剥离版也在相同地址崩溃：

    __libc_early_init+0x93 (Ubuntu 24.04 libc.so.6, offset 0x185503)
      div %r8                         # r8 = 0
    dl_open_worker_begin
    _dl_open
    dlopen
    SDL_LoadObject_REAL
    SDL_DBus_Init
    SDL_InitSubSystem_REAL.part.0
    Sexy::SexyAppBase::SexyAppBase    src/SexyAppFramework/SexyAppBase.cpp:168
    Sexy::SexyApp::SexyApp
    LawnApp::LawnApp
    main                            src/main.cpp:2804

反汇编显示除数从 _rtld_global_ro 的内部字段读取。静态引擎加载不同版本的共享
libc 后，该字段为零，符合 glibc 内部布局不兼容导致的早期初始化故障。
台式机现有库环境未覆盖这条跨版本路径，单纯设置 DISPLAY 或启动 DBus 会话不足以复现。

台式机复现材料均在 ~/pvz-logs/llmplay-build/task256/：
noble-crash-gdb.txt（带符号栈）、minimal-crash-gdb.txt（剥离版、无桌面变量），
以及 noble/root/usr/lib/x86_64-linux-gnu/（隔离的 noble 库）。
没有替换系统库或修改生产工作树。

## 构建改动

唯一构建实现在 scripts/llmplay/build_static_engine.sh。
台式机旧 ~/pvz-logs/llmplay-build/build_static.sh 已改为调用仓库脚本。
源码使用独立工作树 ~/PvZ-search/.claude/worktrees/staticfix256，
构建目录为 ~/pvz-logs/llmplay-build/task256/engine。
SDL 单独安装到 ~/pvz-logs/llmplay-build/sdl-headless-install，
不会复用此前带桌面后端的静态库。构建使用 nice -n 19 和 -j 4。

禁用 DBus、IBus 和 udev。显示后端关闭 X11、Wayland 和 KMSDRM，
同时关闭 DirectFB、RPI 和 Vivante。禁用 OpenGL、OpenGL ES 和 Vulkan。音频子系统及所有 Linux 音频后端关闭，
摇杆、触觉、传感器和 HIDAPI 同时关闭。视频仅保留 dummy，offscreen 关闭。

SDL_LOADSO=OFF 之外，构建脚本按 SDL 源码要求将 SDL_dynapi.h 中
SDL_DYNAMIC_API 的默认值改为 0，并在修改处注明原因。
只关 loadso 时，SDL 的动态 API 仍会直接调用 dlopen，不能满足任务要求。
SDL_internal.h 在 SDL_config.h 之前包含动态 API 头，需要在该头中统一关闭。

-env 在 Init() 中跳过窗口、声音和输入初始化，仅需要 SDL 定时器。
无需改引擎游戏逻辑，也无需调用方设置 SDL 驱动。README 已写明包内引擎仅支持
无窗口模式，不支持窗口或声音。

## 防回归

check_engine() 复用已经读取的 ELF 数据，追加一次字节扫描，
拒收 SDL_DYNAMIC_API 或已知 SDL 桌面/音频共享库名。
Mac 上拒收 238d 实测约 0.105 秒，不启动游戏、不重放存档。copy_engine() 与
accept_package.py 共用此入口。构建脚本作为主机工具排除出接口包。

## 238e 交付验收

新引擎 SHA256：
1c3680f36aa5bd5fef97d035a97a1cdd9ca89fce2493a87f125c7bd507d29c07。

在台式机独立目录中解出 238d，只替换引擎与 README 后运行 accept_package.py，
退出码为 0。三个并发领题互不重复，drill 跨进程保存/恢复后达到视界，
resume 推进到 tick 900，scenario 导出成功，两次导出前后的进度条目均为 12866。
drill 验收使用等待操作，结果为失败，导入器同样判失败，与 238d 验收结果一致。
完整报告在 ~/pvz-logs/llmplay-build/task256/accept238e.txt。

随后保留 DISPLAY=:0、真实 DBus 会话与私有 XDG_RUNTIME_DIR，
给引擎设置同一 noble 库目录的 LD_LIBRARY_PATH，使用包内 PvZEnv 在 L1、seed 256
从开局推进到 tick 1800、wave 1，已出现一个僵尸，无崩溃。
读取 /proc/<引擎PID>/maps 未发现共享库，strace 的该引擎进程 openat 记录中
共享库打开尝试为 0。记录为 first-wave.txt 与 first-wave-open.txt。
测试前可用内存约 18 GB，测试进程限制为 8 GB，没有运行 pytest。

238e 由 Mac 原始 238d 接口压缩包解出后，仅替换上述两个文件重新压缩，
资源包原样复制。交付位置为 ~/Downloads/pvz-llmplay-238e/，
包含 llmplay_pkg.7z.001 与 llmplay_res.zip。
新引擎压缩后通过 HF transfer/ 取回，取用后立即删除远端中转文件。
