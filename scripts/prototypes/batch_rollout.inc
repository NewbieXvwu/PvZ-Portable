// task35 experiment only. Included in an exported copy of main.cpp by
// build_batch_rollout.py; never compiled into the production executable.
#include <cblas.h>
#include <ctime>

thread_local const std::vector<float>* gPrototypeHidden = nullptr;

static double PrototypeCpu()
{
    timespec now{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
    return now.tv_sec + now.tv_nsec * 1e-9;
}

// Each convolution becomes W x K times K x (lanes*60), with a union of the
// sparse first-layer channels. The rest of Encode, Heads and action selection
// remain the existing implementation through the generated hidden-input hook.
static std::vector<std::vector<float>> PrototypeConvolutions(
    EnvironmentRollout& net, std::vector<std::vector<float>> boards)
{
    const int lanes = boards.size(), width = net.width;
    for (int layer = 0; layer < 3; ++layer)
    {
        const int channels = boards[0].size() / 60;
        const int rows = layer == 2 ? 1 : 3, cols = layer == 2 ? 5 : 3;
        std::vector<int> active;
        for (int channel = 0; channel < channels; ++channel)
        {
            bool used = layer != 0;
            for (const auto& board : boards)
                used = used || std::any_of(board.data() + channel * 60, board.data() + (channel + 1) * 60,
                    [](float value) { return value != 0; });
            if (used) active.push_back(channel);
        }
        const int count = active.size() * rows * cols, cells = lanes * 60;
        std::vector<float> columns(count * cells, 0), matrix(width * count), output(width * cells);
        const auto& weights = net.weights[layer * 2];
        int feature = 0;
        for (int channel : active)
            for (int kr = 0; kr < rows; ++kr)
                for (int kc = 0; kc < cols; ++kc, ++feature)
                {
                    const int dy = kr - rows / 2, dx = kc - cols / 2;
                    const int first = std::max(0, -dx), last = std::min(10, 10 - dx);
                    for (int lane = 0; lane < lanes; ++lane)
                        for (int row = std::max(0, -dy); row < std::min(6, 6 - dy); ++row)
                            std::copy_n(boards[lane].data() + channel * 60 + (row + dy) * 10 + first + dx,
                                last - first, columns.data() + feature * cells + lane * 60 + row * 10 + first);
                    for (int target = 0; target < width; ++target)
                        matrix[target * count + feature] =
                            weights[((target * channels + channel) * rows + kr) * cols + kc];
                }
        for (int target = 0; target < width; ++target)
            std::fill_n(output.data() + target * cells, cells, net.weights[layer * 2 + 1][target]);
        if (count)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width, cells, count,
                1, matrix.data(), count, columns.data(), cells, 1, output.data(), cells);
        for (int lane = 0; lane < lanes; ++lane)
        {
            boards[lane].resize(width * 60);
            for (int channel = 0; channel < width; ++channel)
                for (int cell = 0; cell < 60; ++cell)
                    boards[lane][channel * 60 + cell] = std::max(0.0f, output[channel * cells + lane * 60 + cell]);
        }
    }
    return boards;
}

struct PrototypeLane
{
    EnvironmentSnapshot snapshot;
    EnvironmentRollout::History student;
    std::deque<std::pair<int, uint32_t>> income;
    std::vector<std::pair<int, uint32_t>> substeps;
    std::array<int, 10> planted{};
    int previousTick = -1, decisions = 0, netCalls = 0, mowers = 0, result = -1, tick = 0;
    bool previousPlant = false, done = false;
    std::string observation;
    double rate = 0;
};

static int PrototypeMowers(LawnApp* app)
{
    int count = 0;
    for (const LawnMower* mower : app->mBoard->mLawnMowers)
        count += !mower->mDead && mower->mMowerState == LawnMowerState::MOWER_READY;
    return count;
}

static int PrototypeEarlyResult(LawnApp* app, int mask)
{
    if (!(mask & ENV_EARLY_RESULT) || app->mBoard->mLevel != 7 || app->EnvironmentTerminal()) return -1;
    bool protectedRows[6]{};
    for (const LawnMower* mower : app->mBoard->mLawnMowers)
        if (!mower->mDead && mower->mRow >= 0 && mower->mRow < 6 && mower->mMowerState == LawnMowerState::MOWER_READY)
            protectedRows[mower->mRow] = true;
    bool safe = app->mBoard->mCurrentWave == app->mBoard->mNumWaves && app->mBoard->mRiseFromGraveCounter == 0;
    for (Zombie* zombie : app->mBoard->mZombies)
    {
        if (zombie->mDead || zombie->IsDeadOrDying() || zombie->mMindControlled) continue;
        const bool protectedRow = zombie->mRow >= 0 && zombie->mRow < 6 && protectedRows[zombie->mRow];
        safe = safe && protectedRow;
        if (!protectedRow && zombie->mPosX < 80.0f && zombie->mHasHead) return 0;
    }
    return safe ? 1 : -1;
}

static void PrototypeBatchCommand(LawnApp* app, EnvironmentRollout& net,
    const std::unordered_map<int, EnvironmentSnapshot>& snapshots, std::istream& input)
{
    int root = 0, horizon = 0, mask = 0, count = 0, batch = 0, historyStart = 0, historyCount = 0;
    uint32_t seed = 0;
    input >> root >> horizon >> mask >> count >> batch >> seed >> historyStart >> historyCount;
    if (!input || !snapshots.contains(root) || !net.Loaded() || !net.identityPacket ||
        horizon <= 0 || count < 1 || count > 32 || batch < 1 || batch > count ||
        (mask & ~ENV_SKIP_ALLOWED) || (mask & ENV_ADAPTIVE_HORIZON) || historyCount < 0 || historyCount > 100000)
        throw std::runtime_error("invalid prototype batch command");
    std::deque<std::pair<int, uint32_t>> history;
    for (int i = 0; i < historyCount; ++i)
    {
        int tick; uint32_t amount;
        if (!(input >> tick >> amount)) throw std::runtime_error("invalid prototype income");
        history.emplace_back(tick, amount);
    }
    const auto start = PrototypeCpu();
    const int savedMask = gEnvSimSkip;
    struct RestoreGlobals { int mask; ~RestoreGlobals() { gEnvSimSkip = mask; gPrototypeHidden = nullptr; } } restore{savedMask};
    gEnvSimSkip = mask;
    std::vector<PrototypeLane> lanes(count);
    int startTick = 0;
    double network = 0, simulation = 0, snapshotsCpu = 0, features = 0;
    for (int i = 0; i < count; ++i)
    {
        if (!RestoreEnvironmentSnapshot(app, snapshots.at(root))) throw std::runtime_error("restore root failed");
        ReseedEnvironment(app, seed + i);
        if (!SaveEnvironmentSnapshot(app, lanes[i].snapshot)) throw std::runtime_error("snapshot lane failed");
        lanes[i].income = history;
        lanes[i].mowers = PrototypeMowers(app);
        startTick = lanes[i].tick = app->mBoard->mMainCounter;
    }
    snapshotsCpu += PrototypeCpu() - start;
    for (;;)
    {
        std::vector<int> active;
        std::vector<std::vector<float>> boards;
        for (int i = 0; i < count; ++i) if (!lanes[i].done)
        {
            auto& lane = lanes[i];
            double before = PrototypeCpu();
            if (!RestoreEnvironmentSnapshot(app, lane.snapshot)) throw std::runtime_error("restore features failed");
            snapshotsCpu += PrototypeCpu() - before;
            before = PrototypeCpu();
            while (!lane.income.empty() && lane.income.front().first < lane.tick - 6000) lane.income.pop_front();
            double total = 0;
            for (auto event : lane.income) total += event.second;
            int elapsed = std::max(0, std::min(6000, lane.tick - historyStart));
            lane.rate = elapsed ? total * 1000.0 / elapsed : 0;
            if (app->EnvironmentAnyLegalPlant())
            {
                std::vector<float> board, scalars;
                auto historyCopy = lane.student;
                net.Features(app, lane.rate, lane.previousTick, lane.previousPlant, board, scalars, historyCopy);
                active.push_back(i);
                boards.push_back(std::move(board));
            }
            features += PrototypeCpu() - before;
        }
        if (std::all_of(lanes.begin(), lanes.end(), [](const auto& lane) { return lane.done; })) break;
        std::vector<std::vector<float>> hidden(count);
        double before = PrototypeCpu();
        for (size_t first = 0; first < active.size(); first += batch)
        {
            const size_t last = std::min(active.size(), first + batch);
            std::vector<std::vector<float>> group(boards.begin() + first, boards.begin() + last);
            auto result = PrototypeConvolutions(net, std::move(group));
            for (size_t index = first; index < last; ++index) hidden[active[index]] = std::move(result[index - first]);
        }
        network += PrototypeCpu() - before;
        for (int i = 0; i < count; ++i) if (!lanes[i].done)
        {
            auto& lane = lanes[i];
            before = PrototypeCpu();
            if (!RestoreEnvironmentSnapshot(app, lane.snapshot)) throw std::runtime_error("restore step failed");
            snapshotsCpu += PrototypeCpu() - before;
            const int tick = app->mBoard->mMainCounter;
            const uint32_t producedBefore = app->mBoard->mSunMoneyProduced;
            before = PrototypeCpu();
            gPrototypeHidden = hidden[i].empty() ? nullptr : &hidden[i];
            const int calls = net.NetworkCalls();
            const auto action = net.Predict(app, lane.rate, lane.previousTick, lane.previousPlant, lane.student).action;
            lane.netCalls += net.NetworkCalls() - calls;
            gPrototypeHidden = nullptr;
            network += PrototypeCpu() - before;
            before = PrototypeCpu();
            lane.previousPlant = action.packet >= 0;
            bool accepted = true;
            if (lane.previousPlant) accepted = app->EnvironmentPlant(action.packet, action.col, action.row);
            else app->EnvironmentWaitUntil(60, 0);
            if (!accepted) { app->EnvironmentWait(60); lane.previousPlant = false; }
            if (lane.previousPlant) ++lane.planted[action.packet];
            lane.previousTick = tick;
            lane.tick = app->mBoard->mMainCounter;
            const uint32_t produced = app->mBoard->mSunMoneyProduced - producedBefore;
            if (produced) lane.income.emplace_back(lane.tick, produced);
            lane.substeps.emplace_back(lane.tick, produced);
            ++lane.decisions;
            if (!app->EnvironmentTerminal()) lane.mowers = PrototypeMowers(app);
            // The production loop tests the horizon before its early-result shortcut.
            if (lane.tick < startTick + horizon) lane.result = PrototypeEarlyResult(app, mask);
            lane.done = app->EnvironmentTerminal() || lane.decisions >= 1000 ||
                lane.tick >= startTick + horizon || lane.result >= 0;
            simulation += PrototypeCpu() - before;
            before = PrototypeCpu();
            if (lane.done) lane.observation = app->EnvironmentLeanObservation();
            else if (!SaveEnvironmentSnapshot(app, lane.snapshot)) throw std::runtime_error("save step failed");
            snapshotsCpu += PrototypeCpu() - before;
        }
    }
    const double total = PrototypeCpu() - start;
    std::cout << "PVZENV {\"ok\":true,\"prototype_version\":1,\"cpu_seconds\":" << total
        << ",\"network_cpu_seconds\":" << network << ",\"simulation_cpu_seconds\":" << simulation
        << ",\"snapshot_cpu_seconds\":" << snapshotsCpu << ",\"feature_cpu_seconds\":" << features << ",\"lanes\":[";
    for (int i = 0; i < count; ++i)
    {
        const auto& lane = lanes[i];
        std::cout << (i ? ",{" : "{") << "\"observation\":" << lane.observation
            << ",\"estimated_result\":" << lane.result << ",\"decisions\":" << lane.decisions
            << ",\"net_calls\":" << lane.netCalls << ",\"mowers_left\":" << lane.mowers
            << ",\"ticks\":" << lane.tick - startTick << ",\"planted\":[";
        for (int card = 0; card < 10; ++card) std::cout << (card ? "," : "") << lane.planted[card];
        std::cout << "],\"substeps\":[";
        for (size_t j = 0; j < lane.substeps.size(); ++j)
            std::cout << (j ? ",[" : "[") << lane.substeps[j].first << ',' << lane.substeps[j].second << ']';
        std::cout << "]}";
    }
    std::cout << "]}" << std::endl;
}
