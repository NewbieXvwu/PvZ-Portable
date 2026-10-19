// Link with the simulator's objects except main.cpp.o (see the experiment build command).
// Exercises live pointer mutation, restore invalidation, ordered insertions and fallback.
#include "../../src/PvzpLib/Reanimator.h"
#include "../../src/PvzpLib/PvzpCommon.h"
#include <bit>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Fixture {
    std::vector<std::string> names;
    std::vector<ReanimatorTrack> definitions;
    std::vector<ReanimatorTrackInstance> tracks;
    ReanimatorDefinition definition;
    Reanimation animation;
    explicit Fixture(int count) : names(count), definitions(count), tracks(count) {
        for (int i=0;i<count;++i) {
            names[i]="t"+std::to_string(i);
            definitions[i].mName=names[i].c_str();
        }
        definition.mTracks={definitions.data(),count};
        animation.mDefinition=&definition;
        animation.mTrackInstances=tracks.data();
        animation.mFrameCount=1;
        animation.mAnimRate=0;
        animation.mLoopType=ReanimLoopType::REANIM_LOOP;
        animation.mNeedsTrackUpdates=true;
    }
    ~Fixture() { // Borrowed fixture arrays do not belong to the game's allocator.
        animation.mDead=true;
        animation.mTrackInstances=nullptr;
    }
};
static void tick(Fixture& full,Fixture& sparse,int originalMask=61) {
    gEnvSimSkip=originalMask;full.animation.Update();
    gEnvSimSkip=originalMask|256;sparse.animation.Update();
    require(full.animation.mLoopCount==sparse.animation.mLoopCount,"loop count differs");
    require(std::bit_cast<uint32_t>(full.animation.mAnimTime)==std::bit_cast<uint32_t>(sparse.animation.mAnimTime),"animation clock differs");
    require(std::bit_cast<uint32_t>(full.animation.mLastFrameTime)==std::bit_cast<uint32_t>(sparse.animation.mLastFrameTime),"previous clock differs");
    for (size_t i=0;i<full.tracks.size();++i) {
        require(full.tracks[i].mBlendCounter==sparse.tracks[i].mBlendCounter,"blend count differs");
        require(full.tracks[i].mShakeX==sparse.tracks[i].mShakeX && full.tracks[i].mShakeY==sparse.tracks[i].mShakeY,"shake differs");
        require(full.tracks[i].mAttachmentID==sparse.tracks[i].mAttachmentID,"attachment differs");
    }
}
int main() {
    try {
        for(int count:{0,1,63,64,65,255,256,257}) {
            Fixture a(count),b(count);
            for(int i=0;i<count;++i) a.tracks[i].mBlendCounter=b.tracks[i].mBlendCounter=(i%7==0?3:0);
            for(int i=0;i<8;++i)tick(a,b);
            // Native sparse mode requires LIGHT_ANIMATION. Without it use the original loop.
            if(count) a.tracks.back().mBlendCounter=b.tracks.back().mBlendCounter=2;
            for(int i=0;i<3;++i)tick(a,b,0);
        }
        {
            Fixture a(256),b(256);
            auto* oldPointer=a.animation.GetTrackInstanceByName("t255");
            auto* newPointer=b.animation.GetTrackInstanceByName("t255");
            oldPointer->mBlendCounter=newPointer->mBlendCounter=2;
            for(int i=0;i<6;++i)tick(a,b);
            // Pointer still escapes after the count reaches zero: it must remain registered.
            oldPointer->mBlendCounter=newPointer->mBlendCounter=4;
            for(int i=0;i<6;++i)tick(a,b);
            b.animation.ResetEnvTrackCache(); // Portable restore replaces the old pointers.
            a.tracks[129].mBlendCounter=b.tracks[129].mBlendCounter=5;
            for(int i=0;i<6;++i)tick(a,b);
        }
        {
            Fixture a(65),b(65);
            a.animation.mNeedsTrackUpdates=b.animation.mNeedsTrackUpdates=false;
            a.tracks[64].mBlendCounter=b.tracks[64].mBlendCounter=3;
            tick(a,b);require(a.tracks[64].mBlendCounter==3,"original early return changed");
            a.animation.GetTrackInstanceByName("t0");b.animation.GetTrackInstanceByName("t0");
            for(int i=0;i<4;++i)tick(a,b);
        }
        {
            Fixture b(160);
            b.animation.RememberEnvTrack(130);b.animation.RememberEnvTrack(5);b.animation.RememberEnvTrack(64);
            require(b.animation.NextEnvTrack(0)==5,"order changed");
            b.animation.RememberEnvTrack(96); // A callback inserts a higher track this frame.
            require(b.animation.NextEnvTrack(6)==64 && b.animation.NextEnvTrack(65)==96,"new higher track missed");
            b.animation.RememberEnvTrack(2); // A lower track must wait until next frame.
            require(b.animation.NextEnvTrack(97)==130 && b.animation.NextEnvTrack(131)==160,"lower track replayed");
            require(b.animation.NextEnvTrack(0)==2,"lower track lost next frame");
        }
        gEnvSimSkip=0;
        std::cout<<"sparse-track contracts passed (8 count boundaries, escaped pointers, restore, ordering)\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
