// Standalone host tool: pull the measured animation numbers out of the game's `.reanim` data so
// cuda/spec/reanim.md can quote them as measurements instead of as the `0.47 * mVelX` estimate.
//
// It reads two independent copies of every asset out of main.pak -- the plain-text
// `reanim/<Name>.reanim` and the pre-compiled `compiled/reanim/<Name>.reanim.compiled` -- parses
// both, applies the same post-parse pass the engine applies (`ReanimationLoadDefinition`'s
// fill-in-missing sweep), and reports any disagreement.  Parsing the compiled member is the
// cross-check: it is the byte stream `DefinitionReadCompiledFile` would hand to
// `DefMapReadFromCache`, decoded here with the *real* struct definitions
// (`#include "PvzpLib/Reanimator.h"`, sizes via `offsetof`/`sizeof` rather than by hand), so a
// wrong field size shows up immediately as a length mismatch or garbage track names.  Note the
// shipped blobs turn out to be 32-bit-schema and this build rejects them, so at runtime the
// definition really does come from the text XML -- see the "compiled member is in the ..." line.
//
// Build (host g++ only -- no CUDA, no SDL, no link against the game; zlib is the only dependency):
//   c++ -O2 -std=c++20 -I src -I src/SexyAppFramework -o /tmp/reanim_dump cuda/ReanimDump.cpp -lz
// Run:
//   /tmp/reanim_dump <path/to/main.pak> Zombie [anim_walk anim_walk2 ...]
//   /tmp/reanim_dump <path/to/main.pak> --list
//   /tmp/reanim_dump <path/to/main.pak> Zombie anim_walk --velx 0.37
//   /tmp/reanim_dump <path/to/main.pak> Zombie anim_death --rate 24      (death: caller-supplied rate)
//   /tmp/reanim_dump <path/to/main.pak> LawnMoweredZombie --ground locator locator
//
// Format authorities, all in this repo:
//   pak index        src/SexyAppFramework/paklib/PakInterface.cpp:84-216   (XOR 0xF7, magic
//                          0xBAC04AC0, version 0, records {u8 flags,u8 len,name,i32 size,i64 mtime},
//                          bit 0x80 of flags = end; file data begins right after the index)
//   XML -> def       src/PvzpLib/Definition.cpp:1087-1171 (field tables 142-168), element order
//   fill-in-missing  src/PvzpLib/Reanimator.cpp:205-261
//   compiled blob    src/PvzpLib/Definition.cpp:1180-1275 (write) / 379-484 (read),
//                          CompressedDefinitionHeader {u32 cookie 0xDEADFED4, u32 raw size} + zlib
//   frames per layer src/PvzpLib/Reanimator.cpp:1091-1127
//   velocity         src/PvzpLib/Reanimator.cpp:1315-1325 + Zombie.cpp:6643-6696, 4104-4141
#include "PvzpLib/Reanimator.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <zlib.h>

// ---------------------------------------------------------------- pak -------------------------
// Mirrors PakInterface::AddPakFile: the whole pak is XOR-mixed with 0xF7, so decoding is one pass.
struct Pak
{
	std::vector<uint8_t>			mData;
	struct Entry { size_t mStart; size_t mSize; };
	std::map<std::string, Entry>	mIndex;

	static std::string Normalize(std::string thePath)
	{
		for (char& aChar : thePath)
		{
			if (aChar == '\\')
				aChar = '/';
			aChar = static_cast<char>(std::toupper(static_cast<unsigned char>(aChar)));
		}
		while (thePath.rfind("./", 0) == 0)
			thePath = thePath.substr(2);
		return thePath;
	}

	bool Open(const char* theFileName)
	{
		std::ifstream aFile(theFileName, std::ios::binary);
		if (!aFile)
			return false;
		aFile.seekg(0, std::ios::end);
		mData.resize(static_cast<size_t>(aFile.tellg()));
		aFile.seekg(0, std::ios::beg);
		aFile.read(reinterpret_cast<char*>(mData.data()), static_cast<std::streamsize>(mData.size()));
		if (!aFile)
			return false;
		for (uint8_t& aByte : mData)
			aByte ^= 0xF7;

		size_t aPos = 0;
		auto ReadU32 = [&](uint32_t& theOut) {
			std::memcpy(&theOut, mData.data() + aPos, 4); aPos += 4;
			theOut = static_cast<uint32_t>(theOut);
			return theOut;
		};
		uint32_t aMagic = 0, aVersion = 0;
		if (mData.size() < 8 || ReadU32(aMagic) != 0xBAC04AC0u)
			return false;
		if (ReadU32(aVersion) != 0)
			return false;

		size_t aCursor = 0;						// running offset of the packed file data
		for (;;)
		{
			uint8_t aFlags = mData[aPos++];
			if (aFlags & 0x80)
				break;
			uint8_t aNameWidth = mData[aPos++];
			std::string aName(reinterpret_cast<const char*>(mData.data() + aPos), aNameWidth);
			aPos += aNameWidth;
			uint32_t aSize = 0;
			std::memcpy(&aSize, mData.data() + aPos, 4); aPos += 4;
			aPos += 8;							// file time, unused here
			mIndex[Normalize(aName)] = Entry { aCursor, aSize };
			aCursor += aSize;
		}
		mDataStart = aPos;
		return aCursor + aPos == mData.size();	// the same completeness check the engine asserts
	}

	size_t mDataStart = 0;

	bool Slurp(const std::string& theName, std::vector<uint8_t>& theOut) const
	{
		auto anItr = mIndex.find(Normalize(theName));
		if (anItr == mIndex.end())
			return false;
		theOut.assign(mData.begin() + static_cast<ptrdiff_t>(mDataStart + anItr->second.mStart),
			mData.begin() + static_cast<ptrdiff_t>(mDataStart + anItr->second.mStart + anItr->second.mSize));
		return true;
	}
};

// ------------------------------------------------- definition data ------------------------------
// One parsed frame of one track.  `mPresent` records which fields the file actually wrote: the
// engine's fill-in sweep inherits anything left out (or left at DEFAULT_FIELD_PLACEHOLDER) from the
// previous frame, so presence has to survive parsing to reproduce that.
struct Xform
{
	float			mValue[8] = { DEFAULT_FIELD_PLACEHOLDER, DEFAULT_FIELD_PLACEHOLDER, DEFAULT_FIELD_PLACEHOLDER,
					DEFAULT_FIELD_PLACEHOLDER, DEFAULT_FIELD_PLACEHOLDER, DEFAULT_FIELD_PLACEHOLDER,
					DEFAULT_FIELD_PLACEHOLDER, DEFAULT_FIELD_PLACEHOLDER };
	bool			mPresent[8] = { false, false, false, false, false, false, false, false };
	std::string		mImage, mFont, mText;

	// Field order of gReanimatorTransformDefFields (Definition.cpp:142-153) minus the pointer types.
	enum { X = 0, Y = 1, KX = 2, KY = 3, SX = 4, SY = 5, FRAME = 6, ALPHA = 7 };
	static const char* const* Names()
	{
		static const char* const kNames[8] = { "x", "y", "kx", "ky", "sx", "sy", "f", "a" };
		return kNames;
	}
};

struct TrackDef { std::string mName; std::vector<Xform> mTransforms; };
struct Definition { float mFPS = 12.0f; std::vector<TrackDef> mTracks; };

// ReanimationFillInMissingData, Reanimator.cpp:205-211 + the per-track sweep at 226-261.
// In-place, exactly like the engine: a field that is PLACEHOLDER inherits, otherwise it becomes
// the new "previous" value.  Returns the number of inherited slots.
static long FillInMissing(Definition& theDef)
{
	long aInherited = 0;
	static const float kDefault[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };
	for (TrackDef& aTrack : theDef.mTracks)
	{
		float aPrev[8];
		std::memcpy(aPrev, kDefault, sizeof(aPrev));
		for (Xform& aXform : aTrack.mTransforms)
		{
			for (int i = 0; i < 8; i++)
			{
				if (aXform.mValue[i] == DEFAULT_FIELD_PLACEHOLDER)
				{
					aXform.mValue[i] = aPrev[i];
					++aInherited;
				}
				else
					aPrev[i] = aXform.mValue[i];
			}
		}
	}
	return aInherited;
}

// Reanimation::GetFramesForLayer, Reanimator.cpp:1091-1127 (first..last frame with mFrame >= 0).
static void GetFramesForLayer(const TrackDef& theTrack, int& theStart, int& theCount)
{
	theStart = 0;
	theCount = 1;
	if (theTrack.mTransforms.empty())
		return;
	for (size_t i = 0; i < theTrack.mTransforms.size(); i++)
		if (theTrack.mTransforms[i].mValue[Xform::FRAME] >= 0.0f)
		{
			theStart = static_cast<int>(i);
			break;
		}
	for (size_t j = static_cast<size_t>(theStart); j < theTrack.mTransforms.size(); j++)
		if (theTrack.mTransforms[j].mValue[Xform::FRAME] >= 0.0f)
			theCount = static_cast<int>(j) - theStart + 1;
}

// ----------------------------------------------------- text/XML ---------------------------------
// Grammar actually present in the paks: <fps>N</fps> then <track><name>NAME</name><t>...</t>*
// </track>*, every value a child element (no attributes, no self-closing tags -- asserted below).
static bool ParseText(const std::vector<uint8_t>& theBytes, Definition& theOut, std::string& theError)
{
	std::string aText(theBytes.begin(), theBytes.end());
	if (aText.find("/>") != std::string::npos)
	{
		theError = "self-closing tags present, this parser assumes <t></t> pairs";
		return false;
	}
	size_t aPos = 0;
	// read <fps>
	for (;;)
	{
		size_t aNext = aText.find('<', aPos);
		if (aNext == std::string::npos) { theError = "no elements"; return false; }
		aPos = aNext;
		if (aText.compare(aPos, 4, "<fps") == 0)
		{
			size_t aGt = aText.find('>', aPos), aClose = aText.find("</fps>", aGt);
			theOut.mFPS = static_cast<float>(std::atof(aText.substr(aGt + 1, aClose - aGt - 1).c_str()));
			aPos = aClose + 6;
			break;
		}
		if (aText.compare(aPos, 7, "<track>") == 0)
			break;						// files without <fps> start straight at the tracks
		aPos += 4;
	}

	TrackDef aTrack;
	bool aInTrack = false;
	for (;;)
	{
		size_t aNext = aText.find('<', aPos);
		if (aNext == std::string::npos)
			break;
		aPos = aNext;
		if (aText.compare(aPos, 7, "<track>") == 0)
		{
			if (aInTrack && !aTrack.mName.empty())
				theOut.mTracks.push_back(aTrack);
			aTrack = TrackDef();
			aInTrack = true;
			aPos += 7;
			continue;
		}
		if (aText.compare(aPos, 8, "</track>") == 0)
		{
			if (aInTrack && !aTrack.mName.empty())
				theOut.mTracks.push_back(aTrack);
			aTrack = TrackDef();
			aInTrack = false;
			aPos += 8;
			continue;
		}
		if (aText.compare(aPos, 6, "<name>") == 0)
		{
			size_t aClose = aText.find("</name>", aPos);
			aTrack.mName = aText.substr(aPos + 6, aClose - aPos - 6);
			aPos = aClose + 7;
			continue;
		}
		if (aText.compare(aPos, 3, "<t>") == 0 || aText.compare(aPos, 4, "<t/>") == 0)
		{
			Xform aXform;
			size_t aEnd;
			if (aText.compare(aPos, 4, "<t/>") == 0)
				aEnd = aPos + 4;
			else
			{
				aEnd = aText.find("</t>", aPos);
				size_t aCursor = aPos + 3;
				while (aCursor < aEnd)
				{
					size_t aTagStart = aText.find('<', aCursor);
					if (aTagStart == std::string::npos || aTagStart >= aEnd)
						break;
					size_t aGt = aText.find('>', aTagStart);
					std::string aName = aText.substr(aTagStart + 1, aGt - aTagStart - 1);
					size_t aClose = aText.find("</" + aName + ">", aGt);
					std::string aBody = aText.substr(aGt + 1, aClose - aGt - 1);
					if (aName == "i")
						aXform.mImage = aBody;
					else if (aName == "font")
						aXform.mFont = aBody;
					else if (aName == "text")
						aXform.mText = aBody;
					else
					{
						for (int i = 0; i < 8; i++)
							if (aName == Xform::Names()[i])
							{
								aXform.mValue[i] = static_cast<float>(std::atof(aBody.c_str()));
								aXform.mPresent[i] = true;
								break;
							}
					}
					aCursor = aClose + aName.size() + 3;
				}
				aEnd += 4;
			}
			aTrack.mTransforms.push_back(aXform);
			aPos = aEnd;
			continue;
		}
		aPos += 1;
	}
	if (aInTrack && !aTrack.mName.empty())
		theOut.mTracks.push_back(aTrack);
	return !theOut.mTracks.empty();
}

// ------------------------------------------------------ compiled --------------------------------
namespace
{
	// Sanity anchors: the blob is laid out with the *real* structs, so a compile-time mismatch
	// against the sizes this file was written against is a hard error, not a silent misread.
	static_assert(sizeof(ReanimatorDefinition) == 32, "ReanimatorDefinition layout changed");
	static_assert(offsetof(ReanimatorDefinition, mTracks.tracks) == 0);
	static_assert(offsetof(ReanimatorDefinition, mTracks.count) == 8);
	static_assert(offsetof(ReanimatorDefinition, mFPS) == 16);
	static_assert(sizeof(ReanimatorTrack) == 32, "ReanimatorTrack layout changed");
	static_assert(offsetof(ReanimatorTrack, mName) == 0);
	static_assert(offsetof(ReanimatorTrack, mTransforms.mTransforms) == 8);
	static_assert(offsetof(ReanimatorTrack, mTransforms.count) == 16);
	static_assert(offsetof(ReanimatorTrack, mIsAttacher) == 24);
	static_assert(sizeof(ReanimatorTransform) == 56, "ReanimatorTransform layout changed");
	static_assert(offsetof(ReanimatorTransform, mTransX) == 0);
	static_assert(offsetof(ReanimatorTransform, mAlpha) == 28);
	static_assert(offsetof(ReanimatorTransform, mImage) == 32);
	static_assert(offsetof(ReanimatorTransform, mFont) == 40);
	static_assert(offsetof(ReanimatorTransform, mText) == 48);

	struct Cursor
	{
		const uint8_t*			mPtr;
		const uint8_t*			mEnd;
		bool					mFailed = false;

		uint32_t U32()
		{
			uint32_t aResult = 0;
			if (mPtr + 4 > mEnd) { mFailed = true; return 0; }
			std::memcpy(&aResult, mPtr, 4);
			mPtr += 4;
			return aResult;
		}
		void Blob(void* theDst, size_t theSize)
		{
			if (mPtr + theSize > mEnd) { mFailed = true; return; }
			std::memcpy(theDst, mPtr, theSize);
			mPtr += theSize;
		}
		std::string String()
		{
			uint32_t aLen = U32();
			if (mFailed || mPtr + aLen > mEnd) { mFailed = true; return ""; }
			std::string aResult(reinterpret_cast<const char*>(mPtr), aLen);
			mPtr += aLen;
			return aResult;
		}
	};

	// The two layouts a compiled member can be in.  `kNative` is this build's, taken from the real
	// structs rather than guessed; `kLegacy32` is the 32-bit layout the stock PopCap-era paks were
	// packed with (4-byte pointers, and no mIsAttacher in ReanimatorTrack -> 12-byte track records).
	// DefinitionReadCompiledFile only accepts a blob whose leading u32 equals
	// DefWriteToCacheArray's aDefMap->mDefSize *of the running build*, and DefinitionCalcHash folds
	// mDefSize + every mFieldOffset into the schema hash, so a kLegacy32 blob is rejected outright:
	// the engine then falls through to DefinitionCompileFile, i.e. it parses the TEXT reanim.
	struct Layout
	{
		int			mDefRaw;			// sizeof(ReanimatorDefinition)
		int			mCountAt;			// offsetof(mTracks.count)
		int			mFPSAt;				// offsetof(mFPS)
		int			mTrackRaw;			// sizeof(ReanimatorTrack)
		int			mTrackCountAt;		// offsetof(mTransforms.count) within a track record
		int			mXformRaw;			// sizeof(ReanimatorTransform)
		const char*	mName;
	};

	// DefWriteToCache* / DefMapReadFromCache field order: ReanimatorDefinition = {track, fps};
	// ReanimatorTrack = {name, t}; ReanimatorTransform = {8 floats, i, font, text}.
	bool ParseCompiledWith(const uint8_t* aRaw, size_t theRawSize, const Layout& theLayout,
		Definition& theOut, std::string& theError)
	{
		Cursor aCursor { aRaw + 4, aRaw + theRawSize };		// +4: stored schema hash
		std::vector<uint8_t> aDefRaw(static_cast<size_t>(theLayout.mDefRaw));
		aCursor.Blob(aDefRaw.data(), aDefRaw.size());
		int aTrackCount = 0;
		float aFPS = 0.0f;
		std::memcpy(&aTrackCount, aDefRaw.data() + theLayout.mCountAt, 4);
		std::memcpy(&aFPS, aDefRaw.data() + theLayout.mFPSAt, 4);
		if (aCursor.mFailed || aTrackCount <= 0 || aTrackCount > 100000)
		{
			theError = "definition header unreadable";
			return false;
		}
		theOut.mFPS = aFPS;

		const uint32_t aTrackSize = aCursor.U32();
		if (aTrackSize != static_cast<uint32_t>(theLayout.mTrackRaw))
		{
			theError = "track record size " + std::to_string(aTrackSize) +
				" != " + std::to_string(theLayout.mTrackRaw) + " for this layout";
			return false;
		}
		std::vector<uint8_t> aTrackBlob(static_cast<size_t>(aTrackSize) * static_cast<size_t>(aTrackCount));
		aCursor.Blob(aTrackBlob.data(), aTrackBlob.size());
		theOut.mTracks.resize(static_cast<size_t>(aTrackCount));
		for (int i = 0; i < aTrackCount; i++)
		{
			theOut.mTracks[static_cast<size_t>(i)].mName = aCursor.String();
			int aCount = 0;
			std::memcpy(&aCount, aTrackBlob.data() + static_cast<size_t>(aTrackSize) * static_cast<size_t>(i)
				+ theLayout.mTrackCountAt, 4);
			if (aCount < 0 || aCount > 1000000)
			{
				theError = "transform count nonsense";
				return false;
			}
			const uint32_t aXformSize = aCursor.U32();
			if (aXformSize != static_cast<uint32_t>(theLayout.mXformRaw))
			{
				theError = "transform record size mismatch";
				return false;
			}
			std::vector<uint8_t> aBlob(static_cast<size_t>(aXformSize) * static_cast<size_t>(aCount));
			aCursor.Blob(aBlob.data(), aBlob.size());
			std::vector<Xform>& aOut = theOut.mTracks[static_cast<size_t>(i)].mTransforms;
			aOut.resize(static_cast<size_t>(aCount));
			for (int j = 0; j < aCount; j++)
			{
				// The 8 floats are the leading bytes of the record in both layouts (x,y,kx,ky,sx,sy,f,a
				// == offsetof(mTransX)..offsetof(mAlpha)+4, verified by the static_asserts above).
				const uint8_t* aSrc = aBlob.data() + static_cast<size_t>(aXformSize) * static_cast<size_t>(j);
				std::memcpy(aOut[static_cast<size_t>(j)].mValue, aSrc, sizeof(float) * 8);
				for (int k = 0; k < 8; k++)
					aOut[static_cast<size_t>(j)].mPresent[k] = true;
			}
			// The pointer fields are appended after the whole raw blob, per transform, in
			// gReanimatorTransformDefFields order (i, font, text).
			for (int j = 0; j < aCount; j++)
			{
				aOut[static_cast<size_t>(j)].mImage = aCursor.String();
				aOut[static_cast<size_t>(j)].mFont = aCursor.String();
				aOut[static_cast<size_t>(j)].mText = aCursor.String();
			}
		}
		if (aCursor.mFailed)
		{
			theError = "ran off the end";
			return false;
		}
		if (static_cast<size_t>(aCursor.mPtr - aRaw) != theRawSize)
		{
			theError = "trailing bytes: consumed " + std::to_string(aCursor.mPtr - aRaw)
				+ " of " + std::to_string(theRawSize);
			return false;
		}
		return true;
	}

	// Which layout a blob is in is decided by what actually decodes cleanly (the exact-length
	// consumption check above is the discriminator: a wrong record stride cannot land exactly on
	// the end of the buffer).
	bool ParseCompiled(const std::vector<uint8_t>& theBytes, Definition& theOut, std::string& theError,
		std::string& theLayoutName, bool& theIsNativeLayout)
	{
		if (theBytes.size() < 8)
		{
			theError = "compiled member shorter than CompressedDefinitionHeader";
			return false;
		}
		uint32_t aCookie = 0, aRawSize = 0;
		std::memcpy(&aCookie, theBytes.data(), 4);
		std::memcpy(&aRawSize, theBytes.data() + 4, 4);
		if (aCookie != 0xDEADFED4u)
		{
			theError = "compile def cookie wrong";
			return false;
		}
		std::vector<uint8_t> aRaw(aRawSize);
		uLongf anUncompressed = aRawSize;
		if (uncompress(aRaw.data(), &anUncompressed, theBytes.data() + 8,
				static_cast<uLong>(theBytes.size() - 8)) != Z_OK || anUncompressed != aRawSize)
		{
			theError = "zlib inflate failed";
			return false;
		}
		const Layout kNative = {
			static_cast<int>(sizeof(ReanimatorDefinition)),
			static_cast<int>(offsetof(ReanimatorDefinition, mTracks.count)),
			static_cast<int>(offsetof(ReanimatorDefinition, mFPS)),
			static_cast<int>(sizeof(ReanimatorTrack)),
			static_cast<int>(offsetof(ReanimatorTrack, mTransforms.count)),
			static_cast<int>(sizeof(ReanimatorTransform)), "this build (64-bit)" };
		const Layout kLegacy32 = { 16, 4, 8, 12, 8, 44, "32-bit (pre-port)" };
		std::string anErrorNative, anErrorLegacy;
		if (ParseCompiledWith(aRaw.data(), aRaw.size(), kNative, theOut, anErrorNative))
		{
			theLayoutName = kNative.mName;
			theIsNativeLayout = true;
			return true;
		}
		if (ParseCompiledWith(aRaw.data(), aRaw.size(), kLegacy32, theOut, anErrorLegacy))
		{
			theLayoutName = kLegacy32.mName;
			theIsNativeLayout = false;
			return true;
		}
		theError = "native layout: " + anErrorNative + "; legacy32 layout: " + anErrorLegacy;
		return false;
	}
}

// -------------------------------------------------------- report --------------------------------
// Everything below reproduces the engine's arithmetic literally, including where it is float and
// where it widens to double, because that is what "bit-exact x" turns on:
//   Zombie::UpdateAnimSpeed (Zombie.cpp:6684-6693), all float:
//     aDistance   = x[mFrameStart + mFrameCount - 1] - x[mFrameStart]
//     aOneOverSpeed = (float)mFrameCount / aDistance
//     mAnimRate   = fl(fl(fl(mVelX * aOneOverSpeed) * 47.0f) / mScaleZombie)
//   Reanimation::Update (Reanimator.cpp:431): mAnimTime (float) += (double)SECONDS_PER_UPDATE *
//     (double)mAnimRate / (double)mFrameCount, the sum formed in double and narrowed once.
//     SECONDS_PER_UPDATE is declared `constexpr const double ... = 0.01f` (Reanimator.h:52), i.e.
//     the double 0.0099999997764825824, not 0.01.
//   Reanimation::GetFrameTime (Reanimator.cpp:906-927): aFrameCount = mFrameCount - 1 for
//     REANIM_LOOP and REANIM_PLAY_ONCE_AND_HOLD; aAnimPosition (float) = mFrameStart +
//     mAnimTime * aFrameCount; before = FloatRoundToInt(floor(aAnimPosition)); after = before + 1
//     unless before is the last frame of the range, where after == before (so dx == 0 there).
//   Reanimation::GetTrackVelocity (Reanimator.cpp:1315-1325): returns
//     (float)((double)(x[after] - x[before]) * (double)SECONDS_PER_UPDATE * (double)mAnimRate).
//   Zombie::UpdateZombieWalking (:4129, :4146): aSpeed = (float)velocity * mScaleZombie;
//     mPosX -= aSpeed (float). Bodies are attachments: EffectSystem skips them, and
//     Zombie::UpdateReanim advances them AFTER movement. The loop mean below is only
//     a diagnostic of the resource recurrence, not a substitute for imported animation phase.
static void ReportAnim(const Definition& theDef, const char* const theGroundName, const char* theAnimName,
	float theVelX, float theScale, float theRateOverride)
{
	const TrackDef* aGround = nullptr;
	const TrackDef* aAnim = nullptr;
	for (const TrackDef& aTrack : theDef.mTracks)
	{
		if (aTrack.mName == theGroundName)
			aGround = &aTrack;
		if (aTrack.mName == theAnimName)
			aAnim = &aTrack;
	}
	if (aAnim == nullptr)
	{
		std::printf("\n== %s ==  ABSENT from this reanim\n", theAnimName);
		return;
	}
	int aStart = 0, aCount = 0;
	GetFramesForLayer(*aAnim, aStart, aCount);
	std::printf("\n== %s [%d..%d]  mFrameStart=%d mFrameCount=%d ==\n", theAnimName, aStart,
		aStart + aCount - 1, aStart, aCount);
	if (aGround == nullptr)
	{
		std::printf("  no %s track: TrackExists is false, so UpdateZombieWalking takes the raw\n"
			"  mVelX branch (:4133-4141) and UpdateAnimSpeed's _ground branch never runs.\n", theGroundName);
		return;
	}
	if (aStart + aCount > static_cast<int>(aGround->mTransforms.size()))
	{
		std::printf("  the range runs past the end of the %s track\n", theGroundName);
		return;
	}

	const std::vector<Xform>& aX = aGround->mTransforms;
	const int aLast = aStart + aCount - 1;
	const float aFirstX = aX[static_cast<size_t>(aStart)].mValue[Xform::X];
	const float aLastX = aX[static_cast<size_t>(aLast)].mValue[Xform::X];
	const float aDistance = aLastX - aFirstX;			// exactly what UpdateAnimSpeed computes
	const bool aUsesGround = aDistance >= 1e-6f;
	std::printf("  %s x[%d]=%.9g  x[%d]=%.9g  aDistance=%.9g%s\n", theGroundName, aStart, aFirstX,
		aLast, aLastX, aDistance, aUsesGround ? "" : "  (< 1e-6: the _ground branch is skipped)");

	// mAnimRate, in the engine's float precedence order, then the rate actually in force.  An
	// explicit --rate always wins: it is what the caller hardcoded (death 24..30 / 14 / 18, mowed 8).
	float aRate = 0.0f;
	if (theRateOverride > 0.0f)
	{
		aRate = theRateOverride;
		std::printf("  mAnimRate = %.9g  (forced with --rate: this is the rate the caller set, "
			"UpdateAnimSpeed's _ground branch %s)\n", aRate,
			aUsesGround ? "would overwrite it while the zombie walks" : "is skipped entirely here");
	}
	else if (aUsesGround)
	{
		const float aOneOverSpeed = static_cast<float>(aCount) / aDistance;
		aRate = static_cast<float>(static_cast<float>(static_cast<float>(theVelX * aOneOverSpeed) * 47.0f) / theScale);
		std::printf("  mAnimRate = mVelX*(%d/aDistance)*47/mScaleZombie;  %.9g per unit mVelX"
			"  -> %.9g at mVelX=%.3f mScaleZombie=%.3f\n", aCount,
			static_cast<double>(aRate) / theVelX, aRate, theVelX, theScale);
	}
	else
	{
		aRate = theDef.mFPS;
		std::printf("  mAnimRate not derived from _ground here (aDistance==0): it stays whatever PlayZombieReanim"
			" set (mOriginalAnimRate), the caller's aAnimRate argument.  Using %.9g%s for the advance column.\n",
			aRate, " (= the definition fps, a PLACEHOLDER: pass --rate for the real one)");
	}

	// Replay the update recurrence so the per-frame visits and the loop length are measured, not
	// estimated.  REANIM_LOOP (walk) and REANIM_PLAY_ONCE_AND_HOLD (death) share GetFrameTime's
	// aFrameCount = mFrameCount - 1; only the end condition differs, and one loop is what is asked.
	// theLoops > 1 keeps going so the mean converges: after a wrap mAnimTime does not return to 0
	// (the remainder carries), so a single loop is a slightly biased sample of the steady state.
	std::vector<int> aVisits(static_cast<size_t>(aCount), 0);
	int anUpdatesInLoop = 0;					// updates in the first loop only
	double aLoopAdvance = 0.0;					// advance over all replayed loops
	double aFirstLoopAdvance = 0.0;
	int aCycles = 0, anUpdates = 0;
	bool aFirstLoopDone = false;
	constexpr int theLoops = 64;
	{
		const double aStep = static_cast<double>(0.01f) * static_cast<double>(aRate) / static_cast<double>(aCount);
		float aAnimTime = 0.0f;
		const int aSpan = aCount - 1;
		while (anUpdates < 4000000)
		{
			aAnimTime = static_cast<float>(static_cast<double>(aAnimTime) + aStep);
			if (aAnimTime >= 1.0f)						// Reanimation::Update REANIM_LOOP wrap
			{
				aAnimTime = static_cast<float>(aAnimTime - 1.0f);
				++aCycles;
			}
			const float aPosition = static_cast<float>(static_cast<float>(aAnimTime * aSpan) + aStart);
			const float aFloor = std::floor(aPosition);
			int aBefore = static_cast<int>(aFloor > 0.0f ? aFloor + 0.5f : aFloor - 0.5f);	// FloatRoundToInt
			if (aBefore >= aLast)
				aBefore = aLast;
			int aAfter = aBefore == aLast ? aLast : aBefore + 1;
			if (aBefore < aStart)
				aBefore = aStart;
			if (aAfter < aStart)
				aAfter = aStart;
			const float aDx = aX[static_cast<size_t>(aAfter)].mValue[Xform::X]
				- aX[static_cast<size_t>(aBefore)].mValue[Xform::X];
			const float aVelocity = static_cast<float>(static_cast<double>(aDx) * static_cast<double>(0.01f)
				* static_cast<double>(aRate));
			const float aSpeed = static_cast<float>(aVelocity * theScale);
			if (!aFirstLoopDone) aVisits[static_cast<size_t>(aBefore - aStart)]++;
			aLoopAdvance += -aSpeed;
			if (!aFirstLoopDone)
				aFirstLoopAdvance += -aSpeed;
			++anUpdates;
			if (aCycles == 1 && aAnimTime < 1.0f && !aFirstLoopDone)
			{
				aFirstLoopDone = true;
				anUpdatesInLoop = anUpdates;
			}
			if (aCycles >= theLoops) break;
		}
		if (aCycles == 0)							// never wrapped inside the cap
		{
			aFirstLoopDone = true;
			anUpdatesInLoop = anUpdates;
			aCycles = 1;
		}
	}

	std::printf("  frame   f           x              dx     advance px/update   visits\n");
	double aSumDx = 0.0;
	float aMinDx = 0.0f, aMaxDx = 0.0f;
	bool aSeen = false;
	int aZeroDx = 0;
	for (int i = aStart; i <= aLast; i++)
	{
		const int anAfter = std::min(i + 1, aLast);
		const float aDx = aX[static_cast<size_t>(anAfter)].mValue[Xform::X]
			- aX[static_cast<size_t>(i)].mValue[Xform::X];
		const float aSpeed = static_cast<float>(static_cast<float>(static_cast<double>(aDx)
			* static_cast<double>(0.01f) * static_cast<double>(aRate)) * theScale);
		if (i < aLast)
		{
			aSumDx += aDx;
			if (!aSeen) { aMinDx = aMaxDx = aDx; aSeen = true; }
			else { aMinDx = std::min(aMinDx, aDx); aMaxDx = std::max(aMaxDx, aDx); }
			if (aDx == 0.0f)
				++aZeroDx;
		}
		std::printf("  %5d  %4.0f  %12.9g  %12.9g  %14.9g  %6d%s\n", i,
			aX[static_cast<size_t>(i)].mValue[Xform::FRAME], aX[static_cast<size_t>(i)].mValue[Xform::X],
			aDx, aSpeed, aVisits[static_cast<size_t>(i - aStart)],
			anAfter == i ? "   <- last frame: after==before, dx==0" : "");
		// Machine-readable coordinates for gameplay tracks such as a shooter's animated stem.
		std::printf("xy %d %.9g %.9g\n", i, aX[static_cast<size_t>(i)].mValue[Xform::X],
			aX[static_cast<size_t>(i)].mValue[Xform::Y]);
	}
	std::printf("  sum dx over the %d transitions = %.9g (aDistance %.9g);  dx range [%.9g, %.9g],"
		" %d of %d transitions are zero\n", aCount - 1, aSumDx, aDistance, aMinDx, aMaxDx, aZeroDx, aCount - 1);
	if (anUpdatesInLoop > 0)
	{
		const double aMean = aLoopAdvance / anUpdates;
		std::printf("  replay: first loop %d updates / %.9g px, %d loops / %d updates / %.9g px total; "
			"mean %.9g px/update = %.9f * mVelX (spec claim 0.47 * mVelX = %.9g)\n",
			anUpdatesInLoop, aFirstLoopAdvance, aCycles, anUpdates, aLoopAdvance, aMean,
			aMean / theVelX, 0.47 * theVelX);
		std::printf("  loop length = mFrameCount/(SECONDS_PER_UPDATE*mAnimRate) [Zombie.cpp:1729 formula]"
			" = %d*100/%.9g = %.6f updates at the rate in force;  raw mFrameCount*100/rate: at rate 24 ->"
			" %.6g, at 27 -> %.6g, at 30 -> %.6g updates\n", aCount, aRate, aCount * 100.0 / aRate,
			aCount * 100.0 / 24.0, aCount * 100.0 / 27.0, aCount * 100.0 / 30.0);
	}
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::fprintf(stderr, "usage: %s MAIN.PAK (<Name> [--velx F] [--scale F] [--rate F] [--ground NAME]|--list)\n", argv[0]);
		return 2;
	}
	Pak aPak;
	if (!aPak.Open(argv[1]))
	{
		std::fprintf(stderr, "cannot open/parse pak %s\n", argv[1]);
		return 2;
	}
	std::fprintf(stderr, "pak %s: %zu members, %zu bytes, index ends at %zu\n", argv[1],
		aPak.mIndex.size(), aPak.mData.size(), aPak.mDataStart);

	if (argc >= 3 && !std::strcmp(argv[2], "--list"))
	{
		for (const auto& [aKey, anEntry] : aPak.mIndex)
			if (aKey.find("REANIM/") != std::string::npos && aKey.size() > 7 &&
				aKey.compare(aKey.size() - 7, 7, ".REANIM") == 0)
				std::printf("%-52s %zu\n", aKey.c_str(), anEntry.mSize);
		return 0;
	}
	if (argc < 3)
		return 2;

	const std::string aName = argv[2];
	float aVelX = 1.0f, aScale = 1.0f, aRate = 0.0f;
	std::string aGround = "_ground";
	std::vector<std::string> aAnims;
	for (int i = 3; i < argc; i++)
	{
		if (!std::strcmp(argv[i], "--velx") && i + 1 < argc)
			aVelX = std::strtof(argv[++i], nullptr);
		else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc)
			aScale = std::strtof(argv[++i], nullptr);
		else if (!std::strcmp(argv[i], "--rate") && i + 1 < argc)
			aRate = std::strtof(argv[++i], nullptr);
		else if (!std::strcmp(argv[i], "--ground") && i + 1 < argc)
			aGround = argv[++i];
		else
			aAnims.push_back(argv[i]);
	}

	std::vector<uint8_t> aTextBytes, aCompBytes;
	const std::string aTextMember = "reanim/" + aName + ".reanim";
	const std::string aCompMember = "compiled/reanim/" + aName + ".reanim.compiled";
	if (!aPak.Slurp(aTextMember, aTextBytes))
	{
		std::fprintf(stderr, "missing %s\n", aTextMember.c_str());
		return 2;
	}
	const bool aHaveCompiled = aPak.Slurp(aCompMember, aCompBytes);

	Definition aFromText;
	std::string anError;
	if (!ParseText(aTextBytes, aFromText, anError))
	{
		std::fprintf(stderr, "text parse failed: %s\n", anError.c_str());
		return 2;
	}
	const long aTextInherited = FillInMissing(aFromText);

	Definition aFromCompiled;
	bool aCompiledParsed = false, aCompiledIsNative = false;
	std::string aCompiledLayout;
	if (aHaveCompiled && !ParseCompiled(aCompBytes, aFromCompiled, anError, aCompiledLayout, aCompiledIsNative))
		std::fprintf(stderr, "compiled parse of %s failed: %s\n", aCompMember.c_str(), anError.c_str());
	else if (aHaveCompiled)
	{
		aCompiledParsed = true;
		FillInMissing(aFromCompiled);
	}

	std::printf("%s: fps=%.9g tracks=%zu frames/track=%zu   (fill-in sweep inherited %ld slots)\n",
		aTextMember.c_str(), aFromText.mFPS, aFromText.mTracks.size(),
		aFromText.mTracks.empty() ? 0 : aFromText.mTracks[0].mTransforms.size(), aTextInherited);

	if (aCompiledParsed)
	{
		size_t aDiffs = 0;
		if (aFromCompiled.mTracks.size() != aFromText.mTracks.size() || aFromCompiled.mFPS != aFromText.mFPS)
			++aDiffs;
		for (size_t i = 0; i < std::min(aFromCompiled.mTracks.size(), aFromText.mTracks.size()); i++)
		{
			const TrackDef& a = aFromText.mTracks[i];
			const TrackDef& b = aFromCompiled.mTracks[i];
			if (a.mName != b.mName || a.mTransforms.size() != b.mTransforms.size())
			{
				++aDiffs;
				std::printf("  MISMATCH track %zu: text %s/%zu  compiled %s/%zu\n", i, a.mName.c_str(),
					a.mTransforms.size(), b.mName.c_str(), b.mTransforms.size());
				continue;
			}
			for (size_t j = 0; j < a.mTransforms.size(); j++)
				for (int k = 0; k < 8; k++)
					if (std::memcmp(a.mTransforms[j].mValue + k, b.mTransforms[j].mValue + k, 4) != 0)
					{
						if (++aDiffs <= 12)
							std::printf("  MISMATCH %s[%zu].%s: %.9g vs %.9g\n", a.mName.c_str(), j,
								Xform::Names()[k], a.mTransforms[j].mValue[k], b.mTransforms[j].mValue[k]);
					}
		}
		std::printf("text vs %s: %s (%zu field differences)\n", aCompMember.c_str(),
			aDiffs == 0 ? "IDENTICAL" : "DIFFERS", aDiffs);
		std::printf("  compiled member is in the %s layout; the engine's reader would %s it, so at "
			"runtime the definition comes from %s (%s)\n",
			aCompiledLayout.c_str(), aCompiledIsNative ? "accept" : "REJECT (schema hash and mDefSize "
			"are computed from the running build's struct sizes, Definition.cpp:609-635 / 383-387)",
			aCompiledIsNative ? "the compiled blob" : "the text XML", aTextMember.c_str());
	}
	else
		std::printf("text vs compiled: %s -- only the text parse is available\n",
			aHaveCompiled ? "compiled member present but unreadable" : "no compiled member in this pak");

	std::printf("\ntrack table (Reanimation::GetFramesForLayer):\n");
	std::printf("%-34s %6s %6s %8s\n", "name", "start", "count", "blank in range");
	for (const TrackDef& aTrack : aFromText.mTracks)
	{
		int aStart = 0, aCount = 0;
		GetFramesForLayer(aTrack, aStart, aCount);
		int aBlank = 0;
		for (int i = aStart; i < aStart + aCount && i < static_cast<int>(aTrack.mTransforms.size()); i++)
			if (aTrack.mTransforms[static_cast<size_t>(i)].mValue[Xform::FRAME] < 0.0f)
				++aBlank;
		std::printf("%-34s %6d %6d %8d\n", aTrack.mName.c_str(), aStart, aCount, aBlank);
	}

	if (aAnims.empty())
	{
		std::printf("\n(no anim tracks named; pass them as extra arguments for the dx tables)\n");
		return 0;
	}
	for (const std::string& anAnim : aAnims)
		ReportAnim(aFromText, aGround.c_str(), anAnim.c_str(), aVelX, aScale, aRate);
	return 0;
}
