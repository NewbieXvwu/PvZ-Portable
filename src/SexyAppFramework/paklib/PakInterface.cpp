/*
 * Portions of this file are based on the PopCap Games Framework
 * Copyright (C) 2005-2009 PopCap Games, Inc.
 *
 * Copyright (C) 2026 Zhou Qiankang <wszqkzqk@qq.com>
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later AND LicenseRef-PopCap
 *
 * This file is part of PvZ-Portable.
 *
 * PvZ-Portable is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PvZ-Portable is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with PvZ-Portable. If not, see <https://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <filesystem>
#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "Common.h"
#include "PakInterface.h"
#include "fcaseopen/fcaseopen.h"

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned long ulong;

enum
{
	FILEFLAGS_END = 0x80
};

PakInterface* gPakInterface = new PakInterface();

PakInterface::PakInterface()
{
}

#ifndef LOW_MEMORY
// The pak is a read-only ~45 MB blob that every worker needs and none of them writes: map it instead of
// reading a private copy, and decode the XOR on the way out. The pages the loader never touches (most of
// the image data, now that headless decodes sizes only) stay on disk and are shared by every process.
static bool OpenPakStorage(const std::string& theFileName, void** theData, size_t* theSize, bool* isMapped)
{
	FILE* aFileHandle = fcaseopen(theFileName.c_str(), "rb");
	if (aFileHandle == nullptr)
		return false;

	fseek(aFileHandle, 0, SEEK_END);
	long aLength = ftell(aFileHandle);
	fseek(aFileHandle, 0, SEEK_SET);
	if (aLength <= 0)
	{
		fclose(aFileHandle);
		return false;
	}

#if !defined(_WIN32)
	if (void* aMapping = mmap(nullptr, static_cast<size_t>(aLength), PROT_READ, MAP_PRIVATE, fileno(aFileHandle), 0);
		aMapping != MAP_FAILED)
	{
		fclose(aFileHandle);
		*theData = aMapping;
		*theSize = static_cast<size_t>(aLength);
		*isMapped = true;
		return true;
	}
#endif

	void* aCopy = malloc(static_cast<size_t>(aLength));
	bool aRead = aCopy != nullptr && fread(aCopy, 1, static_cast<size_t>(aLength), aFileHandle) == static_cast<size_t>(aLength);
	fclose(aFileHandle);
	if (!aRead)
	{
		free(aCopy);
		return false;
	}

	*theData = aCopy;
	*theSize = static_cast<size_t>(aLength);
	*isMapped = false;
	return true;
}

PakCollection::~PakCollection()
{
#if !defined(_WIN32)
	if (mMapped)
	{
		munmap(mDataPtr, mSize);
		return;
	}
#endif
	free(mDataPtr);
}

size_t PakCollection::ReadAt(void* thePtr, int theOffset, int theSize)
{
	if (theOffset < 0 || theSize <= 0 || static_cast<size_t>(theOffset) >= mSize)
		return 0;

	size_t aSize = std::min(static_cast<size_t>(theSize), mSize - static_cast<size_t>(theOffset));
	memcpy(thePtr, static_cast<uint8_t*>(mDataPtr) + theOffset, aSize);
	for (auto& aByte : std::span{static_cast<uint8_t*>(thePtr), aSize})
		aByte ^= 0xF7;
	return aSize;
}
#endif

PakInterface::~PakInterface()
{
}

// Normalize path for pak lookup.
std::string PakInterface::NormalizePakPath(std::string_view theFileName)
{
	std::filesystem::path aFilePath = Sexy::PathFromU8(theFileName);

	// Make rooted paths relative to resource folder.
	if (Sexy::IsPathRooted(theFileName))
	{
		const std::string& aResourceFolder = Sexy::GetResourceFolder();
		if (!aResourceFolder.empty())
		{
			std::filesystem::path aResPath = Sexy::PathFromU8(aResourceFolder);
			auto [aResEnd, aFileIt] = std::mismatch(aResPath.begin(), aResPath.end(),
												   aFilePath.begin(), aFilePath.end());
			if (aResEnd == aResPath.end())
			{
				std::filesystem::path aRelativePath;
				for (; aFileIt != aFilePath.end(); ++aFileIt)
					aRelativePath /= *aFileIt;
				aFilePath = aRelativePath;
			}
		}
	}

	std::string aResult = Sexy::PathToU8(aFilePath.lexically_normal());

	if (aResult.size() >= 2 && aResult[0] == '.' && aResult[1] == '/')
		aResult = aResult.substr(2);

	std::transform(aResult.begin(), aResult.end(), aResult.begin(),
				   [](unsigned char c) { return std::toupper(c); });

	return aResult;
}

bool PakInterface::AddPakFile(const std::string& theFileName)
{
#ifdef LOW_MEMORY
	FILE *aFileHandle = fcaseopen(theFileName.c_str(), "rb");
	if (!aFileHandle)
		return false;

	fseek(aFileHandle, 0, SEEK_END);
	size_t aFileSize = ftell(aFileHandle);
	fseek(aFileHandle, 0, SEEK_SET);

	mPakCollectionList.emplace_back(aFileHandle);
#else
	void* aData = nullptr;
	size_t aFileSize = 0;
	bool aMapped = false;
	if (!OpenPakStorage(theFileName, &aData, &aFileSize, &aMapped))
		return false;

	mPakCollectionList.emplace_back(aData, aFileSize, aMapped);
#endif
	PakCollection* aPakCollection = &mPakCollectionList.back();

	std::string aPakKey = NormalizePakPath(theFileName);
	auto aEmplaceResult = mPakRecordMap.emplace(aPakKey, PakRecord());
	PakRecord* aPakRecord = &aEmplaceResult.first->second;
	aPakRecord->mCollection = aPakCollection;
	aPakRecord->mFileName = aPakKey;
	aPakRecord->mStartPos = 0;
	aPakRecord->mSize = aFileSize;

	PFILE* aFP = FOpen(theFileName.c_str(), "rb");
	if (aFP == nullptr)
	{
#ifdef LOW_MEMORY
		if (aEmplaceResult.second)
		{
			mPakRecordMap.erase(aPakKey);
			mPakCollectionList.pop_back();
		}
#endif
		return false;
	}

	uint32_t aMagic = 0;
	FRead(&aMagic, sizeof(uint32_t), 1, aFP);
	aMagic = Sexy::FromLE32(aMagic);
	if (aMagic != 0xBAC04AC0)
	{
		FClose(aFP);
#ifdef LOW_MEMORY
		if (aEmplaceResult.second)
		{
			mPakRecordMap.erase(aPakKey);
			mPakCollectionList.pop_back();
		}
#endif
		return false;
	}

	uint32_t aVersion = 0;
	FRead(&aVersion, sizeof(uint32_t), 1, aFP);
	aVersion = Sexy::FromLE32(aVersion);
	if (aVersion > 0)
	{
		FClose(aFP);
#ifdef LOW_MEMORY
		if (aEmplaceResult.second)
		{
			mPakRecordMap.erase(aPakKey);
			mPakCollectionList.pop_back();
		}
#endif
		return false;
	}

	int aPos = 0;
	for (;;)
	{
		uchar aFlags = 0;
		int aCount = FRead(&aFlags, 1, 1, aFP);
		if ((aFlags & FILEFLAGS_END) || (aCount == 0))
			break;

		uchar aNameWidth = 0;
		char aName[256];
		FRead(&aNameWidth, 1, 1, aFP);
		FRead(aName, 1, aNameWidth, aFP);
		aName[aNameWidth] = 0;

		int aSrcSize = 0;
		FRead(&aSrcSize, sizeof(int), 1, aFP);
		aSrcSize = static_cast<int>(Sexy::FromLE32(static_cast<uint32_t>(aSrcSize)));
		int64_t aFileTime;
		FRead(&aFileTime, sizeof(int64_t), 1, aFP);
		aFileTime = static_cast<int64_t>(Sexy::FromLE64(static_cast<uint64_t>(aFileTime)));

		for (int i = 0; i < aNameWidth; i++)
		{
			if (aName[i] == '\\')
				aName[i] = '/';
		}

		std::string aKey = NormalizePakPath(aName);
		auto aRecordItr = mPakRecordMap.emplace(aKey, PakRecord()).first;
		PakRecord* aPakRecord = &aRecordItr->second;
		aPakRecord->mCollection = aPakCollection;
		aPakRecord->mFileName = aKey;
		aPakRecord->mStartPos = aPos;
		aPakRecord->mSize = aSrcSize;
		aPakRecord->mFileTime = aFileTime;

		aPos += aSrcSize;
	}

	int anOffset = FTell(aFP);

	for (auto& [key, record] : mPakRecordMap)
	{
		if (record.mCollection == aPakCollection)
			record.mStartPos += anOffset;
	}

	FClose(aFP);
	return true;
}

PFILE* PakInterface::FOpen(const char* theFileName, const char* anAccess)
{
	if ((strcasecmp(anAccess, "r") == 0) || (strcasecmp(anAccess, "rb") == 0) || (strcasecmp(anAccess, "rt") == 0))
	{
		std::string aKey = NormalizePakPath(theFileName);
		auto anItr = mPakRecordMap.find(aKey);
		if (anItr != mPakRecordMap.end())
		{
			PFILE* aPFP = new PFILE;
			aPFP->mRecord = &anItr->second;
			aPFP->mPos = 0;
			aPFP->mFP = nullptr;
#ifdef LOW_MEMORY
			aPFP->mBufferStart = 0;
			aPFP->mBufferLen = 0;
#endif
			return aPFP;
		}
	}

	const std::string& aResourceBase = Sexy::GetResourceFolder();
	FILE* aFP = nullptr;
	if (!aResourceBase.empty() && !Sexy::IsPathRooted(theFileName))
	{
		aFP = fcaseopenat(aResourceBase.c_str(), theFileName, anAccess);
	}
	else
	{
		aFP = fcaseopen(theFileName, anAccess);
	}

	if (aFP == nullptr)
		return nullptr;

	PFILE* aPFP = new PFILE;
	aPFP->mRecord = nullptr;
	aPFP->mPos = 0;
	aPFP->mFP = aFP;
#ifdef LOW_MEMORY
	aPFP->mBufferStart = 0;
	aPFP->mBufferLen = 0;
#endif
	return aPFP;
}

int PakInterface::FClose(PFILE* theFile)
{
	if (theFile->mRecord == nullptr)
		fclose(theFile->mFP);
	delete theFile;
	return 0;
}

int PakInterface::FSeek(PFILE* theFile, long theOffset, int theOrigin)
{
	if (theFile->mRecord != nullptr)
	{
		if (theOrigin == SEEK_SET)
			theFile->mPos = theOffset;
		else if (theOrigin == SEEK_END)
			theFile->mPos = theFile->mRecord->mSize - theOffset;
		else if (theOrigin == SEEK_CUR)
			theFile->mPos += theOffset;

		theFile->mPos = std::clamp(theFile->mPos, 0, theFile->mRecord->mSize);
#ifdef LOW_MEMORY
		theFile->mBufferLen = 0;
#endif
		return 0;
	}
	else
		return fseek(theFile->mFP, theOffset, theOrigin);
}

int PakInterface::FTell(PFILE* theFile)
{
	if (theFile->mRecord != nullptr)
		return theFile->mPos;
	else
		return ftell(theFile->mFP);
}

static size_t ReadRecordBytes(PFILE* theFile, void* thePtr, int theSize)
{
#ifdef LOW_MEMORY
	if (theSize >= PFILE::BUFFER_SIZE)
	{
		int aSizeBytes = std::min(theSize, theFile->mRecord->mSize - theFile->mPos);
		size_t aRead = theFile->mRecord->mCollection->ReadAt(thePtr, theFile->mRecord->mStartPos + theFile->mPos, aSizeBytes);
		theFile->mPos += (int) aRead;
		theFile->mBufferLen = 0;
		return aRead;
	}

	auto* aDest = static_cast<uint8_t*>(thePtr);
	size_t aDone = 0;
	while (aDone < static_cast<size_t>(theSize))
	{
		int aBufOffset = theFile->mPos - theFile->mBufferStart;
		if (aBufOffset < 0 || aBufOffset >= theFile->mBufferLen)
		{
			int aFill = std::min(theFile->mRecord->mSize - theFile->mPos, PFILE::BUFFER_SIZE);
			if (aFill <= 0)
				break;
			theFile->mBufferStart = theFile->mPos;
			theFile->mBufferLen = (int) theFile->mRecord->mCollection->ReadAt(theFile->mBuffer.data(), theFile->mRecord->mStartPos + theFile->mPos, aFill);
			if (theFile->mBufferLen == 0)
				break;
			aBufOffset = 0;
		}
		int aChunk = std::min(theFile->mBufferLen - aBufOffset, (int) (theSize - aDone));
		memcpy(aDest + aDone, theFile->mBuffer.data() + aBufOffset, aChunk);
		theFile->mPos += aChunk;
		aDone += aChunk;
	}
	return aDone;
#else
	int aSizeBytes = std::min(theSize, theFile->mRecord->mSize - theFile->mPos);

	size_t aRead = theFile->mRecord->mCollection->ReadAt(thePtr, theFile->mRecord->mStartPos + theFile->mPos, aSizeBytes);
	theFile->mPos += static_cast<int>(aRead);
	return aRead;
#endif
}

size_t PakInterface::FRead(void* thePtr, int theElemSize, int theCount, PFILE* theFile)
{
	if (theFile->mRecord != nullptr)
		return ReadRecordBytes(theFile, thePtr, theElemSize*theCount) / theElemSize;

	return fread(thePtr, theElemSize, theCount, theFile->mFP);
}

int PakInterface::FGetC(PFILE* theFile)
{
	if (theFile->mRecord != nullptr)
	{
		for (;;)
		{
			if (theFile->mPos >= theFile->mRecord->mSize)
				return EOF;
			char aChar;
			if (ReadRecordBytes(theFile, &aChar, 1) == 0)
				return EOF;
			if (aChar != '\r')
				return (uchar) aChar;
		}
	}

	return fgetc(theFile->mFP);
}

int PakInterface::UnGetC(int theChar, PFILE* theFile)
{
	if (theFile->mRecord != nullptr)
	{
		// This won't work if we're not pushing the same chars back in the stream
		theFile->mPos = std::max(theFile->mPos - 1, 0);
		return theChar;
	}

	return ungetc(theChar, theFile->mFP);
}

char* PakInterface::FGetS(char* thePtr, int theSize, PFILE* theFile)
{
	if (theFile->mRecord != nullptr)
	{
		int anIdx = 0;
		while (anIdx < theSize - 1) // fgets semantics: at most theSize-1 chars plus the NUL
		{
			if (theFile->mPos >= theFile->mRecord->mSize)
			{
				if (anIdx == 0)
					return nullptr;
				break;
			}
			char aChar;
			if (ReadRecordBytes(theFile, &aChar, 1) == 0)
				break;
			if (aChar != '\r')
				thePtr[anIdx++] = aChar;
			if (aChar == '\n')
				break;
		}
		thePtr[anIdx] = 0;
		return thePtr;
	}

	return fgets(thePtr, theSize, theFile->mFP);
}

int PakInterface::FEof(PFILE* theFile)
{
	if (theFile->mRecord != nullptr)
		return theFile->mPos >= theFile->mRecord->mSize;
	else
		return feof(theFile->mFP);
}
