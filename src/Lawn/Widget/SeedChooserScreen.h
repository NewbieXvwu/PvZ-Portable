/*
 * Copyright (C) 2026 Zhou Qiankang <wszqkzqk@qq.com>
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
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

#ifndef __SEEDCHOOSERSCREEN_H__
#define __SEEDCHOOSERSCREEN_H__

#include "../../ConstEnums.h"
#include "../../PvzpLib/PvzpCommon.h"
#include "widget/Widget.h"
#include <functional>
#include <memory>
#include <vector>
using namespace Sexy;

class Board;
class LawnApp;
class GameButton;
class ToolTipWidget;
namespace Sexy
{
	class MTRand;
}

class ChosenSeed
{
public:
	int                     mX;
	int                     mY;
	int                     mTimeStartMotion;
	int                     mTimeEndMotion;
	int                     mStartX;
	int                     mStartY;
	int                     mEndX;
	int                     mEndY;
	SeedType                mSeedType;
	ChosenSeedState         mSeedState;
	int                     mSeedIndexInBank;
	bool                    mRefreshing;
	int                     mRefreshCounter;
	SeedType                mImitaterType;
	bool                    mCrazyDavePicked;
};

class SeedChooserScreen : public Widget
{
private:
	enum
	{
		SeedChooserScreen_Start = 100,
		SeedChooserScreen_Random = 101,
		SeedChooserScreen_ViewLawn = 102,
		SeedChooserScreen_Almanac = 103,
		SeedChooserScreen_Menu = 104,
		SeedChooserScreen_Store = 105,
		SeedChooserScreen_Imitater = 106
	};

public:
	std::unique_ptr<GameButton> mStartButton;
	std::unique_ptr<GameButton> mRandomButton;
	std::unique_ptr<GameButton> mViewLawnButton;
	std::unique_ptr<GameButton> mStoreButton;
	std::unique_ptr<GameButton> mAlmanacButton;
	std::unique_ptr<GameButton> mMenuButton;
	std::unique_ptr<GameButton> mImitaterButton;
	ChosenSeed              mChosenSeeds[NUM_SEED_TYPES];
	LawnApp*                mApp;
	Board*                  mBoard;
	int                     mNumSeedsToChoose;
	int                     mSeedChooserAge;
	int                     mSeedsInFlight;
	int                     mSeedsInBank;
	std::unique_ptr<ToolTipWidget> mToolTip;
	int                     mToolTipSeed;
	int                     mLastMouseX;
	int                     mLastMouseY;
	SeedChooserState        mChooseState;
	int                     mViewLawnTime;

public:
	// Whole-deck checks from OnStartButton, exposed without any dialog so the environment can ask
	// "would this deck warn?" for an arbitrary deck (task28).  `picked` answers "is this plant in
	// the deck"; the interactive path passes PickedPlantType, the environment passes a set lookup.
	struct DeckWarningSet
	{
		bool sun = false;          // SEED_CHOOSER_SUN_WARNING / _NIGHT_SUN_WARNING
		bool pool = false;         // SEED_CHOOSER_POOL_WARNING / _LILY_WARNING
		bool roof = false;         // SEED_CHOOSER_ROOF_WARNING
		bool flyer = false;        // SEED_CHOOSER_FLYER_WARNING
		bool upgrade = false;      // SEED_CHOOSER_UPGRADE_WARNING
		bool seeingStars = false;  // SEED_CHOOSER_SEEING_STARS_WARNING
		bool puffshroom = false;   // SEED_CHOOSER_PUFFSHROOM_WARNING
		bool artWallnut = false;   // SEED_CHOOSER_ART_WALLNUT_WARNING
		bool art2 = false;         // SEED_CHOOSER_ART_2_WARNING
	};
	static DeckWarningSet CheckDeckWarnings(LawnApp* theApp, Board* theBoard,
		const std::function<bool(SeedType)>& picked, bool survivalRepick);
	// CrazyDavePickSeeds' candidate list (weight 1), without picking anything.
	static std::vector<SeedType> CrazyDaveSeedPool(LawnApp* theApp, Board* theBoard);

	SeedChooserScreen();
	~SeedChooserScreen() override;

	static int   PickFromWeightedArrayUsingSpecialRandSeed(PvzpWeightedArray* theArray, int theCount, MTRand& theLevelRNG);
	void                    CrazyDavePickSeeds();
	bool                    Has7Rows();
	void                    GetSeedPositionInChooser(int theIndex, int& x, int& y);
	void         GetSeedPositionInBank(int theIndex, int& x, int& y);
	unsigned int SeedNotRecommendedToPick(SeedType theSeedType);
	bool         SeedNotAllowedToPick(SeedType theSeedType);
	bool         SeedNotAllowedDuringTrial(SeedType theSeedType);
	void                    Draw(Graphics* g) override;
	void                    UpdateViewLawn();
	void                    LandFlyingSeed(ChosenSeed& theChosenSeed);
	void                    UpdateCursor();
	void                    Update() override;
	bool         DisplayRepickWarningDialog(const char* theMessage);
	bool                    FlyersAreComming();
	bool                    FlyProtectionCurrentlyPlanted();
	bool                    CheckSeedUpgrade(SeedType theSeedTypeTo, SeedType theSeedTypeFrom);
	void                    OnStartButton();
	void                    PickRandomSeeds();
	virtual void            ButtonDepress(int theId);
	SeedType                SeedHitTest(int x, int y);
	SeedType                FindSeedInBank(int theIndexInBank);
	void         EnableStartButton(bool theEnabled);
	void                    ClickedSeedInBank(ChosenSeed& theChosenSeed);
	void                    ClickedSeedInChooser(ChosenSeed& theChosenSeed);
	void                    ShowToolTip();
	void         RemoveToolTip();
	void         CancelLawnView();
	void                    MouseUp(int x, int y, int theClickCount) override;
	void                    UpdateImitaterButton();
	void                    MouseDown(int x, int y, int theClickCount) override;
	bool         PickedPlantType(SeedType theSeedType);
	void                    CloseSeedChooser();
	void                    KeyDown(KeyCode theKey) override;
	void                    KeyChar(char theChar) override;
	void                    UpdateAfterPurchase();
};

#endif
