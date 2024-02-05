/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _ _                     */
/*                                                                 */
/* Copyright 2007 Adobe Systems Incorporated                       */
/* All Rights Reserved.                                            */
/*                                                                 */
/* NOTICE:  All information contained herein is, and remains the   */
/* property of Adobe Systems Incorporated and its suppliers, if    */
/* any.  The intellectual and technical concepts contained         */
/* herein are proprietary to Adobe Systems Incorporated and its    */
/* suppliers and may be covered by U.S. and Foreign Patents,       */
/* patents in process, and are protected by trade secret or        */
/* copyright law.  Dissemination of this information or            */
/* reproduction of this material is strictly forbidden unless      */
/* prior written permission is obtained from Adobe Systems         */
/* Incorporated.                                                   */
/*                                                                 */
/*******************************************************************/

#pragma once
#ifndef NTFinisher_H
#define NTFinisher_H

typedef unsigned char		u_char;
typedef unsigned short		u_short;
typedef unsigned short		u_int16;
typedef unsigned long		u_long;
typedef short int			int16;
#define PF_TABLE_BITS	12
#define PF_TABLE_SZ_16	4096

#define PF_DEEP_COLOR_AWARE 1

#include "NTGlow_Kernel.cl.h"
#include "AEConfig.h"
#include "entry.h"
#include "AEFX_SuiteHelper.h"
#include "PrSDKAESupport.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_Macros.h"
#include "AEGP_SuiteHandler.h"
#include "String_Utils.h"
#include "Param_Utils.h"
#include "Smart_Utils.h"
#include "AEFX_SuiteHelper.h"
#include "AEFX_ChannelDepthTpl.h"
#include "AE_GeneralPlug.h"
#include "PrSDKAESupport.h"


#if _WIN32
#include <CL/cl.h>
#define HAS_METAL 0
#else
#include <OpenCL/cl.h>
#define HAS_METAL 1
#include <Metal/Metal.h>
#include "NTGlow_Kernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
	#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2018 Adobe Systems Incorporated.\rSample Invert ProcAmp effect."

#define NAME			"NTGlow"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


enum {
	NTFINISHER_INPUT = 0,
	NTFINISHER_GRAIN,
	NTFINISHER_GRADE,
	NTFINISHER_GLOW_TOPIC_START,
		NTFINISHER_GLOW_THRESHOLD,
		NTFINISHER_GLOW_RADIUS,
		NTFINISHER_GLOW_INTENSITY,
	NTFINISHER_GLOW_TOPIC_END,
	NTFINISHER_BLUR_TOPIC_START,
		NTFINISHER_BLUR_RADIUS,
		NTFINISHER_BLUR_INTENSITY,
	NTFINISHER_BLUR_TOPIC_END,
	NTFINISHER_SHAKE_TOPIC_START,
		NTFINISHER_SHAKE_MOTION_BLUR,
		NTFINISHER_SHAKE_XAMOUNT,
		NTFINISHER_SHAKE_XFREQ,
		NTFINISHER_SHAKE_YAMOUNT,
		NTFINISHER_SHAKE_YFREQ,
	NTFINISHER_SHAKE_TOPIC_END,
	NTFINISHER_NUM_PARAMS
};

enum {
	NTFINISHER_GRAIN_DISK_ID = 1,
	NTFINISHER_GRADE_DISK_ID,
	NTFINISHER_GLOW_TOPIC_START_DISK_ID,
		NTFINISHER_GLOW_THRESHOLD_DISK_ID,
		NTFINISHER_GLOW_RADIUS_DISK_ID,
		NTFINISHER_GLOW_INTENSITY_DISK_ID,
	NTFINISHER_GLOW_TOPIC_END_DISK_ID,
	NTFINISHER_BLUR_TOPIC_START_DISK_ID,
		NTFINISHER_BLUR_RADIUS_DISK_ID,
		NTFINISHER_BLUR_INTENSITY_DISK_ID,
	NTFINISHER_BLUR_TOPIC_END_DISK_ID,
	NTFINISHER_SHAKE_TOPIC_START_DISK_ID,
		NTFINISHER_SHAKE_MOTION_BLUR_DISK_ID,
		NTFINISHER_SHAKE_XAMOUNT_DISK_ID,
		NTFINISHER_SHAKE_XFREQ_DISK_ID,
		NTFINISHER_SHAKE_YAMOUNT_DISK_ID,
		NTFINISHER_SHAKE_YFREQ_DISK_ID,
	NTFINISHER_SHAKE_TOPIC_END_DISK_ID,
};


#define	BRIGHTNESS_MIN_VALUE		-100
#define	BRIGHTNESS_MAX_VALUE		100
#define	BRIGHTNESS_MIN_SLIDER		-100
#define	BRIGHTNESS_MAX_SLIDER		100
#define	BRIGHTNESS_DFLT				0

#define	CONTRAST_MIN_VALUE			0
#define	CONTRAST_MAX_VALUE			200
#define	CONTRAST_MIN_SLIDER			0
#define	CONTRAST_MAX_SLIDER			200
#define	CONTRAST_DFLT				100

#define	HUE_DFLT					0

#define	SATURATION_MIN_VALUE		0
#define	SATURATION_MAX_VALUE		200
#define	SATURATION_MIN_SLIDER		0
#define	SATURATION_MAX_SLIDER		200
#define	SATURATION_DFLT				100


extern "C" {

	DllExport 
	PF_Err
	EffectMain (	
		PF_Cmd			cmd,
		PF_InData		*in_data,
		PF_OutData		*out_data,
		PF_ParamDef		*params[],
		PF_LayerDef		*output,
		void			*extra);

}

#if HAS_METAL
	/*
	 ** Plugins must not rely on a host autorelease pool.
	 ** Create a pool if autorelease is used, or Cocoa convention calls, such as Metal, might internally autorelease.
	 */
	struct ScopedAutoreleasePool
	{
		ScopedAutoreleasePool()
		:  mPool([[NSAutoreleasePool alloc] init])
		{
		}
	
		~ScopedAutoreleasePool()
		{
			[mPool release];
		}
	
		NSAutoreleasePool *mPool;
	};
#endif 


typedef struct NTInfo {
	PF_FpLong	threshold,
		amount,
		radius;
} NTInfo, *NTInfoP, **NTInfoH;

typedef struct {
	A_u_char	blue, green, red, alpha;
} PF_Pixel_BGRA_8u;

typedef struct {
	A_u_char	Pr, Pb, luma, alpha;
} PF_Pixel_VUYA_8u;

typedef struct {
	PF_FpShort	blue, green, red, alpha;
} PF_Pixel_BGRA_32f;

typedef struct {
	PF_FpShort	Pr, Pb, luma, alpha;
} PF_Pixel_VUYA_32f;

#endif // NTGlow_H
