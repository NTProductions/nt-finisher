/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _ _                     */
/*                                                                 */
/* Copyright 2007-2018 Adobe Systems Incorporated                  */
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

/*
	NTGlow.cpp
	
	A simple Invert ProcAmp effect. This effect adds color invert and ProcAmp to the layer.
	
	Revision History
		
	Version		Change													Engineer	Date
	=======		======													========	======
	1.0			created													ykuang		09/10/2018

*/



#if HAS_CUDA
	#include <cuda_runtime.h>
	// NTGlow.h defines these and are needed whereas the cuda_runtime ones are not.
	#undef MAJOR_VERSION
	#undef MINOR_VERSION
#endif

#include "NTFinisher.h"
#include <iostream>

// brings in M_PI on Windows
#define _USE_MATH_DEFINES
#include <math.h>

typedef struct
{
	int mSrcPitch;
	int mDstPitch;
	int m16f;
	int mWidth;
	int mHeight;
	float threshold;
	float amount;
	float radius;
} NTParams;

inline PF_Err CL2Err(cl_int cl_result) {
	if (cl_result == CL_SUCCESS) {
		return PF_Err_NONE;
	} else {
		// set a breakpoint here to pick up OpenCL errors.
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
}

#define CL_ERR(FUNC) ERR(CL2Err(FUNC))


//  CUDA kernel; see NTGlow.cu.


extern void Exposure_CUDA(
	float const *src,
	float *dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float threshold,
	float amount,
	float radius);


static PF_Err 
About (	
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_SPRINTF(	out_data->return_msg, 
				"%s, v%d.%d\r%s",
				NAME, 
				MAJOR_VERSION, 
				MINOR_VERSION, 
				DESCRIPTION);

	return PF_Err_NONE;
}

static PF_Err 
GlobalSetup (
	PF_InData		*in_dataP,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err	err				= PF_Err_NONE;

	out_data->my_version	= PF_VERSION(	MAJOR_VERSION, 
											MINOR_VERSION,
											BUG_VERSION, 
											STAGE_VERSION, 
											BUILD_VERSION);
	
	out_data->out_flags		=	PF_OutFlag_PIX_INDEPENDENT	|
								PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_I_DO_DIALOG;

	out_data->out_flags2	=	PF_OutFlag2_FLOAT_COLOR_AWARE	|
								PF_OutFlag2_SUPPORTS_SMART_RENDER	|
								PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

	// For Premiere - declare supported pixel formats
	if (in_dataP->appl_id == 'PrMr'){


		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite = AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_dataP, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);

		// add supported pixel formats
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_dataP->effect_ref);

		(*pixelFormatSuite->AddSupportedPixelFormat) (in_dataP->effect_ref, PrPixelFormat_VUYA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat) (in_dataP->effect_ref, PrPixelFormat_BGRA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat) (in_dataP->effect_ref, PrPixelFormat_VUYA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat) (in_dataP->effect_ref, PrPixelFormat_BGRA_4444_8u);

	} else {
		out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}

	return err;
}

inline bool checkCross(A_long x, A_long y, A_long width, A_long height)
{
	float ypr = y / (float)height;
	A_long w = 1 + width / 250;
	return (abs(x - (A_long)(ypr * width)) < w || abs(x - (width - (A_long)(ypr * width))) < w);
}

static PF_Err 
ParamsSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_Err			err = PF_Err_NONE;
	PF_ParamDef		def;
	
	AEFX_CLR_STRUCT(def);
	
	AEFX_CLR_STRUCT(def);

	PF_ADD_FLOAT_SLIDERX("Grain",
		0.0,
		5.0,
		0.0,
		5.0,
		0.0,
		PF_Precision_TENTHS,
		0,
		0,
		NTFINISHER_GRAIN_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_POPUP("Grade", 3, 1, "Grade1|Grade2|Grade3", NTFINISHER_GRADE_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_TOPIC("Glow", NTFINISHER_GLOW_TOPIC_START_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_FLOAT_SLIDERX("Threshold",
		0.000,
		1.000,
		0.000,
		1.000,
		0.500,
		PF_Precision_THOUSANDTHS,
		0,
		0,
		NTFINISHER_GLOW_THRESHOLD_DISK_ID);

	PF_ADD_FLOAT_SLIDERX("Radius",
		0,
		1000,
		0,
		1000,
		50,
		PF_Precision_INTEGER,
		0,
		0,
		NTFINISHER_GLOW_RADIUS_DISK_ID);

	PF_ADD_FLOAT_SLIDERX("Intensity",
		0.0,
		100.0,
		0.0,
		100.0,
		0.0,
		PF_Precision_TENTHS,
		0,
		0,
		NTFINISHER_GLOW_INTENSITY_DISK_ID);

	PF_END_TOPIC(NTFINISHER_GLOW_TOPIC_END_DISK_ID);


	AEFX_CLR_STRUCT(def);

	PF_ADD_TOPIC("Glow", NTFINISHER_BLUR_TOPIC_START_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_FLOAT_SLIDERX("Radius",
		0,
		1000,
		0,
		1000,
		50,
		PF_Precision_INTEGER,
		0,
		0,
		NTFINISHER_BLUR_RADIUS_DISK_ID);

	PF_ADD_FLOAT_SLIDERX("Intensity",
		0.0,
		100.0,
		0.0,
		100.0,
		0.0,
		PF_Precision_TENTHS,
		0,
		0,
		NTFINISHER_BLUR_INTENSITY_DISK_ID);

	PF_END_TOPIC(NTFINISHER_BLUR_TOPIC_END_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_TOPIC("Shake", NTFINISHER_SHAKE_TOPIC_START_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_CHECKBOX("Motion Blur", "", TRUE, NULL, NTFINISHER_SHAKE_MOTION_BLUR_DISK_ID);

	PF_ADD_FLOAT_SLIDERX("xAmount (pxls)",
		0,
		500,
		0,
		500,
		0,
		PF_Precision_INTEGER,
		0,
		0,
		NTFINISHER_SHAKE_XAMOUNT_DISK_ID);

	PF_ADD_FLOAT_SLIDERX("xFrequency (x/sec)",
		0.0,
		20.0,
		0.0,
		20.0,
		0.0,
		PF_Precision_TENTHS,
		0,
		0,
		NTFINISHER_SHAKE_XFREQ_DISK_ID);

	PF_ADD_FLOAT_SLIDERX("yAmount (pxls)",
		0,
		500,
		0,
		500,
		0,
		PF_Precision_INTEGER,
		0,
		0,
		NTFINISHER_SHAKE_YAMOUNT_DISK_ID);

	PF_ADD_FLOAT_SLIDERX("yFrequency (y/sec)",
		0.0,
		20.0,
		0.0,
		20.0,
		0.0,
		PF_Precision_TENTHS,
		0,
		0,
		NTFINISHER_SHAKE_YFREQ_DISK_ID);

	PF_END_TOPIC(NTFINISHER_SHAKE_TOPIC_END_DISK_ID);

	out_data->num_params = NTFINISHER_NUM_PARAMS;

	return err;
}

static PF_FpLong
get32Luma(PF_Pixel32 thisPixel) {
	return (thisPixel.red + thisPixel.green + thisPixel.blue) * .333;
}

static PF_FpLong
get16Luma(PF_Pixel16 thisPixel) {
	return (thisPixel.red + thisPixel.green + thisPixel.blue) * .333 * .0000305;
}

static PF_FpLong
get8Luma(PF_Pixel8 thisPixel) {
	return (thisPixel.red + thisPixel.green + thisPixel.blue) * .333 * .003921568627451;
}

PF_FpLong clamp32(PF_FpLong data) {
	if (data < 0.0) {
		data = 0.0;
	}

	if (data > 1.0) {
		data = 1.0;
	}
	return data;
}

PF_FpLong clamp16(PF_FpLong data) {
	if (data < 0.0) {
		data = 0.0;
	}

	if (data > 32768) {
		data = 32768;
	}
	return data;
}

PF_FpLong clamp8(PF_FpLong data) {
	if (data < 0.0) {
		data = 0.0;
	}

	if (data > 255) {
		data = 255;
	}
	return data;
}

PF_Pixel32 changePropsToVUYA32f(PF_Pixel32 inputPixel) {
	PF_Pixel32 returnPixel;
	PF_Pixel_VUYA_32f convertedPixel;

	convertedPixel.luma = inputPixel.green;
	convertedPixel.Pb = inputPixel.red;
	convertedPixel.Pr = inputPixel.alpha;
	convertedPixel.alpha = inputPixel.blue;

	PF_FpLong R = convertedPixel.luma + 1.403 * convertedPixel.Pr;
	PF_FpLong G = convertedPixel.luma - 0.344 * convertedPixel.Pb - 0.714 * convertedPixel.Pr;
	PF_FpLong B = convertedPixel.luma + 1.77 * convertedPixel.Pb;

	if (R < 0) {
		R = 0;
	}
	if (R > 1.0) {
		R = 1.0;
	}

	if (G < 0) {
		G = 0;
	}
	if (G > 1.0) {
		G = 1.0;
	}

	if (B < 0) {
		B = 0;
	}
	if (B > 1.0) {
		B = 1.0;
	}

	returnPixel.red = R;
	returnPixel.green = G;
	returnPixel.blue = B;
	returnPixel.alpha = 1.0;

	return returnPixel;
}

static PF_Err
NTGlowFunc32(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel32* inP,
	PF_Pixel32* outP)
{
	PF_Err		err = PF_Err_NONE;

	NTParams* giP = reinterpret_cast<NTParams*>(refcon);
	PF_FpLong	tempF = 0;
	PF_FpLong	luma = get32Luma(*inP);
	//PF_FpLong	moonlight = giP->moonlight;
	PF_FpLong	exposure = .1;
	PF_FpLong	multiplier = .8;
	double r, g, b, a;
	if (giP) {
		outP = inP;
	}

	return err;
}



static PF_Err
NTGlowFunc16(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel16* inP,
	PF_Pixel16* outP)
{
	PF_Err		err = PF_Err_NONE;

	NTParams* giP = reinterpret_cast<NTParams*>(refcon);
	PF_FpLong	tempF = 0;
	PF_FpLong	luma = get16Luma(*inP);
	//PF_FpLong	moonlight = giP->moonlight;
	PF_FpLong	exposure = .1;
	PF_FpLong	multiplier = .8;
	double r, g, b, a;
	if (giP) {
		outP = inP;
	}

	return err;
}

static PF_Err
NTGlowFunc8(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel8* inP,
	PF_Pixel8* outP)
{
	PF_Err		err = PF_Err_NONE;

	NTParams* giP = reinterpret_cast<NTParams*>(refcon);
	PF_FpLong	tempF = 0;
	PF_FpLong	luma = get8Luma(*inP);
	//PF_FpLong	moonlight = giP->moonlight;
	PF_FpLong	exposure = .1;
	PF_FpLong	multiplier = .8;
	double r, g, b, a;
	if (giP) {
		outP = inP;
	}

	return err;
}

static PF_Err
NTGlowFuncBGRA8(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel8* inP,
	PF_Pixel8* outP)
{
	PF_Err		err = PF_Err_NONE;

	NTInfo* giP = reinterpret_cast<NTInfo*>(refcon);
	PF_FpLong	tempF = 0;
	PF_FpLong	luma = (inP->red + inP->green + inP->blue) * .3333 * .0039;
	//PF_FpLong	moonlight = giP->moonlight;
	PF_FpLong	exposure = .1;
	PF_FpLong	multiplier = .8;
	double r, g, b, a;
	if (giP) {
		outP = inP;
	}

	return err;
}

static PF_Err
NTGlowFuncBGRA32(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel32* inP,
	PF_Pixel32* outP)
{
	PF_Err		err = PF_Err_NONE;

	NTInfo* giP = reinterpret_cast<NTInfo*>(refcon);
	PF_FpLong	tempF = 0;
	PF_FpLong	luma = (inP->red + inP->green + inP->blue) * .3333;
	//PF_FpLong	moonlight = giP->moonlight;
	PF_FpLong	exposure = .1;
	PF_FpLong	multiplier = .8;
	double r, g, b, a;
	if (giP) {
		outP = inP;
	}

	return err;
}

static PF_Err
NTGlowFuncVUYA8(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel8* inP,
	PF_Pixel8* outP)
{
	PF_Err		err = PF_Err_NONE;

	NTInfo* giP = reinterpret_cast<NTInfo*>(refcon);
	PF_FpLong	tempF = 0;
	PF_FpLong	luma = get8Luma(*inP);
	//PF_FpLong	moonlight = giP->moonlight;
	PF_FpLong	exposure = .1;
	PF_FpLong	multiplier = .8;
	double r, g, b, a;
	if (giP) {
		outP = inP;
	}

	return err;
}

static PF_Err
NTGlowFuncVUYA32(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel32* inP,
	PF_Pixel32* outP)
{
	PF_Err		err = PF_Err_NONE;

	NTInfo* giP = reinterpret_cast<NTInfo*>(refcon);

	PF_Pixel_VUYA_32f  inVUYA_32fP, * outVUYA_32fP, * rgbToVUYA;
	inVUYA_32fP.luma = inP->green;
	inVUYA_32fP.Pb = inP->red;
	inVUYA_32fP.Pr = inP->alpha;
	inVUYA_32fP.alpha = inP->blue;

	PF_Pixel_VUYA_32f colourOneConverted, colourTwoConverted, colourThreeConverted, colourFourConverted, colourFiveConverted, outputPixel;

	PF_Pixel32  convertedPixel = changePropsToVUYA32f(*inP);


	PF_FpLong	tempF = 0;
	PF_FpLong	luma = (convertedPixel.red + convertedPixel.green + convertedPixel.blue) * .3333;
	//PF_FpLong	moonlight = giP->moonlight;
	PF_FpLong	exposure = .1;
	PF_FpLong	multiplier = .8;
	double r, g, b, a;
	if (giP) {
		outP = inP;
	}

	return err;
}

#if HAS_METAL
	PF_Err NSError2PFErr(NSError *inError)
	{
		if (inError)
		{
			return PF_Err_INTERNAL_STRUCT_DAMAGED;  //For debugging, uncomment above line and set breakpoint here
		}
		return PF_Err_NONE;
	}
#endif //HAS_METAL


// GPU data initialized at GPU setup and used during render.
struct OpenCLGPUData
{
	cl_kernel invert_kernel;
	cl_kernel procamp_kernel;
};

#if HAS_METAL
	struct MetalGPUData
	{
		id<MTLComputePipelineState>invert_pipeline;
		id<MTLComputePipelineState>procamp_pipeline;
	};
#endif


static PF_Err
GPUDeviceSetup(
	PF_InData	*in_dataP,
	PF_OutData	*out_dataP,
	PF_GPUDeviceSetupExtra *extraP)
{
	PF_Err err = PF_Err_NONE;

	PF_GPUDeviceInfo device_info;
	AEFX_CLR_STRUCT(device_info);

	AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(	in_dataP,
																					   kPFHandleSuite,
																					   kPFHandleSuiteVersion1,
																					   out_dataP);

	AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpuDeviceSuite =
	AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP,
										 kPFGPUDeviceSuite,
										 kPFGPUDeviceSuiteVersion1,
										 out_dataP);

	gpuDeviceSuite->GetDeviceInfo(in_dataP->effect_ref,
								  extraP->input->device_index,
								  &device_info);

	// Load and compile the kernel - a real plugin would cache binaries to disk

	if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
		// Nothing to do here. CUDA Kernel statically linked
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	} else if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {

		PF_Handle gpu_dataH	= handle_suite->host_new_handle(sizeof(OpenCLGPUData));
		OpenCLGPUData *cl_gpu_data = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);

		cl_int result = CL_SUCCESS;

		char const *k16fString = "#define GF_OPENCL_SUPPORTS_16F 0\n";

		size_t sizes[] = { strlen(k16fString), strlen(kNTGlow_Kernel_OpenCLString) };
		char const *strings[] = { k16fString, kNTGlow_Kernel_OpenCLString };
		cl_context context = (cl_context)device_info.contextPV;
		cl_device_id device = (cl_device_id)device_info.devicePV;

		cl_program program;
		if(!err) {
			program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
			CL_ERR(result);
		}

		CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

		if (!err) {
			cl_gpu_data->procamp_kernel = clCreateKernel(program, "ProcAmp2Kernel", &result);
			CL_ERR(result);
		}

		extraP->output->gpu_data = gpu_dataH;

		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		//Create a library from source
		NSString *source = [NSString stringWithCString:kNTGlow_Kernel_MetalString encoding:NSUTF8StringEncoding];
		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;

		NSError *error = nil;
		id<MTLLibrary> library = [[device newLibraryWithSource:source options:nil error:&error] autorelease];

		// An error code is set for Metal compile warnings, so use nil library as the error signal
		if(!err && !library) {
			err = NSError2PFErr(error);
		}

		// For debugging only. This will contain Metal compile warnings and erorrs.
		NSString *getError = error.localizedDescription;

		PF_Handle metal_handle = handle_suite->host_new_handle(sizeof(MetalGPUData));
		MetalGPUData *metal_data = reinterpret_cast<MetalGPUData *>(*metal_handle);

		//Create pipeline state from function extracted from library
		if (err == PF_Err_NONE)
		{
			id<MTLFunction> procamp_function = nil;
			NSString *procamp_name = [NSString stringWithCString:"ProcAmp2Kernel" encoding:NSUTF8StringEncoding];

			procamp_function = [ [library newFunctionWithName:procamp_name] autorelease];

			if (!err) {
				metal_data->procamp_pipeline = [device newComputePipelineStateWithFunction:procamp_function error:&error];
				err = NSError2PFErr(error);
			}

			if(!err) {
				extraP->output->gpu_data = metal_handle;
				out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
			}
		}
	}
#endif
	return err;
}


static PF_Err
GPUDeviceSetdown(
	PF_InData	*in_dataP,
	PF_OutData	*out_dataP,
	PF_GPUDeviceSetdownExtra *extraP)
{
	PF_Err err = PF_Err_NONE;

	if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData *cl_gpu_dataP = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);

		(void)clReleaseKernel (cl_gpu_dataP->invert_kernel);
		(void)clReleaseKernel (cl_gpu_dataP->procamp_kernel);
		
		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(	in_dataP,
																						   kPFHandleSuite,
																						   kPFHandleSuiteVersion1,
																						   out_dataP);
		
		handle_suite->host_dispose_handle(gpu_dataH);
	}
	
	return err;
}

static PF_Err
IterateFloat(
	PF_InData* in_data,
	long				progress_base,
	long				progress_final,
	PF_EffectWorld* src,
	void* refcon,
	PF_Err(*pix_fn)(void* refcon, A_long x, A_long y, PF_PixelFloat* in, PF_PixelFloat* out),
	PF_EffectWorld* dst)
{
	PF_Err	err = PF_Err_NONE;
	char* localSrc, * localDst;
	localSrc = reinterpret_cast<char*>(src->data);
	localDst = reinterpret_cast<char*>(dst->data);

	for (int y = progress_base; y < progress_final; y++)
	{
		for (int x = 0; x < src->width; x++)
		{
			pix_fn(refcon,
				static_cast<A_long> (x),
				static_cast<A_long> (y),
				reinterpret_cast<PF_PixelFloat*>(localSrc),
				reinterpret_cast<PF_PixelFloat*>(localDst));
			localSrc += 16;
			localDst += 16;
		}
		localSrc += (src->rowbytes - src->width * 16);
		localDst += (dst->rowbytes - src->width * 16);
	}

	return err;
}

static PF_Err
Render ( 
	PF_InData		*in_dataP,
	PF_OutData		*out_dataP,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err				err = PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_dataP->pica_basicP);

	PF_WorldSuite2* wsP = NULL;

	ERR(AEFX_AcquireSuite(in_dataP,
		out_dataP,
		kPFWorldSuite,
		kPFWorldSuiteVersion2,
		"Couldn't load suite.",
		(void**)&wsP));

	/*	Put interesting code here. */
	NTInfo			biP;
	AEFX_CLR_STRUCT(biP);
	A_long				linesL = 0;

	linesL = output->extent_hint.bottom - output->extent_hint.top;

	//biP.threshold = params[NTFINISHER_THRESHOLD]->u.fs_d.value;
	//biP.amount = params[NTFINISHER_AMOUNT]->u.fs_d.value;
	//biP.radius = params[NTFINISHER_RADIUS]->u.fs_d.value;

	if (in_dataP->appl_id == 'PrMr')
	{
		

		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
			AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_dataP,
				kPFPixelFormatSuite,
				kPFPixelFormatSuiteVersion1,
				out_dataP);

		PrPixelFormat destinationPixelFormat = PrPixelFormat_BGRA_4444_8u;

		pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat);

		AEFX_SuiteScoper<PF_Iterate8Suite1> iterate8Suite =
			AEFX_SuiteScoper<PF_Iterate8Suite1>(in_dataP,
				kPFIterate8Suite,
				kPFIterate8SuiteVersion1,
				out_dataP);

		switch (destinationPixelFormat)
		{

		case PrPixelFormat_BGRA_4444_8u:
			iterate8Suite->iterate(in_dataP,
				0,								// progress base
				linesL,							// progress final
				&params[NTFINISHER_INPUT]->u.ld,		// src 
				NULL,							// area - null for all pixels
				(void*)&biP,					// refcon - your custom data pointer
				NTGlowFuncBGRA8,				// pixel function pointer
				output);

			break;
		case PrPixelFormat_VUYA_4444_8u:
			iterate8Suite->iterate(in_dataP,
				0,								// progress base
				linesL,							// progress final
				&params[NTFINISHER_INPUT]->u.ld,		// src 
				NULL,							// area - null for all pixels
				(void*)&biP,					// refcon - your custom data pointer
				NTGlowFuncVUYA8,				// pixel function pointer
				output);

			break;
		case PrPixelFormat_BGRA_4444_32f:
			// Premiere doesn't support IterateFloatSuite1, so we've rolled our own
			IterateFloat(in_dataP,
				0,								// progress base
				linesL,							// progress final
				&params[NTFINISHER_INPUT]->u.ld,		// src 
				(void*)&biP,					// refcon - your custom data pointer
				NTGlowFuncBGRA32,			// pixel function pointer
				output);

			break;
		case PrPixelFormat_VUYA_4444_32f:
			// Premiere doesn't support IterateFloatSuite1, so we've rolled our own
			/////////////////////////
			IterateFloat(in_dataP,
				0,								// progress base
				linesL,							// progress final
				&params[NTFINISHER_INPUT]->u.ld,		// src 
				(void*)&biP,					// refcon - your custom data pointer
				NTGlowFuncVUYA32,			// pixel function pointer
				output);
			break;
		default:
			//	Return error, because we don't know how to handle the specified pixel type
			return PF_Err_UNRECOGNIZED_PARAM_TYPE;
		}
	}

	return PF_Err_NONE;
}


static void
DisposePreRenderData(
	void *pre_render_dataPV)
{
	if(pre_render_dataPV) {
		NTInfo *infoP = reinterpret_cast<NTInfo*>(pre_render_dataPV);
		free(infoP);
	}
}


static PF_Err
PreRender(
	PF_InData			*in_dataP,
	PF_OutData			*out_dataP,
	PF_PreRenderExtra	*extraP)
{
	PF_Err err = PF_Err_NONE;
	PF_CheckoutResult in_result;
	PF_RenderRequest req = extraP->input->output_request;

	extraP->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;

	NTParams *infoP	= reinterpret_cast<NTParams*>( malloc(sizeof(NTParams)) );

	if (infoP) {

		// Querying parameters to demoonstrate they are available at PreRender, and data can be passed from PreRender to Render with pre_render_data.
		PF_ParamDef cur_param;
		ERR(PF_CHECKOUT_PARAM(in_dataP, NTFINISHER_GRAIN, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));

		infoP->threshold = cur_param.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, NTFINISHER_GRADE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));

		infoP->amount = cur_param.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, NTFINISHER_GLOW_THRESHOLD, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));

		ERR(PF_CHECKOUT_PARAM(in_dataP, NTFINISHER_GLOW_RADIUS, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));

		ERR(PF_CHECKOUT_PARAM(in_dataP, NTFINISHER_GLOW_INTENSITY, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));

		ERR(PF_CHECKOUT_PARAM(in_dataP, NTFINISHER_BLUR_RADIUS, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));

		ERR(PF_CHECKOUT_PARAM(in_dataP, NTFINISHER_BLUR_INTENSITY, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));

		ERR(PF_CHECKOUT_PARAM(in_dataP, NTFINISHER_SHAKE_MOTION_BLUR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));

		ERR(PF_CHECKOUT_PARAM(in_dataP, NTFINISHER_SHAKE_XAMOUNT, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));

		ERR(PF_CHECKOUT_PARAM(in_dataP, NTFINISHER_SHAKE_XFREQ, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));

		ERR(PF_CHECKOUT_PARAM(in_dataP, NTFINISHER_SHAKE_YAMOUNT, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));

		ERR(PF_CHECKOUT_PARAM(in_dataP, NTFINISHER_SHAKE_YFREQ, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));

		infoP->radius = cur_param.u.fs_d.value;

		extraP->output->pre_render_data = infoP;
		extraP->output->delete_pre_render_data_func = DisposePreRenderData;
		
		ERR(extraP->cb->checkout_layer(	in_dataP->effect_ref,
									   NTFINISHER_INPUT,
									   NTFINISHER_INPUT,
									   &req,
									   in_dataP->current_time,
									   in_dataP->time_step,
									   in_dataP->time_scale,
									   &in_result));
		
		UnionLRect(&in_result.result_rect, 		&extraP->output->result_rect);
		UnionLRect(&in_result.max_result_rect, 	&extraP->output->max_result_rect);
	} else {
		err = PF_Err_OUT_OF_MEMORY;
	}
	return err;
}


static PF_Err
SmartRenderCPU(
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_PixelFormat			pixel_format,
	PF_EffectWorld			*input_worldP,
	PF_EffectWorld			*output_worldP,
	PF_SmartRenderExtra		*extraP,
	NTParams* infoP)
{
	PF_Err			err		= PF_Err_NONE;


	infoP->mWidth = in_data->width;
	infoP->mHeight = in_data->height;
	if (!err){
		switch (pixel_format) {

			

			case PF_PixelFormat_ARGB128: {
				AEFX_SuiteScoper<PF_iterateFloatSuite1> iterateFloatSuite =
				AEFX_SuiteScoper<PF_iterateFloatSuite1>(in_data,
														kPFIterateFloatSuite,
														kPFIterateFloatSuiteVersion1,
														out_data);
				iterateFloatSuite->iterate(in_data,
										   0,
										   output_worldP->height,
										   input_worldP,
										   NULL,
										   (void*)infoP,
										   NTGlowFunc32,
										   output_worldP);
				break;
			}
				
			case PF_PixelFormat_ARGB64: {
				AEFX_SuiteScoper<PF_iterate16Suite1> iterate16Suite =
				AEFX_SuiteScoper<PF_iterate16Suite1>(in_data,
													 kPFIterate16Suite,
													 kPFIterate16SuiteVersion1,
													 out_data);
				iterate16Suite->iterate(in_data,
										0,
										output_worldP->height,
										input_worldP,
										NULL,
										(void*)infoP,
										NTGlowFunc16,
										output_worldP);
				break;
			}
				
			case PF_PixelFormat_ARGB32: {
				AEFX_SuiteScoper<PF_Iterate8Suite1> iterate8Suite =
				AEFX_SuiteScoper<PF_Iterate8Suite1>(in_data,
													kPFIterate8Suite,
													kPFIterate8SuiteVersion1,
													out_data);

				iterate8Suite->iterate(	in_data,
									   0,
									   output_worldP->height,
									   input_worldP,
									   NULL,
									   (void*)infoP,
										NTGlowFunc8,
									   output_worldP);
				break;
			}

			default:
				err = PF_Err_BAD_CALLBACK_PARAM;
				break;
		}
	}
	
	return err;
}


static size_t
RoundUp(
	size_t inValue,
	size_t inMultiple)
{
	return inValue ? ((inValue + inMultiple - 1) / inMultiple) * inMultiple : 0;
}






size_t DivideRoundUp(
					 size_t inValue,
					 size_t inMultiple)
{
	return inValue ? (inValue + inMultiple - 1) / inMultiple: 0;
}



static PF_Err
SmartRenderGPU(
	PF_InData				*in_dataP,
	PF_OutData				*out_dataP,
	PF_PixelFormat			pixel_format,
	PF_EffectWorld			*input_worldP,
	PF_EffectWorld			*output_worldP,
	PF_SmartRenderExtra		*extraP,
	NTParams *infoP)
{
	PF_Err			err		= PF_Err_NONE;

	AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite = AEFX_SuiteScoper<PF_GPUDeviceSuite1>( in_dataP,
																						  kPFGPUDeviceSuite,
																						  kPFGPUDeviceSuiteVersion1,
																						  out_dataP);

	if(pixel_format != PF_PixelFormat_GPU_BGRA128) {
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}
	A_long bytes_per_pixel = 16;

	PF_GPUDeviceInfo device_info;
	ERR(gpu_suite->GetDeviceInfo(in_dataP->effect_ref, extraP->input->device_index, &device_info));

	// Allocate an intermediate buffer for extra kernel steps
	// Here we use this buffer to invert color in CUDA
	PF_EffectWorld *intermediate_buffer;

	ERR(gpu_suite->CreateGPUWorld(in_dataP->effect_ref,
								  extraP->input->device_index,
								  input_worldP->width,
								  input_worldP->height,
								  input_worldP->pix_aspect_ratio,
								  in_dataP->field,
								  pixel_format,
								  false,
								  &intermediate_buffer));

	void *src_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, input_worldP, &src_mem));

	void *dst_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, output_worldP, &dst_mem));

	void *im_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, intermediate_buffer, &im_mem));

	// read the parameters
	NTParams ntParams;

	ntParams.mWidth = input_worldP->width;
	ntParams.mHeight = input_worldP->height;

	A_long src_row_bytes = input_worldP->rowbytes;
	A_long tmp_row_bytes = intermediate_buffer->rowbytes;
	A_long dst_row_bytes = output_worldP->rowbytes;

	ntParams.mSrcPitch = src_row_bytes / bytes_per_pixel;
	ntParams.mDstPitch = tmp_row_bytes / bytes_per_pixel;
	ntParams.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);
	ntParams.threshold = infoP->threshold;
	ntParams.amount = infoP->amount;
	ntParams.radius = infoP->radius;


	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData *cl_gpu_dataP = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);

		cl_mem cl_src_mem = (cl_mem)src_mem;
		cl_mem cl_im_mem = (cl_mem)im_mem;
		cl_mem cl_dst_mem = (cl_mem)dst_mem;

		cl_uint invert_param_index = 0;
		cl_uint procamp_param_index = 0;

		// Set the arguments
		/*CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(cl_mem), &cl_im_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &ntParams.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &ntParams.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &ntParams.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &ntParams.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->invert_kernel, invert_param_index++, sizeof(int), &ntParams.mHeight));*/

		// Launch the kernel
		size_t threadBlock[2] = { 16, 16 };
		size_t grid[2] = { RoundUp(ntParams.mWidth, threadBlock[0]), RoundUp(ntParams.mHeight, threadBlock[1])};

		/*CL_ERR(clEnqueueNDRangeKernel(
									  (cl_command_queue)device_info.command_queuePV,
									  cl_gpu_dataP->invert_kernel,
									  2,
									  0,
									  grid,
									  threadBlock,
									  0,
									  0,
									  0));*/

		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(cl_mem), &cl_im_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(cl_mem), &cl_dst_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(int), &ntParams.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(int), &ntParams.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(int), &ntParams.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(int), &ntParams.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(int), &ntParams.mHeight));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(float), &ntParams.threshold));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(float), &ntParams.amount));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->procamp_kernel, procamp_param_index++, sizeof(float), &ntParams.radius));
		
		CL_ERR(clEnqueueNDRangeKernel(
									(cl_command_queue)device_info.command_queuePV,
									cl_gpu_dataP->procamp_kernel,
									2,
									0,
									grid,
									threadBlock,
									0,
									0,
									0));
	}
	#if HAS_CUDA
		else if (!err && extraP->input->what_gpu == PF_GPU_Framework_CUDA) {

				Exposure_CUDA(
				(const float *)src_mem,
				(float *)dst_mem,
					ntParams.mSrcPitch,
					ntParams.mDstPitch,
					ntParams.m16f,
					ntParams.mWidth,
					ntParams.mHeight,
					ntParams.threshold,
					ntParams.amount,
					ntParams.radius);

			if (cudaPeekAtLastError() != cudaSuccess) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}

		}
	#endif
	#if HAS_METAL
		else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
		{
			ScopedAutoreleasePool pool;
			
			Handle metal_handle = (Handle)extraP->input->gpu_data;
			MetalGPUData *metal_dataP = reinterpret_cast<MetalGPUData *>(*metal_handle);


			//Set the arguments
			id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
			id<MTLBuffer> procamp_param_buffer = [[device newBufferWithBytes:&ntParams
																length:sizeof(ntParams)
																options:MTLResourceStorageModeManaged] autorelease];
			
			/*id<MTLBuffer> invert_param_buffer = [[device newBufferWithBytes:&invert_params
															    length:sizeof(InvertColorParams)
															    options:MTLResourceStorageModeManaged] autorelease];*/

			//Launch the command
			id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
			id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
			id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
			id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
			id<MTLBuffer> im_metal_buffer = (id<MTLBuffer>)im_mem;
			id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

			//MTLSize threadsPerGroup1 = {[metal_dataP->invert_pipeline threadExecutionWidth], 16, 1};
			//MTLSize numThreadgroups1 = {DivideRoundUp(ntParams.mWidth, threadsPerGroup1.width), DivideRoundUp(ntParams.mHeight, threadsPerGroup1.height), 1};
			
			MTLSize threadsPerGroup2 = {[metal_dataP->procamp_pipeline threadExecutionWidth], 16, 1};
			MTLSize numThreadgroups2 = {DivideRoundUp(ntParams.mWidth, threadsPerGroup2.width), DivideRoundUp(ntParams.mHeight, threadsPerGroup2.height), 1};

			//[computeEncoder setComputePipelineState:metal_dataP->invert_pipeline];
			//[computeEncoder setBuffer:src_metal_buffer offset:0 atIndex:0];
			//[computeEncoder setBuffer:im_metal_buffer offset:0 atIndex:1];
			//[computeEncoder setBuffer:invert_param_buffer offset:0 atIndex:2];
			//[computeEncoder dispatchThreadgroups:numThreadgroups1 threadsPerThreadgroup:threadsPerGroup1];

			err = NSError2PFErr([commandBuffer error]);

			if (!err) {
				[computeEncoder setComputePipelineState:metal_dataP->procamp_pipeline];
				[computeEncoder setBuffer:src_metal_buffer offset:0 atIndex:0];
				[computeEncoder setBuffer:dst_metal_buffer offset:0 atIndex:1];
				[computeEncoder setBuffer:procamp_param_buffer offset:0 atIndex:2];
				[computeEncoder dispatchThreadgroups:numThreadgroups2 threadsPerThreadgroup:threadsPerGroup2];
				[computeEncoder endEncoding];
				[commandBuffer commit];

				err = NSError2PFErr([commandBuffer error]);
			}

		}
	#endif //HAS_METAL

	// Must free up allocated intermediate buffer
	ERR(gpu_suite->DisposeGPUWorld(in_dataP->effect_ref, intermediate_buffer));
	return err;
}


static PF_Err
SmartRender(
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_SmartRenderExtra		*extraP,
	bool					isGPU)
{

	PF_Err			err		= PF_Err_NONE,
					err2 	= PF_Err_NONE;
	
	PF_EffectWorld	*input_worldP	= NULL, 
					*output_worldP  = NULL;

	// Parameters can be queried during render. In this example, we pass them from PreRender as an example of using pre_render_data.
	NTParams* infoP = reinterpret_cast<NTParams*>(extraP->input->pre_render_data);
	NTInfo* infoPI = reinterpret_cast<NTInfo*>(extraP->input->pre_render_data);

	if (infoP) {
		ERR((extraP->cb->checkout_layer_pixels(	in_data->effect_ref, NTFINISHER_INPUT, &input_worldP)));
		ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

		AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
																				kPFWorldSuite,
																				kPFWorldSuiteVersion2,
																				out_data);
		PF_PixelFormat	pixel_format = PF_PixelFormat_INVALID;
		ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

		if(isGPU) {
			
			ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
		} else {
			
			ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
		}
		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, NTFINISHER_INPUT));
	} else {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
	return err;
}


extern "C" DllExport
PF_Err PluginDataEntryFunction(
	PF_PluginDataPtr inPtr,
	PF_PluginDataCB inPluginDataCallBackPtr,
	SPBasicSuite* inSPBasicSuitePtr,
	const char* inHostName,
	const char* inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;

	result = PF_REGISTER_EFFECT(
		inPtr,
		inPluginDataCallBackPtr,
		"NTGlow", // Name
		"NTGlow", // Match Name
		"NT Productions", // Category
		AE_RESERVED_INFO); // Reserved Info

	return result;
}


PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData		*in_dataP,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err		err = PF_Err_NONE;
	
	try {
		switch (cmd) 
		{
			case PF_Cmd_ABOUT:
				err = About(in_dataP,out_data,params,output);
				break;
			case PF_Cmd_GLOBAL_SETUP:
				err = GlobalSetup(in_dataP,out_data,params,output);
				break;
			case PF_Cmd_PARAMS_SETUP:
				err = ParamsSetup(in_dataP,out_data,params,output);
				break;
			case PF_Cmd_GPU_DEVICE_SETUP:
				err = GPUDeviceSetup(in_dataP, out_data, (PF_GPUDeviceSetupExtra *)extra);
				break;
			case PF_Cmd_GPU_DEVICE_SETDOWN:
				err = GPUDeviceSetdown(in_dataP, out_data, (PF_GPUDeviceSetdownExtra *)extra);
				break;
			case PF_Cmd_RENDER:
				err = Render(in_dataP,out_data,params,output);
				break;
			case PF_Cmd_SMART_PRE_RENDER:
				err = PreRender(in_dataP, out_data, (PF_PreRenderExtra*)extra);
				break;
			case PF_Cmd_SMART_RENDER:
				err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra*)extra, false);
				break;
			case PF_Cmd_SMART_RENDER_GPU:
				err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra *)extra, true);
				break;
		}
	} catch(PF_Err &thrown_err) {
		// Never EVER throw exceptions into AE.
		err = thrown_err;
	}
	return err;
}
