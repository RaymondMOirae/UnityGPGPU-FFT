using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class FastFourierTransform: MonoBehaviour
{
    [SerializeField] bool SaveButterfly;

    [SerializeField] RenderTexture SourceData;
    [SerializeField] RenderTexture DestData;

    [SerializeField] RenderTexture RT_ButterflyTex;
    [SerializeField] RenderTexture RT_Pingpong0;
    [SerializeField] RenderTexture RT_Pingpong1;
    [SerializeField] RenderTexture RT_Source;
    [SerializeField] RenderTexture RT_Result;

    [SerializeField] ComputeShader Compute_ButterflyTexGen;
    [SerializeField] ComputeShader Compute_ButterflyCompute;

    private void Awake()
    {
        SetupFFT();        
    }

    private void SetupFFT()
    {
        Setup_ButterflyTexGen();
        Dispatch_ButterflyTexGen();
        AsyncGPUReadback.Request(RT_ButterflyTex, 0, TextureFormat.RGBAFloat, OnButterflyTexGen);
    }

    private void OnButterflyTexGen(AsyncGPUReadbackRequest req)
    {
        if (req.hasError)
        {
            Debug.LogError("Error occurred when read back BUTTERFLY.");
        }
        else
        {
            if (SaveButterfly)
            {
                Texture2D tex2d = new Texture2D(BF_TEXTURE_WIDTH, TEXTURE_HEIGHT, TextureFormat.RGBAFloat, false);
                tex2d.LoadRawTextureData(req.GetData<Color>());
                tex2d.Apply();
                var path = "Assets/Textures/Butterfly.png";
                System.IO.File.WriteAllBytes(path, tex2d.EncodeToPNG());
                UnityEditor.AssetDatabase.ImportAsset(path);
            }
        }
    }

    #region ButterflyTexGen
    private void Setup_ButterflyTexGen()
    {
        RT_ButterflyTex = ReadableRenderTexture(BF_TEXTURE_WIDTH, TEXTURE_HEIGHT);

        KERNEL_ButterflyTexGen = Compute_ButterflyTexGen.FindKernel("ButterflyTexGen");

        Compute_ButterflyTexGen.SetInt("BigN", TEXTURE_WIDTH);
        Compute_ButterflyTexGen.SetTexture(KERNEL_ButterflyTexGen, "ButterflyTexture", RT_ButterflyTex);        
    }

    private void Dispatch_ButterflyTexGen()
    {
        Compute_ButterflyTexGen.Dispatch(KERNEL_ButterflyTexGen, 32, 32, 1);
    }
    #endregion

    #region InverseFourierTransform
    private void Setup_IFFT(RenderTexture source)
    {
        RT_Pingpong0 = source;
        RT_Pingpong1 = ReadableRenderTexture(TEXTURE_WIDTH, TEXTURE_HEIGHT);
        RT_Result = ReadableRenderTexture(TEXTURE_WIDTH, TEXTURE_HEIGHT);

        KERNEL_IFFTButterflyCompute = Compute_ButterflyCompute.FindKernel("IFFTButterflyCompute");
    }

    private void Dispatch_IFFT()
    {
        int stage = (int)Mathf.Log(TEXTURE_WIDTH, 2);
        
        bool pingpong = false; // when pingpong is false, source data should be in pingpong1
        Compute_ButterflyCompute.SetBool("isHorizontal", true);
        Compute_ButterflyCompute.SetTexture(KERNEL_IFFTButterflyCompute, "ButterflyTex", RT_ButterflyTex);
        Compute_ButterflyCompute.SetTexture(KERNEL_IFFTButterflyCompute, "Pingpong0", RT_Pingpong0);
        Compute_ButterflyCompute.SetTexture(KERNEL_IFFTButterflyCompute, "Pingpong1", RT_Pingpong1);

        for (int i = 0; i < stage; i++)
        {
            pingpong = !pingpong;
            Compute_ButterflyCompute.SetInt("stage", i);
            Compute_ButterflyCompute.SetBool("pingpong", pingpong);
            Compute_ButterflyCompute.Dispatch(KERNEL_IFFTButterflyCompute, 32, 32, 1);
        }

        pingpong = false;
        Compute_ButterflyCompute.SetBool("isHorizontal", false);
        Compute_ButterflyCompute.SetTexture(KERNEL_IFFTButterflyCompute, "ButterflyTex", RT_ButterflyTex);
        Compute_ButterflyCompute.SetTexture(KERNEL_IFFTButterflyCompute, "Pingpong0", RT_Pingpong0);
        Compute_ButterflyCompute.SetTexture(KERNEL_IFFTButterflyCompute, "Pingpong1", RT_Pingpong1);

        for(int i = 0; i < stage; i++)
        {
            pingpong = !pingpong;
            Compute_ButterflyCompute.SetInt("stage", i);
            Compute_ButterflyCompute.SetBool("pingpong", pingpong);
            Compute_ButterflyCompute.Dispatch(KERNEL_IFFTButterflyCompute, 
                                              TEXTURE_WIDTH / THREAD_GROUP_WIDTH, 
                                              TEXTURE_WIDTH / THREAD_GROUP_HEIGHT, 1); 
        }

        if (pingpong)
        {
            Graphics.Blit(RT_Pingpong1, RT_Result);
        }
        else
        {
            Graphics.Blit(RT_Pingpong0, RT_Result);
        }
    }
    #endregion

    #region FourierTransform
    private void Setup_FFT(RenderTexture source)
    {
        RT_Pingpong0 = source;
        RT_Pingpong1 = ReadableRenderTexture(TEXTURE_WIDTH, TEXTURE_HEIGHT);
        RT_Result = ReadableRenderTexture(TEXTURE_WIDTH, TEXTURE_HEIGHT);

        KERNEL_FFTButterflyCompute = Compute_ButterflyCompute.FindKernel("FFTButterflyCompute");
    }

    private void Dispatch_FFT()
    {
        int stage = (int)Mathf.Log(TEXTURE_WIDTH, 2);
        
        bool pingpong = false; // when pingpong is false, source data should be in pingpong1
        Compute_ButterflyCompute.SetBool("isHorizontal", true);
        Compute_ButterflyCompute.SetTexture(KERNEL_FFTButterflyCompute, "ButterflyTex", RT_ButterflyTex);
        Compute_ButterflyCompute.SetTexture(KERNEL_FFTButterflyCompute, "Pingpong0", RT_Pingpong0);
        Compute_ButterflyCompute.SetTexture(KERNEL_FFTButterflyCompute, "Pingpong1", RT_Pingpong1);

        for (int i = 0; i < stage; i++)
        {
            pingpong = !pingpong;
            Compute_ButterflyCompute.SetInt("stage", i);
            Compute_ButterflyCompute.SetBool("pingpong", pingpong);
            Compute_ButterflyCompute.Dispatch(KERNEL_FFTButterflyCompute, 32, 32, 1);
        }

        pingpong = false;
        Compute_ButterflyCompute.SetBool("isHorizontal", false);
        Compute_ButterflyCompute.SetTexture(KERNEL_FFTButterflyCompute, "ButterflyTex", RT_ButterflyTex);
        Compute_ButterflyCompute.SetTexture(KERNEL_FFTButterflyCompute, "Pingpong0", RT_Pingpong0);
        Compute_ButterflyCompute.SetTexture(KERNEL_FFTButterflyCompute, "Pingpong1", RT_Pingpong1);

        for(int i = 0; i < stage; i++)
        {
            pingpong = !pingpong;
            Compute_ButterflyCompute.SetInt("stage", i);
            Compute_ButterflyCompute.SetBool("pingpong", pingpong);
            Compute_ButterflyCompute.Dispatch(KERNEL_FFTButterflyCompute, 
                                              TEXTURE_WIDTH / THREAD_GROUP_WIDTH, 
                                              TEXTURE_WIDTH / THREAD_GROUP_HEIGHT, 1); 
        }

        if (pingpong)
        {
            Graphics.Blit(RT_Pingpong1, RT_Result);
        }
        else
        {
            Graphics.Blit(RT_Pingpong0, RT_Result);
        }
    }
    #endregion

    RenderTexture ReadableRenderTexture(int width, int height)
    {
        RenderTexture rt =  new RenderTexture(width, height, 0)
        {
            enableRandomWrite = true,
            format = RenderTextureFormat.ARGBFloat
        };
        if(!rt.IsCreated())
            rt.Create();
        return rt;
    }

    readonly int TEXTURE_WIDTH = 256;
    readonly int TEXTURE_HEIGHT = 256;
    readonly int THREAD_GROUP_WIDTH = 8;
    readonly int THREAD_GROUP_HEIGHT = 8;
    int BF_TEXTURE_WIDTH => (int)Mathf.Log(TEXTURE_WIDTH, 2);

    int KERNEL_ButterflyTexGen;
    int KERNEL_FFTButterflyCompute;
    int KERNEL_IFFTButterflyCompute;
}
