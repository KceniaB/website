using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.AddressableAssets;
using UnityEngine.Networking;
using UnityEngine.ResourceManagement.AsyncOperations;
using UnityEngine.UI;
using UnityEngine.Video;

public class TrialViewerManager : MonoBehaviour
{
    [DllImport("__Internal")]
    private static extern void UpdateTrialTime(float time);
    [DllImport("__Internal")]
    private static extern void ChangeTrial(int trialInc);
    [DllImport("__Internal")]
    private static extern void TrialViewerLoaded();

    #region exposed fields
    [SerializeField] private Button prevTrialButton;
    [SerializeField] private Button nextTrialButton;
    [SerializeField] private Button playButton;
    [SerializeField] private Button stopButton;

    [SerializeField] private GameObject _loadingScreen;

    [SerializeField] private Image infoImage;
    [SerializeField] private Color defaultColor;
    [SerializeField] private Sprite defaultSprite;
    [SerializeField] private Sprite goSprite;
    [SerializeField] private Sprite correctSprite;
    [SerializeField] private Sprite wrongSprite;

    [SerializeField] private VideoPlayer leftVideoPlayer;
    [SerializeField] private VideoPlayer bodyVideoPlayer;
    [SerializeField] private VideoPlayer rightVideoPlayer;

    [SerializeField] private DLCPoint pawLcamR;
    [SerializeField] private DLCPoint pawRcamR;
    [SerializeField] private DLCPoint pawLcamL;
    [SerializeField] private DLCPoint pawRcamL;

    [SerializeField] private GaborStimulus stimulus;

    [SerializeField] private WheelComponent wheel;

    [SerializeField] private AssetReferenceT<TextAsset> trialTextAsset;

    #endregion

    #region data
    //private List<(float time, int leftIdx, int bodyIdx,
    //            float cr_pawlx, float cr_pawly, float cr_pawrx, float cr_pawry,
    //            float cl_pawlx, float cl_pawly, float cl_pawrx, float cl_pawry,
    //            float wheel)> timestampData;
    private Dictionary<string, float[]> timestampData;

    private List<(int start, int stimOn, int feedback, bool right, float contrast, bool correct)> trialData;
    private (int start, int stimOn, int feedback, bool right, float contrast, bool correct) currentTrialData;
    private (int start, int stimOn, int feedback, bool right, float contrast, bool correct) nextTrialData;
    #endregion

    #region local vars
    private float time; // the current time referenced from 0 across the entire session
    private int trial;
    private bool playing;
    private Coroutine infoCoroutine;
    #endregion

    #region trial vars
    private bool playedGo;
    private bool playedFeedback;
    private float sideFlip;
    private float initDeg;
    private float endDeg;
    #endregion

    private void Awake()
    {
#if !UNITY_EDITOR && UNITY_WEBGL
        // disable WebGLInput.captureAllKeyboardInput so elements in web page can handle keyboard inputs
        WebGLInput.captureAllKeyboardInput = false;
#endif
        Addressables.WebRequestOverride = EditWebRequestURL;

        StartCoroutine(LoadData("0802ced5-33a3-405e-8336-b65ebc5cb07c"));   

        Stop();
    }

    //Override the url of the WebRequest, the request passed to the method is what would be used as standard by Addressables.
    private void EditWebRequestURL(UnityWebRequest request)
    {
        if (request.url.Contains("http://"))
            request.url = request.url.Replace("http://", "https://");
        Debug.Log(request.url);
    }

    #region data loading
    public IEnumerator LoadData(string pid)
    {
        _loadingScreen.SetActive(true);

        // for now we ignore the PID and just load the referenced assets
        Debug.Log("Starting async load calls");
        AsyncOperationHandle<TextAsset> trialHandle = trialTextAsset.LoadAssetAsync();

        Debug.Log("Passed initial load");
        // videos
        leftVideoPlayer.url = string.Format("https://viz.internationalbrainlab.org/WebGL/{0}_left_scaled.mp4",pid);
        rightVideoPlayer.url = string.Format("https://viz.internationalbrainlab.org/WebGL/{0}_right_scaled.mp4", pid);
        bodyVideoPlayer.url = string.Format("https://viz.internationalbrainlab.org/WebGL/{0}_body_scaled.mp4", pid);

        timestampData = new Dictionary<string, float[]>();
        string[] dataTypes = {"right_ts","left_idx","body_idx",
            "cr_paw_l_x", "cr_paw_l_y", "cr_paw_r_x", "cr_paw_r_y",
            "cl_paw_l_x", "cl_paw_l_y", "cl_paw_r_x", "cl_paw_r_y",
            "wheel"};
        foreach (string type in dataTypes)
        {
            Debug.Log("Loading: " + type);
            string path = string.Format("Assets/AddressableAssets/{0}/{0}.{1}.bytes", pid, type);
            AsyncOperationHandle<TextAsset> dataHandle = Addressables.LoadAssetAsync<TextAsset>(path);
            if (!dataHandle.IsDone)
                yield return dataHandle;

            int nBytes = dataHandle.Result.bytes.Length;
            Debug.Log(string.Format("Loading {0} with {1} bytes", path, nBytes));
            float[] data = new float[nBytes / 4];

            Buffer.BlockCopy(dataHandle.Result.bytes, 0, data, 0, nBytes);
            Debug.LogFormat("Found {0} floats", data.Length);

            timestampData[type] = data;
        }

        if (!trialHandle.IsDone)
            yield return trialHandle;

        // parse trial data
        trialData = CSVReader.ParseTrialData(trialHandle.Result.text);

        trial = 1;
        UpdateTrial();

        MoveToFrameAndPrepare(currentTrialData.start);

        while (!leftVideoPlayer.isPrepared || !rightVideoPlayer.isPrepared || !bodyVideoPlayer.isPrepared)
            yield return null;

        pawLcamL.gameObject.SetActive(true);
        pawRcamL.gameObject.SetActive(true);
        pawLcamR.gameObject.SetActive(true);
        pawRcamR.gameObject.SetActive(true);

#if !UNITY_EDITOR && UNITY_WEBGL
        TrialViewerLoaded();
#endif

        Debug.Log("LOADED");
        _loadingScreen.SetActive(false);
    }

    #endregion 

    private void Update()
    {
        if (playing)
        {
            int frameIdx = (int)leftVideoPlayer.frame;

            // catch in case the video hasn't finished loading
            if (frameIdx == -1)
                return;

            time += timestampData["right_ts"][frameIdx];

#if !UNITY_EDITOR && UNITY_WEBGL
            UpdateTrialTime(time);
#endif

            if (frameIdx >= nextTrialData.start)
            {
                trial++;
                UpdateTrial();
            }

            // wheel properties
            wheel.SetRotation(timestampData["wheel"][frameIdx]);


            // stimulus properties
            if (currentTrialData.correct)
            {
                if (frameIdx >= currentTrialData.stimOn && frameIdx <= currentTrialData.feedback)
                {
                    stimulus.gameObject.SetActive(true);

                    float deg = wheel.Degrees();
                    Mathf.InverseLerp(initDeg, endDeg, deg);
                    stimulus.SetPosition(sideFlip * Mathf.InverseLerp(endDeg, initDeg, deg));
                }
                else if (frameIdx >= currentTrialData.feedback && frameIdx <= nextTrialData.start)
                {
                    stimulus.gameObject.SetActive(true);
                    stimulus.SetPosition(0f);
                }
                else
                    stimulus.gameObject.SetActive(false);
            }
            else
            {
                if (frameIdx >= currentTrialData.stimOn && frameIdx <= currentTrialData.feedback)
                {
                    stimulus.gameObject.SetActive(true);

                    float deg = wheel.Degrees();
                    Mathf.InverseLerp(initDeg, endDeg, deg);
                    stimulus.SetPosition(1 + -sideFlip * Mathf.InverseLerp(endDeg, initDeg, deg));
                }
                else if (frameIdx >= currentTrialData.feedback && frameIdx <= nextTrialData.start)
                {
                    stimulus.gameObject.SetActive(true);
                    stimulus.SetPosition(2f);
                }
                else
                    stimulus.gameObject.SetActive(false);
            }

            if (frameIdx >= currentTrialData.stimOn && !playedGo)
            {
                // stim on
                playedGo = true;
                infoImage.sprite = goSprite;
                infoImage.color = Color.yellow;

                if (infoCoroutine != null)
                    StopCoroutine(infoCoroutine);
                infoCoroutine = StartCoroutine(ClearSprite(0.2f));
            }

            if (frameIdx >= currentTrialData.feedback && !playedFeedback)
            {
                playedFeedback = true;
                if (currentTrialData.correct)
                {
                    infoImage.sprite = goSprite;
                    infoImage.color = Color.green;

                    if (infoCoroutine != null)
                        StopCoroutine(infoCoroutine);
                    infoCoroutine = StartCoroutine(ClearSprite(0.5f));
                }
                else
                {
                    infoImage.sprite = wrongSprite;
                    infoImage.color = Color.red;

                    if (infoCoroutine != null)
                        StopCoroutine(infoCoroutine);
                    infoCoroutine = StartCoroutine(ClearSprite(0.5f));
                }
            }

            // Set DLC points
            pawLcamR.SetPosition(timestampData["cr_paw_l_x"][frameIdx], timestampData["cr_paw_l_y"][frameIdx]);
            pawRcamR.SetPosition(timestampData["cr_paw_r_x"][frameIdx], timestampData["cr_paw_r_y"][frameIdx]);
            pawLcamL.SetPosition(timestampData["cl_paw_l_x"][frameIdx], timestampData["cl_paw_l_y"][frameIdx]);
            pawRcamL.SetPosition(timestampData["cl_paw_r_x"][frameIdx], timestampData["cl_paw_r_y"][frameIdx]);
        }
    }

    public IEnumerator ClearSprite(float delay)
    {
        yield return new WaitForSeconds(delay);

        infoImage.sprite = defaultSprite;
        infoImage.color = defaultColor;
    }

    Coroutine nextFrameRoutine;

    public void UpdateTrial(bool forceFrame = false)
    {
        //reset trial variables
        playedGo = false;
        playedFeedback = false;

        currentTrialData = trialData[trial];
        nextTrialData = trialData[trial + 1];

#if UNITY_EDITOR
        Debug.Log(string.Format("Starting trial {0}: start {1} end {2} right {3} correct {4}", trial,
            currentTrialData.start, nextTrialData.start, currentTrialData.right, currentTrialData.correct));
#endif

        // set the stimulus properties
        stimulus.SetContrast(currentTrialData.contrast);
        sideFlip = currentTrialData.right ? 1 : -1;

        // set the wheel properties
        initDeg = wheel.CalculateDegrees(timestampData["wheel"][currentTrialData.stimOn]);
        endDeg = wheel.CalculateDegrees(timestampData["wheel"][currentTrialData.feedback]);

        if (!playing || forceFrame)
        {
            if (forceFrame)
            {
                if (nextFrameRoutine != null)
                    StopCoroutine(nextFrameRoutine);
                nextFrameRoutine = StartCoroutine(UpdateVideoFramesAndPlay());
            }
            else
            {
                // if we aren't playing, move the videos to the correct frame
                MoveToFrameAndPrepare(currentTrialData.start);
            }
        }
    }

    private IEnumerator UpdateVideoFramesAndPlay()
    {
        Stop();

        MoveToFrameAndPrepare(currentTrialData.start);

        while (!leftVideoPlayer.isPrepared || !rightVideoPlayer.isPrepared || !bodyVideoPlayer.isPrepared)
            yield return null;

        Play();
    }

    private void MoveToFrameAndPrepare(int frame)
    {
        rightVideoPlayer.frame = frame;
        bodyVideoPlayer.frame = (long)timestampData["body_idx"][frame];
        leftVideoPlayer.frame = (long)timestampData["left_idx"][frame];

        leftVideoPlayer.Prepare();
        rightVideoPlayer.Prepare();
        bodyVideoPlayer.Prepare();

        pawLcamR.SetPosition(timestampData["cr_paw_l_x"][frame], timestampData["cr_paw_l_y"][frame]);
        pawRcamR.SetPosition(timestampData["cr_paw_r_x"][frame], timestampData["cr_paw_r_y"][frame]);
        pawLcamL.SetPosition(timestampData["cl_paw_l_x"][frame], timestampData["cl_paw_l_y"][frame]);
        pawRcamL.SetPosition(timestampData["cl_paw_r_x"][frame], timestampData["cl_paw_r_y"][frame]);
    }

    #region webpage callbacks
    public void SetSession(string pid)
    {
        Debug.LogWarning("Session cannot be changed until additional sessions are created");
    }

    /// <summary>
    /// Callback from the webpage to tell us which trial to go to
    /// 
    /// Stops playback
    /// </summary>
    /// <param name="newTrial"></param>
    public void SetTrial(int newTrial)
    {
        Stop();

        trial = newTrial;
        UpdateTrial();
        
        if (newTrial == 0)
            prevTrialButton.enabled = false;
        if (newTrial == trialData.Count)
            nextTrialButton.enabled = false;
    }
    #endregion

    #region button controls

    public void PrevTrial()
    {
        trial -= 1;
        UpdateTrial(playing);

#if !UNITY_EDITOR && UNITY_WEBGL
        ChangeTrial(trial);
#endif
    }

    public void NextTrial()
    {
        trial += 1;
        UpdateTrial(playing);

#if !UNITY_EDITOR && UNITY_WEBGL
        ChangeTrial(trial);
#endif
    }

    public void Play()
    {
        playing = true;

        rightVideoPlayer.Play();
        bodyVideoPlayer.Play();
        leftVideoPlayer.Play();
    }

    public void Stop()
    {
        playing = false;

        rightVideoPlayer.Stop();
        bodyVideoPlayer.Stop();
        leftVideoPlayer.Stop();
    }
    #endregion
}
