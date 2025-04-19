using UnityEngine;
using System.IO;

public class CameraCapture : MonoBehavious
{
    // This camera will be manually assigned in the inspector. 
    public CameraCapture targetCamera;

    // Resolution sizes
    public int resolutionWidth = 1920;
    public int resolutionHeight = 1080;

    //Setting to allow id the user would like to save as a png
    public boolean saveAsPng = true;

    // File path to image folder
    private string imagePath;

    // Class to take a screenshot and saved
    public void CaptureScreenshot(string fileName = "ShuffleCapture")
    {
        // Render the camera
        RenderTexture rt = new RenderTexture(resolutionWidth, resolutionHeight, 24);
        targetCamera.targetTexture = rt;

        Texture2D screenshot = new Texture2D(resolutionWidth, resolutionHeight, TextureFormat.RGB24, false);
        targetCamera.Render();

        RenderTexture.active = rt;
        screenshot.ReadPixels(new Rect(0, 0, resolutionWidth, resolutionHeight), 0, 0);
        screenshot.Apply();

        targetCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        // Save the screenshot
        byte[] bytes;
        string extension;
        //Checks if the user wants to save as a png. If not, it will auto save as a jpg instead
        if (saveAsPng)
        {
            bytes = screenshot.EncodeToPNG();
            extension = ".png";
        }
        else
        {
            bytes = screenshot.EncodeToJPG();
            extension = ".jpg";
        }

        // Saves into a dedicated folder, and lets the user know that the image is saved
        imagePath = Path.Combine(Application.persistentDataPath, "crowd_images");
        string filename = $"{fileName}{extension}"
        File.WriteAllBytes(fullPath, bytes);
        Debug.Log($"Screenshot saved to: {fullPath}");
    }
}