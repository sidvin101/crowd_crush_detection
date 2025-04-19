// Mainly controls the shuffling and davign data of the characters
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ShuffleManager : MonoBehaviour
{
    // Section to set it for manual shuffling or auto shuffling
    [Header("Simulation Settings")]
    public bool runBatchSimulation = false;
    public int totalSimulations = 1000;
    public float simulationInterval = 0.1f;

    private float simulationTime;
    private bool isSimulating = false;

    // private variables for the two diffeent folders
    private string imageFolder;
    private string jsonFolder;

    // Section to allow the user to set which GameObjects are being used to shuffle
    [Header("Characters To Shuffle")]
    public List<RandomPos> characters = new List<RandomPos>();

    // Section to set the max and minimum range for the characters to be shuffled in
    [Header("Position Range")]
    public Vector3 minPosition = new Vector3(-3f, 0f, -3f);
    public Vector3 maxPosition = new Vector3(3f, 0f, 3f);

    // Section to set the max and minimum rotation
    [Header("Rotation")]
    public float minYRotation = 0f;
    public float maxYRotation = 360f;

    // Section to allow for keyboard input to manually shuffle the stage
    [Header("Key")]
    public KeyCode shuffleKey = KeyCode.Space;
    public KeyCode resetKey = KeyCode.R;

    // CameraCapture class
    public CameraCapture cameraCapture;

    // Variable to check for used positions to prevent clipping
    private List<Vector3> usedPositions = new List<Vector3>();

    void Start()
    {
        // For each character, cache their positions
        foreach(var character in characters)
        {
            character.CacheOriginal();
        }

        // Set the image and json folder
        imageFolder = Path.Combine(Application.persistentDataPath, "crowd_images");
        jsonFolder = Path.Combine(Application.persistentDataPath, "crowd_json");

        // Create Directory in case
        Directory.CreateDirectory(imageFolder);
        Directory.CreateDirectory(jsonFolder);

        // Set variables if the batch simulation is called
        if (runBatchSimulation)
        {
            isSimulating = true;
            simulationTime = Time.time;
        }
    }

    void Update()
    {
        // If we ar eauto running the code
        if (runBatchSimulation && isSimulating)
        {
            // Run at the set interval
            if (Time.time - simulationTime >= simulationInterval)
            {
                simulationTime = Time.time;
                RunSimulationStep();
            }
        }
        // Keycodes
        if (Input.GetKeyDown(shuffleKey))
        {
            ShuffleCharacters();
        }
        if (Input.GetKeyDown(resetKey))
        {
            foreach(var character in characters)
            {
                character.ResetToOriginal();
            }
        }
    }

    // FUnction to run the simulation
    void RunSimulationStep()
    {
        for (int i = 0; i < totalSimulations; i++)
        {
            // Creates an id with a 4 digit placeholder
            string id = i.ToString("D4");

            usedPositions.Clear();
            ShuffleCharacters();

            // Allow the camera to update before taking a screenshot
            new WaitForEndOfFrame();

            if (cameraCapture != null)
            {
                cameraCapture.CaptureScreenshot(Path.Combine(imageFolder, "scene_" + id));
            }

            // Save the json data
            SaveCharacterData(Path.Combine(jsonFolder, "scene_" + id ));

            // End the simulation if complete
            if (i >= totalSimulations - 1)
            {
                isSimulating = false;
                Debug.Log("Simulation complete!");
            }
        }
    }

    // Function to shuffle the characters
    void ShuffleCharacters()
    {
        usedPositions.Clear();

        foreach (var character in characters)
        {
            Vector3 newPosition = GetUniqueRandomPosition();
            float newYRotation = Random.Range(minYRotation, maxYRotation);
            character.ApplyPositionAndRotation(newPosition, newYRotation);
        }
    }

    //Function to ensure each character gets a unique position
    Vector3 GetUniqueRandomPosition()
    {
        const int maxAttempte = 50;

        for (int i = 0; i < maxAttempts; i++)
        {
            Vector3 candidate = new Vector3(
                Random.Range(minPosition.x, maxPosition.x),
                0f,
                Random.Range(minPosition.z, maxPosition.z)
            );

            bool isTooClose = false;
             // checks if each candidate is "too close" which means that the characters ar eliterally on top of each other
            foreach (var pos in usedPositions)
            {
                if (Vector3.Distance(candidate, pos) < 1f)
                {
                    isTooClose = true;
                    break;
                }
            }
            // If the candiate is fine, we add that new candidare
            if (!isTooClose)
            {
                usedPositions.Add(candidate);
                return candidate;
            }

            // Sends out a warning
            Debug.LogWarning("Could not find a unique position for character after " + maxAttempts + " attempts.");
            return minPosition;
        }
    }

    // Structure to create the json data
    [System.Serializable]
    public class CharacterData
    {
        public string id;
        public Vector3 worldPosition;
        public Vector3 screenPosition;
    }

    [System.Serializable]
    public class CharacterDataList
    {
        public List<CharacterData> characters;
    }

    // Function to save the character data
    void SaveCharacterData(string fileName)
    {
        List<CharacterData> characterDataList = new List<CharacterData>();

        // log the info for each character
        foreach (var character in characters)
        {
            if (character == null) continue;

            CharacterData c = new CharacterData();
            c.id = character.name;
            c.worldPosition = character.transform.position;

            // Gets the screen position from the camera
            if (cameraCapture != null && cameraCapture.targetCamera != null)
            {
                Vector3 viewport Pos = cameraCapture.targetCamera.WorldToViewportPoint(character.transform.position);
                c.screenPosition = new Vector3(
                    viewportPos.x * cameraCapture.resolutionWidth,
                    (1f - viewportPos.y) * cameraCapture.resolutionHeight, // We need to flip this because Python and Unity have different screen measurments
                    0
                );
            }
            else
            {
                c.screenPosition = Vector3.zero;
            }

            // We add the data
            data.Add(c);
        }

        // Wrap the list and save the file
        CharacterDataKust wrapper = new CharacterDataList();
        wrapper.characters = data;

        string json = JsonUnity.ToJson(wrapper, true);
        string path = path.Combine(Application.persistentDataPath, fileName + ".json");
        File.WriteAllText(path, json);

        Debug.Log("Character data saved to: " + path);
    }

    // Uses the Gizmo to show the shuffle area. This is useful to fine tune the size of your area
    void OnDrawGizmosSelec()
    {
        Gizmos.color = Color.cyan

        Vector3 center = (minPosition + maxPosition) / 2f;
        Vector3 size = maxPosition - minPosition;
        Gizmos.DrawWireCube(center, size);
    }

}