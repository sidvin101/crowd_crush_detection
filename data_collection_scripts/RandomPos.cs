using UnityEngine;

public class RandomPos : MonoBehaviour
{
    // Original position and rotation of the object
    private Vector3 originalPosition;
    private Quaternion originalRotation;

    // Function to cache the original values
    public void CacheOriginal()
    {
        originalPosition = transform.position;
        originalRotation = transform.rotation;
    }

    // Function to reset the position and rotation
    public void ResetToOriginal()
    {
        transform.position = originalPosition;
        transform.rotation = originalRotation;
    }

    // Function to apply the position and rotation based on the provided values
    public void ApplyPositionAndRotation(Vector3 pos, float yRotation)
    {
        transform.position = pos;
        transform.rotation = Quaternion.Euler(0, yRotation, 0);
    }
}