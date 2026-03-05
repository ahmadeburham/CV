from deepface import DeepFace


def verify_faces(id_path, selfie_path):

    result = DeepFace.verify(
        img1_path=id_path,
        img2_path=selfie_path,
        model_name="ArcFace",
        distance_metric="cosine"
    )

    similarity = 1 - result["distance"]

    return {
        "verified": result["verified"],
        "similarity_score": round(similarity, 3)
    }