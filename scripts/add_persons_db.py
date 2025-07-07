import argparse
import os
import shutil

import cv2
import numpy as np
import torch
from torchvision import transforms

from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import read_features


# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the face detector (Choose one of the detectors)
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# Initialize the face recognizer
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)


@torch.no_grad()
def get_feature(face_image):
    """
    Extract facial features from an image using the face recognition model.

    Args:
        face_image (numpy.ndarray): Input facial image.

    Returns:
        numpy.ndarray: Extracted facial features.
    """
    # Define a series of image preprocessing steps
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Convert the image to RGB format
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Apply the defined preprocessing to the image
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # Use the model to obtain facial features
    emb_img_face = recognizer(face_image)[0].cpu().numpy()

    # Normalize the features
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    return images_emb


def add_persons(backup_dir, add_persons_dir, faces_save_dir, features_path):
    """Add a new person to the face recognition database"""
    db = next(get_db())  # Get database session
    
    for name_person in os.listdir(add_persons_dir):
        person_image_path = os.path.join(add_persons_dir, name_person)
        
        # Check if person exists, if not create
        person = db.query(Person).filter(Person.name == name_person).first()
        if not person:
            person = Person(name=name_person)
            db.add(person)
            db.commit()
        
        for image_name in os.listdir(person_image_path):
            if image_name.endswith(("png", "jpg", "jpeg")):
                input_image = cv2.imread(os.path.join(person_image_path, image_name))
                
                # Detect faces and landmarks
                bboxes, landmarks = detector.detect(image=input_image)
                
                for i in range(len(bboxes)):
                    x1, y1, x2, y2, score = bboxes[i]
                    face_image = input_image[y1:y2, x1:x2]
                    
                    # Generate unique image path
                    img_filename = f"{name_person}_{int(time.time())}.jpg"
                    img_path = os.path.join(faces_save_dir, img_filename)
                    os.makedirs(faces_save_dir, exist_ok=True)
                    cv2.imwrite(img_path, face_image)
                    
                    # Get face embedding
                    embedding = get_feature(face_image=face_image)
                    
                    # Store in database
                    try:
                        # Add face feature
                        feature = FaceFeature(
                            person_id=person.id,
                            embedding=embedding.tobytes()  # Store as bytes
                        )
                        db.add(feature)
                        
                        # Add face image (with thumbnail)
                        thumbnail = cv2.resize(face_image, (100, 100))
                        face_img = FaceImage(
                            person_id=person.id,
                            image_path=img_path,
                            thumbnail=thumbnail.tobytes()
                        )
                        db.add(face_img)
                        
                        db.commit()
                    except Exception as e:
                        db.rollback()
                        print(f"[ERROR] Failed to save to database: {e}")
    
    print("Successfully added new person(s)!")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="./datasets/backup",
        help="Directory to save person data.",
    )
    parser.add_argument(
        "--add-persons-dir",
        type=str,
        default="./datasets/new_persons",
        help="Directory to add new persons.",
    )
    parser.add_argument(
        "--faces-save-dir",
        type=str,
        default="./datasets/data/",
        help="Directory to save faces.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="./datasets/face_features/feature",
        help="Path to save face features.",
    )
    opt = parser.parse_args()

    # Run the main function
    add_persons(**vars(opt))
