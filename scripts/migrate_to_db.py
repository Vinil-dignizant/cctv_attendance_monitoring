# scripts/migrate_to_db.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2

import os
import numpy as np
from app.db.database import SessionLocal
from app.db.crud import create_person, add_face_feature, add_face_image
from app.db.models import Person
from scripts.add_persons_db import get_feature



def migrate_existing_data():
    db = SessionLocal()
    
    # Path to your existing data
    data_path = "datasets/data"
    features_path = "datasets/face_features/feature.npz"
    
    # Load existing features
    if os.path.exists(features_path):
        data = np.load(features_path)
        names = data['images_name']
        embeddings = data['images_emb']
        
        # Get unique person names
        unique_names = np.unique(names)
        
        for name in unique_names:
            # Check if person already exists
            person = db.query(Person).filter(Person.name == name).first()
            if not person:
                # Create new person
                person = create_person(db, name=name)
                
                # Add all embeddings for this person
                for emb in embeddings[names == name]:
                    add_face_feature(db, person.id, emb)
                
                # Add face images if available
                person_image_dir = os.path.join(data_path, name)
                if os.path.exists(person_image_dir):
                    for img_file in os.listdir(person_image_dir):
                        if img_file.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(person_image_dir, img_file)
                            img = cv2.imread(img_path)
                            thumbnail = cv2.resize(img, (100, 100))
                            add_face_image(db, person.id, img_path, thumbnail)
    
    db.close()
    print("Migration completed successfully")

if __name__ == "__main__":
    migrate_existing_data()