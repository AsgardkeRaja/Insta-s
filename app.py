import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db, auth, firestore
import face_recognition
from PIL import Image
import numpy as np
import cv2
import dlib

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))
skp_path = os.path.join(current_directory, "projectinsta-s-firebase-adminsdk-y6vlu-9a1345f468.json")

# Check if the app is already initialized
if not firebase_admin._apps:
    # Initialize Firebase Admin SDK
    cred = credentials.Certificate(skp_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://projectinsta-s-default-rtdb.firebaseio.com/',
        'projectId': 'projectinsta-s'
    })

# Reference to the root of your Firebase Realtime Database
ref = db.reference('/')

# Initialize Firestore client
db_firestore = firestore.client()

# Streamlit session state
if "auth_state" not in st.session_state:
    st.session_state.auth_state = {
        "user": None,
        "signed_in": False,
    }

# Add the shape predictor model for face alignment
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)

# Firebase Authentication
def authenticate_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        # The user is successfully fetched, meaning the email and password are valid.
        return True, user
    except auth.AuthError as e:
        print(f"Authentication error: {str(e)}")
        return False, None
        
# Sign-up Functionality
def create_user(email, password):
    try:
        user = auth.create_user(
            email=email,
            password=password
        )
        return True, user.uid
    except Exception as e:
        print(f"User creation error: {str(e)}")
        return False, None
        
# Update load_and_encode function to use the aligned face without normalization
def load_and_encode(image_path):
    try:
        aligned_face = detect_and_align_faces(image_path)

        if aligned_face is not None:
            encoding = face_recognition.face_encodings(aligned_face)

            if encoding:
                return encoding
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"Error loading and encoding image: {str(e)}")
        return None

# Function to detect and align faces in an image with preprocessing
def detect_and_align_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    
    # Resize the image to a fixed width (you can adjust the width as needed)
    target_width = 800
    aspect_ratio = image.shape[1] / image.shape[0]
    target_height = int(target_width / aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height))

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using dlib
    faces = detector(gray)

    if not faces:
        return None

    # Use the first face found (you can modify this to handle multiple faces)
    face = faces[0]
    
    # Use dlib for face alignment
    landmarks = shape_predictor(gray, face)
    aligned_face = dlib.get_face_chip(resized_image, landmarks, size=256)  # Adjust the size as needed
    
    return aligned_face

# Add person to database
def add_person(name, image_path, instagram_handle):
    try:
        encoding = load_and_encode(image_path)
        if not encoding:
            return "No face found in the provided image."

        # Convert NumPy arrays to lists for JSON serialization
        encoding = encoding[0].tolist()

        # Save data to Firebase Realtime Database
        ref.child(name).set({
            "encoding": encoding,
            "info": {
                "instagram_handle": instagram_handle,
                "instagram_link": f"https://www.instagram.com/{instagram_handle}/"
            }
        })

        return f"Success: {name} added to the database!"
    except Exception as e:
        return f"Failed to add person: {str(e)}"

# Recognize face from image
def recognize_face(image_path):
    if not image_path:
        return "Please upload an image."

    try:
        unknown_encoding = load_and_encode(image_path)
        if not unknown_encoding:
            return "No face found in the provided image."

        matches = []
        for name, data in ref.get().items():
            known_encoding = np.array(data["encoding"])
            if face_recognition.compare_faces([known_encoding], unknown_encoding[0])[0]:
                matches.append((name, data["info"]))

        if matches:
            results = []
            for name, info in matches:
                insta_handle = info["instagram_handle"]
                insta_link = info["instagram_link"]
                insta_link_html = f'<a href="{insta_link}" target="_blank"><font color="red">{insta_handle}</font></a>'
                results.append(f"- It's a picture of {name}! Insta handle: {insta_link_html}")
            return "\n".join(results)
        else:
            return "Face not found in the database."
    except Exception as e:
        return f"Failed to recognize face: {str(e)}"

# Recognize face from image and return optimal or highest matching ID
def recognize_face_optimal(image_path):
    if not image_path:
        return "Please upload an image."

    try:
        unknown_encoding = load_and_encode(image_path)
        if not unknown_encoding:
            return "No face found in the provided image."

        matches = []
        for name, data in ref.get().items():
            known_encoding = np.array(data["encoding"])
            similarity_score = face_recognition.face_distance([known_encoding], unknown_encoding[0])[0]
            matches.append((name, similarity_score))

        if matches:
            best_match = min(matches, key=lambda x: x[1])
            best_name, best_score = best_match
            info = ref.child(best_name).child("info").get()
            insta_handle = info["instagram_handle"]
            insta_link = info["instagram_link"]
            insta_link_html = f'<a href="{insta_link}" target="_blank"><font color="red">{insta_handle}</font></a>'
            return f"Best match: {best_name} with a similarity score of {1 - best_score:.2%}. Insta handle: {insta_link_html}"
        else:
            return "Face not found in the database."
    except Exception as e:
        return f"Failed to recognize face: {str(e)}"

# Delete person from database
def delete_person(name):
    try:
        ref.child(name).delete()
        return f"{name} deleted from the database!"
    except Exception as e:
        return f"Failed to delete person: {str(e)}"

# Send feedback to Firebase
def send_feedback(feedback_data):
    try:
        db_firestore.collection('feedback').add(feedback_data)
    except Exception as e:
        st.error(f"Failed to submit feedback: {str(e)}")

# Streamlit interface for adding a person
def add_person_ui():
    st.title("Add Person")
    name = st.text_input("Enter Name", help="Enter the name of the person")
    image_path = st.file_uploader("Upload Image", help="Upload an image containing the person's face")
    instagram_handle = st.text_input("Enter Instagram Handle", help="Enter the person's Instagram handle")
    if st.button("Add Person"):
        if not name or not image_path or not instagram_handle:
            st.error("Please fill all the fields.")
        else:
            result = add_person(name, image_path, instagram_handle)
            st.success(result)

# Streamlit interface for recognizing face
def recognize_face_ui():
    st.title("Recognize Face")
    image_path = st.file_uploader("Upload Image", help="Upload an image for face recognition")
    if st.button("Recognize Face"):
        result = recognize_face(image_path)
        st.write(result, unsafe_allow_html=True)

# Streamlit interface for recognizing face with optimal ID
def recognize_face_optimal_ui():
    st.title("Recognize Face (Optimal)")
    image_path = st.file_uploader("Upload Image", help="Upload an image for optimal face recognition")
    if st.button("Recognize Face (Optimal)"):
        result = recognize_face_optimal(image_path)
        st.write(result, unsafe_allow_html=True)
        
# Streamlit interface for deleting a person
def delete_person_ui():
    st.title("Delete Person")
    name = st.text_input("Enter Name", help="Enter the name of the person to delete")
    if st.button("Delete Person"):
        if not name:
            st.error("Please enter a name.")
        else:
            result = delete_person(name)
            st.success(result)

# Streamlit interface for feedback
def feedback_ui():
    st.title("Feedback")
    st.write("Your feedback is important to us! Please fill out the form below:")

    name = st.text_input("Name (optional)")
    email = st.text_input("Email (optional)")
    category = st.selectbox("Category", ["Bug Report", "Feature Request", "General Feedback"])
    message = st.text_area("Feedback Message")

    if st.button("Submit Feedback"):
        if not message:
            st.error("Please enter your feedback message.")
        else:
            feedback_data = {
                "name": name,
                "email": email,
                "category": category,
                "message": message,
            }
            send_feedback(feedback_data)
            st.success("Feedback submitted successfully! Thank you for your feedback.")

def tour_guide_ui():
    st.title("Tour Guide")
    st.markdown("This tour will guide you through the application.")

    with st.expander("Welcome"):
        st.write("This is a tour guide to help you navigate through the application.")

    with st.expander("Options Sidebar"):
        st.write("Here you can select different options such as adding a person, recognizing a face, deleting a person, or recognizing a face with optimal identification.")

    with st.expander("Main Interface"):
        st.write("This is where the main functionality of the application is displayed.")

    with st.expander("Upload Image"):
        st.write("You can upload an image here for face recognition or adding a person.")

    with st.expander("Text Input"):
        st.write("Enter text here such as the person's name or Instagram handle.")

    with st.expander("Buttons"):
        st.write("Click on these buttons to perform actions like adding a person or recognizing a face.")

# Streamlit interface for user authentication
def authenticate_user_ui():
    st.title("Insta's EYE")
    # Display the logo
    logo_path = os.path.join(current_directory, "Explore+.png")
    logo = Image.open(logo_path)
    st.image(logo, width=30) 
    st.sidebar.title("Options")

    if st.session_state.auth_state["signed_in"]:
        st.sidebar.button("Sign Out", on_click=logout)
        st.title("Welcome!")
        main()
    else:
        option = st.sidebar.radio("Select Option", ["Login", "Sign-Up"])

        email = st.text_input("Enter Email", help="Enter your email address")
        password = st.text_input("Enter Password", type="password", help="Enter your password")

        if option == "Login":
            if st.button("Login"):
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    success, user = authenticate_user(email, password)
                    if success:
                        st.session_state.auth_state["user"] = user
                        st.session_state.auth_state["signed_in"] = True
                        st.success("Authentication successful! You can now manage your set of images and profiles.")
                        main()
                    else:
                        st.error("Authentication failed. Please check your email and password.")

        elif option == "Sign-Up":
            confirm_password = st.text_input("Confirm Password", type="password", help="Re-enter your password for confirmation")
            if st.button("Sign-Up"):
                if not email or not password or not confirm_password:
                    st.error("Please fill all the fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    success, uid = create_user(email, password)
                    if success:
                        st.success(f"User with UID: {uid} created successfully! You can now log in.")
                    else:
                        st.error("User creation failed. Please try again.")

# Log out user
def logout():
    st.session_state.auth_state["user"] = None
    st.session_state.auth_state["signed_in"] = False

# Define tour steps
steps = [
    {
        "title": "Welcome to Insta's EYE",
        "content": "This is a tour guide to help you navigate through the application.",
    },
    {
        "title": "Options Sidebar",
        "content": "Here you can select different options such as adding a person, recognizing a face, deleting a person, or recognizing a face with optimal identification.",
    },
    {
        "title": "Main Interface",
        "content": "This is where the main functionality of the application is displayed.",
    },
    {
        "title": "Upload Image",
        "content": "You can upload an image here for face recognition or adding a person.",
    },
    {
        "title": "Text Input",
        "content": "Enter text here such as the person's name or Instagram handle.",
    },
    {
        "title": "Buttons",
        "content": "Click on these buttons to perform actions like adding a person or recognizing a face.",
    },
]
    
# Function to display tour steps
def display_tour_steps(steps):
    st.markdown("# Tour Guide")
    st.markdown("This tour will guide you through the application.")
    st.markdown("---")

    for step in steps:
        st.markdown(f"## {step['title']}")
        st.write(step['content'])
        st.markdown("---")
    
# Update the main function to include the feedback option
def main():
    st.sidebar.title("Options")
    option = st.sidebar.radio("Select Option", ["Add Person", "Recognize Face", "Delete Person", "Recognize Face (Optimal)", "Tour Guide", "Feedback"])
    
    if option == "Add Person":
        add_person_ui()
    elif option == "Recognize Face":
        recognize_face_ui()
    elif option == "Delete Person":
        delete_person_ui()
    elif option == "Recognize Face (Optimal)":
        recognize_face_optimal_ui()
    elif option == "Tour Guide":
        tour_guide_ui()
    elif option == "Feedback":
        feedback_ui()

# Run the tour guide
if __name__ == "__main__":
    authenticate_user_ui()