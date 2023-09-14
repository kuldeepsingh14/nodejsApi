const express = require('express');
const cv = require('opencv4nodejs');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

// Middleware for parsing JSON request bodies
app.use(express.json());

// Load known face images
const knownFaces = [
  { name: 'Universal', imagePath: 'C:/Users/Kuldeep Singh/OneDrive/Desktop/nodejsapi/know_face/universalpassimage.jpg' },
  
  // Add more known faces with their image paths
];

// Load a Haar Cascade for face detection
const faceClassifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

// Endpoint for face recognition
app.post('/recognize', (req, res) => {
  try {
    const { imageBase64 } = req.body;

    // Decode base64 image
    const buffer = Buffer.from(imageBase64, 'base64');
    const image = cv.imdecode(buffer);

    // Detect faces in the image
    const faces = faceClassifier.detectMultiScale(image.bgrToGray());

    const recognizedFaces = [];

    // Iterate over detected faces
    for (const faceRect of faces) {
      const faceRegion = image.getRegion(faceRect);

      // Compare the detected face with known faces
      for (const knownFace of knownFaces) {
        const knownFaceImage = cv.imread(knownFace.imagePath);
        const match = faceRegion.matchTemplate(knownFaceImage, cv.TM_CCOEFF_NORMED);

        if (match.maxLoc.confidence > 0.7) {
          recognizedFaces.push(knownFace.name);
        }
      }
    }

    res.json({ message: 'Recognition results', data: recognizedFaces });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
