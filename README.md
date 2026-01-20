<h1>🧭 Navigation System for Visually Impaired Individuals</h1>

<p>
This project focuses on building a smartphone-based indoor navigation system for visually impaired individuals.
It uses <b>Computer Vision</b>, <b>Machine Learning</b>, and <b>Artificial Intelligence</b> to detect nearby objects
and provide real-time voice feedback, helping users navigate indoor environments independently.
</p>

<hr>

<h2>📌 Project Overview</h2>
<p>
Indoor navigation is challenging for visually impaired individuals due to changing environments and lack of
real-time guidance. Traditional tools like white canes and guide dogs have limitations, especially indoors.
This project introduces a cost-effective and portable solution that works using a smartphone camera without
any additional hardware or infrastructure.
</p>

<hr>

<h2>🎯 Project Objectives</h2>
<ul>
  <li>Assist visually impaired individuals in indoor navigation</li>
  <li>Detect nearby objects in real time using a smartphone camera</li>
  <li>Provide voice-based feedback for environmental awareness</li>
  <li>Eliminate the need for GPS, internet, or external sensors</li>
  <li>Create an affordable and easy-to-use assistive system</li>
</ul>

<hr>

<h2>🛠 Technologies Used</h2>
<ul>
  <li>Python</li>
  <li>Computer Vision (OpenCV)</li>
  <li>Machine Learning</li>
  <li>Deep Learning</li>
  <li>YOLOv8</li>
  <li>Faster R-CNN</li>
  <li>Pyttsx3 (Text-to-Speech)</li>
</ul>

<hr>

<h2>📊 System Architecture</h2>
<ul>
  <li>Smartphone camera captures real-time video feed</li>
  <li>Frames are processed using object detection models</li>
  <li>Detected objects are identified and classified</li>
  <li>Object names are announced using voice output</li>
</ul>

<hr>

<h2>📌 Methodology (Step-by-Step)</h2>

<h3>1️⃣ Data Preparation</h3>
<ul>
  <li>Used YOLO-based datasets containing over 120,000 images</li>
  <li>Mapped images with corresponding annotation files</li>
  <li>Filtered dataset to retain meaningful object samples</li>
  <li>Final dataset reduced to ~88,000 cleaned images</li>
</ul>

<h3>2️⃣ Initial Model Experiments</h3>
<ul>
  <li>Trained standard CNN models</li>
  <li>Observed underfitting and unstable accuracy (39–60%)</li>
  <li>Concluded CNNs were insufficient for complex detection</li>
</ul>

<h3>3️⃣ R-CNN and YOLOv5 Trials</h3>
<ul>
  <li>R-CNN failed due to software and hardware issues</li>
  <li>YOLOv5 struggled with indoor object detection</li>
  <li>Faced memory, compatibility, and performance limitations</li>
</ul>

<h3>4️⃣ YOLOv8 Implementation</h3>
<ul>
  <li>Integrated YOLOv8 for real-time object detection</li>
  <li>Optimized performance by resizing frames to 640×480</li>
  <li>Skipped frames to reduce processing delay</li>
  <li>Implemented bounding boxes and object labels</li>
  <li>Used multithreading for non-blocking voice output</li>
  <li>Reduced repetitive announcements for better user experience</li>
</ul>

<h3>5️⃣ Faster R-CNN Breakthrough</h3>
<ul>
  <li>Used ResNet-50 backbone with Region Proposal Network</li>
  <li>Trained on a curated dataset of 200 images</li>
  <li>Achieved 100% accuracy on test data</li>
  <li>Maintained ~90% accuracy with reduced datasets</li>
  <li>Provided best balance of accuracy and performance</li>
</ul>

<hr>

<h2>🔊 Voice Feedback System</h2>
<ul>
  <li>Implemented using Pyttsx3</li>
  <li>Works completely offline</li>
  <li>Supports speech rate and volume control</li>
  <li>Announces detected objects in real time</li>
  <li>Runs asynchronously to avoid system lag</li>
</ul>

<hr>

<h2>📈 Results</h2>
<ul>
  <li>CNN accuracy: ~50%</li>
  <li>YOLOv8: Improved detection but limited real-time speed</li>
  <li>Faster R-CNN: Up to 100% accuracy on indoor dataset</li>
  <li>Stable and responsive real-time performance</li>
</ul>

<hr>

<h2>🔍 Key Insights</h2>
<ul>
  <li>Model choice is critical for real-time applications</li>
  <li>Dataset quality impacts accuracy more than size</li>
  <li>Asynchronous voice output improves usability</li>
  <li>Indoor navigation requires specialized training data</li>
</ul>

<hr>

<h2>⚠ Challenges Faced</h2>
<ul>
  <li>Messy and unstructured datasets</li>
  <li>Low accuracy with basic models</li>
  <li>GPU and library compatibility issues</li>
  <li>Lag in real-time detection and voice feedback</li>
  <li>Incorrect labels from pre-trained models</li>
</ul>

<hr>

<h2>🚀 Future Work</h2>
<ul>
  <li>Train custom YOLOv8 model for indoor environments</li>
  <li>Expand dataset with real-world indoor objects</li>
  <li>Deploy model on mobile or embedded devices</li>
  <li>Add haptic or vibration feedback</li>
  <li>Conduct testing with visually impaired users</li>
</ul>

<hr>

<h2>✅ Conclusion</h2>
<p>
This project demonstrates how AI, Machine Learning, and Computer Vision can be combined to create an
effective assistive navigation system. The Faster R-CNN model delivered excellent accuracy and performance,
making the system practical for real-world indoor use. The solution promotes independence, safety
