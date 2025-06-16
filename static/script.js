// document.getElementById('imageInput').addEventListener('change', function (e) {
//   const preview = document.getElementById('preview');
//   const file = e.target.files[0];
  
//   if (file) {
//     preview.src = URL.createObjectURL(file);
//     preview.hidden = false;
//   } else {
//     preview.src = "#";
//     preview.hidden = true;
//   }
// });

// document.getElementById('predictBtn').addEventListener('click', async () => {
//   const fileInput = document.getElementById('imageInput');
//   const resultBox = document.getElementById('result');

//   if (!fileInput.files[0]) {
//     resultBox.textContent = "⚠️ Please select an image first.";
//     return;
//   }

//   const formData = new FormData();
//   formData.append("image", fileInput.files[0]);

//   resultBox.textContent = "🔄 Predicting...";

//   try {
//     const response = await fetch("/predict", {
//       method: "POST",
//       body: formData
//     });

//     const data = await response.json();
//     resultBox.textContent = `📌 Result: ${data.result}`;
//   } catch (error) {
//     resultBox.textContent = "❌ Error during prediction.";
//   }
// });




// document.getElementById('imageInput').addEventListener('change', function (e) {
//   const preview = document.getElementById('preview');
//   const file = e.target.files[0];

//   if (file) {
//     preview.src = URL.createObjectURL(file);
//     preview.hidden = false;
//   } else {
//     preview.src = "#";
//     preview.hidden = true;
//   }
// });

// document.getElementById('predictBtn').addEventListener('click', async () => {
//   const fileInput = document.getElementById('imageInput');
//   const resultBox = document.getElementById('result');

//   if (!fileInput.files[0]) {
//     resultBox.textContent = "⚠️ Please select an image first.";
//     return;
//   }

//   const formData = new FormData();
//   formData.append("image", fileInput.files[0]);

//   resultBox.textContent = "🔄 Predicting...";

//   try {
//     const response = await fetch("/predict", {
//       method: "POST",
//       body: formData,
//     });

//     if (!response.ok) {
//       throw new Error("Server error");
//     }

//     const data = await response.json();
//     resultBox.textContent = `📌 Result: ${data.result}`;
//   } catch (error) {
//     resultBox.textContent = "❌ Error during prediction.";
//     console.error("Prediction error:", error);
//   }
// });

















// Preview image only after selection
document.getElementById('imageInput').addEventListener('change', function (e) {
  const file = e.target.files[0];
  const previewContainer = document.getElementById('previewContainer');
  previewContainer.innerHTML = ''; // Remove any existing preview

  if (file) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.alt = 'Image Preview';
    img.style.maxWidth = '100%';
    img.style.borderRadius = '8px';
    img.style.boxShadow = '0 0 10px rgba(0, 0, 0, 0.1)';
    previewContainer.appendChild(img);
  }
});

// Predict logic
document.getElementById('predictBtn').addEventListener('click', async () => {
  const fileInput = document.getElementById('imageInput');
  const resultBox = document.getElementById('result');

  if (!fileInput.files[0]) {
    resultBox.textContent = "⚠️ Please select an image first.";
    return;
  }

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  resultBox.textContent = "🔄 Predicting...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Server error");
    }

    const data = await response.json();
    resultBox.textContent = `📌 Result: ${data.result}`;
  } catch (error) {
    resultBox.textContent = "❌ Error during prediction.";
    console.error("Prediction error:", error);
  }
});
