document.getElementById('uploadForm').onsubmit = async function (e) {
    e.preventDefault();
    const file = e.target.file.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/predict', { method: 'POST', body: formData });
    const result = await response.json();
    document.getElementById('result').innerText = `Prediction: ${result.prediction}`;
};
