const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const loader = document.getElementById("loader");
const resultDiv = document.getElementById("result");

imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];
    const reader = new FileReader();

    reader.onload = function (e) {
        preview.src = e.target.result;
        preview.style.display = "block";
    };

    reader.readAsDataURL(file);
});

function predictImage() {
    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    loader.style.display = "inline-block";
    resultDiv.innerHTML = "";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loader.style.display = "none";
        resultDiv.innerHTML = data.result;
    })
    .catch(error => {
        loader.style.display = "none";
        resultDiv.innerHTML = "Error occurred.";
    });
}
